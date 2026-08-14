#!/usr/bin/env python3
"""
Two-stage training driver for the NAFNet+MDTA speckle restoration model.
Tuned for a Kaggle "GPU T4 x2" accelerator.

Stage 1 (pixel/structure prior):
    Charbonnier + LPIPS + Sobel edge + MS-SSIM, generator only.

Stage 2 (GAN fine-tune), from the Stage 1 checkpoint:
    Adds UNetDiscriminatorSN + RaGAN + Focal Frequency Loss. Charbonnier is
    kept (reduced) as an anchor so the GAN can't reintroduce speckle-like
    artifacts while it pushes texture realism back in.

Usage:
    python train.py --config configs/base.yaml --stage 1
    python train.py --config configs/base.yaml --stage 2 \
        --overrides train.stage1_ckpt=/kaggle/working/kla-restoration/artifacts/best_nafnet.pt

--------------------------------------------------------------------------
FIXES vs. the previous version, specifically for T4 x2:

1. cuDNN was fully disabled (`cudnn.enabled = False`, `deterministic = True`,
   `benchmark = False`). That kills cuDNN's tensor-core conv kernels
   entirely -- on a T4 this is the single biggest performance bug in the
   old script. Now: `benchmark = True`, everything else left at default.

2. AMP dtype was hardcoded to bfloat16. T4 is Turing (compute capability
   7.5) and has **no native bf16 Tensor Core path** -- bf16 autocast
   there gives no speedup. T4's Tensor Cores are fp16-only. Fixed:
   auto-detect bf16 support and fall back to fp16 + GradScaler on
   Turing/Pascal cards. This makes the script correct on T4, P100, A100,
   and H100 without editing it.

3. `channels_last` was applied to the model but never to the input
   tensors, so cuDNN was silently reconverting layout every forward pass.
   Now applied to both LR and HR batches.

4. The discriminator was wrapped in `nn.DataParallel`. `spectral_norm`'s
   power-iteration buffer is updated in-place during forward, and
   DataParallel's per-call replica recreation makes that update unreliable
   across GPUs (the same class of issue that makes BatchNorm running
   stats unreliable under DataParallel without SyncBatchNorm). The
   discriminator is cheap -- it doesn't need multi-GPU. Fixed: only the
   generator is wrapped; the discriminator stays on a single device.

5. Added GradScaler (needed once we're on fp16 -- bf16 doesn't need one,
   fp16 does, because of its much smaller exponent range).

6. DataLoader given `persistent_workers` + `prefetch_factor` so two T4s
   (which chew through batches fast) don't sit idle waiting on a handful
   of Kaggle CPU workers.
--------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import add_config_args, load_config
from src.dataset import RestorationDataset
from src.splits import load_splits
from src.model import MODELS, UNetDiscriminatorSN
from src.eval_utils import stratified_ssim

import lpips
from pytorch_msssim import SSIM, MS_SSIM

# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred.float() - target.float()
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_x = F.conv2d(pred, self.weight_x, padding=1)
        pred_y = F.conv2d(pred, self.weight_y, padding=1)
        target_x = F.conv2d(target, self.weight_x, padding=1)
        target_y = F.conv2d(target, self.weight_y, padding=1)
        return self.l1(pred_x, target_x) + self.l1(pred_y, target_y)


class FocalFrequencyLoss(nn.Module):
    """Jiang et al. 2021 -- weighted L2 distance in the 2D FFT domain.
    Always computed in float32: cuFFT's fp16 path is unreliable/unsupported
    on several driver versions, and FFT error accumulation is exactly the
    kind of thing you don't want happening in reduced precision.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        pred_freq = torch.fft.rfft2(pred.float(), norm="ortho")
        target_freq = torch.fft.rfft2(target.float(), norm="ortho")
        diff = pred_freq - target_freq
        freq_distance = diff.real ** 2 + diff.imag ** 2

        weight = freq_distance.clone().detach() ** self.alpha
        weight = weight / (weight.amax(dim=(-2, -1), keepdim=True) + self.eps)

        return (weight * freq_distance).mean()


class RelativisticAverageGANLoss(nn.Module):
    """RaGAN (ESRGAN / Real-ESRGAN)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits, fake_logits):
        real_rel = real_logits - fake_logits.mean(dim=0, keepdim=True)
        fake_rel = fake_logits - real_logits.mean(dim=0, keepdim=True)
        loss_real = self.bce(real_rel, torch.ones_like(real_rel))
        loss_fake = self.bce(fake_rel, torch.zeros_like(fake_rel))
        return (loss_real + loss_fake) * 0.5

    def generator_loss(self, real_logits, fake_logits):
        real_rel = real_logits - fake_logits.mean(dim=0, keepdim=True)
        fake_rel = fake_logits - real_logits.mean(dim=0, keepdim=True)
        loss_real = self.bce(real_rel, torch.zeros_like(real_rel))
        loss_fake = self.bce(fake_rel, torch.ones_like(fake_rel))
        return (loss_real + loss_fake) * 0.5


# --------------------------------------------------------------------------
# AMP / device helpers
# --------------------------------------------------------------------------

def get_amp_dtype(device: torch.device) -> torch.dtype:
    """bf16 Tensor Cores require Ampere+ (compute capability >= 8.0).
    torch.cuda.is_bf16_supported() is NOT a reliable check for this --
    it falls back to 'can I allocate a bf16 tensor at all', which
    succeeds on Turing/Volta via unaccelerated software paths and will
    happily tell you bf16 is 'supported' on a T4. Check the hardware
    generation directly instead.
    """
    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8:  # Ampere (A100/RTX30xx+), Hopper (H100), etc.
        return torch.bfloat16
    return torch.float16  # Turing (T4), Volta (V100), Pascal (P100)


def to_channels_last(*tensors, enabled: bool):
    if not enabled:
        return tensors if len(tensors) > 1 else tensors[0]
    out = tuple(t.to(memory_format=torch.channels_last) for t in tensors)
    return out if len(out) > 1 else out[0]


def print_gpu_summary():
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available -- running on CPU.")
        return
    n = torch.cuda.device_count()
    print(f"[gpu] {n} CUDA device(s) visible:")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {props.name} | {props.total_memory / 1e9:.1f} GB | "
              f"sm_{props.major}{props.minor}")
    if n < 2:
        print("[gpu] Only 1 GPU visible -- check Notebook Settings -> "
              "Accelerator is set to 'GPU T4 x2' if you expected 2.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get_batch(batch, device, channels_last: bool = False):
    if isinstance(batch, dict):
        lr = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
        hr = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
    else:
        lr = batch[0].to(device, non_blocking=True)
        hr = batch[1].to(device, non_blocking=True)
    if channels_last:
        lr, hr = to_channels_last(lr, hr, enabled=True)
    return lr, hr


def _make_structural_loss(hr_hw, device, channel: int = 1):
    """MS-SSIM needs >=~160px on the short side; falls back to single-scale
    SSIM on smaller crops instead of erroring out."""
    min_side = min(hr_hw)
    if min_side >= 160:
        fn = MS_SSIM(data_range=1.0, size_average=True, channel=channel).to(device)
        print(f"[structural loss] using MS-SSIM (HR patch min side = {min_side})")
    else:
        fn = SSIM(data_range=1.0, size_average=True, channel=channel).to(device)
        print(f"[structural loss] HR patch min side = {min_side} < 160, "
              f"falling back to single-scale SSIM")
    return fn


@torch.no_grad()
def evaluate_ood(model, dataloader, device, amp_dtype, use_amp, channels_last):
    model.eval()
    total_ssim = 0.0
    total_ssim_edge = 0.0

    pbar = tqdm(dataloader, desc="Validating (OOD)", leave=False)
    for batch in pbar:
        lr, hr = _get_batch(batch, device, channels_last)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = torch.clamp(model(lr), 0.0, 1.0)

        pred_np = pred.float().cpu().numpy()[0, 0]
        clean_np = hr.float().cpu().numpy()[0, 0]

        metrics = stratified_ssim(pred_np, clean_np)
        if isinstance(metrics, dict):
            total_ssim += float(metrics["ssim"])
            total_ssim_edge += float(metrics["ssim_edge"])
        else:
            total_ssim += float(metrics[0])
            total_ssim_edge += float(metrics[1])

    n = len(dataloader)
    return total_ssim / n, total_ssim_edge / n


# --------------------------------------------------------------------------
# Stage 1: pixel / structure prior
# --------------------------------------------------------------------------

def train_one_epoch_stage1(model, dataloader, optimizer, scaler, loss_fns, weights,
                            device, amp_dtype, use_amp, channels_last):
    model.train()
    charbonnier, lpips_fn, sobel_loss, structural_fn = loss_fns
    w_char, w_lpips, w_edge, w_struct = weights
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Stage1 train", leave=False)
    for batch in pbar:
        lr, hr = _get_batch(batch, device, channels_last)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(lr)
            l_char = charbonnier(pred, hr)
            pred_norm, hr_norm = pred * 2.0 - 1.0, hr * 2.0 - 1.0
            l_lpips = lpips_fn(pred_norm, hr_norm).mean()
            l_edge = sobel_loss(pred, hr)
            l_struct = 1.0 - structural_fn(pred.float(), hr.float())
            loss = w_char * l_char + w_lpips * l_lpips + w_edge * l_edge + w_struct * l_struct

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


# --------------------------------------------------------------------------
# Stage 2: GAN fine-tune
# --------------------------------------------------------------------------

def train_one_epoch_stage2(gen, disc, dataloader, opt_g, opt_d, scaler_g, scaler_d,
                            loss_fns, weights, device, amp_dtype, use_amp, channels_last):
    gen.train()
    disc.train()
    charbonnier, lpips_fn, sobel_loss, structural_fn, ffl, ragan = loss_fns
    w_char, w_lpips, w_edge, w_struct, w_ffl, w_gan = weights
    total_g, total_d = 0.0, 0.0

    pbar = tqdm(dataloader, desc="Stage2 GAN train", leave=False)
    for batch in pbar:
        lr, hr = _get_batch(batch, device, channels_last)

        # --- Discriminator step ---
        opt_d.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                fake_hr = gen(lr)
            real_logits = disc(hr)
            fake_logits = disc(fake_hr.detach())
            d_loss = ragan.discriminator_loss(real_logits, fake_logits)
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- Generator step ---
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            fake_hr = gen(lr)
            # Reuse real_logits from the D step (detached; the real image
            # doesn't depend on the generator) instead of a second disc(hr).
            fake_logits_for_g = disc(fake_hr)

            l_char = charbonnier(fake_hr, hr)
            fake_norm, hr_norm = fake_hr * 2.0 - 1.0, hr * 2.0 - 1.0
            l_lpips = lpips_fn(fake_norm, hr_norm).mean()
            l_edge = sobel_loss(fake_hr, hr)
            l_struct = 1.0 - structural_fn(fake_hr.float(), hr.float())
            l_ffl = ffl(fake_hr, hr)
            l_gan = ragan.generator_loss(real_logits.detach(), fake_logits_for_g)

            g_loss = (w_char * l_char + w_lpips * l_lpips + w_edge * l_edge
                      + w_struct * l_struct + w_ffl * l_ffl + w_gan * l_gan)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        total_g += g_loss.item()
        total_d += d_loss.item()
        pbar.set_postfix({"g": f"{g_loss.item():.4f}", "d": f"{d_loss.item():.4f}"})

    n = len(dataloader)
    return total_g / n, total_d / n


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None,
                    help="1 = pixel/structure prior, 2 = GAN fine-tune. "
                         "Overrides train.stage in config if given.")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_summary()

    if device.type == "cuda":
        # FIX: previous version disabled cuDNN entirely -- forces slow
        # fallback kernels for every conv. Fixed patch sizes per run make
        # benchmark=True a straightforward win.
        torch.backends.cudnn.benchmark = True

    def get_cfg(key, default=None):
        if hasattr(cfg, "get_path"):
            try:
                val = cfg.get_path(key)
                if val is not None:
                    return val
            except Exception:
                pass
        if hasattr(cfg, "get"):
            val = cfg.get(key)
            if val is not None:
                return val
        return default

    stage = args.stage if args.stage is not None else int(get_cfg("train.stage", 1))

    # FIX: auto-detect AMP dtype instead of hardcoding bfloat16. T4 (Turing,
    # sm_75) has no bf16 Tensor Core path; fp16 + GradScaler is correct
    # there, bf16 (no scaler needed) is picked automatically on Ampere+.
    use_amp = bool(get_cfg("train.use_amp", get_cfg("train.amp_bf16", True))) and device.type == "cuda"
    amp_dtype = get_amp_dtype(device) if use_amp else torch.float32
    scaler_enabled = use_amp and amp_dtype == torch.float16
    print(f"[amp] enabled={use_amp} dtype={amp_dtype} grad_scaler={scaler_enabled}")

    # FIX: channels_last must also be applied to input tensors, not just
    # the model, or cuDNN silently reconverts layout every forward pass.
    channels_last = bool(get_cfg("train.channels_last", True)) and device.type == "cuda"

    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(get_cfg("train.seed", 42))
    sp = load_splits()
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")

    train_ds = RestorationDataset(cache_dir, stems=sp["train"])
    val_ood_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    num_gpus = max(torch.cuda.device_count(), 1)
    # Default batch size scales with visible GPUs so each T4 gets a
    # reasonable per-replica chunk -- raise if you have memory headroom.
    default_batch = 8 * num_gpus
    batch_size = get_cfg("train.batch_size", default_batch)
    workers = get_cfg("train.num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ood_ds, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )

    scale_factor = get_cfg("dataset.scale", 2)
    model_kwargs = dict(
        scale=scale_factor,
        dim=get_cfg("model.dim", 64),
        num_transformer_blocks=get_cfg("model.num_transformer_blocks", 2),
        num_heads=get_cfg("model.num_heads", 4),
        use_log_input=get_cfg("model.use_log_input", True),
    )

    model = MODELS["nafnet"](**model_kwargs).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Multi-GPU: wrap only the generator (see docstring point 4 for why the
    # discriminator stays single-device).
    if num_gpus > 1:
        print(f"[gpu] Wrapping generator with nn.DataParallel across {num_gpus} GPUs "
              f"(effective batch size {batch_size} = {batch_size // num_gpus} per GPU).")
        model = nn.DataParallel(model)

    sample_lr, sample_hr = _get_batch(next(iter(train_loader)), device, channels_last)
    structural_fn = _make_structural_loss(sample_hr.shape[-2:], device)

    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad = False

    best_ood_ssim = 0.0

    if stage == 1:
        epochs = get_cfg("train.epochs", 100)
        optimizer = optim.AdamW(model.parameters(), lr=get_cfg("train.lr", 2e-4), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        weights = (
            get_cfg("train.w_char", 1.0),
            get_cfg("train.w_lpips", 0.35),
            get_cfg("train.w_edge", 0.5),
            get_cfg("train.w_struct", 0.3),
        )
        loss_fns = (charbonnier, lpips_fn, sobel_loss, structural_fn)

        print(f"--- Stage 1: pixel/structure prior, {epochs} epochs on {device} "
              f"(weights char/lpips/edge/struct = {weights}) ---")
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch_stage1(
                model, train_loader, optimizer, scaler, loss_fns, weights,
                device, amp_dtype, use_amp, channels_last)
            scheduler.step()

            ood_ssim, ood_ssim_edge = evaluate_ood(
                model, val_loader, device, amp_dtype, use_amp, channels_last)
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
                  f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}")

            if ood_ssim > best_ood_ssim:
                best_ood_ssim = ood_ssim
                save_path = output_dir / "best_nafnet.pt"
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, save_path)
                print(f" -> Checkpoint saved to {save_path}")

    else:  # stage == 2
        stage1_ckpt = Path(get_cfg("train.stage1_ckpt", output_dir / "best_nafnet.pt"))
        if not stage1_ckpt.exists():
            raise FileNotFoundError(
                f"Stage 2 needs a Stage 1 checkpoint -- none found at {stage1_ckpt}. "
                f"Run `--stage 1` first, or pass train.stage1_ckpt=<path> via --overrides.")

        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f"Loaded Stage 1 weights from {stage1_ckpt}")

        disc = UNetDiscriminatorSN(in_channels=1, base_dim=get_cfg("model.disc_base_dim", 32)).to(device)
        if channels_last:
            disc = disc.to(memory_format=torch.channels_last)
        # NOTE: intentionally NOT wrapped in DataParallel -- see docstring.

        epochs = get_cfg("train.stage2_epochs", 20)
        lr_g = get_cfg("train.stage2_lr_g", 1e-4)
        lr_d = get_cfg("train.stage2_lr_d", 4e-4)  # TTUR: D learns faster than G
        opt_g = optim.AdamW(model.parameters(), lr=lr_g, weight_decay=1e-4)
        opt_d = optim.AdamW(disc.parameters(), lr=lr_d, weight_decay=1e-4)
        scaler_g = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        scaler_d = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        weights = (
            get_cfg("train.stage2_w_char", 0.5),
            get_cfg("train.stage2_w_lpips", 0.4),
            get_cfg("train.stage2_w_edge", 0.3),
            get_cfg("train.stage2_w_struct", 0.2),
            get_cfg("train.stage2_w_ffl", 0.5),
            get_cfg("train.stage2_w_gan", 0.1),
        )
        ffl = FocalFrequencyLoss().to(device)
        ragan = RelativisticAverageGANLoss().to(device)
        loss_fns = (charbonnier, lpips_fn, sobel_loss, structural_fn, ffl, ragan)

        print(f"--- Stage 2: GAN fine-tune, {epochs} epochs on {device} "
              f"(lr_g={lr_g}, lr_d={lr_d}, weights char/lpips/edge/struct/ffl/gan = {weights}) ---")
        for epoch in range(1, epochs + 1):
            g_loss, d_loss = train_one_epoch_stage2(
                model, disc, train_loader, opt_g, opt_d, scaler_g, scaler_d,
                loss_fns, weights, device, amp_dtype, use_amp, channels_last)

            ood_ssim, ood_ssim_edge = evaluate_ood(
                model, val_loader, device, amp_dtype, use_amp, channels_last)
            print(f"Epoch {epoch:03d}/{epochs:03d} | G: {g_loss:.4f} | D: {d_loss:.4f} | "
                  f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}")

            if ood_ssim > best_ood_ssim:
                best_ood_ssim = ood_ssim
                save_path = output_dir / "best_nafnet_gan.pt"
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, save_path)
                print(f" -> Checkpoint saved to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())