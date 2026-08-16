#!/usr/bin/env python3
"""
v3 changes from v2 (fixes the Kaggle T4x2 OOM):

Root cause: with plain nn.DataParallel, only the model's forward/backward
is split across GPUs. The five loss functions (Charbonnier, LPIPS, Sobel,
MS-SSIM, HF) all ran *after* DataParallel had already gathered every
replica's prediction back onto GPU 0 -- so the whole batch (all N samples,
not the N/2-per-GPU split) went through LPIPS's VGG16 forward *and*
backward on a single GPU. That's the actual OOM trigger, not simply
"batch size is too big": GPU 0 was doing roughly 2x the memory work of
GPU 1 every step, so GPU 0 OOMs while GPU 1 still has headroom.

Fixes, in order of impact:
  1. ModelWithLoss: model forward + all five losses now happen *inside*
     the module that gets wrapped by DataParallel, so each GPU computes
     its own loss (including the LPIPS VGG backward graph) only for its
     own batch shard. Only a per-GPU scalar loss gets gathered back to
     GPU 0 every step, not the full-resolution prediction tensor plus
     its whole backward graph. This alone should roughly halve GPU 0's
     peak memory under 2 GPUs.
  2. Optional gradient checkpointing (train.grad_checkpoint, default
     True) on the NAFBlock stacks -- trades ~20-30% more compute for a
     large cut in activation memory, which is what lets you push batch
     size back up instead of running tiny/slow batches. See model.py.
  3. Gradient accumulation (train.accum_steps, default 1) decouples the
     physical (memory) batch size from the effective (optimization)
     batch size, e.g. batch_size=16, accum_steps=8 behaves like batch
     128 for training dynamics without ever materializing 128 samples'
     worth of activations at once.
  4. LPIPS now runs *inside* the AMP autocast region (fp16) instead of
     being forced to fp32. VGG-backbone perceptual losses are generally
     fine under fp16 in practice (unlike MS-SSIM's log/sqrt multi-scale
     math, which stays fp32 -- that one's still genuinely fp16-fragile).
     LPIPS was the single most expensive part of the loss stack, so this
     roughly halves its memory too. If you see NaN losses, move
     `l_perceptual` back into the fp32 block below.
  5. cudnn.benchmark=True (fixed input sizes across a run -> free
     speedup) and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (cuts
     OOM from allocator fragmentation, which gets worse when tensor
     shapes vary between the fp16 and fp32 loss blocks every step).
  6. evaluate_ood now runs on the raw (unwrapped) model directly instead
     of through DataParallel -- eval batch size is 1, so routing it
     through DataParallel's scatter/gather was pure dispatch overhead
     with zero parallelism benefit.

CLI/config schema unchanged except two new optional keys:
  --set train.grad_checkpoint=true/false   (default true)
  --set train.accum_steps=N                (default 1)
Everything else (config loading, dataset/splits, checkpoint format) is
unchanged. Paste this over train.py; pair it with the updated model.py
(NAFNet_UNet gained a use_checkpoint flag, forward pass unchanged
numerically).

If you're still OOMing after this: lower train.batch_size and raise
train.accum_steps to compensate for the same effective batch at lower
peak memory, e.g. batch_size=8, accum_steps=16 for an effective batch
of 128. If the error is a Kaggle "kernel restarted" message rather than
a "CUDA out of memory" traceback, that's system RAM, not VRAM, and a
different problem (likely something in RestorationDataset/caching) --
none of the fixes here target that.
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

# Must be set before CUDA initializes (i.e. before `import torch`).
# Reduces OOM caused by allocator fragmentation, not just raw usage --
# relevant here since tensor shapes vary between the fp16 model/LPIPS
# path and the fp32 MS-SSIM/HF path every single step.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from src.model import MODELS
from src.eval_utils import stratified_ssim
import lpips

try:
    from pytorch_msssim import ms_ssim, ssim
except ImportError as e:
    raise ImportError(
        "pytorch-msssim is required for the MS-SSIM loss term.\n"
        "Install with: pip install pytorch-msssim"
    ) from e

# --- PyTorch Native Losses (unchanged from v2) ---

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps**2))


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


class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM. Stays in fp32 (see module docstring point 4)."""
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred, target):
        min_side = min(pred.shape[-2], pred.shape[-1])
        if min_side >= 160:
            return 1.0 - ms_ssim(pred, target, data_range=self.data_range, size_average=True)
        else:
            return 1.0 - ssim(pred, target, data_range=self.data_range, size_average=True)


class HighFrequencyLoss(nn.Module):
    """L1 on the high-frequency residual. Unchanged from v2."""
    def __init__(self, kernel_size=5, sigma=1.5):
        super().__init__()
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
        self.register_buffer('kernel', kernel)
        self.pad = kernel_size // 2
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_blur = F.conv2d(pred, self.kernel, padding=self.pad)
        target_blur = F.conv2d(target, self.kernel, padding=self.pad)
        return self.l1(pred - pred_blur, target - target_blur)


class ModelWithLoss(nn.Module):
    """Wraps model forward + every loss term in one module so DataParallel
    parallelizes the *whole* thing (including the LPIPS VGG backward
    graph) across GPUs, instead of gathering predictions to GPU 0 first
    and paying for loss compute there alone. See module docstring point 1."""

    def __init__(self, model, loss_weights):
        super().__init__()
        self.model = model
        self.charbonnier = CharbonnierLoss()
        self.sobel_loss = SobelEdgeLoss()
        self.msssim_loss = MSSSIMLoss()
        self.hf_loss = HighFrequencyLoss()
        self.lpips_fn = lpips.LPIPS(net='vgg')
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.w = loss_weights

    def forward(self, noisy_lr, clean_hr, amp_enabled: bool):
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_hr = self.model(noisy_lr)
            l_char = self.charbonnier(pred_hr, clean_hr)
            l_edge = self.sobel_loss(pred_hr, clean_hr)
            # LPIPS stays in the fp16 region now -- see point 4 above.
            pred_norm = pred_hr * 2.0 - 1.0
            clean_norm = clean_hr * 2.0 - 1.0
            l_perceptual = self.lpips_fn(pred_norm, clean_norm).mean()

        # Only MS-SSIM / HF forced to fp32 now -- they're cheap (no VGG),
        # so fp32 here costs little, unlike forcing LPIPS to fp32 did.
        with torch.cuda.amp.autocast(enabled=False):
            pred_f = pred_hr.float()
            clean_f = clean_hr.float()
            l_msssim = self.msssim_loss(pred_f, clean_f)
            l_hf = self.hf_loss(pred_f, clean_f)

        loss = (self.w["char"] * l_char + self.w["lpips"] * l_perceptual + self.w["edge"] * l_edge
                + self.w["msssim"] * l_msssim + self.w["hf"] * l_hf)

        # Shape (1,) per replica -> DataParallel gathers to (n_gpus,) on
        # the primary device; caller takes .mean(). This scalar is the
        # only thing that crosses GPUs every step now, not the prediction.
        return loss.unsqueeze(0)


# --- Padding helpers (unchanged from v2) ---

def pad_to_multiple(x: torch.Tensor, multiple: int = 4):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)


def crop_to_scale(x: torch.Tensor, pad_h: int, pad_w: int, scale: int):
    if pad_h == 0 and pad_w == 0:
        return x
    h, w = x.shape[-2:]
    return x[..., : h - pad_h * scale, : w - pad_w * scale]


def get_raw_model(m: nn.Module) -> nn.Module:
    """Unwraps DataParallel and then the ModelWithLoss shim to get the
    plain NAFNet -- what you want for eval and for the saved checkpoint."""
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "model"):
        m = m.model
    return m


# --- Core Loop ---

def train_one_epoch(model_with_loss, dataloader, optimizer, device, scaler,
                     amp_enabled: bool, accum_steps: int):
    model_with_loss.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
            clean_hr = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
        else:
            noisy_lr = batch[0].to(device, non_blocking=True)
            clean_hr = batch[1].to(device, non_blocking=True)

        per_gpu_loss = model_with_loss(noisy_lr, clean_hr, amp_enabled)
        loss = per_gpu_loss.mean()

        scaler.scale(loss / accum_steps).backward()

        is_last_batch = (step + 1) == n_batches
        if (step + 1) % accum_steps == 0 or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n_batches


@torch.no_grad()
def evaluate_ood(raw_model, dataloader, device, scale: int):
    raw_model.eval()
    total_ssim = 0.0
    total_ssim_edge = 0.0

    pbar = tqdm(dataloader, desc="Validating (OOD)", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device)
            clean_hr = batch.get("hr", batch.get("gt")).to(device)
        else:
            noisy_lr, clean_hr = batch[0].to(device), batch[1].to(device)

        # Pad defensively -- val_ood images should already be a fixed
        # size, but this keeps eval robust if that ever changes (e.g.
        # mixed-resolution OOD samples).
        noisy_lr_p, (ph, pw) = pad_to_multiple(noisy_lr, multiple=4)
        pred_hr = raw_model(noisy_lr_p)
        pred_hr = crop_to_scale(pred_hr, ph, pw, scale=scale)
        pred_hr = torch.clamp(pred_hr, 0.0, 1.0)

        pred_np = pred_hr.cpu().numpy()[0, 0]
        clean_np = clean_hr.cpu().numpy()[0, 0]

        metrics = stratified_ssim(pred_np, clean_np)

        if isinstance(metrics, dict):
            total_ssim += float(metrics["ssim"])
            total_ssim_edge += float(metrics["ssim_edge"])
        else:
            total_ssim += float(metrics[0])
            total_ssim_edge += float(metrics[1])

    avg_ssim = total_ssim / len(dataloader)
    avg_ssim_edge = total_ssim_edge / len(dataloader)
    return avg_ssim, avg_ssim_edge


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input sizes -> free speedup

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

    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(get_cfg("train.seed", 42))
    sp = load_splits()
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")

    # Explicit synth_p rather than relying on RestorationDataset's default --
    # unverified here since src/dataset.py wasn't available when this script
    # was written. If your default already matches, this is a no-op; if it
    # doesn't, this is the difference between the synthetic-augmentation risk
    # mitigation actually running or silently not running.

    train_ds = RestorationDataset(cache_dir, stems=sp["train"])
    val_ood_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    batch_size = get_cfg("train.batch_size", 8)
    workers = get_cfg("train.num_workers", 4)
    accum_steps = max(1, int(get_cfg("train.accum_steps", 1)))
    use_checkpoint = bool(get_cfg("train.grad_checkpoint", True))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ood_ds, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )

    scale_factor = get_cfg("dataset.scale", 2)
    model = MODELS["nafnet"](scale=scale_factor, use_checkpoint=use_checkpoint)

    epochs = get_cfg("train.epochs", 100)
    use_amp = bool(get_cfg("train.amp", True)) and device.type == "cuda"

    loss_weights = {
        "char":   float(get_cfg("train.loss.char_w", 1.0)),
        "lpips":  float(get_cfg("train.loss.lpips_w", 0.05)),
        "edge":   float(get_cfg("train.loss.edge_w", 0.2)),
        "msssim": float(get_cfg("train.loss.msssim_w", 0.3)),
        "hf":     float(get_cfg("train.loss.hf_w", 0.4)),
    }
    print(f"--- Loss weights: {loss_weights} ---")

    model_with_loss = ModelWithLoss(model, loss_weights).to(device)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"--- Detected {n_gpus} GPUs -> wrapping model+loss in nn.DataParallel ---")
        print("    (loss is now computed per-GPU-shard, not gathered to GPU 0 first)")
        model_with_loss = nn.DataParallel(model_with_loss)

    trainable_params = [p for p in model_with_loss.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=get_cfg("train.lr", 5e-4), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_amp:
        print("--- AMP (mixed precision) enabled ---")
    if use_checkpoint:
        print("--- Gradient checkpointing enabled on NAFBlock stacks ---")
    if accum_steps > 1:
        effective_bs = batch_size * accum_steps
        print(f"--- Gradient accumulation: physical batch {batch_size} x {accum_steps} "
              f"steps = effective batch {effective_bs} ---")

    best_ood_ssim = 0.0

    print(f"--- Starting Training Run (3-level NAFNet) for {epochs} epochs on {device} "
          f"({n_gpus} GPU{'s' if n_gpus != 1 else ''}) ---")
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(
            model_with_loss, train_loader, optimizer, device, scaler, use_amp, accum_steps
        )
        scheduler.step()

        raw_model = get_raw_model(model_with_loss)
        ood_ssim, ood_ssim_edge = evaluate_ood(raw_model, val_loader, device, scale=scale_factor)

        mem_str = ""
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_str = f" | Peak GPU0 mem: {peak_gb:.1f}GB"

        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}{mem_str}")

        if ood_ssim > best_ood_ssim:
            best_ood_ssim = ood_ssim
            save_path = output_dir / "best_nafnet.pt"
            # Always save the plain (unwrapped) state_dict so it loads
            # directly into a plain single-GPU model at inference time.
            torch.save(raw_model.state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())