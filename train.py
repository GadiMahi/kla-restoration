#!/usr/bin/env python3
"""
Two-stage training driver for the NAFNet+MDTA speckle restoration model.

Usage:
    python train.py --config configs/base.yaml --stage 1
    python train.py --config configs/base.yaml --stage 2 \
        --overrides train.stage1_ckpt=/kaggle/working/kla-restoration/artifacts_stage1/best_nafnet.pt
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
        # FIX: Compute in float32 to prevent 1e-6 from rounding to 0 in bfloat16
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
# Helpers
# --------------------------------------------------------------------------

def _get_batch(batch, device):
    if isinstance(batch, dict):
        lr = batch.get("lr", batch.get("noisy")).to(device)
        hr = batch.get("hr", batch.get("gt")).to(device)
    else:
        lr, hr = batch[0].to(device), batch[1].to(device)
    return lr, hr

def _make_structural_loss(hr_hw, device, channel: int = 1):
    min_side = min(hr_hw)
    if min_side >= 160:
        fn = MS_SSIM(data_range=1.0, size_average=True, channel=channel).to(device)
        print(f"[structural loss] using MS-SSIM (HR patch min side = {min_side})")
    else:
        fn = SSIM(data_range=1.0, size_average=True, channel=channel).to(device)
        print(f"[structural loss] HR patch min side = {min_side} < 160, falling back to single-scale SSIM")
    return fn

@torch.no_grad()
def evaluate_ood(model, dataloader, device):
    model.eval()
    
    # FIX: Unwrap DataParallel for batch_size=1 evaluation to prevent NCCL crashes
    base_model = model.module if hasattr(model, "module") else model
    
    total_ssim = 0.0
    total_ssim_edge = 0.0

    pbar = tqdm(dataloader, desc="Validating (OOD)", leave=False)
    for batch in pbar:
        lr, hr = _get_batch(batch, device)
        pred = torch.clamp(base_model(lr), 0.0, 1.0)
        
        pred_np = pred.cpu().numpy()[0, 0]
        clean_np = hr.cpu().numpy()[0, 0]

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
# Training Loops
# --------------------------------------------------------------------------

def train_one_epoch_stage1(model, dataloader, optimizer, loss_fns, weights, device, use_amp):
    model.train()
    charbonnier, lpips_fn, sobel_loss, structural_fn = loss_fns
    w_char, w_lpips, w_edge, w_struct = weights
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Stage1 train", leave=False)
    for batch in pbar:
        lr, hr = _get_batch(batch, device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(lr)
            l_char = charbonnier(pred, hr)
            pred_norm, hr_norm = pred * 2.0 - 1.0, hr * 2.0 - 1.0
            l_lpips = lpips_fn(pred_norm, hr_norm).mean()
            l_edge = sobel_loss(pred, hr)
            l_struct = 1.0 - structural_fn(pred, hr)
            loss = w_char * l_char + w_lpips * l_lpips + w_edge * l_edge + w_struct * l_struct

        loss.backward()
        # FIX: Clip gradients to max norm of 1.0 to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)

def train_one_epoch_stage2(gen, disc, dataloader, opt_g, opt_d, loss_fns, weights, device, use_amp):
    gen.train()
    disc.train()
    charbonnier, lpips_fn, sobel_loss, structural_fn, ffl, ragan = loss_fns
    w_char, w_lpips, w_edge, w_struct, w_ffl, w_gan = weights
    total_g, total_d = 0.0, 0.0

    pbar = tqdm(dataloader, desc="Stage2 GAN train", leave=False)
    for batch in pbar:
        lr, hr = _get_batch(batch, device)

        # --- Discriminator step ---
        opt_d.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                fake_hr = gen(lr)
            real_logits = disc(hr)
            fake_logits = disc(fake_hr.detach())
            d_loss = ragan.discriminator_loss(real_logits, fake_logits)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        opt_d.step()

        # --- Generator step ---
        opt_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            fake_hr = gen(lr)
            real_logits = disc(hr).detach()
            fake_logits_for_g = disc(fake_hr)

            l_char = charbonnier(fake_hr, hr)
            fake_norm, hr_norm = fake_hr * 2.0 - 1.0, hr * 2.0 - 1.0
            l_lpips = lpips_fn(fake_norm, hr_norm).mean()
            l_edge = sobel_loss(fake_hr, hr)
            l_struct = 1.0 - structural_fn(fake_hr, hr)
            l_ffl = ffl(fake_hr, hr)
            l_gan = ragan.generator_loss(real_logits, fake_logits_for_g)

            g_loss = (w_char * l_char + w_lpips * l_lpips + w_edge * l_edge
                      + w_struct * l_struct + w_ffl * l_ffl + w_gan * l_gan)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt_g.step()

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
                    help="1 = pixel/structure prior, 2 = GAN fine-tune. Overrides train.stage in config if given.")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FIX: Absolute cuDNN bypass to prevent Depthwise Conv failures on T4s
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

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
    use_amp = bool(get_cfg("train.amp_bf16", True)) and device.type == "cuda"

    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(get_cfg("train.seed", 42))
    sp = load_splits()
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")

    train_ds = RestorationDataset(cache_dir, stems=sp["train"])
    val_ood_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    batch_size = get_cfg("train.batch_size", 8)
    workers = get_cfg("train.num_workers", 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ood_ds, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)

    scale_factor = get_cfg("dataset.scale", 2)
    model_kwargs = dict(
        scale=scale_factor,
        dim=get_cfg("model.dim", 64),
        num_transformer_blocks=get_cfg("model.num_transformer_blocks", 2),
        num_heads=get_cfg("model.num_heads", 4),
        use_log_input=get_cfg("model.use_log_input", True),
    )
    
    model = MODELS["nafnet"](**model_kwargs).to(device)
    if use_amp:
        model = model.to(memory_format=torch.channels_last)

    # 🚀 MULTI-GPU WRAPPER FOR GENERATOR
    if torch.cuda.device_count() > 1:
        print(f"Accelerating Generator with {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    sample_lr, sample_hr = _get_batch(next(iter(train_loader)), device)
    structural_fn = _make_structural_loss(sample_hr.shape[-2:], device)

    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad = False

    best_ood_ssim = 0.0

    if stage == 1:
        epochs = get_cfg("train.epochs", 100)
        # FIX: Force float casting for parsed YAML configurations
        optimizer = optim.AdamW(model.parameters(), lr=float(get_cfg("train.lr", 2e-4)), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        weights = (
            get_cfg("train.w_char", 1.0),
            get_cfg("train.w_lpips", 0.35),
            get_cfg("train.w_edge", 0.5),
            get_cfg("train.w_struct", 0.3),
        )
        loss_fns = (charbonnier, lpips_fn, sobel_loss, structural_fn)

        print(f"--- Stage 1: pixel/structure prior, {epochs} epochs on {device} (weights char/lpips/edge/struct = {weights}) ---")
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch_stage1(model, train_loader, optimizer, loss_fns, weights, device, use_amp)
            scheduler.step()

            ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device)
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}")

            if ood_ssim > best_ood_ssim:
                best_ood_ssim = ood_ssim
                save_path = output_dir / "best_nafnet.pt"
                # Safely extract state_dict bypassing DataParallel wrapper
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, save_path)
                print(f" -> Checkpoint saved to {save_path}")

    else:  # stage == 2
        stage1_ckpt = Path(get_cfg("train.stage1_ckpt", output_dir / "best_nafnet.pt"))
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"Stage 2 needs a Stage 1 checkpoint -- none found at {stage1_ckpt}.")
        
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f"Loaded Stage 1 weights from {stage1_ckpt}")

        disc = UNetDiscriminatorSN(in_channels=1, base_dim=get_cfg("model.disc_base_dim", 32)).to(device)
        if use_amp:
            disc = disc.to(memory_format=torch.channels_last)
            
        # 🚀 MULTI-GPU WRAPPER FOR DISCRIMINATOR
        if torch.cuda.device_count() > 1:
            print(f"Accelerating Discriminator with {torch.cuda.device_count()} GPUs!")
            disc = nn.DataParallel(disc)

        epochs = get_cfg("train.stage2_epochs", 20)
        # FIX: Force float casting for GAN learning rates
        lr_g = float(get_cfg("train.stage2_lr_g", 1e-4))
        lr_d = float(get_cfg("train.stage2_lr_d", 4e-4))
        opt_g = optim.AdamW(model.parameters(), lr=lr_g, weight_decay=1e-4)
        opt_d = optim.AdamW(disc.parameters(), lr=lr_d, weight_decay=1e-4)

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

        print(f"--- Stage 2: GAN fine-tune, {epochs} epochs on {device} (lr_g={lr_g}, lr_d={lr_d}, weights = {weights}) ---")
        for epoch in range(1, epochs + 1):
            g_loss, d_loss = train_one_epoch_stage2(model, disc, train_loader, opt_g, opt_d, loss_fns, weights, device, use_amp)

            ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device)
            print(f"Epoch {epoch:03d}/{epochs:03d} | G: {g_loss:.4f} | D: {d_loss:.4f} | OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}")

            if ood_ssim > best_ood_ssim:
                best_ood_ssim = ood_ssim
                save_path = output_dir / "best_nafnet_gan.pt"
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, save_path)
                print(f" -> Checkpoint saved to {save_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())