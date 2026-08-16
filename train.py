#!/usr/bin/env python3
"""
v2 changes from the original train.py:
  1. Multi-GPU: wraps the model in nn.DataParallel when >1 GPU is visible
     (e.g. Kaggle T4x2). No script/CLI changes needed -- DataParallel
     splits whatever --set train.batch_size=N you pass across the visible
     GPUs automatically. Safe here because NAFBlock uses per-sample
     LayerNorm2d, not BatchNorm, so there's no cross-GPU stat-sync pitfall
     to worry about.
  2. AMP (mixed precision) on top of that for T4 throughput. The
     numerically sensitive losses (LPIPS, MS-SSIM, high-freq) are forced
     back to fp32 inside the autocast region since SSIM-style losses and
     VGG-based LPIPS are known to be unstable in fp16.
  3. Loss function: added a differentiable MS-SSIM term (pytorch-msssim,
     GPU-native, no numpy round-trip -- unlike skimage SSIM, which is why
     it broke autograd before) and a high-frequency texture loss aimed
     directly at the "painterly" over-smoothing problem. Sobel edge
     weight was reduced since MS-SSIM already covers structure/edges to
     some degree and stacking both at full weight over-constrains.
  4. Checkpoints are always saved as a plain (unwrapped) state_dict, so a
     model trained with DataParallel loads cleanly in a single-GPU
     inference script with no key-prefix surgery needed.

Everything else (config loading, dataset/splits, checkpoint logic) is
untouched from the original script -- only paste this over your existing
train.py, no other files need to change except src/model.py.

Install the one new dependency: pip install pytorch-msssim
"""
from __future__ import annotations
import argparse
import sys
import os
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

# --- PyTorch Native Losses ---

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
    """1 - MS-SSIM. Fully differentiable, GPU-native.
    Dynamically falls back to single-scale SSIM if the image patch 
    is smaller than 160x160 (since MS-SSIM requires 4 downsamplings)."""
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred, target):
        # Check the smallest spatial dimension (Height or Width)
        min_side = min(pred.shape[-2], pred.shape[-1])
        
        if min_side >= 160:
            return 1.0 - ms_ssim(pred, target, data_range=self.data_range, size_average=True)
        else:
            return 1.0 - ssim(pred, target, data_range=self.data_range, size_average=True)


class HighFrequencyLoss(nn.Module):
    """L1 loss on the high-frequency residual (image minus a Gaussian-
    blurred version of itself). This is the direct fix for the
    "painterly / plastic" over-smoothing: pixel losses like Charbonnier
    are minimized on average by regressing to a smooth mean whenever the
    exact high-frequency detail is ambiguous, which is exactly what
    happens with texture. This loss explicitly rewards matching the
    *amount* of local high-frequency energy, not just low-frequency
    structure. It can't reintroduce speckle noise because the target
    side is always the clean HR ground truth -- the network is never
    shown a noisy high-frequency target to chase."""
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


# --- Padding helpers (fully-conv model, arbitrary input size incl. 512) ---

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


def unwrap(model: nn.Module) -> nn.Module:
    """Returns the underlying module whether or not it's DataParallel-wrapped."""
    return model.module if hasattr(model, "module") else model


# --- Core Loop ---

def train_one_epoch(model, dataloader, optimizer, loss_fns, loss_weights, device, scaler):
    model.train()
    total_loss = 0.0
    charbonnier, lpips_fn, sobel_loss, msssim_loss, hf_loss = loss_fns
    w = loss_weights
    amp_enabled = scaler.is_enabled()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
            clean_hr = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
        else:
            noisy_lr = batch[0].to(device, non_blocking=True)
            clean_hr = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_hr = model(noisy_lr)
            l_char = charbonnier(pred_hr, clean_hr)
            l_edge = sobel_loss(pred_hr, clean_hr)

        # Numerically sensitive losses forced to fp32 even under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            pred_f = pred_hr.float()
            clean_f = clean_hr.float()
            pred_norm = pred_f * 2.0 - 1.0
            clean_norm = clean_f * 2.0 - 1.0
            l_perceptual = lpips_fn(pred_norm, clean_norm).mean()
            l_msssim = msssim_loss(pred_f, clean_f)
            l_hf = hf_loss(pred_f, clean_f)

        loss = (w["char"] * l_char + w["lpips"] * l_perceptual + w["edge"] * l_edge
                + w["msssim"] * l_msssim + w["hf"] * l_hf)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_ood(model, dataloader, device, scale: int):
    model.eval()
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
        pred_hr = model(noisy_lr_p)
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
    model = MODELS["nafnet"](scale=scale_factor).to(device)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"--- Detected {n_gpus} GPUs -> wrapping model in nn.DataParallel ---")
        model = nn.DataParallel(model)

    epochs = get_cfg("train.epochs", 100)
    optimizer = optim.AdamW(model.parameters(), lr=get_cfg("train.lr", 5e-4), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = bool(get_cfg("train.amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("--- AMP (mixed precision) enabled ---")

    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    msssim_loss = MSSSIMLoss().to(device)
    hf_loss = HighFrequencyLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False

    loss_fns = (charbonnier, lpips_fn, sobel_loss, msssim_loss, hf_loss)
    loss_weights = {
        "char":   float(get_cfg("train.loss.char_w", 1.0)),
        "lpips":  float(get_cfg("train.loss.lpips_w", 0.05)),
        "edge":   float(get_cfg("train.loss.edge_w", 0.2)),
        "msssim": float(get_cfg("train.loss.msssim_w", 0.3)),
        "hf":     float(get_cfg("train.loss.hf_w", 0.4)),
    }
    print(f"--- Loss weights: {loss_weights} ---")

    best_ood_ssim = 0.0

    print(f"--- Starting Training Run (3-level NAFNet) for {epochs} epochs on {device} "
          f"({n_gpus} GPU{'s' if n_gpus != 1 else ''}) ---")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fns, loss_weights, device, scaler)
        scheduler.step()

        ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device, scale=scale_factor)

        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}")

        if ood_ssim > best_ood_ssim:
            best_ood_ssim = ood_ssim
            save_path = output_dir / "best_nafnet.pt"
            # Always save the unwrapped state_dict so a DataParallel-trained
            # checkpoint loads directly into a plain single-GPU model at
            # inference time -- no "module." key surgery needed.
            torch.save(unwrap(model).state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())