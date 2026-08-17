#!/usr/bin/env python3
"""
T4x2 Kaggle changes (see inline comments for details):
  1. Multi-GPU: wraps the model in nn.DataParallel when >1 GPU is visible,
     so `train.batch_size=32` gets split ~16/16 across the two T4s.
     (DataParallel, not DDP -- this keeps the single `!python train.py`
     invocation working with no torchrun/launcher needed.)
  2. Mixed precision (autocast + GradScaler) on the NAFNet forward pass,
     which is the expensive part -- perceptual/edge/pixel losses are
     computed in fp32 for stability (LPIPS-vgg can be flaky under fp16).
  3. cudnn.benchmark=True, since batch shape is fixed every step
     (drop_last=True), so cudnn's autotuned kernels pay off immediately.
  4. GPU (+ host RSS) memory logging: once every `train.log_interval`
     steps during training, and a summary at the end of every epoch.
  5. Checkpoint saving unwraps `.module` when DataParallel is active, so
     the saved state_dict loads on 1-GPU / CPU without a "module." prefix
     mismatch.
"""
from __future__ import annotations
import argparse
import sys
import os

# Must be set before the first CUDA call (not necessarily before `import
# torch`, but doing it here is simplest) -- reduces allocator fragmentation
# on 16GB T4s under AMP's mixed alloc sizes. Harmless if already set.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
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
    import resource  # POSIX only (fine on Kaggle's Linux images)
    _HAVE_RESOURCE = True
except ImportError:
    _HAVE_RESOURCE = False

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


# --- Memory logging helpers ---

def _gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def reset_peak_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def log_memory(tag: str = "", printer=print):
    """Logs per-GPU allocated/reserved/peak memory plus host peak RSS."""
    parts = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = _gb(torch.cuda.memory_allocated(i))
            reserved = _gb(torch.cuda.memory_reserved(i))
            peak = _gb(torch.cuda.max_memory_allocated(i))
            parts.append(f"GPU{i} alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")
    host = ""
    if _HAVE_RESOURCE:
        # ru_maxrss is KB on Linux, bytes on macOS -- Kaggle is Linux.
        host_gb = _gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        host = f" | host_peak_rss={host_gb:.2f}GB"
    label = f"[mem{' ' + tag if tag else ''}] "
    printer(label + (" | ".join(parts) if parts else "no CUDA device") + host)


# --- Core Loop ---

def train_one_epoch(model, dataloader, optimizer, loss_fns, device,
                     epoch, use_amp, scaler, log_interval):
    model.train()
    total_loss = 0.0
    charbonnier, lpips_fn, sobel_loss = loss_fns

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
            clean_hr = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
        else:
            noisy_lr = batch[0].to(device, non_blocking=True)
            clean_hr = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Only the NAFNet forward pass runs under autocast -- it's the
        # dominant cost. Losses (esp. LPIPS-vgg) are computed in fp32.
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred_hr = model(noisy_lr)
        pred_hr = pred_hr.float()

        # 1. Pixel Fidelity Loss (Primary driver for contrast and exact intensities)
        l_char = charbonnier(pred_hr, clean_hr)

        # 2. Perceptual Loss (Scaled down to 0.05 so it doesn't destroy pixel contrast)
        pred_norm = pred_hr * 2.0 - 1.0
        clean_norm = clean_hr * 2.0 - 1.0
        l_perceptual = lpips_fn(pred_norm, clean_norm).mean()

        # 3. Structural Edge Loss
        l_edge = sobel_loss(pred_hr, clean_hr)

        # Balanced Loss Combination
        loss = (1.0 * l_char) + (0.05 * l_perceptual) + (0.5 * l_edge)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if log_interval > 0 and (step + 1) % log_interval == 0:
            log_memory(tag=f"epoch {epoch} step {step + 1}/{len(dataloader)}", printer=tqdm.write)

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_ood(model, dataloader, device, use_amp):
    model.eval()
    total_ssim = 0.0
    total_ssim_edge = 0.0

    pbar = tqdm(dataloader, desc="Validating (OOD)", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
            clean_hr = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
        else:
            noisy_lr = batch[0].to(device, non_blocking=True)
            clean_hr = batch[1].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred_hr = model(noisy_lr)
        pred_hr = torch.clamp(pred_hr.float(), 0.0, 1.0)

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

    # Fixed batch shape every step (drop_last=True below) -> let cudnn
    # autotune and reuse the fastest conv algorithms instead of picking
    # generic ones. Big win on T4s for the strided/PixelShuffle convs.
    torch.backends.cudnn.benchmark = True

    sp = load_splits()
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")

    train_ds = RestorationDataset(cache_dir, stems=sp["train"])
    val_ood_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    batch_size = get_cfg("train.batch_size", 8)
    workers = get_cfg("train.num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=workers > 0,
    )

    val_workers = min(workers, 1)
    val_loader = DataLoader(
        val_ood_ds, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )

    scale_factor = get_cfg("dataset.scale", 2)
    model = MODELS["nafnet"](scale=scale_factor).to(device)

    n_gpus = torch.cuda.device_count()
    multi_gpu = n_gpus > 1 and device.type == "cuda"
    if multi_gpu:
        print(f"--- Detected {n_gpus} GPUs -> wrapping model in nn.DataParallel "
              f"(effective per-GPU batch ~= {batch_size // n_gpus}) ---")
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))

    if torch.cuda.is_available():
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU{i}: {props.name}, {_gb(props.total_memory):.1f}GB total")

    use_amp = bool(get_cfg("train.amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_interval = int(get_cfg("train.log_interval", 50))
    print(f"--- Mixed precision (AMP): {use_amp} | memory log every {log_interval} steps ---")

    epochs = get_cfg("train.epochs", 100)
    optimizer = optim.AdamW(model.parameters(), lr=get_cfg("train.lr", 5e-4), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    msssim_loss = MSSSIMLoss().to(device)
    hf_loss = HighFrequencyLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False

    loss_fns = (charbonnier, lpips_fn, sobel_loss)
    best_ood_ssim = 0.0

    print(f"--- Starting Training Run (High-Capacity U-Net NAFNet) for {epochs} epochs on {device} ---")
    for epoch in range(1, epochs + 1):
        reset_peak_memory_stats()
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fns, device,
                                      epoch, use_amp, scaler, log_interval)
        scheduler.step()

        ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device, use_amp)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f} | "
              f"Time: {epoch_time:.1f}s")
        log_memory(tag=f"epoch {epoch} end")

        if ood_ssim > best_ood_ssim:
            best_ood_ssim = ood_ssim
            save_path = output_dir / "best_nafnet.pt"
            state_dict = model.module.state_dict() if multi_gpu else model.state_dict()
            torch.save(state_dict, save_path)
            print(f" -> Checkpoint saved to {save_path}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log_mem(f"after epoch {epoch:03d}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())