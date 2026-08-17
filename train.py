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

v2 fixes (diagnosed from a training run that OOM'd on Kaggle T4x2):
  6. Loss computation now happens INSIDE the DataParallel-wrapped module
     (see ModelWithLoss) instead of after gathering every GPU's output
     back to the default device. Previously Charbonnier/LPIPS/Sobel ran
     only on GPU0 for the *full* batch -- LPIPS in particular is a full
     VGG forward+backward -- which serialized most of the per-step cost
     onto GPU0 while GPU1 sat idle after its half of the forward pass.
     This is visible in the old logs as a persistent GPU0/GPU1 memory
     imbalance (~6.7GB vs ~2.9GB). Now each GPU computes its own local
     loss and only a small per-GPU scalar gets gathered.
  7. `model` (the bare NAFNet) is never itself wrapped in DataParallel
     any more -- only `ModelWithLoss(model, ...)` is. That means eval no
     longer needs `.module` unwrapping, and more importantly no longer
     pays DataParallel's per-call replicate-to-every-GPU overhead for
     727 batch-of-1 iterations that gain nothing from being split.
  8. torch.cuda.amp.GradScaler -> torch.amp.GradScaler('cuda', ...) to
     drop the FutureWarning; behavior is unchanged.

v3 fix (cudnn.benchmark + DataParallel threading race):
  9. cudnn.benchmark is now off everywhere (was: on for training, off for
     eval). Moving LPIPS/Charbonnier/Sobel inside ModelWithLoss (fix 6)
     means DataParallel's parallel_apply now runs LPIPS concurrently on
     BOTH GPUs' threads during training, not just on GPU0. cudnn's FIND
     algorithm search is not safe against two threads hitting a
     never-before-benchmarked shape at the same instant -- that's exactly
     what "FIND was unable to find an engine to execute this computation"
     on the very first step is. benchmark=True was never actually
     validated against concurrent multi-GPU forward passes before (LPIPS
     used to run single-threaded on GPU0 only); the "shape is fixed so
     benchmark is safe" reasoning in (3) didn't account for that. Given
     the forward pass is a small fraction of wall time here anyway (data
     loading dominates), losing autotuning is a low-cost trade for not
     crashing on step 1.

Root cause of the actual OOM (for the record): GPU memory was flat across
epochs in the logs, but host RSS climbed ~1GB/epoch with no plateau, and
the crash happened mid-epoch with no CUDA OOM traceback -- the signature
of the Linux/Kaggle OOM killer running out of *host* RAM, not GPU memory.
That was traced to src/dataset.py mmapping every group in index.json
regardless of the `stems` filter (fixed separately, see dataset.py).
Fixes 6-8 above are real efficiency/correctness fixes but are not what
caused that specific crash.
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


class ModelWithLoss(nn.Module):
    """Wraps the restoration model together with the loss functions so that,
    under nn.DataParallel, each GPU runs its own forward pass AND computes
    its own loss on its own local shard of the batch -- see fix (6) above.

    Only a (1,)-shaped per-GPU loss tensor and pred_hr get gathered back to
    the default device, instead of full activations/gradients for the whole
    batch funneling through one GPU's loss computation.
    """
    def __init__(self, model, charbonnier, lpips_fn, sobel_loss, use_amp):
        super().__init__()
        self.model = model
        self.charbonnier = charbonnier
        self.lpips_fn = lpips_fn
        self.sobel_loss = sobel_loss
        self.use_amp = use_amp

    def forward(self, noisy_lr, clean_hr):
        # autocast is thread-local and DataParallel dispatches each
        # replica's forward on its own thread via parallel_apply, so
        # autocast must be entered *inside* forward -- entering it at the
        # call site (outside DataParallel) would silently not apply to the
        # replica threads. Losses stay in fp32 for stability (LPIPS-vgg can
        # be flaky under fp16).
        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            pred_hr = self.model(noisy_lr)
        pred_hr = pred_hr.float()

        # 1. Pixel Fidelity Loss (Primary driver for contrast and exact intensities)
        l_char = self.charbonnier(pred_hr, clean_hr)

        # 2. Perceptual Loss (Scaled down to 0.05 so it doesn't destroy pixel contrast)
        pred_norm = pred_hr * 2.0 - 1.0
        clean_norm = clean_hr * 2.0 - 1.0
        l_perceptual = self.lpips_fn(pred_norm, clean_norm).mean()

        # 3. Structural Edge Loss
        l_edge = self.sobel_loss(pred_hr, clean_hr)

        # Balanced Loss Combination
        loss = (1.0 * l_char) + (0.05 * l_perceptual) + (0.5 * l_edge)

        # unsqueeze so DataParallel concatenates per-GPU scalars along dim 0
        # correctly regardless of device count; caller takes .mean().
        return loss.unsqueeze(0), pred_hr


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

def train_one_epoch(train_module, dataloader, optimizer, device,
                     epoch, scaler, log_interval):
    train_module.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
            clean_hr = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
        else:
            noisy_lr = batch[0].to(device, non_blocking=True)
            clean_hr = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        per_gpu_loss, pred_hr = train_module(noisy_lr, clean_hr)
        loss = per_gpu_loss.mean()

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
                if val is not None: return val
            except Exception: pass
        if hasattr(cfg, "get"):
            val = cfg.get(key)
            if val is not None: return val
        return default

    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(get_cfg("train.seed", 42))

    # Left off deliberately -- see fix (9) in the module docstring.
    # DataParallel now runs LPIPS concurrently on both GPUs' threads during
    # training (since it lives inside ModelWithLoss), and cudnn's benchmark
    # FIND search is not thread-safe against two GPUs hitting a
    # never-before-seen shape at the same instant -> "FIND was unable to
    # find an engine to execute this computation" on step 1.
    torch.backends.cudnn.benchmark = False

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
    val_loader = DataLoader(
        val_ood_ds, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True,
        # Validation only runs once per epoch, so the per-epoch worker
        # respawn cost is trivial -- persistent_workers=False here gives
        # each epoch's eval workers a clean slate instead of accumulating
        # state for the entire 100-epoch run.
        persistent_workers=False,
    )

    scale_factor = get_cfg("dataset.scale", 2)
    model = MODELS["nafnet"](scale=scale_factor).to(device)

    n_gpus = torch.cuda.device_count()
    multi_gpu = n_gpus > 1 and device.type == "cuda"

    if torch.cuda.is_available():
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU{i}: {props.name}, {_gb(props.total_memory):.1f}GB total")

    use_amp = bool(get_cfg("train.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    log_interval = int(get_cfg("train.log_interval", 50))
    print(f"--- Mixed precision (AMP): {use_amp} | memory log every {log_interval} steps ---")

    epochs = get_cfg("train.epochs", 100)
    # Optimizer sees only the bare NAFNet's parameters, same as before --
    # `model` is never itself wrapped in DataParallel now (only
    # ModelWithLoss(model, ...) is), so model.parameters() is unaffected.
    optimizer = optim.AdamW(model.parameters(), lr=get_cfg("train.lr", 5e-4), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False

    # See fix (6): wrap model+losses together so DataParallel splits the
    # *whole* per-step workload -- forward AND loss -- across both GPUs,
    # instead of gathering every GPU's output onto GPU0 and running
    # Charbonnier/LPIPS/Sobel there alone for the full batch.
    train_module = ModelWithLoss(model, charbonnier, lpips_fn, sobel_loss, use_amp)
    if multi_gpu:
        print(f"--- Detected {n_gpus} GPUs -> wrapping model+loss in nn.DataParallel "
              f"(effective per-GPU batch ~= {batch_size // n_gpus}) ---")
        train_module = nn.DataParallel(train_module, device_ids=list(range(n_gpus)))
    else:
        train_module = train_module.to(device)

    best_ood_ssim = 0.0

    print(f"--- Starting Training Run (High-Capacity U-Net NAFNet) for {epochs} epochs on {device} ---")
    for epoch in range(1, epochs + 1):
        reset_peak_memory_stats()
        epoch_start = time.time()

        train_loss = train_one_epoch(train_module, train_loader, optimizer, device,
                                      epoch, scaler, log_interval)
        scheduler.step()

        # `model` is the bare NAFNet -- never wrapped in DataParallel itself
        # (see fix 7), so it evaluates directly on a single GPU with no
        # `.module` unwrapping and no per-call replicate-to-every-GPU
        # overhead for these 727 batch-of-1 iterations. cudnn.benchmark is
        # left off (see fix 9) so no toggle is needed here any more.
        ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device, use_amp)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f} | "
              f"Time: {epoch_time:.1f}s")
        log_memory(tag=f"epoch {epoch} end")

        if ood_ssim > best_ood_ssim:
            best_ood_ssim = ood_ssim
            save_path = output_dir / "best_nafnet.pt"
            # model is always the unwrapped NAFNet now, so state_dict()
            # loads on 1-GPU / CPU with no "module." prefix mismatch --
            # no conditional unwrap needed any more.
            torch.save(model.state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())