#!/usr/bin/env python3
"""
Training script for joint denoising + NxSR (KLA hackathon).

Two-stage curriculum (answers "how do I stop the painterly/over-smoothed
texture" without touching the backbone, so H100 inference speed is
unaffected):

  Stage 1 (--stage 1): fidelity-first. Gets contrast/geometry/macro-structure
      right and gives the network a stable starting point. Heavy Charbonnier,
      light everything else.
  Stage 2 (--stage 2): texture recovery. Fine-tunes the stage-1 checkpoint
      with more weight on SSIM / perceptual / high-frequency terms to recover
      the micro-texture stage 1 tends to over-smooth. Requires --resume
      pointing at the stage-1 checkpoint, and uses a lower LR.

Runs on 1 or 2 GPUs automatically via nn.DataParallel -- built for Kaggle's
T4x2 runtime. Uses AMP (fp16) so a batch of 128 at dim=64 fits comfortably
across two 16GB T4s.

NOTE ON THE CONFIG SYSTEM: this version uses plain argparse instead of your
previous add_config_args/load_config layer, since that module wasn't
provided and its --set-based override syntax didn't actually match anything
train.py was reading (there's no --stage handling anywhere in the original
loop). If you still want the YAML config layer for other scripts, keep it --
just point it at this script's flags below instead of the old ones.
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset import RestorationDataset
from src.splits import load_splits
from src.model import MODELS
from src.eval_utils import stratified_ssim
import lpips

try:
    from pytorch_msssim import SSIM
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


# --- Losses ---

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("wx", kx)
        self.register_buffer("wy", ky)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        px, py = F.conv2d(pred, self.wx, padding=1), F.conv2d(pred, self.wy, padding=1)
        tx, ty = F.conv2d(target, self.wx, padding=1), F.conv2d(target, self.wy, padding=1)
        return self.l1(px, tx) + self.l1(py, ty)


class FrequencyLoss(nn.Module):
    """L1 distance between the high-frequency FFT bands of pred and target.

    Complements the Sobel loss (local gradients only) by comparing the full
    high-frequency spectrum -- closer to what "hallucinate missing texture"
    actually requires.
    """

    def __init__(self, cutoff_ratio=0.25):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.l1 = nn.L1Loss()

    def _high_freq(self, img):
        fft = torch.fft.fftshift(torch.fft.fft2(img))
        _, _, h, w = img.shape
        cy, cx = h // 2, w // 2
        ry = max(1, int(h * self.cutoff_ratio / 2))
        rx = max(1, int(w * self.cutoff_ratio / 2))
        mask = torch.ones_like(fft.real)
        mask[..., cy - ry:cy + ry, cx - rx:cx + rx] = 0.0
        return torch.fft.ifft2(torch.fft.ifftshift(fft * mask)).real

    def forward(self, pred, target):
        return self.l1(self._high_freq(pred), self._high_freq(target))


class SSIMLoss(nn.Module):
    """Single-scale SSIM (not MS-SSIM) -- robust regardless of training
    patch size. Swap for pytorch_msssim.ms_ssim yourself if your crops are
    consistently large (roughly >=160px) and you want the multi-scale
    version."""

    def __init__(self, data_range=1.0):
        super().__init__()
        self.ssim = SSIM(data_range=data_range, size_average=True, channel=1)

    def forward(self, pred, target):
        return 1.0 - self.ssim(pred, target)


STAGE_WEIGHTS = {
    1: dict(char=1.0, ssim=0.20, edge=0.30, lpips=0.02, freq=0.0),
    2: dict(char=0.6, ssim=0.35, edge=0.40, lpips=0.15, freq=0.15),
}


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(model, loader, optimizer, scaler, losses, weights, device):
    model.train()
    charbonnier, ssim_loss, sobel, freq_loss, lpips_fn = losses
    total = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            noisy = batch.get("lr", batch.get("noisy")).to(device, non_blocking=True)
            clean = batch.get("hr", batch.get("gt")).to(device, non_blocking=True)
        else:
            noisy = batch[0].to(device, non_blocking=True)
            clean = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            pred = model(noisy)
            pred_c = torch.clamp(pred, 0.0, 1.0)

            l_char = charbonnier(pred, clean)
            l_ssim = ssim_loss(pred_c, clean) if ssim_loss is not None else torch.zeros((), device=device)
            l_edge = sobel(pred, clean)
            l_freq = freq_loss(pred, clean) if weights["freq"] > 0 else torch.zeros((), device=device)

            pred_norm = pred_c * 2.0 - 1.0
            clean_norm = clean * 2.0 - 1.0
            l_lpips = lpips_fn(pred_norm, clean_norm).mean()

            loss = (
                weights["char"] * l_char
                + weights["ssim"] * l_ssim
                + weights["edge"] * l_edge
                + weights["lpips"] * l_lpips
                + weights["freq"] * l_freq
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total / len(loader)


@torch.no_grad()
def evaluate_ood(model, loader, device):
    model.eval()
    total_ssim = total_edge = 0.0
    pbar = tqdm(loader, desc="Validating (OOD)", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            noisy = batch.get("lr", batch.get("noisy")).to(device)
            clean = batch.get("hr", batch.get("gt")).to(device)
        else:
            noisy, clean = batch[0].to(device), batch[1].to(device)

        pred = torch.clamp(model(noisy), 0.0, 1.0)
        pred_np = pred.cpu().numpy()[0, 0]
        clean_np = clean.cpu().numpy()[0, 0]

        metrics = stratified_ssim(pred_np, clean_np)
        if isinstance(metrics, dict):
            total_ssim += float(metrics["ssim"])
            total_edge += float(metrics["ssim_edge"])
        else:
            total_ssim += float(metrics[0])
            total_edge += float(metrics[1])

    n = len(loader)
    return total_ssim / n, total_edge / n


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, required=True, help="Root dir your caching step reads from")
    ap.add_argument("--cache-dir", type=str, default="/kaggle/working/cache")
    ap.add_argument("--output-dir", type=str, default="/kaggle/working/kla-restoration/artifacts")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--resume", type=str, default=None, help="Checkpoint to start from (required for --stage 2)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64, help="Global batch size, split across all visible GPUs")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> int:
    args = build_argparser().parse_args()

    if args.stage == 2 and not args.resume:
        raise SystemExit("--stage 2 needs --resume pointing at your stage-1 checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"--- Device: {device} | GPUs visible: {n_gpus} | Stage: {args.stage} ---")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    sp = load_splits()

    train_ds = RestorationDataset(args.cache_dir, stems=sp["train"])
    val_ds = RestorationDataset(args.cache_dir, stems=sp["val_ood"], train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = MODELS["nafnet_v2"](scale=args.scale).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed weights from {args.resume}")

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped in DataParallel across {n_gpus} GPUs "
              f"(~{args.batch_size // n_gpus} samples/GPU per step)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    charbonnier = CharbonnierLoss().to(device)
    sobel = SobelEdgeLoss().to(device)
    freq_loss = FrequencyLoss().to(device)
    ssim_loss = SSIMLoss().to(device) if _HAS_MSSSIM else None
    if ssim_loss is None:
        print("pytorch-msssim not installed (pip install pytorch-msssim) -- "
              "SSIM term disabled, falling back on Charbonnier + Sobel only.")

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad = False

    weights = STAGE_WEIGHTS[args.stage]
    losses = (charbonnier, ssim_loss, sobel, freq_loss, lpips_fn)

    best_ssim = 0.0
    print(f"--- Stage {args.stage} | weights: {weights} | {args.epochs} epochs ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, losses, weights, device)
        scheduler.step()
        ssim_val, edge_val = evaluate_ood(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ssim_val:.4f} | OOD Edge: {edge_val:.4f}")

        if ssim_val > best_ssim:
            best_ssim = ssim_val
            save_path = output_dir / f"best_nafnet_stage{args.stage}.pt"
            torch.save(unwrap(model).state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())