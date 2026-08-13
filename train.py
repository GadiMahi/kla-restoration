#!/usr/bin/env python3
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

# --- PyTorch Native Losses ---

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps**2))

class SobelEdgeLoss(nn.Module):
    """
    Computes spatial gradients (edges) directly in PyTorch.
    Forces the network to output sharp edges instead of blurring them.
    """
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

# --- Core Loop ---

def train_one_epoch(model, dataloader, optimizer, loss_fns, device):
    model.train()
    total_loss = 0.0
    charbonnier, lpips_fn, sobel_loss = loss_fns
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device)
            clean_hr = batch.get("hr", batch.get("gt")).to(device)
        else:
            noisy_lr, clean_hr = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        pred_hr = model(noisy_lr)
        
        # 1. Base Pixel Loss
        l_char = charbonnier(pred_hr, clean_hr)
        
        # 2. Perceptual Loss (Scaled to [-1, 1])
        pred_norm = pred_hr * 2.0 - 1.0
        clean_norm = clean_hr * 2.0 - 1.0
        l_perceptual = lpips_fn(pred_norm, clean_norm).mean()
        
        # 3. Structural Edge Loss (Differentiable)
        l_edge = sobel_loss(pred_hr, clean_hr)
        
        # The Re-balanced Composite Loss
        loss = (1.0 * l_char) + (1.5 * l_perceptual) + (0.5 * l_edge)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_ood(model, dataloader, device):
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
            
        pred_hr = model(noisy_lr)
        pred_hr = torch.clamp(pred_hr, 0.0, 1.0)
        
        # Convert to numpy strictly for the evaluation metrics
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
    sp = load_splits()
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")
    
    train_ds = RestorationDataset(cache_dir, stems=sp["train"])
    val_ood_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)
    
    batch_size = get_cfg("train.batch_size", 8)
    workers = get_cfg("train.num_workers", 4)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ood_ds, batch_size=1, shuffle=False, 
        num_workers=workers, pin_memory=True
    )

    scale_factor = get_cfg("dataset.scale", 2)
    model = MODELS["nafnet"](scale=scale_factor).to(device)
    
    epochs = get_cfg("train.epochs", 100)
    optimizer = optim.AdamW(model.parameters(), lr=get_cfg("train.lr", 2e-4), weight_decay=1e-4)
    
    # Cosine annealing ensures the model settles into fine details in later epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    for param in lpips_fn.parameters():
        param.requires_grad = False
        
    loss_fns = (charbonnier, lpips_fn, sobel_loss)
    best_ood_ssim = 0.0
    
    print(f"--- Starting Training Run (U-Net NAFNet) for {epochs} epochs on {device} ---")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fns, device)
        scheduler.step()
        
        ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device)
        
        print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ood_ssim:.4f} | OOD Edge: {ood_ssim_edge:.4f}")
        
        if ood_ssim > best_ood_ssim:
            best_ood_ssim = ood_ssim
            save_path = output_dir / "best_nafnet.pt"
            torch.save(model.state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())