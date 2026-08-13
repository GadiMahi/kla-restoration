#!/usr/bin/env python3
"""
Training entry point - OWNED BY THE MODEL TEAM.
Trains the NAFNet architecture using a composite Charbonnier + SSIM + LPIPS loss.
Evaluates and saves checkpoints strictly based on the val_ood split.

Run via:
    python train.py --set data.root=/kaggle/input/kla-dataset output.dir=/kaggle/working/kla-restoration/artifacts
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

import numpy as np
from src.config import add_config_args, load_config
from src.dataset import RestorationDataset
from src.splits import load_splits
from src.model import MODELS
from src.eval_utils import edge_weight, stratified_ssim

# Third-party library for perceptual loss. 
# (Run: pip install lpips)
import lpips 


class CharbonnierLoss(nn.Module):
    """Robust L1 loss that handles outliers like extreme speckle noise."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, weight_map=None):
        loss = torch.sqrt((pred - target)**2 + self.eps**2)
        if weight_map is not None:
            loss = loss * weight_map
        return torch.mean(loss)


def train_one_epoch(model, dataloader, optimizer, loss_fns, device):
    model.train()
    total_loss = 0.0
    
    charbonnier, lpips_fn = loss_fns
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        # 1. Handle dictionary unpacking safely
        if isinstance(batch, dict):
            noisy_lr = batch.get("lr", batch.get("noisy")).to(device)
            clean_hr = batch.get("hr", batch.get("gt")).to(device)
        else:
            noisy_lr, clean_hr = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_hr = model(noisy_lr)
        
        # 2. Edge-weighted Charbonnier Loss
        clean_hr_np = clean_hr.detach().cpu().numpy()
        
        # FIX: The data team's edge_weight expects a 2D image (H, W). 
        # We loop through the batch to apply it correctly per image.
        import numpy as np # Fallback import just in case
        edges_np = np.zeros_like(clean_hr_np)
        for i in range(clean_hr_np.shape[0]):
            edges_np[i, 0] = edge_weight(clean_hr_np[i, 0])
            
        edges = torch.from_numpy(edges_np).to(device)
        
        l_char = charbonnier(pred_hr, clean_hr, weight_map=edges)
        
        # 3. Perceptual Loss (LPIPS)
        pred_norm = pred_hr * 2.0 - 1.0
        clean_norm = clean_hr * 2.0 - 1.0
        l_perceptual = lpips_fn(pred_norm, clean_norm).mean()
        
        loss = (1.0 * l_char) + (0.1 * l_perceptual)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_ood(model, dataloader, device):
    """Evaluates strictly on the out-of-distribution validation split."""
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
        
        # FIX: Squeeze out the Batch and Channel dimensions (1, 1, H, W) -> (H, W)
        # so skimage structural_similarity works correctly.
        pred_np = pred_hr.cpu().numpy()[0, 0]
        clean_np = clean_hr.cpu().numpy()[0, 0]
        
        ssim_val, ssim_edge, _ = stratified_ssim(pred_np, clean_np)
        
        metrics = stratified_ssim(pred_np, clean_np)
        
        # Safely extract the values (handling both dict and tuple returns just in case)
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
    print(f"--- Booting Training Pipeline on {device} ---")

    # Helper function to safely traverse nested configurations
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

    # 1. Setup Output Directory
    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving artifacts to: {output_dir}")

    # 2. Load Data Splits & Initialize Datasets
    torch.manual_seed(get_cfg("train.seed", 42))
    sp = load_splits()
    
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")
    print(f"Reading cache from: {cache_dir}")
    
    train_ds = RestorationDataset(cache_dir, stems=sp["train"])
    # Primary Metric: Held-out structure cluster
    val_ood_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)
    
    # 3. Create Dataloaders (Optimized for H100 I/O)
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

    # 4. Initialize Architecture
    scale_factor = get_cfg("dataset.scale", 2)
    model = MODELS["nafnet"](scale=scale_factor).to(device)

    print("\n--- Model Architecture Summary ---")
    # Assuming input shape is (Batch, Channels, Height, Width) -> e.g., (8, 1, 128, 128)
    summary(model, input_size=(batch_size, 1, 128, 128), device=device)
    print("----------------------------------\n")
    # 5. Optimization & Loss setup
    learning_rate = get_cfg("train.lr", 1e-4)
    epochs = get_cfg("train.epochs", 20)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    charbonnier = CharbonnierLoss().to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    # Freeze LPIPS network weights
    for param in lpips_fn.parameters():
        param.requires_grad = False
        
    loss_fns = (charbonnier, lpips_fn)

    # 6. Main Training Loop
    best_ood_ssim = 0.0
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fns, device)
        scheduler.step()
        
        ood_ssim, ood_ssim_edge = evaluate_ood(model, val_loader, device)
        
        print(f"Epoch {epoch:03d}/{epochs:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"OOD SSIM: {ood_ssim:.4f} | "
              f"OOD Edge SSIM: {ood_ssim_edge:.4f}")
        
        # 7. Checkpointing: Save only if it beats the previous OOD score
        if ood_ssim > best_ood_ssim:
            best_ood_ssim = ood_ssim
            save_path = output_dir / "best_nafnet.pt"
            
            # Save the raw state dictionary for easy loading in inference.py
            torch.save(model.state_dict(), save_path)
            print(f" -> New Best OOD SSIM! Checkpoint saved to {save_path}")

    print("Training Complete. Final Best OOD SSIM:", best_ood_ssim)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())