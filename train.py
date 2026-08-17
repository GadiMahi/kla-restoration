#!/usr/bin/env python3
import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import add_config_args, load_config
from src.splits import load_splits
from src.dataset import RestorationDataset
from src.model import MODELS
from src.eval_utils import stratified_ssim

# --- NATIVE LOSSES ---

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

class HighFrequencyLoss(nn.Module):
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


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    def get_cfg(key, default=None):
        if hasattr(cfg, "get_path"):
            try: return cfg.get_path(key)
            except: pass
        if hasattr(cfg, "get"): return cfg.get(key)
        return default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")
    
    batch_size = min(get_cfg("train.batch_size", 32), 32)
    epochs = get_cfg("train.epochs", 100)

    print("--- Starting Minimal Training Script (Native Losses) ---")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs}")

    torch.manual_seed(42)
    sp = load_splits()

    train_ds = RestorationDataset(cache_dir, stems=sp["train"], train=True)
    val_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    # STRICT ANTI-LEAK SETTINGS from simple script
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=0, pin_memory=False)

    scale_factor = get_cfg("dataset.scale", 2)
    model = MODELS["nafnet"](scale=scale_factor).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Wrapping model in nn.DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4)

    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    hf_loss = HighFrequencyLoss().to(device)

    w_char, w_edge, w_hf = 1.0, 0.5, 0.5
    best_ssim = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        # NO TQDM, NO AMP, NO MULTIPLE ITEMS
        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad()
            pred = model(lr)
            
            l_char = charbonnier(pred, hr)
            l_edge = sobel_loss(pred, hr)
            l_hf = hf_loss(pred, hr)
            
            loss = (w_char * l_char) + (w_edge * l_edge) + (w_hf * l_hf)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Evaluation Phase ---
        model.eval()
        epoch_ssim = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                lr = batch["lr"].to(device)
                hr = batch["hr"].to(device)
                
                pred = torch.clamp(model(lr), 0.0, 1.0)
                
                pred_np = pred.cpu().numpy()[0, 0]
                hr_np = hr.cpu().numpy()[0, 0]
                
                metrics = stratified_ssim(pred_np, hr_np)
                if isinstance(metrics, dict):
                    epoch_ssim += float(metrics["ssim"])
                else:
                    epoch_ssim += float(metrics[0])

        avg_val_ssim = epoch_ssim / len(val_loader)
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.4f} | Val SSIM: {avg_val_ssim:.4f}")

        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            save_path = output_dir / "best_nafnet.pt"
            unwrapped_model = model.module if hasattr(model, "module") else model
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

        # Clean up ONLY at the epoch boundary
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())