#!/usr/bin/env python3
import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

# Use the stateful modules, NOT the functional calls!
from pytorch_msssim import SSIM, MS_SSIM

from src.config import add_config_args, load_config
from src.splits import load_splits
from src.dataset import RestorationDataset
from src.model import MODELS
from src.eval_utils import stratified_ssim

# --- Memory-Safe PyTorch Losses ---

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
    def __init__(self, data_range=1.0, channel=1):
        super().__init__()
        # 🚀 CPU OOM FIX: Use the stateful SSIM module. 
        # This permanently caches the kernel on the GPU instead of recreating it on the CPU every forward pass.
        self.ssim_module = SSIM(data_range=data_range, size_average=True, channel=channel)

    def forward(self, pred, target):
        return 1.0 - self.ssim_module(pred, target)

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
            try: 
                val = cfg.get_path(key)
                if val is not None: return val
            except: pass
        if hasattr(cfg, "get"): 
            val = cfg.get(key)
            if val is not None: return val
        return default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(get_cfg("output.dir", "/kaggle/working/kla-restoration/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = get_cfg("cache.dir", "/kaggle/working/cache")
    
    batch_size = min(int(get_cfg("train.batch_size", 32)), 32)
    epochs = int(get_cfg("train.epochs", 100))
    use_amp = bool(get_cfg("train.amp", True)) and device.type == "cuda"

    print(f"--- Starting Training Run (Full Losses, CPU-Safe) ---")
    print(f"Device: {device} | Batch Size: {batch_size} | AMP: {use_amp}")

    torch.manual_seed(42)
    sp = load_splits()

    train_ds = RestorationDataset(cache_dir, stems=sp["train"], train=True)
    val_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=0, pin_memory=False)

    scale_factor = int(get_cfg("dataset.scale", 2))
    model = MODELS["nafnet"](scale=scale_factor).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    lr_val = float(get_cfg("train.lr", 5e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr_val, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Initialize all losses
    charbonnier = CharbonnierLoss().to(device)
    sobel_loss = SobelEdgeLoss().to(device)
    msssim_loss = MSSSIMLoss(channel=1).to(device)
    hf_loss = HighFrequencyLoss().to(device)
    
    # 🚀 CPU OOM FIX 2: Explicitly force LPIPS into .eval() to stop PyTorch tracking dropout/batch statistics
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False

    w = {
        "char":   float(get_cfg("train.loss.char_w", 1.0)),
        "lpips":  float(get_cfg("train.loss.lpips_w", 0.05)),
        "edge":   float(get_cfg("train.loss.edge_w", 0.2)),
        "msssim": float(get_cfg("train.loss.msssim_w", 0.3)),
        "hf":     float(get_cfg("train.loss.hf_w", 0.4)),
    }

    best_ssim = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Train]", leave=False) as pbar:
            for batch in pbar:
                lr = batch["lr"].to(device, non_blocking=True)
                hr = batch["hr"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(lr)
                    l_char = charbonnier(pred, hr)
                    l_edge = sobel_loss(pred, hr)

                with torch.amp.autocast('cuda', enabled=False):
                    pred_f = pred.float()
                    hr_f = hr.float()
                    
                    pred_norm = pred_f * 2.0 - 1.0
                    hr_norm = hr_f * 2.0 - 1.0
                    
                    l_perceptual = lpips_fn(pred_norm, hr_norm).mean()
                    l_msssim = msssim_loss(pred_f, hr_f)
                    l_hf = hf_loss(pred_f, hr_f)

                loss = (w["char"] * l_char + w["lpips"] * l_perceptual + 
                        w["edge"] * l_edge + w["msssim"] * l_msssim + w["hf"] * l_hf)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                
                # Format string manually to avoid holding references
                pbar.set_postfix_str(f"Tot:{loss.item():.3f} Char:{(w['char']*l_char).item():.3f} MSSSIM:{(w['msssim']*l_msssim).item():.3f} LPIPS:{(w['lpips']*l_perceptual).item():.3f}")
                
                # Instantly destroy variables
                del batch, lr, hr, pred, pred_f, hr_f, pred_norm, hr_norm
                del l_char, l_edge, l_perceptual, l_msssim, l_hf, loss

        avg_train_loss = total_loss / len(train_loader)

        # --- Evaluation Phase ---
        model.eval()
        epoch_ssim = 0.0
        
        with tqdm(val_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Eval]", leave=False) as pbar:
            with torch.no_grad():
                for batch in pbar:
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
                        
                    del batch, lr, hr, pred, pred_np, hr_np

        avg_val_ssim = epoch_ssim / len(val_loader)
        
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Loss: {avg_train_loss:.4f} | Val SSIM: {avg_val_ssim:.4f}")

        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            save_path = output_dir / "best_nafnet.pt"
            unwrapped_model = model.module if hasattr(model, "module") else model
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())