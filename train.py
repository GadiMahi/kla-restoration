#!/usr/bin/env python3
import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import add_config_args, load_config
from src.splits import load_splits
from src.dataset import RestorationDataset
from src.model import MODELS
from src.eval_utils import stratified_ssim

def main() -> int:
    ap = add_config_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    # Safe config getter
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
    
    # Force batch size down if 64 is requested; 64 is too large for 2x T4s and causes separate VRAM OOMs
    batch_size = min(get_cfg("train.batch_size", 32), 32)
    epochs = get_cfg("train.epochs", 100)

    print("--- Starting Minimal, Leak-Proof Training Script ---")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs}")

    torch.manual_seed(42)
    sp = load_splits()

    train_ds = RestorationDataset(cache_dir, stems=sp["train"], train=True)
    val_ds = RestorationDataset(cache_dir, stems=sp["val_ood"], train=False)

    # STRICT ANTI-LEAK SETTINGS: 0 workers, no pin_memory, no persistent_workers
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
    criterion = nn.L1Loss() # Keep it simple to avoid graph retention

    best_ssim = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            # Only extract the tensors. Ignore the "stem" string entirely to prevent IPC leaks.
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad()
            pred = model(lr)
            
            loss = criterion(pred, hr)
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
                
                # Forward pass and clamp
                pred = torch.clamp(model(lr), 0.0, 1.0)
                
                pred_np = pred.cpu().numpy()[0, 0]
                hr_np = hr.cpu().numpy()[0, 0]
                
                metrics = stratified_ssim(pred_np, hr_np)
                
                # Handle whether stratified_ssim returns a dict or tuple
                if isinstance(metrics, dict):
                    epoch_ssim += float(metrics["ssim"])
                else:
                    epoch_ssim += float(metrics[0])

        avg_val_ssim = epoch_ssim / len(val_loader)
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train L1: {avg_train_loss:.4f} | Val SSIM: {avg_val_ssim:.4f}")

        # Save Best Model
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            save_path = output_dir / "best_nafnet.pt"
            
            # Save unwrapped state_dict
            unwrapped_model = model.module if hasattr(model, "module") else model
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f" -> Checkpoint saved to {save_path}")

        # --- Brutal Memory Cleanup ---
        # Destroy all references to the current batch and flush the GC
        del batch, lr, hr, pred, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())