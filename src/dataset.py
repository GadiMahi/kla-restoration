from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from .transforms import normalize

def _grad_energy(patch: np.ndarray) -> float:
    return float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())

class RestorationDataset(Dataset):
    def __init__(self, cache_dir, stems=None, lr_patch=64, scale=2,
                 grad_thresh=0.0, crop_tries=8, train=True, seed=1337):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "index.json") as f:
            self.index = json.load(f)

        self.lr_patch, self.scale = lr_patch, scale
        self.grad_thresh, self.crop_tries = grad_thresh, crop_tries
        self.train = train
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)
        self._rng_worker_id = None 

        self.group_meta = [] 
        self.items = []
        keep = set(stems) if stems is not None else None
        
        for gi, g in enumerate(self.index):
            self.group_meta.append(g)
            for i, stem in enumerate(g["stems"]):
                if keep is None or stem in keep:
                    self.items.append((gi, i, stem))

    def _ensure_worker_rng(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else -1
        if self._rng_worker_id != worker_id:
            self.rng = np.random.default_rng(self._base_seed + worker_id + 1)
            self._rng_worker_id = worker_id

    def __len__(self) -> int:
        return len(self.items)

    def _read(self, gi: int, i: int):
        g = self.group_meta[gi]
        gt_path = str(self.cache_dir / g["gt_file"])
        lr_path = str(self.cache_dir / g["lr_file"])
        
        gt_mm = np.load(gt_path, mmap_mode="r")
        lr_mm = np.load(lr_path, mmap_mode="r")
        
        # Copy into physical RAM as an isolated array
        gt = np.array(gt_mm[i], dtype=np.float32)
        lr = np.array(lr_mm[i], dtype=np.float32)
        
        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
            
        # BRUTAL FIX: Force Linux to close the memory mapping immediately
        if hasattr(gt_mm, '_mmap') and gt_mm._mmap is not None: gt_mm._mmap.close()
        if hasattr(lr_mm, '_mmap') and lr_mm._mmap is not None: lr_mm._mmap.close()
        del gt_mm, lr_mm
        
        return gt, lr

    def _read_train_crop(self, gi: int, i: int):
        g = self.group_meta[gi]
        gt_path = str(self.cache_dir / g["gt_file"])
        lr_path = str(self.cache_dir / g["lr_file"])
        
        gt_mm = np.load(gt_path, mmap_mode="r")
        lr_mm = np.load(lr_path, mmap_mode="r")
        
        p, s = self.lr_patch, self.scale
        H, W = lr_mm.shape[-2], lr_mm.shape[-1]

        if H <= p or W <= p:
            gt = np.array(gt_mm[i], dtype=np.float32)
            lr = np.array(lr_mm[i], dtype=np.float32)
        else:
            best = None
            for _ in range(self.crop_tries):
                y = int(self.rng.integers(0, H - p + 1))
                x = int(self.rng.integers(0, W - p + 1))
                cand = np.array(lr_mm[i, y:y + p, x:x + p], dtype=np.float32)
                e = _grad_energy(cand)
                if e >= self.grad_thresh:
                    best = (e, y, x)
                    break
                if best is None or e > best[0]:
                    best = (e, y, x)
            _, y, x = best  
            lr = np.array(lr_mm[i, y:y + p, x:x + p], dtype=np.float32)
            gt = np.array(gt_mm[i, y * s:(y + p) * s, x * s:(x + p) * s], dtype=np.float32)

        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
            
        # BRUTAL FIX: Force Linux to close the memory mapping immediately
        if hasattr(gt_mm, '_mmap') and gt_mm._mmap is not None: gt_mm._mmap.close()
        if hasattr(lr_mm, '_mmap') and lr_mm._mmap is not None: lr_mm._mmap.close()
        del gt_mm, lr_mm
        
        return gt, lr

    def __getitem__(self, idx: int):
        self._ensure_worker_rng()
        gi, i, stem = self.items[idx]
        if self.train:
            gt, lr = self._read_train_crop(gi, i)
        else:
            gt, lr = self._read(gi, i)
        return {
            "lr": torch.from_numpy(np.ascontiguousarray(normalize(lr))).unsqueeze(0),
            "hr": torch.from_numpy(np.ascontiguousarray(normalize(gt))).unsqueeze(0),
            "stem": stem,
        }


def estimate_grad_threshold(cache_dir, percentile=40, lr_patch=64, n=2000, seed=0) -> float:
    """Measure the crop-gradient distribution instead of guessing a threshold."""
    ds = RestorationDataset(cache_dir, lr_patch=lr_patch, train=False, seed=seed)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        gi, i, _ = ds.items[int(rng.integers(0, len(ds.items)))]
        _, lr = ds._read(gi, i)
        H, W = lr.shape
        if H <= lr_patch or W <= lr_patch:
            continue
        y = int(rng.integers(0, H - lr_patch + 1))
        x = int(rng.integers(0, W - lr_patch + 1))
        vals.append(_grad_energy(lr[y:y + lr_patch, x:x + lr_patch]))
    return float(np.percentile(vals, percentile)) if vals else 0.0