"""Training dataset.

Mixed source resolutions (128->256 and 256->512) are unified by cropping every
sample to a fixed LR patch, so both groups are indistinguishable inside a batch.
The network is fully convolutional, so full-size inference is unaffected.

Crops are drawn with gradient-based rejection sampling: wafer imagery is mostly
flat die area and uniform random crops would spend most of training on blank
regions.

--- Host-RAM fix (Kaggle T4x2) ---
__init__ used to mmap *every* group in index.json regardless of `stems`, so
e.g. train_ds also held open mmap handles to the val/OOD files (and vice
versa) that it would never read. With both loaders using
persistent_workers=True across a 100-epoch run, the unnecessary extra
touchable file surface let the OS page cache (counted as RSS in a
memory-cgroup-limited container like a Kaggle session) grow steadily until
the process got OOM-killed -- visible in the training logs as host_peak_rss
climbing ~1GB/epoch with no plateau, while GPU memory stayed flat. Now each
RestorationDataset instance only mmaps the groups its own `stems` filter
actually needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
        self.rng = np.random.default_rng(seed)

        self._groups, self.items = [], []
        keep = set(stems) if stems is not None else None
        for g in self.index:
            # Skip groups this split will never read -- avoids mmapping (and
            # eventually paging into RSS) file content that isn't ours.
            if keep is not None and not any(stem in keep for stem in g["stems"]):
                continue
            gi = len(self._groups)
            self._groups.append((
                np.load(self.cache_dir / g["gt_file"], mmap_mode="r"),
                np.load(self.cache_dir / g["lr_file"], mmap_mode="r"),
                g,
            ))
            for i, stem in enumerate(g["stems"]):
                if keep is None or stem in keep:
                    self.items.append((gi, i, stem))

    def __len__(self) -> int:
        return len(self.items)

    def _read(self, gi: int, i: int):
        gt_mm, lr_mm, g = self._groups[gi]
        gt = np.asarray(gt_mm[i], dtype=np.float32)
        lr = np.asarray(lr_mm[i], dtype=np.float32)
        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
        return gt, lr

    def _crop(self, gt: np.ndarray, lr: np.ndarray):
        p, s = self.lr_patch, self.scale
        H, W = lr.shape
        if H <= p or W <= p:
            return gt, lr
        best = None
        for _ in range(self.crop_tries):
            y = int(self.rng.integers(0, H - p + 1))
            x = int(self.rng.integers(0, W - p + 1))
            e = _grad_energy(lr[y:y + p, x:x + p])
            if e >= self.grad_thresh:
                best = (e, y, x)
                break
            if best is None or e > best[0]:
                best = (e, y, x)
        _, y, x = best  # always terminates
        return (gt[y * s:(y + p) * s, x * s:(x + p) * s], lr[y:y + p, x:x + p])

    def __getitem__(self, idx: int):
        gi, i, stem = self.items[idx]
        gt, lr = self._read(gi, i)
        if self.train:
            gt, lr = self._crop(gt, lr)
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