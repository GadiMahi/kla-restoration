"""Training dataset.

Mixed source resolutions (128->256 and 256->512) are unified by cropping every
sample to a fixed LR patch, so both groups are indistinguishable inside a batch.
The network is fully convolutional, so full-size inference is unaffected.

Crops are drawn with gradient-based rejection sampling: wafer imagery is mostly
flat die area and uniform random crops would spend most of training on blank
regions.

Fix notes (host-RAM OOM):
  The training path used to call np.asarray(gt_mm[i], dtype=np.float32) /
  np.asarray(lr_mm[i], dtype=np.float32) on the *whole* source image before
  cropping down to a small lr_patch x lr_patch region. mmap_mode="r" on load
  makes reads lazy (only touched bytes get faulted in from disk), but the
  eager float32 cast on the full image defeated that -- every __getitem__
  call pulled an entire source image through the page cache just to keep a
  small crop of it. Under a container memory limit, that page-cache growth
  counts against the limit and can trigger an OOM well before "self.rng
  peak" memory would suggest, especially at high read throughput (fast
  GPU + large batch_size means workers cycle through far more images per
  second). Fixed by doing crop-coordinate search and the crop itself while
  still memmap-backed, and only float32-casting the small patch that's
  actually kept. Eval (train=False) still reads the full image, since eval
  needs the whole frame anyway and only does it once per sample per epoch.

  Also: self.rng was a np.random.default_rng() Generator created before
  DataLoader workers fork. PyTorch's automatic per-worker reseeding only
  touches the legacy global np.random state, not a Generator instance held
  as an attribute, so every worker inherited the identical RNG state and
  produced identical crop sequences. Now reseeded per worker on first use.
"""
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
        self._rng_worker_id = None  # reseeded lazily once we know our worker id

        self._groups, self.items = [], []
        keep = set(stems) if stems is not None else None
        for gi, g in enumerate(self.index):
            self._groups.append((
                np.load(self.cache_dir / g["gt_file"], mmap_mode="r"),
                np.load(self.cache_dir / g["lr_file"], mmap_mode="r"),
                g,
            ))
            for i, stem in enumerate(g["stems"]):
                if keep is None or stem in keep:
                    self.items.append((gi, i, stem))

    def _ensure_worker_rng(self):
        """Give each DataLoader worker process its own RNG stream.

        self.rng is created in the main process before workers fork, so
        without this every worker would otherwise inherit and reuse the
        exact same Generator state and sample identical "random" crops.
        """
        info = get_worker_info()
        worker_id = info.id if info is not None else -1
        if self._rng_worker_id != worker_id:
            self.rng = np.random.default_rng(self._base_seed + worker_id + 1)
            self._rng_worker_id = worker_id

    def __len__(self) -> int:
        return len(self.items)

    def _read(self, gi: int, i: int):
        """Full-image read (materializes the whole frame). Used for eval,
        where the entire image is needed anyway, and by
        estimate_grad_threshold. NOT used on the train __getitem__ path --
        see _read_train_crop below."""
        gt_mm, lr_mm, g = self._groups[gi]
        gt = np.asarray(gt_mm[i], dtype=np.float32)
        lr = np.asarray(lr_mm[i], dtype=np.float32)
        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
        return gt, lr

    def _crop(self, gt: np.ndarray, lr: np.ndarray):
        """Kept for backwards compatibility / external callers. Operates on
        already-materialized arrays -- prefer _read_train_crop for the
        training hot path, which never materializes more than the patch."""
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

    def _read_train_crop(self, gi: int, i: int):
        """Crop-coordinate search + crop, done directly on the memmap views
        so only the chosen lr_patch x lr_patch window (and its matching HR
        region) ever gets faulted into RAM / float32-cast -- not the whole
        source image."""
        gt_mm, lr_mm, g = self._groups[gi]
        gt_img = gt_mm[i]  # still memmap-backed, no data touched yet
        lr_img = lr_mm[i]  # still memmap-backed

        p, s = self.lr_patch, self.scale
        H, W = lr_img.shape

        if H <= p or W <= p:
            # Whole image is smaller than the patch -- nothing to crop,
            # have to take it all (rare edge case).
            gt = np.asarray(gt_img, dtype=np.float32)
            lr = np.asarray(lr_img, dtype=np.float32)
        else:
            best = None
            for _ in range(self.crop_tries):
                y = int(self.rng.integers(0, H - p + 1))
                x = int(self.rng.integers(0, W - p + 1))
                # Only this small window is faulted in from the memmap.
                cand = np.asarray(lr_img[y:y + p, x:x + p], dtype=np.float32)
                e = _grad_energy(cand)
                if e >= self.grad_thresh:
                    best = (e, y, x)
                    break
                if best is None or e > best[0]:
                    best = (e, y, x)
            _, y, x = best  # always terminates
            lr = np.asarray(lr_img[y:y + p, x:x + p], dtype=np.float32)
            gt = np.asarray(
                gt_img[y * s:(y + p) * s, x * s:(x + p) * s], dtype=np.float32
            )

        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
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