"""Training dataset.

Mixed source resolutions (128->256 and 256->512) are unified by cropping every
sample to a fixed LR patch, so both groups are indistinguishable inside a batch.
The network is fully convolutional, so full-size inference is unaffected.

Crops are drawn with gradient-based rejection sampling: wafer imagery is mostly
flat die area and uniform random crops would spend most of training on blank
regions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import scale_jitter
from .degrade import degrade
from .transforms import load_stats, normalize


def degrade_cfg_from_stats(stats=None, width=1.0, jitter=0.30) -> dict:
    """Build a degrade() config from the measured constants in stats.json."""
    s = stats or load_stats()
    return {
        "width": width,
        "jitter": jitter,
        "kernels": s.get("kernels", ["area", "bicubic_aa", "lanczos", "bilinear_aa"]),
        "kernel_p": s.get("kernel_p", [0.3, 0.3, 0.2, 0.2]),
        "order_mix": s.get("order_mix", [0.5, 0.5]),
        "noise_var_fit": s.get("noise_var_fit"),
        "meas_mult": s.get("meas_mult", 0.0),
        "meas_add": s.get("meas_add", 0.0),
    }


def _grad_energy(patch: np.ndarray) -> float:
    return float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())


class RestorationDataset(Dataset):
    def __init__(self, cache_dir, stems=None, lr_patch=64, scale=2,
                 grad_thresh=0.0, crop_tries=8, train=True, seed=1337,
                 synth_p=0.0, degrade_cfg=None, jitter_range=(0.7, 1.4), margin=8):
        """
        synth_p       probability a training sample is SYNTHESISED from the GT
                      instead of using the provided degraded pair
        degrade_cfg   from degrade_cfg_from_stats(); required when synth_p > 0
        jitter_range  GT rescaling applied BEFORE degradation. This is the main
                      defence against unfamiliar feature scales - the training
                      set is entirely 256->128, while evaluation may include
                      512x512 content.
        margin        HR pixels of context kept around the crop while degrading,
                      then trimmed, so the resampling kernel never sees the crop
                      boundary. Measured effect on border-row statistics was
                      small (<0.2%), so this is cheap insurance rather than a
                      fix for an observed problem.
        """
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "index.json") as f:
            self.index = json.load(f)

        self.lr_patch, self.scale = lr_patch, scale
        self.grad_thresh, self.crop_tries = grad_thresh, crop_tries
        self.train = train
        self.seed = seed
        self._rng = None                       # created per worker, see _ensure_rng
        self.synth_p = float(synth_p)
        self.degrade_cfg = degrade_cfg
        self.jitter_range = tuple(jitter_range)
        self.margin = int(margin) - int(margin) % scale     # keep divisible by scale
        if self.synth_p > 0 and self.degrade_cfg is None:
            raise ValueError("synth_p > 0 requires degrade_cfg (see degrade_cfg_from_stats)")

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

    def __len__(self) -> int:
        return len(self.items)

    def _ensure_rng(self):
        """Per-worker RNG.

        A single RNG created in __init__ is COPIED into every dataloader worker,
        so all workers would draw identical crops and identical noise. Seed by
        worker id instead.
        """
        if self._rng is None:
            try:
                import torch.utils.data as _tud

                info = _tud.get_worker_info()
                wid = info.id if info is not None else 0
            except Exception:
                wid = 0
            self._rng = np.random.default_rng([self.seed, wid])
        return self._rng

    def set_width(self, width: float) -> None:
        """Curriculum knob: ramp synthetic degradation variety 0.3 -> 1.0."""
        if self.degrade_cfg is not None:
            self.degrade_cfg = {**self.degrade_cfg, "width": float(width)}

    def _read(self, gi: int, i: int):
        gt_mm, lr_mm, g = self._groups[gi]
        gt = np.asarray(gt_mm[i], dtype=np.float32)
        lr = np.asarray(lr_mm[i], dtype=np.float32)
        if g["gt_dtype"] == "uint16":
            gt = gt / 65535.0
        return gt, lr

    def _crop_hr(self, hr: np.ndarray, size: int):
        """Gradient-rejection crop in HR space (used by the synthetic path)."""
        rng = self._ensure_rng()
        H, W = hr.shape
        if H <= size or W <= size:
            pad_h, pad_w = max(0, size - H), max(0, size - W)
            hr = np.pad(hr, ((0, pad_h), (0, pad_w)), mode="reflect")
            H, W = hr.shape
        best = None
        for _ in range(self.crop_tries):
            y = int(rng.integers(0, H - size + 1))
            x = int(rng.integers(0, W - size + 1))
            e = _grad_energy(hr[y:y + size, x:x + size])
            if e >= self.grad_thresh:
                best = (e, y, x)
                break
            if best is None or e > best[0]:
                best = (e, y, x)
        _, y, x = best
        return hr[y:y + size, x:x + size]

    def _synth(self, gt_full: np.ndarray):
        """Generate a fresh (lr, hr) pair from a clean GT image.

        Order matters: rescale -> crop with margin -> degrade -> trim margin.
        Degrading before cropping would waste work; cropping without a margin
        would bake resampling edge artefacts into every training patch.
        """
        rng = self._ensure_rng()
        s, m = self.scale, self.margin
        gt = scale_jitter(gt_full, rng, *self.jitter_range)

        hr_size = self.lr_patch * s
        crop = self._crop_hr(gt, hr_size + 2 * m)
        lr = degrade(crop, rng, self.degrade_cfg)

        lm = m // s
        lr = lr[lm:lm + self.lr_patch, lm:lm + self.lr_patch]
        hr = crop[m:m + hr_size, m:m + hr_size]
        return hr, lr

    def _crop(self, gt: np.ndarray, lr: np.ndarray):
        rng = self._ensure_rng()
        p, s = self.lr_patch, self.scale
        H, W = lr.shape
        if H <= p or W <= p:
            return gt, lr
        best = None
        for _ in range(self.crop_tries):
            y = int(rng.integers(0, H - p + 1))
            x = int(rng.integers(0, W - p + 1))
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
            if self.synth_p > 0 and float(self._ensure_rng().random()) < self.synth_p:
                gt, lr = self._synth(gt)
            else:
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
