#!/usr/bin/env python3
"""Generate a tiny synthetic dataset so the pipeline can be smoke-tested
without the real KLA data. NOT for training - structure only.

    python scripts/make_dummy_data.py --out /tmp/dummy --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import ImageFormat, save_image  # noqa: E402


def synth_wafer(size, rng):
    """Axis-aligned line/grid structure, loosely like a die layout."""
    img = np.full((size, size), 0.15, dtype=np.float32)
    for _ in range(rng.integers(4, 10)):
        w = int(rng.integers(2, 7))
        if rng.random() < 0.5:
            c = int(rng.integers(0, size - w))
            img[:, c:c + w] = rng.uniform(0.5, 0.95)
        else:
            r = int(rng.integers(0, size - w))
            img[r:r + w, :] = rng.uniform(0.5, 0.95)
    for _ in range(rng.integers(2, 6)):
        r, c = rng.integers(0, size - 20, 2)
        s = int(rng.integers(8, 20))
        img[r:r + s, c:c + s] = rng.uniform(0.2, 0.9)
    return np.clip(img, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/dummy")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--format", default="npy", choices=["npy", "tiff32", "png16"])
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    fmt = {"npy": ImageFormat("npy", ".npy", "float32", 1.0),
           "tiff32": ImageFormat("tiff32", ".tif", "float32", 1.0),
           "png16": ImageFormat("png16", ".png", "uint16", 65535.0)}[args.format]

    out = Path(args.out)
    gt_dir, lr_dir = out / "GT", out / "NoisyLR"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    from src.degrade import degrade

    cfg = dict(width=1.0, jitter=0.30, kernels=["bicubic_aa", "area"],
               kernel_p=[0.7, 0.3], order_mix=[0.4, 0.6],
               meas_mult=0.12, meas_add=0.03)

    for i in range(args.n):
        size = 512 if i % 2 == 0 else 256
        gt = synth_wafer(size, rng)
        lr = degrade(gt, rng, cfg)
        save_image(gt, gt_dir / f"img_{i:04d}{fmt.suffix}", fmt)
        # NoisyLR must keep its out-of-range values -> raw float, never clipped.
        np.save(lr_dir / f"img_{i:04d}.npy", lr.astype(np.float32))

    print(f"wrote {args.n} pairs to {out}  (GT format={args.format}, LR=npy float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
