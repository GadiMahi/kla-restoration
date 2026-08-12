"""Decode once into a memmap. Per-item file decode will GPU-starve training.

Dtype policy:
  float32  - default, zero precision risk
  uint16   - GT only, if float32 does not fit (error ~1.5e-5, negligible)
  float16  - never; ~5e-4 relative precision is too close to the 8-bit floor
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .io_utils import load_image


def _store(arr: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "uint16":
        return np.rint(np.clip(arr, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if dtype == "float16":
        raise ValueError("float16 caching is banned: precision too close to the 8-bit floor")
    return arr.astype(dtype, copy=False)


def build_cache(pairs, out_dir, gt_dtype="float32", lr_dtype="float32"):
    """pairs: list[(gt_path, lr_path)]. Groups by (gt_shape, lr_shape)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if lr_dtype != "float32":
        raise ValueError(
            "NoisyLR must be cached as float32: integer dtypes clip the\n"
            "out-of-[0,1] values that the KLA spec says are intentional."
        )

    groups: dict[tuple, list] = {}
    for gt_p, lr_p in tqdm(pairs, desc="scan shapes"):
        g, l = load_image(gt_p), load_image(lr_p)
        groups.setdefault((g.shape, l.shape), []).append((gt_p, lr_p))

    index = []
    for (gshape, lshape), items in groups.items():
        tag = f"{gshape[0]}x{gshape[1]}_from_{lshape[0]}x{lshape[1]}"
        gt_mm = np.lib.format.open_memmap(
            out_dir / f"gt_{tag}.npy", mode="w+",
            dtype=gt_dtype, shape=(len(items), *gshape))
        lr_mm = np.lib.format.open_memmap(
            out_dir / f"lr_{tag}.npy", mode="w+",
            dtype=lr_dtype, shape=(len(items), *lshape))
        for i, (gt_p, lr_p) in enumerate(tqdm(items, desc=f"cache {tag}")):
            gt_mm[i] = _store(load_image(gt_p), gt_dtype)
            lr_mm[i] = _store(load_image(lr_p), lr_dtype)  # LR may exceed [0,1]
        gt_mm.flush(); lr_mm.flush()
        index.append({
            "tag": tag, "n": len(items),
            "gt_file": f"gt_{tag}.npy", "lr_file": f"lr_{tag}.npy",
            "gt_shape": list(gshape), "lr_shape": list(lshape),
            "gt_dtype": gt_dtype, "lr_dtype": lr_dtype,
            "stems": [Path(p).stem for p, _ in items],
        })

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    return index
