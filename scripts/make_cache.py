#!/usr/bin/env python3
"""Decode the dataset once into memmaps, then measure dataloader throughput.

    python scripts/make_cache.py --set data.root=/kaggle/input/kla-dataset \
                                      cache.dir=/kaggle/working/cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader  # noqa: E402

from src.cache import build_cache        # noqa: E402
from src.config import add_config_args, load_config  # noqa: E402
from src.dataset import RestorationDataset, estimate_grad_threshold  # noqa: E402
from src.io_utils import pair_by_stem    # noqa: E402


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    cache_dir = Path(cfg.get_path("cache.dir"))

    if not args.skip_build:
        pairs = pair_by_stem(root / cfg.get_path("data.gt_subdir"),
                             root / cfg.get_path("data.lr_subdir"))
        print(f"caching {len(pairs)} pairs -> {cache_dir}")
        build_cache(pairs, cache_dir,
                    gt_dtype=cfg.get_path("cache.gt_dtype"),
                    lr_dtype=cfg.get_path("cache.lr_dtype"))

    thr = estimate_grad_threshold(cache_dir,
                                  percentile=cfg.get_path("dataset.grad_percentile"),
                                  lr_patch=cfg.get_path("dataset.lr_patch"))
    print(f"grad_thresh (p{cfg.get_path('dataset.grad_percentile')}) = {thr:.6f}")

    ds = RestorationDataset(cache_dir,
                            lr_patch=cfg.get_path("dataset.lr_patch"),
                            scale=cfg.get_path("dataset.scale"),
                            grad_thresh=thr,
                            crop_tries=cfg.get_path("dataset.crop_tries"))
    dl = DataLoader(ds, batch_size=cfg.get_path("train.batch_size"), shuffle=True,
                    num_workers=cfg.get_path("train.num_workers"),
                    pin_memory=True, persistent_workers=True, drop_last=True)

    n, t0 = 0, time.perf_counter()
    for i, b in enumerate(dl):
        n += b["lr"].shape[0]
        if i >= 50:
            break
    dt = time.perf_counter() - t0
    print(f"dataloader: {n / dt:,.0f} samples/s over {n} samples "
          f"(workers={cfg.get_path('train.num_workers')})")
    print("If this is below the model's step rate, the GPU is starving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
