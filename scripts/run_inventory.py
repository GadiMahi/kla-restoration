#!/usr/bin/env python3
"""BLOCK A - run this before anything else.

Answers, in order of consequence:
  1. What format/dtype is the ground truth, and can we reproduce it bit-exactly?
     (KLA scores images exactly as saved and never renormalises.)
  2. Are pairs complete, grayscale, and exactly 2x?
  3. How far outside [0,1] does NoisyLR actually go?
  4. Is the intensity range stable enough for a fixed global normalisation?

Writes artifacts/inventory.csv and seeds artifacts/stats.json.

    python scripts/run_inventory.py --set data.root=/kaggle/input/kla-dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, load_config          # noqa: E402
from src.io_utils import (detect_format, load_image, list_images,  # noqa: E402
                          pair_by_stem, round_trip_check)
from src.transforms import save_stats                         # noqa: E402


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--limit", type=int, default=0, help="inspect only N pairs (quick pass)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    gt_dir = root / cfg.get_path("data.gt_subdir")
    lr_dir = root / cfg.get_path("data.lr_subdir")

    if not gt_dir.exists() or not lr_dir.exists():
        print(f"!! Not found:\n   GT: {gt_dir}\n   LR: {lr_dir}")
        print("   Existing subdirectories of data.root:")
        for p in sorted(Path(root).glob("*"))[:40] if Path(root).exists() else []:
            print("     ", p.name)
        return 2

    pairs = pair_by_stem(gt_dir, lr_dir)
    n_gt, n_lr = len(list_images(gt_dir)), len(list_images(lr_dir))
    print(f"GT files: {n_gt}   LR files: {n_lr}   matched pairs: {len(pairs)}")
    if len(pairs) < min(n_gt, n_lr):
        print(f"!! {min(n_gt, n_lr) - len(pairs)} file(s) failed to pair by stem")
    if not pairs:
        return 2

    # ---- 1. Format contract -------------------------------------------------
    print("\n=== FORMAT CONTRACT (highest-risk check) ===")
    fmt = detect_format(pairs[0][0])
    rt = round_trip_check(pairs[0][0])
    print(f"GT format : {fmt}")
    print(f"LR format : {detect_format(pairs[0][1])}")
    print(f"round-trip: max_abs_error={rt['max_abs_error']:.3e} "
          f"allowed={rt['allowed']:.3e} -> {'PASS' if rt['passed'] else 'FAIL'}")
    if not rt["passed"]:
        print("!! STOP. Saving does not reproduce the GT values. Fix io_utils "
              "before training - every downstream metric is capped by this.")

    # ---- 2-4. Per-pair statistics ------------------------------------------
    todo = pairs[: args.limit] if args.limit else pairs
    rows = []
    for gt_p, lr_p in todo:
        g, l = load_image(gt_p), load_image(lr_p)
        rows.append(dict(
            stem=gt_p.stem, gt_h=g.shape[0], gt_w=g.shape[1],
            lr_h=l.shape[0], lr_w=l.shape[1],
            scale_h=g.shape[0] / l.shape[0], scale_w=g.shape[1] / l.shape[1],
            gt_min=g.min(), gt_max=g.max(), gt_mean=g.mean(), gt_std=g.std(),
            lr_min=l.min(), lr_max=l.max(), lr_mean=l.mean(), lr_std=l.std(),
            frac_above1=float((l > 1).mean()), frac_below0=float((l < 0).mean()),
        ))
    df = pd.DataFrame(rows)
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    df.to_csv(out / "inventory.csv", index=False)

    print("\n=== SHAPES ===")
    print(df.groupby(["gt_h", "gt_w", "lr_h", "lr_w"]).size().rename("count").to_string())
    bad = df[(df.scale_h != cfg.get_path("dataset.scale")) |
             (df.scale_w != cfg.get_path("dataset.scale"))]
    print(f"pairs not exactly {cfg.get_path('dataset.scale')}x: {len(bad)}")

    print("\n=== RANGES ===")
    print(f"GT  : [{df.gt_min.min():.4f}, {df.gt_max.max():.4f}]  "
          f"(spec says GT is normalised to [0,1])")
    print(f"LR  : [{df.lr_min.min():.4f}, {df.lr_max.max():.4f}]")
    print(f"LR pixels > 1 : {df.frac_above1.mean() * 100:.3f}% of pixels "
          f"(max per-image {df.frac_above1.max() * 100:.3f}%)")
    print(f"LR pixels < 0 : {df.frac_below0.mean() * 100:.3f}%")

    cv = float(df.gt_mean.std() / (df.gt_mean.mean() + 1e-9))
    print(f"\n=== NORMALISATION DECISION ===")
    print(f"CV of per-image GT means = {cv:.4f}")
    print("-> per-image GT means are tightly grouped" if cv < 0.15
          else "-> per-image GT means vary widely")
    print("   NOTE: GT is already normalised to [0,1] by KLA, so a high CV here")
    print("   reflects genuine CONTENT diversity (different structures have")
    print("   different brightness), not a calibration problem to correct.")
    print("   Keep scale_constant=1.0. Per-image normalisation is the classic")
    print("   out-of-distribution failure mode and there is nothing to gain here.")

    stats = {
        "scale_constant": 1.0,
        "log_transform": False,
        "log_eps": 0.01,
        "gt_valid_range": [0.0, 1.0],
        "gt_format": fmt.to_dict(),
        "roundtrip_passed": bool(rt["passed"]),
        "n_pairs": len(pairs),
        "cv_image_means": cv,
        "frac_lr_above_1": float(df.frac_above1.mean()),
    }
    save_stats(stats)
    print(f"\nwrote artifacts/stats.json and artifacts/inventory.csv")
    print(json.dumps({k: v for k, v in stats.items() if k != "gt_format"}, indent=2))
    return 0 if rt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
