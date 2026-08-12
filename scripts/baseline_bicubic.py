#!/usr/bin/env python3
"""Mandatory baseline (KLA spec 4D: "compare at least one baseline").

Bicubic upsampling needs no model, so this gives the team a number to beat on
day one and validates eval_utils before any training happens.

    python scripts/baseline_bicubic.py --set data.root=/kaggle/input/kla-dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, load_config  # noqa: E402
from src.eval_utils import evaluate                   # noqa: E402
from src.io_utils import load_image, pair_by_stem     # noqa: E402


def bicubic_up(lr: np.ndarray, scale: int = 2) -> np.ndarray:
    t = torch.from_numpy(lr)[None, None]
    up = F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False)
    return up[0, 0].numpy().clip(0, 1)


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    pairs = pair_by_stem(root / cfg.get_path("data.gt_subdir"),
                         root / cfg.get_path("data.lr_subdir"))[: args.limit]

    scale = cfg.get_path("dataset.scale")
    results = evaluate(((bicubic_up(load_image(l), scale), load_image(g)) for g, l in pairs),
                       with_lpips=not args.no_lpips)

    print(json.dumps(results, indent=2))
    out = Path("results"); out.mkdir(exist_ok=True)
    with open(out / "baseline_bicubic.json", "w") as f:
        json.dump({"n": len(pairs), "metrics": results}, f, indent=2)
    print(f"\nwrote results/baseline_bicubic.json  (n={len(pairs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
