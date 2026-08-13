#!/usr/bin/env python3
"""Cluster images by structure and hold out one cluster as an OOD proxy.

    python scripts/make_splits.py --set data.root=/kaggle/input/kla-dataset
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, load_config   # noqa: E402
from src.io_utils import load_image, pair_by_stem     # noqa: E402
from src.splits import make_splits, save_splits, structure_features  # noqa: E402


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    root = Path(cfg.get_path("data.root"))
    pairs = pair_by_stem(root / cfg.get_path("data.gt_subdir"),
                         root / cfg.get_path("data.lr_subdir"))
    stems = [g.stem for g, _ in pairs]

    print(f"featurising {len(pairs)} images ...")
    feats = structure_features(load_image(g) for g, _ in pairs)

    sp = make_splits(stems, feats,
                     n_clusters=cfg.get_path("split.n_clusters"),
                     ood_cluster=cfg.get_path("split.ood_cluster"),
                     val_frac=cfg.get_path("split.val_frac"),
                     seed=cfg.get_path("split.seed"),
                     max_ood_frac=cfg.get_path("split.max_ood_frac", 0.25),
                     min_ood_n=cfg.get_path("split.min_ood_n", 150))
    save_splits(sp)

    print("held-out OOD cluster:", sp["ood_cluster"])
    print("cluster sizes:", dict(sorted(Counter(sp["clusters"].values()).items())))
    print(f"train={len(sp['train'])}  val_id={len(sp['val_id'])}  val_ood={len(sp['val_ood'])}")
    print("\nEyeball a few images per cluster before trusting this split.")
    print("val_ood is the PRIMARY metric; val_id is a sanity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
