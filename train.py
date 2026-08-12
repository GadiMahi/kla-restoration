#!/usr/bin/env python3
"""Training entry point - OWNED BY THE MODEL TEAM.

The data side is ready; this stub shows how to consume it.

    python train.py --set data.root=/kaggle/input/kla-dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import add_config_args, load_config


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    # import torch
    # from torch.utils.data import DataLoader
    # from src.dataset import RestorationDataset
    # from src.splits import load_splits
    # from src.model import build_model
    # from src.eval_utils import edge_weight, stratified_ssim
    #
    # torch.manual_seed(cfg.get_path("train.seed"))
    # sp = load_splits()
    # train_ds = RestorationDataset(cfg.get_path("cache.dir"), stems=sp["train"], ...)
    # val_ood  = RestorationDataset(cfg.get_path("cache.dir"), stems=sp["val_ood"],
    #                               train=False)          # <- PRIMARY metric
    # model = build_model("nafnet", scale=cfg.get_path("dataset.scale"))
    # loss  = l1*charbonnier(w=edge_weight(hr)) + l2*ssim_loss + l3*lpips_loss

    raise SystemExit("train.py not implemented yet - model team")


if __name__ == "__main__":
    raise SystemExit(main())
