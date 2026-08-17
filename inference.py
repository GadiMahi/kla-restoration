#!/usr/bin/env python3
"""Standalone inference (KLA spec, section 4C).

    python inference.py --input_dir <degraded> --output_dir <restored>

Requirements this satisfies:
  * input-directory and output-directory arguments, no source edits needed
  * loads every degraded image, restores it, writes to the output directory
  * preserves filename stems and the ground-truth file format
  * batched GPU execution, grouped by shape
  * reports the per-stage timing breakdown, because KLA's runtime measurement
    includes disk read, transfers, pre/post-processing and saving - not just
    the forward pass
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import add_config_args, load_config
from src.io_utils import ImageFormat, detect_format, list_images, load_image, output_path, save_image
from src.model import build_model
from src.transforms import denormalize, load_stats, normalize


class Stopwatch:
    def __init__(self):
        self.t = defaultdict(float)

    def add(self, key, dt):
        self.t[key] += dt

    def report(self, n):
        total = sum(self.t.values())
        lines = [f"{'stage':<18}{'seconds':>10}{'%':>8}{'ms/img':>10}"]
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            lines.append(f"{k:<18}{v:>10.3f}{100 * v / max(total, 1e-9):>7.1f}%{1000 * v / max(n, 1):>10.2f}")
        lines.append(f"{'TOTAL':<18}{total:>10.3f}{100:>7.1f}%{1000 * total / max(n, 1):>10.2f}")
        return "\n".join(lines)


def pad_to_multiple(x: torch.Tensor, m: int):
    if m <= 1:
        return x, (0, 0)
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)


def main() -> int:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    
    # Updated Defaults: Automatically load the nafnet model and the pushed LFS weights
    default_weights_path = str(Path(__file__).resolve().parent / "weights" / "best_nafnet.pt")
    ap.add_argument("--weights", default=default_weights_path)
    ap.add_argument("--model", default="nafnet")
    
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--timing_json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    stats = load_stats()
    device = torch.device(args.device or cfg.get_path("inference.device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    bs = args.batch_size or cfg.get_path("inference.batch_size", 8)
    pad_m = cfg.get_path("inference.pad_multiple", 1)
    amp = bool(cfg.get_path("inference.amp", True)) and device.type == "cuda"

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_images(in_dir)
    if not files:
        print(f"No images found in {in_dir}")
        return 2

    # Output format: whatever the GT format was, recorded at inventory time.
    gf = stats.get("gt_format")
    fmt = ImageFormat.from_dict(gf) if gf else detect_format(files[0])

    # Instantiate model
    model = build_model(args.model, scale=cfg.get_path("dataset.scale", 2)).to(device).eval()
    
    if args.weights and Path(args.weights).exists():
        print(f"Loading weights from: {args.weights}")
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd.get("model", sd))
    else:
        print(f"WARNING: Weights file not found at {args.weights}. Running with uninitialized weights!")

    sw = Stopwatch()
    pool = ThreadPoolExecutor(max_workers=8)
    wall = time.perf_counter()

    # Group by shape so batches are rectangular.
    t0 = time.perf_counter()
    loaded = list(pool.map(load_image, files))
    sw.add("disk_read", time.perf_counter() - t0)

    by_shape: dict = defaultdict(list)
    for p, a in zip(files, loaded):
        by_shape[a.shape].append((p, a))

    pending = []
    for shape, items in by_shape.items():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]

            t0 = time.perf_counter()
            batch = np.stack([normalize(a, stats) for _, a in chunk])[:, None]
            x = torch.from_numpy(batch)
            sw.add("preprocess", time.perf_counter() - t0)

            t0 = time.perf_counter()
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            sw.add("h2d", time.perf_counter() - t0)

            t0 = time.perf_counter()
            with torch.no_grad():
                xp, (ph, pw) = pad_to_multiple(x, pad_m)
                with torch.autocast("cuda", enabled=amp):
                    y = model(xp)
                s = cfg.get_path("dataset.scale", 2)
                if ph or pw:
                    y = y[..., : y.shape[-2] - ph * s, : y.shape[-1] - pw * s]
            if device.type == "cuda":
                torch.cuda.synchronize()
            sw.add("model", time.perf_counter() - t0)

            t0 = time.perf_counter()
            y = denormalize(y.float(), stats).cpu().numpy()[:, 0]
            sw.add("d2h_post", time.perf_counter() - t0)

            t0 = time.perf_counter()
            for (p, _), arr in zip(chunk, y):
                pending.append(pool.submit(save_image, arr, output_path(p, out_dir, fmt), fmt))
            sw.add("save_submit", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for f in pending:
        f.result()
    pool.shutdown()
    sw.add("save_flush", time.perf_counter() - t0)

    wall = time.perf_counter() - wall
    n = len(files)
    print(f"\nrestored {n} images -> {out_dir}")
    print(f"format: {fmt}")
    print(f"\n{sw.report(n)}")
    print(f"\nwall clock: {wall:.3f}s   throughput: {n / wall:.2f} img/s   "
          f"({1000 * wall / n:.2f} ms/img)   device={device}  batch={bs}")

    if args.timing_json:
        with open(args.timing_json, "w") as f:
            json.dump({"n": n, "wall_s": wall, "img_per_s": n / wall,
                       "device": str(device), "batch_size": bs,
                       "stages": dict(sw.t)}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())