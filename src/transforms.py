"""Normalisation contract. Every constant lives in artifacts/stats.json.

GT is already normalised to [0,1] by KLA, so scale_constant is expected to be
1.0 and this is mostly a verification layer plus a switch for an optional
log (homomorphic) transform if the noise turns out to be strongly
multiplicative.

Invariants:
  * The INPUT is never clipped. NoisyLR values outside [0,1] are intentional.
  * The OUTPUT is always clipped to [0,1]. GT is guaranteed to lie there and
    KLA does not renormalise for us.
  * denormalize(normalize(x)) == x  (unit tested in tests/test_transforms.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_STATS = {
    "scale_constant": 1.0,
    "log_transform": False,
    "log_eps": 0.01,
    "gt_valid_range": [0.0, 1.0],
}

_STATS: dict | None = None


def load_stats(path: str | Path | None = None) -> dict:
    global _STATS
    if _STATS is not None and path is None:
        return _STATS
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "artifacts" / "stats.json"
    if p.exists():
        with open(p) as f:
            stats = {**DEFAULT_STATS, **json.load(f)}
    else:
        stats = dict(DEFAULT_STATS)
    _STATS = stats
    return stats


def save_stats(stats: dict, path: str | Path | None = None) -> Path:
    global _STATS
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "artifacts" / "stats.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_STATS, **stats}
    with open(p, "w") as f:
        json.dump(merged, f, indent=2)
    _STATS = merged
    return p


def normalize(x, stats: dict | None = None):
    """float32 image on GT scale -> model input space. Never clips."""
    s = stats or load_stats()
    is_torch = hasattr(x, "clamp")
    y = x * float(s["scale_constant"])
    if s["log_transform"]:
        eps = float(s["log_eps"])
        if is_torch:
            import torch

            y = torch.log1p(torch.clamp(y, min=0.0) / eps)
        else:
            y = np.log1p(np.clip(y, 0.0, None) / eps)
    return y if is_torch else np.asarray(y, dtype=np.float32)


def denormalize(y, stats: dict | None = None):
    """Model output -> GT scale, clipped to the valid GT range."""
    s = stats or load_stats()
    lo, hi = s["gt_valid_range"]
    is_torch = hasattr(y, "clamp")
    if s["log_transform"]:
        eps = float(s["log_eps"])
        if is_torch:
            import torch

            y = torch.expm1(y) * eps
        else:
            y = np.expm1(y) * eps
    x = y / float(s["scale_constant"])
    return x.clamp(lo, hi) if is_torch else np.clip(x, lo, hi)
