"""Model registry.

PLACEHOLDER: `bicubic` makes the full pipeline runnable end-to-end today, so
the I/O harness, padding, batching and timing can all be validated before the
network exists. The model team registers the real architecture here and
inference.py needs no changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(name: str = "bicubic", **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


class BicubicUpsample(nn.Module):
    """Zero-parameter baseline. Same interface as the real model."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)


# --- model team: add the real architecture below -------------------------
# @register("nafnet")
# def _nafnet(width=32, blocks=(2,2,4,8), scale=2, **_):
#     from .nafnet import NAFNetSR
#     return NAFNetSR(width=width, blocks=blocks, scale=scale)
