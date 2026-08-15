"""Model registry.

Final architecture: NAFNet-based U-Net for joint denoising + NxSR on grayscale
images. Same backbone as the earlier version (kept for the same H100
throughput) with one correctness fix -- see NAFNet_UNet docstring.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS: dict = {}


def register(name):
    """Decorator to register models for easy access via config strings."""
    def decorator(cls_or_func):
        MODELS[name] = cls_or_func
        return cls_or_func
    return decorator


def build_model(name: str = "nafnet", **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


class BicubicUpsample(nn.Module):
    """Zero-parameter baseline. Same interface as the real model."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw_channel = c * 2
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)

        ffn_channel = c * 2
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.conv3(self.sca(self.sg1(self.conv2(self.conv1(out)))))
        x = identity + out * self.beta

        identity2 = x
        out = self.norm2(x)
        out = self.conv5(self.sg2(self.conv4(out)))
        return identity2 + out * self.gamma


class NAFNet_UNet(nn.Module):
    """
    4-level NAFNet U-Net for joint denoise + scale-x super-resolution.

    Fix vs. the previous version: the global residual shortcut now clamps the
    raw noisy input to [0, 1] before bilinear-upsampling it back into the
    output. The degraded inputs can carry speckle spikes that exceed the
    clean intensity range (per the challenge's own data notes). Passing those
    spikes straight through an unclamped skip connection, then adding them to
    the network's output, is what produced the blown-out / clipped-white
    patches after the final clamp(0, 1) at eval time -- visible in several
    rows of your 30-sample OOD grid (the model-output column going flat white
    where input/GT still show texture). The encoder still sees the raw
    (unclamped) input, since the actual noise magnitude is useful signal for
    learning to denoise -- only the passthrough skip is clamped.
    """

    def __init__(self, in_channels=1, out_channels=1, dim=64, scale=2):
        super().__init__()
        self.scale = scale

        self.intro = nn.Conv2d(in_channels, dim, 3, 1, 1)

        self.enc1 = nn.Sequential(NAFBlock(dim), NAFBlock(dim))
        self.down = nn.Conv2d(dim, dim * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(NAFBlock(dim * 2), NAFBlock(dim * 2))

        self.middle = nn.Sequential(NAFBlock(dim * 2), NAFBlock(dim * 2))

        self.up = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.reduce = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.dec1 = nn.Sequential(NAFBlock(dim), NAFBlock(dim))

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        # Clamp only the passthrough skip -- see class docstring.
        x_safe = torch.clamp(x, 0.0, 1.0)
        shortcut = F.interpolate(x_safe, scale_factor=self.scale, mode="bilinear", align_corners=False)

        out = self.intro(x)  # encoder still sees the raw, unclamped signal

        skip = self.enc1(out)
        out = self.down(skip)
        out = self.enc2(out)

        out = self.middle(out)

        out = self.up(out)
        out = torch.cat([out, skip], dim=1)
        out = self.reduce(out)
        out = self.dec1(out)

        out = self.upsample(out)
        return out + shortcut


@register("nafnet")
def _nafnet(scale=2, **kwargs):
    return NAFNet_UNet(in_channels=1, out_channels=1, dim=64, scale=scale)


@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)