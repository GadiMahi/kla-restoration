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
    NAFNet U-Net for joint denoise + scale-x super-resolution.

    Note: this is a 2-level U-Net (one down/up stage) -- despite the "4-level"
    description in the original writeup, the code only ever had a single
    downsample/upsample pair. See NAFNet_UNet_v2 below for an actual 3-level
    version with PixelUnshuffle downsampling, a bottleneck attention block,
    and a refinement head, aimed at the texture-hallucination problem.

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


class GlobalContextAttention(nn.Module):
    """Single-head full self-attention. Only ever applied at the bottleneck
    (smallest spatial resolution, e.g. 32x32 = 1024 tokens for a 128px LR
    input), where O(n^2) attention is cheap. NAFBlock's channel attention is
    global-average-pooled -- it can't let a texture patch in one corner
    inform synthesis elsewhere in the image. This can, which is what
    repeating/self-similar textures (stone grain, grass, brick) need."""

    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.q(y).reshape(b, c, h * w).permute(0, 2, 1)  # B, N, C
        k = self.k(y).reshape(b, c, h * w)                   # B, C, N
        v = self.v(y).reshape(b, c, h * w).permute(0, 2, 1)  # B, N, C
        attn = torch.softmax((q @ k) * self.scale, dim=-1)   # B, N, N
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


def _pixel_unshuffle_down(in_c, out_c):
    """Lossless downsample: PixelUnshuffle(2) is a pure reshape (no dropped
    pixels, no learned params, zero FLOPs) that turns (C, H, W) into
    (4C, H/2, W/2); the 1x1 conv after it just fixes the channel count.
    Strictly more information-preserving than the strided conv v1 uses, and
    cheaper."""
    return nn.Sequential(
        nn.PixelUnshuffle(2),
        nn.Conv2d(in_c * 4, out_c, 1, 1, 0),
    )


def _pixel_shuffle_up(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c * 4, 1, 1, 0),
        nn.PixelShuffle(2),
    )


class NAFNet_UNet_v2(nn.Module):
    """
    3-level NAFNet U-Net (two down/up stages instead of v1's one) aimed
    squarely at the "painterly / over-smoothed micro-texture" problem:

      - A real third resolution level (bottleneck at 1/4 res, not 1/2) for a
        meaningfully larger receptive field -- added where it's cheap, since
        deeper levels run on 4x fewer pixels each step down.
      - PixelUnshuffle/PixelShuffle downsampling instead of strided conv --
        lossless and cheaper than v1's stride-2 conv.
      - One GlobalContextAttention block at the bottleneck only, so it stays
        H100-throughput-friendly (runs once, at the smallest resolution).
      - A small residual refinement head right before the SR tail, whose
        only job is sharpening -- rather than asking the same blocks to
        denoise, reconstruct, and hallucinate detail all at once.

    NOT weight-compatible with v1 (channel layout differs) -- registered
    separately as "nafnet_v2" rather than replacing "nafnet", so v1 stays
    available as a baseline to compare against.
    """

    def __init__(self, in_channels=1, out_channels=1, dim=64, scale=2, mid_blocks=4):
        super().__init__()
        self.scale = scale

        self.intro = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Level 0: full res, dim channels
        self.enc0 = nn.Sequential(NAFBlock(dim), NAFBlock(dim))
        self.down0 = _pixel_unshuffle_down(dim, dim * 2)

        # Level 1: 1/2 res, dim*2 channels
        self.enc1 = nn.Sequential(NAFBlock(dim * 2), NAFBlock(dim * 2))
        self.down1 = _pixel_unshuffle_down(dim * 2, dim * 4)

        # Level 2 (bottleneck): 1/4 res, dim*4 channels
        self.middle = nn.Sequential(*[NAFBlock(dim * 4) for _ in range(mid_blocks)])
        self.attn = GlobalContextAttention(dim * 4)

        self.up1 = _pixel_shuffle_up(dim * 4, dim * 2)
        self.reduce1 = nn.Conv2d(dim * 4, dim * 2, 1, 1, 0)
        self.dec1 = nn.Sequential(NAFBlock(dim * 2), NAFBlock(dim * 2))

        self.up0 = _pixel_shuffle_up(dim * 2, dim)
        self.reduce0 = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.dec0 = nn.Sequential(NAFBlock(dim), NAFBlock(dim))

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        x_safe = torch.clamp(x, 0.0, 1.0)
        shortcut = F.interpolate(x_safe, scale_factor=self.scale, mode="bilinear", align_corners=False)

        out = self.intro(x)

        skip0 = self.enc0(out)
        out = self.down0(skip0)

        skip1 = self.enc1(out)
        out = self.down1(skip1)

        out = self.middle(out)
        out = self.attn(out)

        out = self.up1(out)
        out = self.reduce1(torch.cat([out, skip1], dim=1))
        out = self.dec1(out)

        out = self.up0(out)
        out = self.reduce0(torch.cat([out, skip0], dim=1))
        out = self.dec0(out)

        out = out + self.refine(out)
        out = self.upsample(out)
        return out + shortcut


@register("nafnet")
def _nafnet(scale=2, **kwargs):
    return NAFNet_UNet(in_channels=1, out_channels=1, dim=64, scale=scale)


@register("nafnet_v2")
def _nafnet_v2(scale=2, **kwargs):
    return NAFNet_UNet_v2(in_channels=1, out_channels=1, dim=64, scale=scale)


@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)