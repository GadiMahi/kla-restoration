"""Model registry.

v3 change (on top of v2's 3-level U-Net fix): optional gradient
checkpointing on each encoder/decoder/bottleneck NAFBlock stack, gated
by `use_checkpoint` (default False here; train.py passes True by
default via train.grad_checkpoint). Trades ~20-30% recompute for a
large cut in activation memory -- the full-res enc1/dec1 stacks hold
the biggest activations (dim channels at full H,W) and benefit the
most; enc3/middle/dec2 help less since they're already at lower
resolution, but are checkpointed too since it's cheap to do. Only
active during training (model.training=True) -- checkpointing during
eval/inference is pure overhead with no memory pressure to relieve,
since there's no backward pass to save activations for. NAFBlock
internals, SimpleGate, LayerNorm2d, SCA, PixelShuffle SR tail, and the
global residual are all unchanged from v2 -- checkpointing does not
change the numerical output, only memory/compute trade-off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Single registry. (The original file had a second `_REGISTRY` dict that
# `register()` never populated, so `build_model()` always raised KeyError
# -- dead code, since train.py / inference.py both use MODELS[...] directly.
# Fixed here so both names point at the same registry.)
MODELS: dict = {}
_REGISTRY = MODELS


def register(name):
    """Decorator to register models for easy access via config strings."""
    def decorator(cls_or_func):
        MODELS[name] = cls_or_func
        return cls_or_func
    return decorator


def build_model(name: str = "bicubic", **kwargs) -> nn.Module:
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
    """3-level encoder/decoder NAFNet for joint denoise + 2x SR.

    Input must have H, W divisible by 4 (two stride-2 downsamples). Use
    `pad_to_multiple` / `crop_to_scale` (see inference.py) to handle
    arbitrary input sizes -- the network itself has no fixed-size
    assumptions, per the spec's requirement to generalize to 512x512
    eval images despite only training on 256->128 pairs.

    `use_checkpoint=True` wraps each stage's NAFBlock stack in gradient
    checkpointing during training (see module docstring) -- no effect
    on outputs, only on the memory/compute trade-off.
    """

    def __init__(self, in_channels=1, out_channels=1, dim=64, scale=2,
                 blocks=(2, 2, 2, 2, 2, 2), use_checkpoint=False):
        super().__init__()
        self.scale = scale
        self.use_checkpoint = use_checkpoint
        b_enc1, b_enc2, b_enc3, b_mid, b_dec2, b_dec1 = blocks

        self.intro = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Encoder
        self.enc1 = nn.Sequential(*[NAFBlock(dim) for _ in range(b_enc1)])
        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(b_enc2)])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1)
        self.enc3 = nn.Sequential(*[NAFBlock(dim * 4) for _ in range(b_enc3)])

        # Bottleneck (runs at 1/4 resolution, 1/16 the pixel count of the input)
        self.middle = nn.Sequential(*[NAFBlock(dim * 4) for _ in range(b_mid)])

        # Decoder level 2: dim*4 @ H/4 -> dim*2 @ H/2
        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, 1, 0)  # after concat with skip2
        self.dec2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(b_dec2)])

        # Decoder level 1: dim*2 @ H/2 -> dim @ H
        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, 1, 0)  # after concat with skip1
        self.dec1 = nn.Sequential(*[NAFBlock(dim) for _ in range(b_dec1)])

        # Upsampling tail for the 2x SR step
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def _run(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Runs a NAFBlock stack, checkpointing it during training if
        use_checkpoint is set. use_reentrant=False is the modern,
        recommended checkpoint mode; needs no other code changes here
        since every checkpointed input already carries requires_grad=True
        (it flows through self.intro's weights before reaching enc1)."""
        if self.use_checkpoint and self.training:
            return checkpoint(seq, x, use_reentrant=False)
        return seq(x)

    def forward(self, x):
        shortcut = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        out = self.intro(x)

        skip1 = self._run(self.enc1, out)          # dim      @ H
        out = self.down1(skip1)                     # dim*2    @ H/2
        skip2 = self._run(self.enc2, out)            # dim*2    @ H/2
        out = self.down2(skip2)                      # dim*4    @ H/4
        out = self._run(self.enc3, out)               # dim*4    @ H/4

        out = self._run(self.middle, out)              # dim*4    @ H/4  (bottleneck)

        out = self.up2(out)                             # dim*2    @ H/2
        out = torch.cat([out, skip2], dim=1)             # dim*4 @ H/2
        out = self.reduce2(out)                          # dim*2    @ H/2
        out = self._run(self.dec2, out)

        out = self.up1(out)                              # dim      @ H
        out = torch.cat([out, skip1], dim=1)             # dim*2 @ H
        out = self.reduce1(out)                          # dim      @ H
        out = self._run(self.dec1, out)

        out = self.upsample(out)                         # out_channels @ H*scale
        return out + shortcut


@register("nafnet")
def _nafnet(scale=2, **kwargs):
    return NAFNet_UNet(in_channels=1, out_channels=1, dim=64, scale=scale, **kwargs)


@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)