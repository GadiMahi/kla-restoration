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

MODELS = {}
def register(name):
    """Decorator to register models for easy access via config strings."""
    def decorator(cls_or_func):
        MODELS[name] = cls_or_func
        return cls_or_func
    return decorator

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



class LayerNorm2d(nn.Module):
    """
    Custom 2D LayerNorm.
    Standard nn.LayerNorm requires permuting tensors from (B, C, H, W) to (B, H, W, C).
    This native 2D implementation avoids memory-intensive permute operations,
    significantly speeding up end-to-end H100 inference times.
    """
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
    """
    Replaces traditional activation functions (ReLU, GELU).
    Splits the feature map in half along the channel dimension and multiplies them.
    Maintains non-linearity while reducing computational overhead.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Computes attention weights without multi-layer perceptrons or Sigmoids.
    Allows the network to isolate speckle noise channels from true structural edges.
    """
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFBlock(nn.Module):
    """
    The core feature extraction block.
    Configured with a DW_Expand ratio of 2 and FFN_Expand ratio of 2.
    """
    def __init__(self, c):
        super().__init__()
        dw_channel = c * 2
        
        # Spatial feature extraction
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, stride=1, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, stride=1, padding=0)

        # Feed Forward Network
        ffn_channel = c * 2
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, stride=1, padding=0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, stride=1, padding=0)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        # Learnable scaling parameters to stabilize deep network training
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        
        # Spatial modeling
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sg1(out)
        out = self.sca(out)
        out = self.conv3(out)
        x = identity + out * self.beta
        
        identity2 = x
        
        # Feed-forward modeling
        out = self.norm2(x)
        out = self.conv4(out)
        out = self.sg2(out)
        out = self.conv5(out)
        
        return identity2 + out * self.gamma


# --- Main Architecture ---

class NAFNetSR(nn.Module):
    """
    End-to-end Joint Denoising and Super-Resolution architecture.
    Takes degraded grayscale inputs and upscales them by `scale`, removing noise.
    """
    def __init__(self, in_channels=1, out_channels=1, dim=64, num_blocks=8, scale=2):
        super().__init__()
        self.scale = scale
        
        # 1. Feature Extraction (projects 1-channel grayscale to dense features)
        self.intro = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        
        # 2. Deep Feature Cleaning (filters out speckle and Gaussian noise)
        self.body = nn.Sequential(*[NAFBlock(dim) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        
        # 3. Upsampling (Sub-pixel convolution to avoid checkerboard artifacts)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, out_channels * (scale ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # Global Residual Shortcut: Upscale the degraded image via standard bilinear interpolation.
        # This acts as the structural baseline so the network only has to learn the high-frequency residual
        # (i.e., restoring missing edges and subtracting the speckle noise).
        shortcut = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        out = self.intro(x)
        res = self.body(out)
        out = out + self.body_tail(res)
        
        # Network predicts the residual detail and adds it to the baseline
        out = self.upsample(out)
        return out + shortcut


# --- Registry Hooks ---

@register("nafnet")
def _nafnet(scale=2, **kwargs):
    """
    Factory function to initialize the model.
    Defaults to 8 blocks and 64 feature dimensions, which strikes an optimal 
    balance between perceptual quality metrics (LPIPS/SSIM) and inference speed.
    """
    return NAFNetSR(in_channels=1, out_channels=1, dim=64, num_blocks=8, scale=scale)

@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)
