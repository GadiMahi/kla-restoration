"""Model registry.

Final architecture: NAFNet U-Net backbone (encoder/decoder NAFBlocks, PixelShuffle SR tail, 
global bilinear residual) plus two targeted additions:
1. Bottleneck hybrid attention (MDTA + GDFN, Restormer-style) for global receptive field.
2. Dual-domain input conditioning (log1p + linear) to isolate multiplicative speckle noise.

Includes UNetDiscriminatorSN for the Stage-2 GAN fine-tune (zero inference cost).
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

def build_model(name: str = "bicubic", **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODELS)}")
    return MODELS[name](**kwargs)

class BicubicUpsample(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

# --------------------------------------------------------------------------
# NAFNet building blocks
# --------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # FIX: Force float32 so 1e-6 doesn't underflow to 0.0 in mixed precision
        x_f32 = x.float()
        mu = x_f32.mean(dim=1, keepdim=True)
        sigma = x_f32.var(dim=1, keepdim=True, unbiased=False)
        out = (x_f32 - mu) / torch.sqrt(sigma + self.eps)
        
        # Cast back to original dtype for speed
        return (out.type_as(x) * self.weight) + self.bias

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
        
        # FIX: Ensure contiguous memory layout for cuDNN depthwise conv engine on T4 GPUs
        out = self.conv1(out).contiguous()
        out = self.conv2(out).contiguous()
        out = self.sg1(out)
        out = self.sca(out)
        out = self.conv3(out)
        x = identity + out * self.beta

        identity2 = x
        out = self.norm2(x)
        out = self.conv4(out).contiguous()
        out = self.sg2(out)
        out = self.conv5(out)
        return identity2 + out * self.gamma

# --------------------------------------------------------------------------
# Transformer Bottleneck (Restormer-style MDTA + GDFN)
# --------------------------------------------------------------------------

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x).contiguous()).contiguous()
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        # FIX: Force float32 for softmax stability, preventing attention collapse
        attn = attn.float().softmax(dim=-1).type_as(attn)
        
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)
class GDFN(nn.Module):
    def __init__(self, dim, expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x.contiguous()).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# --------------------------------------------------------------------------
# Final Generator
# --------------------------------------------------------------------------

class NAFNet_UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, dim: int = 64, scale: int = 2, 
                 num_transformer_blocks: int = 2, num_heads: int = 4, use_log_input: bool = True):
        super().__init__()
        self.scale = scale
        self.use_log_input = use_log_input

        intro_in_channels = in_channels * 2 if use_log_input else in_channels
        self.intro = nn.Conv2d(intro_in_channels, dim, 3, 1, 1)

        self.enc1 = nn.Sequential(NAFBlock(dim), NAFBlock(dim))
        self.down = nn.Conv2d(dim, dim * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(NAFBlock(dim * 2), NAFBlock(dim * 2))

        middle_layers = [NAFBlock(dim * 2)]
        middle_layers += [TransformerBlock(dim * 2, num_heads) for _ in range(num_transformer_blocks)]
        middle_layers += [NAFBlock(dim * 2)]
        self.middle = nn.Sequential(*middle_layers)

        self.up = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.reduce = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        self.dec1 = nn.Sequential(NAFBlock(dim), NAFBlock(dim))

        self.upsample = nn.Sequential(nn.Conv2d(dim, out_channels * (scale ** 2), 3, 1, 1), nn.PixelShuffle(scale))

    def forward(self, x):
        shortcut = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)

        if self.use_log_input:
            x_log = torch.log1p(torch.clamp(x, min=0.0))
            feat_in = torch.cat([x, x_log], dim=1)
        else:
            feat_in = x

        out = self.intro(feat_in)
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

# --------------------------------------------------------------------------
# Training-only discriminator (Stage-2 GAN)
# --------------------------------------------------------------------------

class UNetDiscriminatorSN(nn.Module):
    def __init__(self, in_channels: int = 1, base_dim: int = 32):
        super().__init__()
        norm = nn.utils.spectral_norm
        self.conv0 = norm(nn.Conv2d(in_channels, base_dim, 3, 1, 1))
        self.conv1 = norm(nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1))
        self.conv2 = norm(nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1))
        self.conv3 = norm(nn.Conv2d(base_dim * 4, base_dim * 8, 4, 2, 1))
        self.conv4 = norm(nn.Conv2d(base_dim * 8, base_dim * 4, 3, 1, 1))
        self.conv5 = norm(nn.Conv2d(base_dim * 4, base_dim * 2, 3, 1, 1))
        self.conv6 = norm(nn.Conv2d(base_dim * 2, base_dim, 3, 1, 1))
        self.conv_out1 = norm(nn.Conv2d(base_dim, base_dim, 3, 1, 1))
        self.conv_out2 = norm(nn.Conv2d(base_dim, base_dim, 3, 1, 1))
        self.conv_final = nn.Conv2d(base_dim, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x0 = self.lrelu(self.conv0(x))
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = self.lrelu(self.conv4(x3)) + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = self.lrelu(self.conv5(x4)) + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = self.lrelu(self.conv6(x5)) + x0

        out = self.lrelu(self.conv_out1(x6))
        out = self.lrelu(self.conv_out2(out))
        return self.conv_final(out)

@register("nafnet")
def _nafnet(scale: int = 2, dim: int = 64, num_transformer_blocks: int = 2, num_heads: int = 4, use_log_input: bool = True, **kwargs) -> nn.Module:
    return NAFNet_UNet(in_channels=1, out_channels=1, dim=dim, scale=scale, num_transformer_blocks=num_transformer_blocks, num_heads=num_heads, use_log_input=use_log_input)

@register("bicubic")
def _bicubic(scale: int = 2, **_) -> nn.Module:
    return BicubicUpsample(scale)