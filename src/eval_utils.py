"""Metrics. KLA scores a fixed undisclosed weighting of PSNR, SSIM and LPIPS.

Also provides edge-stratified SSIM: global SSIM hides line smearing on wafer
images because most of the frame is flat. ssim_edge is the number that actually
tells you whether the model is blurring structure.
"""
from __future__ import annotations

import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

_LPIPS = None


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(_psnr(gt, np.clip(pred, 0, 1), data_range=1.0))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(_ssim(gt, np.clip(pred, 0, 1), data_range=1.0))


def lpips_score(pred: np.ndarray, gt: np.ndarray, device="cpu") -> float:
    """LPIPS expects 3-channel input in [-1,1]; grayscale is repeated."""
    global _LPIPS
    if _LPIPS is None:
        import lpips as _l

        _LPIPS = _l.LPIPS(net="alex").to(device).eval()

    def prep(a):
        t = torch.from_numpy(np.clip(a, 0, 1).astype(np.float32))[None, None]
        return (t.repeat(1, 3, 1, 1) * 2 - 1).to(device)

    with torch.no_grad():
        return float(_LPIPS(prep(pred), prep(gt)).item())


def sobel_mag(img: np.ndarray) -> np.ndarray:
    from scipy.ndimage import sobel

    return np.hypot(sobel(img, axis=0), sobel(img, axis=1))


def edge_weight(hr: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    """Per-pixel weight map for the Charbonnier term. Hand to the model team."""
    g = sobel_mag(hr)
    m = g.max()
    return 1.0 + alpha * (g / m if m > 0 else g)


def stratified_ssim(pred: np.ndarray, gt: np.ndarray, q: float = 0.75) -> dict:
    g = sobel_mag(gt)
    mask = g > np.quantile(g, q)
    _, smap = _ssim(gt, np.clip(pred, 0, 1), data_range=1.0, full=True)
    return {
        "ssim": float(smap.mean()),
        "ssim_edge": float(smap[mask].mean()) if mask.any() else float("nan"),
        "ssim_flat": float(smap[~mask].mean()) if (~mask).any() else float("nan"),
    }


def evaluate(pairs, with_lpips=True, device="cpu") -> dict:
    """pairs: iterable of (pred, gt) float arrays in [0,1]."""
    rows = []
    for pred, gt in pairs:
        r = {"psnr": psnr(pred, gt), **stratified_ssim(pred, gt)}
        if with_lpips:
            r["lpips"] = lpips_score(pred, gt, device)
        rows.append(r)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
