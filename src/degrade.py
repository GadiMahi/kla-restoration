"""Synthetic degradation for extra training pairs (explicitly permitted, 4B).

Two design decisions come straight from the problem statement:

  * "The three degradations may have been applied in any order" -> the order is
    sampled per image, using the mix ratio measured in 02_degradation.ipynb.
  * "Noise mechanisms remain the same; sampled levels may vary within a similar
    range" -> jitter is deliberately modest (+/-30%). Widening it further makes
    the model hedge and blur, which the spec explicitly penalises.

Nothing here clips: the out-of-range values are the point.
"""
from __future__ import annotations

import numpy as np

try:  # optional, only needed for some kernels
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from PIL import Image

_PIL_FILTERS = {
    "bicubic_aa": Image.BICUBIC,
    "bilinear_aa": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
    "area": Image.BOX,
    "nearest": Image.NEAREST,
}


def downsample(img: np.ndarray, kernel: str, scale: int = 2) -> np.ndarray:
    h, w = img.shape[-2:]
    size = (max(1, w // scale), max(1, h // scale))  # PIL wants (W, H)
    if kernel not in _PIL_FILTERS:
        raise ValueError(f"Unknown kernel {kernel!r}")
    out = Image.fromarray(img.astype(np.float32), mode="F").resize(size, _PIL_FILTERS[kernel])
    return np.asarray(out, dtype=np.float32)


def add_noise(img: np.ndarray, sigma_mult: float, sigma_add: float, rng) -> np.ndarray:
    """Speckle (multiplicative) then additive Gaussian. No clipping.

    NOTE: this only reproduces the a*mu^2 and c terms of a fitted variance
    curve. If the fit has a significant linear (b) term, use add_noise_varfit
    instead - otherwise most of the measured variance is silently dropped.
    """
    out = img
    if sigma_mult > 0:
        out = out * (1.0 + rng.normal(0.0, sigma_mult, out.shape).astype(np.float32))
    if sigma_add > 0:
        out = out + rng.normal(0.0, sigma_add, out.shape).astype(np.float32)
    return out.astype(np.float32)


def add_noise_varfit(img: np.ndarray, a: float, b: float, c: float, rng,
                     scale: float = 1.0) -> np.ndarray:
    """Signal-dependent noise matching a measured variance curve.

        var(r | mu) = a*mu^2 + b*mu + c

    This reproduces the observed second-order statistics regardless of how the
    variance decomposes into speckle / shot / read noise - which matters because
    those three components are strongly correlated in the fit and cannot be
    separated reliably from paired data alone.

    `scale` multiplies the noise standard deviation (the curriculum knob).
    """
    mu = np.clip(img, 0.0, None)
    var = a * mu * mu + b * mu + c
    sigma = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32) * np.float32(scale)
    return (img + sigma * rng.standard_normal(img.shape).astype(np.float32)).astype(np.float32)


def degrade(gt: np.ndarray, rng, cfg, width: float | None = None) -> np.ndarray:
    """Produce a NoisyLR-like image from a clean GT image.

    If cfg contains 'noise_var_fit' the measured variance curve is used (correct
    whenever the fit has a linear term). Otherwise falls back to the simple
    speckle+Gaussian parameterisation.
    """
    width = cfg["width"] if width is None else width
    j = float(cfg.get("jitter", 0.30)) * float(width)

    kernel = rng.choice(cfg["kernels"], p=np.asarray(cfg["kernel_p"], dtype=float))
    order = rng.choice(["noise_first", "noise_last"], p=np.asarray(cfg["order_mix"], dtype=float))

    fit = cfg.get("noise_var_fit")
    if fit:
        # Jitter the whole noise level, preserving the shape of the curve.
        scale = float(rng.uniform(1.0 - j, 1.0 + j))
        noise = lambda x: add_noise_varfit(  # noqa: E731
            x, float(fit["a_mult"]), float(fit["b_poisson"]), float(fit["c_additive"]),
            rng, scale=scale)
    else:
        s_mult = float(cfg["meas_mult"]) * float(rng.uniform(1.0 - j, 1.0 + j))
        s_add = float(cfg["meas_add"]) * float(rng.uniform(1.0 - j, 1.0 + j))
        noise = lambda x: add_noise(x, s_mult, s_add, rng)  # noqa: E731

    if order == "noise_first":
        return downsample(noise(gt), kernel)
    return noise(downsample(gt, kernel))
