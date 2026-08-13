"""Noise-model estimation from paired GT / NoisyLR data.

Why not block statistics: block_reduce(clean, 8, mean) paired with
block_reduce(r, 8, var) is biased. Inside a block containing an edge, the mean
intensity is not representative, and multiplicative noise variance depends on
E[mu^2] rather than (E[mu])^2. High-contrast blocks therefore show excess
variance, which the quadratic fit absorbs into a spurious linear term.

Binning by per-PIXEL clean intensity avoids that entirely.
"""
from __future__ import annotations

import numpy as np


def binned_variance(clean: np.ndarray, resid: np.ndarray, nbins: int = 50,
                    lo: float = 0.0, hi: float = 1.0, min_count: int = 200):
    """Return (bin_centres, variance_per_bin, count_per_bin)."""
    c = clean.ravel()
    r = resid.ravel()
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(c, edges) - 1, 0, nbins - 1)

    n = np.bincount(idx, minlength=nbins).astype(np.float64)
    s1 = np.bincount(idx, weights=r, minlength=nbins)
    s2 = np.bincount(idx, weights=r * r, minlength=nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = s2 / n - (s1 / n) ** 2
    ok = n >= min_count
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres[ok], var[ok], n[ok]


def fit_variance_curve(centres, var, counts=None):
    """Non-negative fit of var(mu) = a*mu^2 + b*mu + c, weighted by bin count."""
    from scipy.optimize import nnls

    w = np.sqrt(counts) if counts is not None else np.ones_like(centres)
    A = np.stack([centres ** 2, centres, np.ones_like(centres)], axis=1) * w[:, None]
    coef, _ = nnls(A, var * w)
    return tuple(float(x) for x in coef)          # (a, b, c)


def estimate(pairs, downsample_fn, kernel, nbins=50, max_images=40):
    """Accumulate binned statistics across images, then fit once."""
    from .io_utils import load_image

    tot_n = np.zeros(nbins)
    tot_s1 = np.zeros(nbins)
    tot_s2 = np.zeros(nbins)
    edges = np.linspace(0.0, 1.0, nbins + 1)

    for gt_p, lr_p in pairs[:max_images]:
        gt, lr = load_image(gt_p), load_image(lr_p)
        clean = downsample_fn(gt, kernel)
        if clean.shape != lr.shape:
            continue
        r = (lr - clean).ravel()
        idx = np.clip(np.digitize(clean.ravel(), edges) - 1, 0, nbins - 1)
        tot_n += np.bincount(idx, minlength=nbins)
        tot_s1 += np.bincount(idx, weights=r, minlength=nbins)
        tot_s2 += np.bincount(idx, weights=r * r, minlength=nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = tot_s2 / tot_n - (tot_s1 / tot_n) ** 2
    ok = tot_n >= 200
    centres = 0.5 * (edges[:-1] + edges[1:])
    a, b, c = fit_variance_curve(centres[ok], var[ok], tot_n[ok])
    return dict(a_mult=a, b_poisson=b, c_additive=c,
                centres=centres[ok].tolist(), var=var[ok].tolist(),
                counts=tot_n[ok].tolist())


def fit_kernel_lstsq(pairs, load_fn, ksize=5, scale=2, n_images=12, samples=800,
                     seed=0, ridge=1e-4):
    """Kernel estimate by regularised least squares (GT patches -> LR pixels).

    Zero-mean noise averages out of the normal equations, so this is far less
    noise-sensitive than comparing low-passed images. But neighbouring HR pixels
    are highly correlated on smooth wafer imagery, so the design matrix is
    ill-conditioned and plain least squares returns wildly oscillating weights
    that still fit. Ridge regularisation is required, and the result is still
    only weakly identified - treat it as indicative, not definitive.
    """
    rng = np.random.default_rng(seed)
    A, y = [], []
    pad = ksize // 2
    for gt_p, lr_p in pairs[:n_images]:
        gt, lr = load_fn(gt_p), load_fn(lr_p)
        gp = np.pad(gt, pad, mode="reflect")
        H, W = lr.shape
        ys = rng.integers(0, H, samples)
        xs = rng.integers(0, W, samples)
        for yy, xx in zip(ys, xs):
            patch = gp[yy * scale: yy * scale + ksize, xx * scale: xx * scale + ksize]
            if patch.shape != (ksize, ksize):
                continue
            A.append(patch.ravel())
            y.append(lr[yy, xx])
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = A.shape[1]
    G = A.T @ A
    k = np.linalg.solve(G + ridge * np.trace(G) / n * np.eye(n), A.T @ y)
    return k.reshape(ksize, ksize)


def compare_to_known(K, downsample_fn, kernels, size=64, scale=2, seed=0):
    """Which named kernel does the recovered numeric kernel behave like?"""
    rng = np.random.default_rng(seed)
    probe = rng.random((size, size)).astype(np.float32)
    pad = K.shape[0] // 2
    pp = np.pad(probe, pad, mode="reflect")
    est = np.array([[np.sum(pp[i * scale:i * scale + K.shape[0],
                               j * scale:j * scale + K.shape[1]] * K)
                     for j in range(size // scale)] for i in range(size // scale)])
    out = {}
    for name in kernels:
        try:
            ref = downsample_fn(probe, name)
            out[name] = float(np.mean((ref - est) ** 2))
        except Exception:
            pass
    return out
