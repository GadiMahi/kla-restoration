"""Leakage-free splits, including an out-of-distribution proxy.

KLA's hidden test set contains unfamiliar image *content*. We approximate that
by clustering images on structure signature (radial FFT profile + intensity
moments) and holding out one entire cluster. That held-out cluster is the
number worth optimising; the in-distribution split is only a sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def radial_fft_profile(img: np.ndarray, nbins: int = 16) -> np.ndarray:
    f = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())))
    h, w = f.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    r = (r / (r.max() + 1e-9) * (nbins - 1)).astype(int)
    prof = np.array([f[r == b].mean() if np.any(r == b) else 0.0 for b in range(nbins)])
    return np.log1p(prof)


def structure_features(images) -> np.ndarray:
    rows = []
    for im in images:
        rows.append(np.concatenate([
            radial_fft_profile(im),
            [im.mean(), im.std(), np.percentile(im, 95) - np.percentile(im, 5)],
        ]))
    return np.asarray(rows, dtype=np.float32)


def choose_ood_cluster(labels, max_frac=0.25, min_n=150) -> int:
    """Pick a held-out cluster big enough to be a trustworthy metric.

    A tiny cluster (say 35 of 3200 images) is an outlier bucket, not a
    distribution: the variance on it swamps any real difference between models.
    """
    import collections

    counts = collections.Counter(labels.tolist())
    n = len(labels)
    ok = [(c, k) for k, c in counts.items() if min_n <= c <= max_frac * n]
    if ok:
        return int(max(ok)[1])           # largest cluster that still fits the cap
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def make_splits(stems, features, n_clusters=6, ood_cluster=None, val_frac=0.1, seed=1337,
                max_ood_frac=0.25, min_ood_n=150) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x)

    if ood_cluster is None:
        ood_cluster = choose_ood_cluster(labels, max_ood_frac, min_ood_n)

    rng = np.random.default_rng(seed)
    ood = [s for s, l in zip(stems, labels) if l == ood_cluster]
    rest = [s for s, l in zip(stems, labels) if l != ood_cluster]
    rng.shuffle(rest)
    n_val = max(1, int(len(rest) * val_frac))

    if len(ood) < min_ood_n:
        print(f"!! val_ood has only {len(ood)} images - too few for a stable metric. "
              f"Cluster sizes: {dict(sorted(__import__('collections').Counter(labels.tolist()).items()))}")

    return {
        "train": sorted(rest[n_val:]),
        "val_id": sorted(rest[:n_val]),
        "val_ood": sorted(ood),
        "clusters": {s: int(l) for s, l in zip(stems, labels)},
        "ood_cluster": ood_cluster,
        "seed": seed,
    }


def save_splits(splits: dict, path="artifacts/splits.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(splits, f, indent=2)
    return p


def load_splits(path="artifacts/splits.json") -> dict:
    with open(path) as f:
        return json.load(f)
