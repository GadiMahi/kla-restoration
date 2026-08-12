"""Augmentation, ordered by value for THIS challenge.

KLA states the hidden test set contains unfamiliar image *content* but the same
degradation mechanisms. So content diversity matters more than noise diversity:

  1. scale_jitter - varies feature density. Primary generalisation lever.
  2. d4           - free, and valid for axis-aligned semiconductor layouts.
  3. cutblur      - teaches how much / where to restore; counters blurring.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def scale_jitter(gt: np.ndarray, rng, lo: float = 0.7, hi: float = 1.4) -> np.ndarray:
    """Resize the clean GT before degrading it, changing apparent feature size."""
    f = float(rng.uniform(lo, hi))
    if abs(f - 1.0) < 1e-3:
        return gt
    h, w = gt.shape[-2:]
    size = (max(8, int(round(w * f))), max(8, int(round(h * f))))
    out = Image.fromarray(gt.astype(np.float32), mode="F").resize(size, Image.BICUBIC)
    return np.asarray(out, dtype=np.float32)


def d4(lr: torch.Tensor, hr: torch.Tensor, generator=None):
    """Random element of the dihedral group. Apply on GPU - it is free there."""
    k = int(torch.randint(0, 4, (1,), generator=generator))
    flip = bool(torch.randint(0, 2, (1,), generator=generator))
    lr, hr = torch.rot90(lr, k, (-2, -1)), torch.rot90(hr, k, (-2, -1))
    if flip:
        lr, hr = lr.flip(-1), hr.flip(-1)
    return lr.contiguous(), hr.contiguous()


def cutblur(lr_up: torch.Tensor, hr: torch.Tensor, p: float = 0.7, alpha: float = 0.7,
            generator=None):
    """CutBlur (Yoo et al., CVPR 2020), adapted to a batch of tensors.

    lr_up must already be bicubically upsampled to the HR size.
    """
    if float(torch.rand(1, generator=generator)) > p:
        return lr_up, hr

    h, w = hr.shape[-2:]
    cut = float(torch.empty(1).uniform_(0.0, alpha, generator=generator))
    ch, cw = max(1, int(h * cut)), max(1, int(w * cut))
    y = int(torch.randint(0, max(1, h - ch + 1), (1,), generator=generator))
    x = int(torch.randint(0, max(1, w - cw + 1), (1,), generator=generator))

    out = lr_up.clone()
    if float(torch.rand(1, generator=generator)) < 0.5:
        out[..., y:y + ch, x:x + cw] = hr[..., y:y + ch, x:x + cw]
    else:
        out, keep = hr.clone(), lr_up[..., y:y + ch, x:x + cw]
        out[..., y:y + ch, x:x + cw] = keep
    return out, hr
