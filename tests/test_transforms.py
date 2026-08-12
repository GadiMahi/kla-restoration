"""Run before training: python -m pytest tests/ -q"""
import numpy as np
import torch

from src.transforms import denormalize, normalize, save_stats


def _rt(stats):
    save_stats(stats)
    x = np.random.rand(64, 64).astype(np.float32)
    out_np = denormalize(normalize(x, stats), stats)
    assert np.allclose(out_np, x, atol=1e-5), "numpy round-trip failed"
    t = torch.from_numpy(x)
    out_t = denormalize(normalize(t, stats), stats)
    assert torch.allclose(out_t, t, atol=1e-5), "torch round-trip failed"


def test_identity_roundtrip():
    _rt({"scale_constant": 1.0, "log_transform": False})


def test_log_roundtrip():
    _rt({"scale_constant": 1.0, "log_transform": True, "log_eps": 0.01})


def test_output_is_clipped_input_is_not():
    stats = {"scale_constant": 1.0, "log_transform": False}
    noisy = np.array([[-0.2, 1.4]], dtype=np.float32)
    assert normalize(noisy, stats).min() < 0, "input must NOT be clipped"
    assert denormalize(noisy, stats).max() <= 1.0, "output MUST be clipped"
