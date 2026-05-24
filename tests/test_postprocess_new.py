"""Tests for added-mass and FFT post-processing helpers."""
from __future__ import annotations

import math

import torch

from tensorlbm.postprocess import (
    compute_added_mass_2d,
    compute_added_mass_3d,
    compute_strouhal_fft,
)


def test_strouhal_fft_sine() -> None:
    t = torch.arange(0, 1000, dtype=torch.float32)
    signal = torch.sin(2.0 * math.pi * 0.1 * t)
    assert abs(compute_strouhal_fft(signal) - 0.1) < 0.005


def test_added_mass_2d_smoke() -> None:
    t = torch.arange(256, dtype=torch.float32)
    omega = 0.1
    motion = 0.5 * torch.sin(omega * t)
    force = -2.0 * omega**2 * motion - 0.4 * omega * torch.gradient(motion, spacing=1.0)[0]
    added_mass, damping = compute_added_mass_2d(force, force, motion, omega)
    assert isinstance(added_mass, float)
    assert isinstance(damping, float)


def test_added_mass_3d_smoke() -> None:
    t = torch.arange(256, dtype=torch.float32)
    omega = 0.1
    motion = 0.5 * torch.sin(omega * t)
    force = -1.5 * omega**2 * motion - 0.2 * omega * torch.gradient(motion, spacing=1.0)[0]
    added_mass, damping = compute_added_mass_3d(force, force, force, motion, omega)
    assert isinstance(added_mass, float)
    assert isinstance(damping, float)
