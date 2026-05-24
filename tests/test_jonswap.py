"""Tests for JONSWAP wave-boundary helpers."""
from __future__ import annotations

import torch

from tensorlbm import equilibrium3d
from tensorlbm.wave_bc import apply_jonswap_inlet_3d, jonswap_spectrum, jonswap_wave_velocity_3d


def test_jonswap_spectrum_positive() -> None:
    omega = torch.linspace(0.1, 3.0, 64)
    spectrum = jonswap_spectrum(omega, omega_p=1.0)
    assert torch.all(spectrum >= 0.0)


def test_jonswap_spectrum_peak() -> None:
    omega_p = 1.0
    omega = torch.linspace(0.5, 1.5, 201)
    spectrum = jonswap_spectrum(omega, omega_p=omega_p)
    peak_omega = omega[int(torch.argmax(spectrum).item())]
    assert abs(float(peak_omega.item()) - omega_p) < 1e-6


def test_jonswap_wave_velocity_shape() -> None:
    omega = torch.tensor([1.0, 1.2])
    amplitude = torch.tensor([0.05, 0.03])
    phase = torch.tensor([0.0, 0.5])
    k = torch.tensor([0.2, 0.3])
    ux, uy, uz = jonswap_wave_velocity_3d(
        6,
        5,
        0,
        omega,
        amplitude,
        phase,
        k,
        3.0,
        0.0,
        0.1,
        torch.device("cpu"),
    )
    assert ux.shape == uy.shape == uz.shape == (6, 5)


def test_apply_jonswap_inlet_shape() -> None:
    rho = torch.ones((4, 5, 6))
    zeros = torch.zeros_like(rho)
    f = equilibrium3d(rho, zeros, zeros, zeros)
    mask = torch.zeros((4, 5, 6), dtype=torch.bool)
    omega = torch.tensor([1.0])
    amplitude = torch.tensor([0.05])
    phase = torch.tensor([0.0])
    k = torch.tensor([0.2])
    fout = apply_jonswap_inlet_3d(f, 0, mask, mask, omega, amplitude, phase, k, 3.0, 0.0, 0.1)
    assert fout.shape == f.shape
