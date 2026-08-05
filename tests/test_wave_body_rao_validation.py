"""Analytical regular-wave RAO reference checks (no LBM or solver coupling)."""

import math

import pytest
import torch

from tensorlbm.wave_body_rao import (
    SparGeometry,
    compute_heave_rao,
    compute_pitch_rao,
    finite_depth_wave_number,
    spar_heave_added_mass,
)


def _reference_spar() -> SparGeometry:
    radius, draft, rho = 4.0, 120.0, 1025.0
    mass = rho * math.pi * radius**2 * draft
    iyy = mass * (3.0 * radius**2 + draft**2) / 12.0
    return SparGeometry(radius=radius, draft=draft, mass=mass, iyy=iyy, rho_water=rho)


def test_heave_rao_static_limit_and_resonance_peak() -> None:
    spar = _reference_spar()
    assert compute_heave_rao(0.01, spar).item() == pytest.approx(1.0, abs=0.15)

    omega_n = math.sqrt(spar.C33 / (spar.mass + spar_heave_added_mass(0.5, spar)))
    omega = torch.linspace(0.5 * omega_n, 1.5 * omega_n, 50, dtype=torch.float64)
    rao = compute_heave_rao(omega, spar)
    peak_omega = omega[int(rao.argmax())].item()
    assert peak_omega == pytest.approx(omega_n, rel=0.2)
    assert rao.max().item() > 1.0


def test_pitch_rao_is_finite_and_high_frequency_decays() -> None:
    spar = _reference_spar()
    low = compute_pitch_rao(0.3, spar).item()
    high = compute_pitch_rao(3.0, spar).item()
    assert math.isfinite(low)
    assert low > 0.0
    assert high < low


def test_finite_depth_dispersion_residual_is_small() -> None:
    omega, depth = 1.0, 10.0
    k = finite_depth_wave_number(omega, depth)
    assert omega**2 == pytest.approx(9.81 * k * math.tanh(k * depth), rel=1e-8)
