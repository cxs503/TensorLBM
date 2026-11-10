"""Pytest tests for acoustic wave propagation verification.

Tests cover:
1. Lattice sound speed c_s = 1/√3 (relative error < 1%)
2. 1D sinusoidal wave waveform preservation (L2 error < 2%)
3. 2D Gaussian pulse propagation
4. Numerical dissipation rate scaling with k² (second-order accuracy)
5. Von Neumann analysis consistency
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.acoustic_wave import (
    CS_LATTICE,
    CS2_LATTICE,
    GaussianPulseConfig,
    SinusoidalWaveConfig,
    WaveVerificationResult,
    compute_lattice_sound_speed,
    run_1d_sinusoidal_wave,
    run_2d_gaussian_pulse,
    run_dissipation_scan,
    run_full_verification,
)


# ---------------------------------------------------------------------------
# Test 1: Lattice sound speed constant
# ---------------------------------------------------------------------------


def test_lattice_sound_speed_constant():
    """c_s = 1/√3 exactly in lattice units."""
    cs = compute_lattice_sound_speed()
    expected = 1.0 / math.sqrt(3.0)
    assert abs(cs - expected) < 1e-15, f"c_s = {cs}, expected {expected}"


def test_cs2_constant():
    """c_s² = 1/3 exactly."""
    assert abs(CS2_LATTICE - 1.0 / 3.0) < 1e-15


# ---------------------------------------------------------------------------
# Test 2: Sound speed from von Neumann analysis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.8, 1.0, 1.5])
def test_sound_speed_from_von_neumann(tau):
    """Phase speed from von Neumann analysis should match c_s = 1/√3 within 1%."""
    res = run_dissipation_scan(tau=tau, k_min=0.01, k_max=0.3, n_k=15)
    cs_error = res["phase_speed_error"]
    assert cs_error < 0.01, (
        f"Phase speed error {cs_error:.4%} exceeds 1% for τ={tau}. "
        f"Measured: {res['phase_speed']:.6f}, Expected: {CS_LATTICE:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: 1D sinusoidal wave - waveform preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau,wavelength,n_steps", [
    (1.0, 128, 200),
    (1.0, 256, 400),
    (1.5, 128, 200),
])
def test_1d_waveform_preservation(tau, wavelength, n_steps):
    """L2 error between numerical and analytical solution should be < 2%."""
    res = run_1d_sinusoidal_wave(
        nx=512, tau=tau, n_steps=n_steps,
        wave=SinusoidalWaveConfig(amplitude=1e-4, wavelength=wavelength),
    )
    l2_err = res["l2_error"]
    assert l2_err < 0.02, (
        f"L2 error {l2_err:.4%} exceeds 2% for τ={tau}, λ={wavelength}, steps={n_steps}"
    )


# ---------------------------------------------------------------------------
# Test 4: 2D Gaussian pulse
# ---------------------------------------------------------------------------


def test_2d_pulse_runs():
    """2D Gaussian pulse simulation should complete without error."""
    nx, ny = 128, 128
    n_steps = 100
    res = run_2d_gaussian_pulse(
        nx=nx, ny=ny, tau=1.0, n_steps=n_steps,
        pulse=GaussianPulseConfig(amplitude=1e-4, center=(nx / 2, ny / 2), sigma=10.0),
    )
    assert len(res["rho_history"]) > 1
    assert res["rho_final"].shape == (ny, nx)


# ---------------------------------------------------------------------------
# Test 5: Dissipation rate scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.8, 1.0, 1.5])
def test_dissipation_second_order(tau):
    """Im(ω) should scale as k² (R² > 0.99 for small k)."""
    res = run_dissipation_scan(tau=tau, k_min=0.02, k_max=0.5, n_k=20)
    r_sq = res["r_squared"]
    assert r_sq > 0.99, (
        f"Dissipation fit R² = {r_sq:.4f} < 0.99 for τ={tau}. "
        f"Expected second-order scaling Im(ω) ∝ k²"
    )


def test_dissipation_rate_accuracy():
    """Measured dissipation rate should match theoretical ν_num within 5%."""
    tau = 1.0
    res = run_dissipation_scan(tau=tau)
    gamma_rel = res["gamma_relative_error"]
    assert gamma_rel < 0.05, (
        f"Dissipation rate error {gamma_rel:.4%} exceeds 5%. "
        f"Measured: {res['gamma_measured']:.6f}, Theory: {res['gamma_theory']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Full verification pipeline
# ---------------------------------------------------------------------------


def test_full_verification_passes():
    """Full verification should pass all criteria."""
    result = run_full_verification(
        nx_1d=512, nx_2d=128, ny_2d=128,
        tau=1.0, n_steps_1d=200, n_steps_2d=100,
    )
    assert result.passed_cs, f"Sound speed check failed: error={result.cs_relative_error:.4%}"
    assert result.passed_l2, f"L2 error check failed: error={result.l2_error:.4%}"
    assert result.passed_dissipation, "Dissipation scaling check failed"


# ---------------------------------------------------------------------------
# Test 7: Low Mach number regime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", [1e-5, 1e-4, 5e-4])
def test_low_mach_regime(amplitude):
    """Weak compressibility should hold for Ma < 0.1."""
    res = run_1d_sinusoidal_wave(
        nx=512, tau=1.0, n_steps=200,
        wave=SinusoidalWaveConfig(amplitude=amplitude, wavelength=128.0),
    )
    assert res["l2_error"] < 0.02, (
        f"L2 error {res['l2_error']:.4%} at amplitude={amplitude} (Ma ~ {amplitude:.1e})"
    )


# ---------------------------------------------------------------------------
# Test 8: Conservation of mass
# ---------------------------------------------------------------------------


def test_mass_conservation_1d():
    """Total mass should be conserved to machine precision."""
    res = run_1d_sinusoidal_wave(
        nx=256, tau=1.0, n_steps=100,
        wave=SinusoidalWaveConfig(amplitude=1e-4, wavelength=64.0),
    )
    mass_0 = np.sum(res["rho_history"][0])
    mass_final = np.sum(res["rho_history"][-1])
    mass_error = abs(mass_final - mass_0) / mass_0
    assert mass_error < 1e-10, f"Mass conservation error: {mass_error:.2e}"


# ---------------------------------------------------------------------------
# Test 9: Numerical viscosity formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.8, 1.0, 1.5])
def test_viscosity_formula(tau):
    """ν_num = c_s²(τ - 0.5) should match von Neumann fit."""
    res = run_dissipation_scan(tau=tau)
    gamma_theory = CS2_LATTICE * (tau - 0.5)
    gamma_measured = res["gamma_measured"]
    if gamma_theory > 0:
        rel_error = abs(gamma_measured - gamma_theory) / gamma_theory
        assert rel_error < 0.05, (
            f"Viscosity error {rel_error:.4%} for τ={tau}. "
            f"Measured: {gamma_measured:.6f}, Theory: {gamma_theory:.6f}"
        )
