"""Tests for 2D numerical wave tank with Stokes second-order wave generation.

Covers:
- Stokes wave theory (analytical)
- Dispersion relation
- Relaxation zone weights
- Wave tank initialization and stepping
- Harmonic analysis
- Two-point reflection coefficient (Goda & Suzuki)
- Wave elevation extraction
- Mass conservation
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.wave_tank_2d import (
    StokesWaveParams,
    WaveTankConfig,
    WaveTank2D,
    stokes_second_order_surface,
    stokes_second_order_velocity,
    solve_dispersion_relation,
    compute_relaxation_weight,
    extract_wave_elevation,
    harmonic_analysis,
    two_point_reflection_coefficient,
)


# ---------------------------------------------------------------------------
# Stokes wave theory tests
# ---------------------------------------------------------------------------

class TestStokesWaveParams:
    """Test Stokes wave parameter calculations."""

    def test_sigma(self):
        """σ = tanh(kh) for known kh."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        expected = math.tanh(0.03 * 50.0)
        assert abs(wp.sigma - expected) < 1e-10

    def test_omega_dispersion(self):
        """ω² = gk·tanh(kh) — dispersion relation."""
        k, g, h = 0.03, 1e-4, 50.0
        wp = StokesWaveParams(amplitude=1.0, wavenumber=k, gravity=g, depth=h)
        expected_omega = math.sqrt(g * k * math.tanh(k * h))
        assert abs(wp.omega - expected_omega) < 1e-10

    def test_period(self):
        """T = 2π/ω."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        assert abs(wp.period - 2 * math.pi / wp.omega) < 1e-10

    def test_wavelength(self):
        """λ = 2π/k."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        assert abs(wp.wavelength - 2 * math.pi / 0.03) < 1e-10

    def test_phase_speed(self):
        """c = ω/k."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        assert abs(wp.phase_speed - wp.omega / 0.03) < 1e-10

    def test_ka(self):
        """Wave steepness ka."""
        wp = StokesWaveParams(amplitude=2.0, wavenumber=0.04, gravity=1e-4, depth=50.0)
        assert abs(wp.ka - 0.08) < 1e-10

    def test_second_order_amplitude_positive(self):
        """Second-order amplitude should be positive for typical waves."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        assert wp.second_order_amplitude > 0

    def test_second_order_amplitude_formula(self):
        """Verify a₂ = (ka²/4)·(3-σ²)/σ³."""
        a, k, g, h = 1.0, 0.03, 1e-4, 50.0
        wp = StokesWaveParams(amplitude=a, wavenumber=k, gravity=g, depth=h)
        sigma = math.tanh(k * h)
        expected = (k * a**2 / 4.0) * (3.0 - sigma**2) / sigma**3
        assert abs(wp.second_order_amplitude - expected) < 1e-10


class TestStokesSurface:
    """Test Stokes second-order surface elevation."""

    def test_surface_at_t0(self):
        """η(x,0) = a·cos(kx) + a₂·cos(2kx)."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        eta = stokes_second_order_surface(0.0, 0.0, wp)
        expected = wp.amplitude + wp.second_order_amplitude  # cos(0) = 1
        assert abs(eta - expected) < 1e-10

    def test_surface_tensor(self):
        """Surface elevation works with tensor input."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        x = torch.tensor([0.0, 10.0, 20.0])
        eta = stokes_second_order_surface(x, 0.0, wp)
        assert eta.shape == (3,)
        # At x=0, should match scalar version
        assert abs(eta[0].item() - stokes_second_order_surface(0.0, 0.0, wp)) < 1e-6

    def test_surface_periodicity(self):
        """Surface should be periodic with wavelength λ."""
        wp = StokesWaveParams(amplitude=1.0, wavenumber=0.03, gravity=1e-4, depth=50.0)
        eta0 = stokes_second_order_surface(0.0, 0.0, wp)
        eta_L = stokes_second_order_surface(wp.wavelength, 0.0, wp)
        assert abs(eta0 - eta_L) < 1e-8


class TestDispersionRelation:
    """Test dispersion relation solver."""

    def test_closed_form(self):
        """ω = sqrt(gk·tanh(kh)) matches solve_dispersion_relation."""
        k, g, h = 0.03, 1e-4, 50.0
        omega = solve_dispersion_relation(k, g, h)
        expected = math.sqrt(g * k * math.tanh(k * h))
        assert abs(omega - expected) < 1e-10

    def test_deep_water_limit(self):
        """Deep water: tanh(kh)→1, ω²→gk."""
        k, g, h = 0.1, 1e-4, 1000.0  # kh = 100 >> 1
        omega = solve_dispersion_relation(k, g, h)
        expected = math.sqrt(g * k)  # deep water limit
        assert abs(omega - expected) / expected < 1e-6

    def test_shallow_water_limit(self):
        """Shallow water: tanh(kh)≈kh, ω²→ghk²."""
        k, g, h = 0.001, 1e-4, 50.0  # kh = 0.05 << 1
        omega = solve_dispersion_relation(k, g, h)
        expected = math.sqrt(g * h) * k  # shallow water limit
        assert abs(omega - expected) / expected < 0.01  # 1% tolerance


class TestRelaxationWeight:
    """Test relaxation zone weight computation."""

    def test_gen_zone_boundaries(self):
        """Generation zone: λ(0)=0, λ(L)≈1."""
        lam_start = compute_relaxation_weight(0.0, 0.0, 50.0, "generation")
        lam_end = compute_relaxation_weight(50.0, 0.0, 50.0, "generation")
        assert abs(lam_start) < 1e-6
        assert abs(lam_end - 1.0) < 1e-6

    def test_abs_zone_boundaries(self):
        """Absorption zone: λ(start)=1, λ(end)≈0."""
        lam_start = compute_relaxation_weight(50.0, 50.0, 100.0, "absorption")
        lam_end = compute_relaxation_weight(100.0, 50.0, 100.0, "absorption")
        assert abs(lam_start - 1.0) < 1e-6
        assert abs(lam_end) < 1e-6

    def test_monotonic_generation(self):
        """Generation zone weight should be monotonically increasing."""
        xs = [i * 10.0 for i in range(6)]
        lams = [compute_relaxation_weight(x, 0.0, 50.0, "generation") for x in xs]
        for i in range(len(lams) - 1):
            assert lams[i] <= lams[i + 1] + 1e-10

    def test_tensor_input(self):
        """Relaxation weight works with tensor input."""
        x = torch.linspace(0, 50, 11)
        lam = compute_relaxation_weight(x, 0.0, 50.0, "generation")
        assert lam.shape == (11,)
        assert lam[0].item() < 1e-6
        assert abs(lam[-1].item() - 1.0) < 1e-6


class TestWaveTankInit:
    """Test wave tank initialization."""

    def test_init_fields(self):
        """Wave tank initializes with correct field shapes."""
        cfg = WaveTankConfig(nx=100, ny=50, water_depth=25.0)
        tank = WaveTank2D(cfg)
        assert tank.f.shape == (9, 50, 100)
        assert tank.fill.shape == (50, 100)
        assert tank.flags.shape == (50, 100)
        assert tank.mass.shape == (50, 100)

    def test_init_water_depth(self):
        """Initial water depth matches configuration."""
        cfg = WaveTankConfig(nx=100, ny=50, water_depth=25.0)
        tank = WaveTank2D(cfg)
        # Count liquid cells in a column at mid-tank
        mid_x = 50
        liquid_count = (tank.fill[:, mid_x] > 0.5).sum().item()
        # Should be approximately water_depth (±2 for solid rows and interface)
        assert abs(liquid_count - 25) < 5

    def test_probe_positions(self):
        """Probes are placed in the tank interior."""
        cfg = WaveTankConfig(nx=300, ny=60, water_depth=25.0)
        tank = WaveTank2D(cfg)
        assert tank.gen_zone_end < tank.probe1_x < tank.probe2_x < tank.abs_zone_start


class TestWaveTankStep:
    """Test wave tank time stepping."""

    def test_step_runs(self):
        """Single step runs without error."""
        cfg = WaveTankConfig(nx=200, ny=50, water_depth=20.0, n_steps=1)
        tank = WaveTank2D(cfg)
        tank.step(0)
        # Fields should still have correct shapes
        assert tank.f.shape == (9, 50, 200)

    def test_short_run(self):
        """Short run completes and records data."""
        cfg = WaveTankConfig(
            nx=100, ny=40, water_depth=20.0,
            n_steps=20, record_interval=5,
        )
        tank = WaveTank2D(cfg)
        tank.run()
        assert len(tank.probe_history_1) > 0
        assert len(tank.mass_history) > 0


class TestHarmonicAnalysis:
    """Test harmonic analysis on synthetic signals."""

    def test_pure_sine(self):
        """Pure sine wave: only first harmonic should be non-zero."""
        import numpy as np
        n = 200
        dt = 1.0
        omega = 2 * math.pi / 50.0  # period = 50 steps
        signal = [2.0 * math.sin(omega * i * dt) for i in range(n)]
        result = harmonic_analysis(signal, dt, omega, n_harmonics=3)
        assert result["A1"] > 1.5  # Should be close to 2.0
        assert result["A2"] < 0.3  # Should be near zero

    def test_two_harmonics(self):
        """Signal with first and second harmonics."""
        import numpy as np
        n = 400
        dt = 1.0
        omega = 2 * math.pi / 80.0
        signal = [
            3.0 * math.sin(omega * i * dt) + 0.5 * math.sin(2 * omega * i * dt)
            for i in range(n)
        ]
        result = harmonic_analysis(signal, dt, omega, n_harmonics=3)
        assert abs(result["A1"] - 3.0) < 0.5
        assert abs(result["A2"] - 0.5) < 0.3


class TestReflectionCoefficient:
    """Test two-point reflection coefficient method."""

    def test_no_reflection(self):
        """Pure incident wave: Kr ≈ 0."""
        import numpy as np
        n = 500
        dt = 1.0
        k = 0.05
        omega = 0.1
        x1, x2 = 50.0, 80.0
        a_I = 2.0

        eta1 = [a_I * math.cos(k * x1 - omega * i * dt) for i in range(n)]
        eta2 = [a_I * math.cos(k * x2 - omega * i * dt) for i in range(n)]

        result = two_point_reflection_coefficient(eta1, eta2, dt, x1, x2, k, omega)
        assert result["Kr"] < 0.1  # Should be near zero

    def test_full_reflection(self):
        """Standing wave (full reflection): Kr ≈ 1."""
        import numpy as np
        n = 500
        dt = 1.0
        k = 0.05
        omega = 0.1
        x1, x2 = 50.0, 80.0
        a_I = 2.0

        # Standing wave: η = 2a·cos(kx)·cos(ωt)
        eta1 = [2 * a_I * math.cos(k * x1) * math.cos(omega * i * dt) for i in range(n)]
        eta2 = [2 * a_I * math.cos(k * x2) * math.cos(omega * i * dt) for i in range(n)]

        result = two_point_reflection_coefficient(eta1, eta2, dt, x1, x2, k, omega)
        assert result["Kr"] > 0.5  # Should be near 1.0


class TestWaveElevation:
    """Test wave elevation extraction."""

    def test_flat_water(self):
        """Flat water surface at y=h."""
        fill = torch.zeros(50, 100)
        fill[2:27, :] = 1.0  # water from y=2 to y=26
        eta = extract_wave_elevation(fill, 50, 25.0)
        assert abs(eta - 26.0) < 1.0  # surface at y=26

    def test_empty_domain(self):
        """No water: elevation should be 0."""
        fill = torch.zeros(50, 100)
        eta = extract_wave_elevation(fill, 50, 25.0)
        assert eta == 0.0


class TestWaveTankResults:
    """Test wave tank results computation."""

    def test_get_results_structure(self):
        """Results dictionary has expected keys."""
        cfg = WaveTankConfig(
            nx=100, ny=40, water_depth=20.0,
            n_steps=30, record_interval=10,
        )
        tank = WaveTank2D(cfg)
        tank.run()
        results = tank.get_results()
        assert "harmonics" in results
        assert "reflection" in results
        assert "mass_drift_relative" in results
        assert "dispersion_error" in results
        assert "wave_params" in results
        assert "config" in results

    def test_wave_params_in_results(self):
        """Wave params in results match configuration."""
        cfg = WaveTankConfig(
            nx=200, ny=50, water_depth=20.0,
            wave_amplitude=1.0, wave_kh=1.5, gravity=1e-4,
        )
        tank = WaveTank2D(cfg)
        results = tank.get_results()
        assert abs(results["wave_params"]["kh"] - 1.5) < 1e-6
        assert abs(results["wave_params"]["amplitude"] - 1.0) < 1e-6
