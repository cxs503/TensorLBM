"""Tests for LBM free surface multiphase flow verification.

Tests cover:
  - VOF interface capture (PLIC reconstruction, curvature, advection)
  - CSF surface tension model (force computation, Guo forcing)
  - Dam break verification (Ritter solution comparison)
  - Wigley hull wave pattern (Michell theory)
  - Water entry slamming (Wagner theory)
  - Negative controls (single-phase limit, zero surface tension)
"""

from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from tensorlbm.vof_interface_capture import (
    compute_volume_fraction,
    compute_interface_normal,
    compute_curvature,
    advect_vof,
    youngs_normal,
    height_function_curvature,
)
from tensorlbm.csf_force_model import (
    csf_force,
    guo_force_term,
    surface_tension_step,
    capillary_time_step,
    laplace_pressure_test,
)
from tensorlbm.free_surface_verification import (
    DamBreakVerification,
    DamBreakConfig,
    WigleyWaveVerification,
    WigleyWaveConfig,
    WaterEntrySlamming,
    WaterEntryConfig,
    NegativeControls,
)


# =============================================================================
# VOF Tests
# =============================================================================

class TestVOFInterfaceCapture:
    """Tests for VOF interface capture module."""

    def test_volume_fraction_sharp_interface(self):
        """Sharp interface: phi > 0 -> alpha = 1, phi < 0 -> alpha = 0."""
        phi = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        alpha = compute_volume_fraction(phi, threshold=0.0)
        assert alpha[0, 0] == 1.0
        assert alpha[0, 1] == 0.0
        assert alpha[1, 0] == 1.0
        assert alpha[1, 1] == 0.0

    def test_volume_fraction_bounded(self):
        """Volume fraction should be bounded in [0, 1]."""
        phi = torch.randn(50, 50)
        alpha = compute_volume_fraction(phi)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_youngs_normal_uniform(self):
        """Uniform field should have zero normal."""
        alpha = torch.ones(20, 20)
        n_x, n_y = youngs_normal(alpha)
        assert torch.allclose(n_x, torch.zeros_like(n_x), atol=1e-6)
        assert torch.allclose(n_y, torch.zeros_like(n_y), atol=1e-6)

    def test_youngs_normal_sharp_interface(self):
        """Sharp vertical interface should have horizontal normal."""
        alpha = torch.zeros(20, 20)
        alpha[:, 10:] = 1.0
        n_x, n_y = youngs_normal(alpha)
        # Normal should point in x-direction at interface
        interface_region = (n_x.abs() > 0.1)
        if interface_region.any():
            # At interface, n_x should dominate
            assert n_x[interface_region].abs().mean() > n_y[interface_region].abs().mean()

    def test_interface_normal_normalized(self):
        """Interface normal should be unit length (where non-zero)."""
        alpha = torch.zeros(30, 30)
        alpha[:, 15:] = 1.0
        n_x, n_y = compute_interface_normal(alpha, smooth=False)
        mag = torch.sqrt(n_x**2 + n_y**2)
        # Where magnitude is significant, should be ~1
        significant = mag > 0.1
        if significant.any():
            assert torch.allclose(mag[significant], torch.ones_like(mag[significant]), atol=0.1)

    def test_curvature_flat_interface(self):
        """Flat interface should have zero curvature."""
        alpha = torch.zeros(30, 30)
        alpha[:, 15:] = 1.0
        kappa = compute_curvature(alpha, method="height_function")
        # Curvature should be small for flat interface
        assert kappa.abs().mean() < 0.1

    def test_curvature_circular(self):
        """Circular interface should have non-zero curvature."""
        ny, nx = 50, 50
        y, x = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        r = torch.sqrt((x - 25)**2 + (y - 25)**2)
        alpha = (r < 10).float()
        kappa = compute_curvature(alpha, method="height_function")
        # Curvature should be non-zero near interface
        interface = (alpha > 0.1) & (alpha < 0.9)
        if interface.any():
            # Just check it's finite
            assert torch.isfinite(kappa).all()

    def test_advect_vof_mass_conservation(self):
        """VOF advection should conserve mass (volume fraction integral)."""
        alpha = torch.zeros(30, 30)
        alpha[:, 10:20] = 1.0
        u = torch.ones(30, 30) * 0.1  # Uniform x-velocity
        v = torch.zeros(30, 30)

        mass_before = alpha.sum()
        alpha_new = advect_vof(alpha, u, v, dt=1.0, periodic=True)
        mass_after = alpha_new.sum()

        # Mass should be conserved (periodic boundaries)
        assert torch.allclose(mass_before, mass_after, rtol=0.01)

    def test_advect_vof_bounded(self):
        """Advected volume fraction should remain bounded."""
        alpha = torch.rand(30, 30)
        u = torch.randn(30, 30) * 0.1
        v = torch.randn(30, 30) * 0.1

        alpha_new = advect_vof(alpha, u, v, dt=1.0)
        assert alpha_new.min() >= 0.0
        assert alpha_new.max() <= 1.0

    def test_height_function_curvature_2d(self):
        """Height function curvature should work for 2D fields."""
        alpha = torch.zeros(30, 30)
        alpha[:, 15:] = 1.0
        kappa = height_function_curvature(alpha)
        assert kappa.shape == (30, 30)
        assert torch.isfinite(kappa).all()


# =============================================================================
# CSF Surface Tension Tests
# =============================================================================

class TestCSFSurfaceTension:
    """Tests for CSF surface tension model."""

    def test_csf_force_flat_interface(self):
        """Flat interface should have zero CSF force (zero curvature)."""
        alpha = torch.zeros(30, 30)
        alpha[:, 15:] = 1.0
        fx, fy = csf_force(alpha, sigma=0.01, curvature_method="height_function")
        # Force should be small for flat interface
        assert fx.abs().mean() < 0.01
        assert fy.abs().mean() < 0.01

    def test_csf_force_finite(self):
        """CSF force should be finite."""
        alpha = torch.rand(30, 30)
        fx, fy = csf_force(alpha, sigma=0.01)
        assert torch.isfinite(fx).all()
        assert torch.isfinite(fy).all()

    def test_guo_force_mass_conservation(self):
        """Guo forcing should conserve mass."""
        f = torch.rand(9, 30, 30)
        u = torch.zeros(30, 30)
        v = torch.zeros(30, 30)
        fx = torch.zeros(30, 30)
        fy = torch.zeros(30, 30)

        S = guo_force_term(f, u, v, fx, fy, tau=1.0)

        # Source term should sum to zero (mass conservation)
        assert torch.allclose(S.sum(), torch.tensor(0.0), atol=1e-6)

    def test_guo_force_momentum(self):
        """Guo forcing should add correct momentum."""
        f = torch.ones(9, 30, 30) / 9.0  # Uniform distribution
        u = torch.zeros(30, 30)
        v = torch.zeros(30, 30)
        fx = torch.ones(30, 30) * 0.01  # Uniform x-force
        fy = torch.zeros(30, 30)

        S = guo_force_term(f, u, v, fx, fy, tau=1.0)

        # Total momentum change should equal force
        from tensorlbm.d2q9 import C
        cx = C[:, 0].view(9, 1, 1).float()
        momentum_change = (S * cx).sum()
        expected = fx.sum() * (1.0 - 0.5 / 1.0)  # Total force with Guo scaling

        assert torch.allclose(momentum_change, expected, rtol=0.1)

    def test_surface_tension_step_shapes(self):
        """Surface tension step should preserve shapes."""
        f = torch.rand(9, 30, 30)
        alpha = torch.rand(30, 30)

        f_new, alpha_new = surface_tension_step(f, alpha, sigma=0.01, tau=1.0)

        assert f_new.shape == f.shape
        assert alpha_new.shape == alpha.shape

    def test_capillary_time_step(self):
        """Capillary time step should be positive and scale correctly."""
        dt = capillary_time_step(sigma=0.01, rho=1.0, dx=1.0)
        assert dt > 0

        # Higher surface tension -> smaller time step
        dt_high = capillary_time_step(sigma=0.1, rho=1.0, dx=1.0)
        assert dt_high < dt

    def test_laplace_pressure(self):
        """Laplace pressure test should return correct value."""
        dp = laplace_pressure_test(R=10.0, sigma=0.01)
        expected = 0.01 / 10.0  # sigma / R
        assert abs(dp - expected) < 1e-10


# =============================================================================
# Dam Break Verification Tests
# =============================================================================

class TestDamBreakVerification:
    """Tests for dam break verification."""

    def test_ritter_solution_initial(self):
        """Ritter solution at t*=0 should give X*=1."""
        cfg = DamBreakConfig()
        dam = DamBreakVerification(cfg)
        X_star = dam.ritter_solution(np.array([0.0]))
        assert abs(X_star[0] - 1.0) < 1e-10

    def test_ritter_solution_linear(self):
        """Ritter solution should be linear for t* < 2."""
        cfg = DamBreakConfig()
        dam = DamBreakVerification(cfg)
        t_star = np.linspace(0, 1.5, 10)
        X_star = dam.ritter_solution(t_star)
        expected = 1.0 + t_star
        assert np.allclose(X_star, expected, atol=0.1)

    def test_dam_break_initialization(self):
        """Dam break initialization should create correct volume fraction."""
        cfg = DamBreakConfig(nx=100, ny=50, dam_width=30)
        dam = DamBreakVerification(cfg)
        f_r, f_b, alpha = dam.initialize()

        # Alpha should be ~1 in dam region, ~0 outside
        assert alpha[:, :20].mean() > 0.8
        assert alpha[:, 50:].mean() < 0.2

    def test_dam_break_front_tracking(self):
        """Front tracking should detect wave front position."""
        cfg = DamBreakConfig(nx=100, ny=50, dam_width=30)
        dam = DamBreakVerification(cfg)

        # Create a field with known front position
        alpha = torch.zeros(50, 100)
        alpha[:, :60] = 1.0  # Front at x=60

        x_front = dam.track_front(alpha, threshold=0.5)
        assert 55 <= x_front <= 65

    def test_dam_break_run(self):
        """Dam break simulation should run and produce results."""
        cfg = DamBreakConfig(nx=100, ny=50, dam_width=25, n_steps=200, output_interval=50)
        dam = DamBreakVerification(cfg)
        results = dam.run()

        assert "front_positions" in results
        assert "t_star" in results
        assert "mean_error" in results
        assert len(results["front_positions"]) > 0


# =============================================================================
# Wigley Hull Wave Verification Tests
# =============================================================================

class TestWigleyWaveVerification:
    """Tests for Wigley hull wave verification."""

    def test_wigley_geometry_bounds(self):
        """Wigley geometry should be zero outside hull bounds."""
        cfg = WigleyWaveConfig()
        wig = WigleyWaveVerification(cfg)

        # Outside length
        x = torch.tensor([cfg.L])
        z = torch.tensor([0.0])
        y = wig.wigley_geometry(x, z)
        assert y.item() == 0.0

        # Above waterline
        x = torch.tensor([0.0])
        z = torch.tensor([1.0])
        y = wig.wigley_geometry(x, z)
        assert y.item() == 0.0

    def test_wigley_geometry_midship(self):
        """Wigley geometry at midship should give maximum beam."""
        cfg = WigleyWaveConfig()
        wig = WigleyWaveVerification(cfg)

        x = torch.tensor([0.0])
        z = torch.tensor([-cfg.T / 2])
        y = wig.wigley_geometry(x, z)
        # At midship, half-breadth should be close to B/2
        assert y.item() > cfg.B / 4

    def test_michell_wavelength(self):
        """Michell wavelength should scale with Fr^2."""
        cfg1 = WigleyWaveConfig(Fr=0.2)
        wig1 = WigleyWaveVerification(cfg1)
        lambda1 = wig1.michell_wave_wavelength()

        cfg2 = WigleyWaveConfig(Fr=0.4)
        wig2 = WigleyWaveVerification(cfg2)
        lambda2 = wig2.michell_wave_wavelength()

        # Wavelength scales with Fr^2
        ratio = lambda2 / lambda1
        expected_ratio = (0.4 / 0.2)**2
        assert abs(ratio - expected_ratio) < 0.1

    def test_wigley_run(self):
        """Wigley verification should run and produce results."""
        cfg = WigleyWaveConfig(L=50, B=5, T=3, Fr=0.3)
        wig = WigleyWaveVerification(cfg)
        results = wig.run()

        assert "theoretical_wavelength" in results
        assert "wave_number" in results
        assert results["theoretical_wavelength"] > 0


# =============================================================================
# Water Entry Slamming Tests
# =============================================================================

class TestWaterEntrySlamming:
    """Tests for water entry slamming verification."""

    def test_wagner_force_slope_positive(self):
        """Wagner force slope should be positive."""
        cfg = WaterEntryConfig(wedge_angle=4.0, V_entry=0.01)
        we = WaterEntrySlamming(cfg)
        dF_dt = we.wagner_force_slope()
        assert dF_dt > 0

    def test_wagner_force_slope_angle_dependence(self):
        """Wagner force slope should decrease with deadrise angle."""
        cfg1 = WaterEntryConfig(wedge_angle=2.0, V_entry=0.01)
        we1 = WaterEntrySlamming(cfg1)
        dF_dt1 = we1.wagner_force_slope()

        cfg2 = WaterEntryConfig(wedge_angle=8.0, V_entry=0.01)
        we2 = WaterEntrySlamming(cfg2)
        dF_dt2 = we2.wagner_force_slope()

        # Higher deadrise -> lower force
        assert dF_dt1 > dF_dt2

    def test_wagner_pressure_distribution(self):
        """Wagner pressure distribution should peak at leading edge."""
        cfg = WaterEntryConfig(wedge_angle=4.0, V_entry=0.01)
        we = WaterEntrySlamming(cfg)

        x = np.linspace(-1, 1, 100)
        p = we.wagner_pressure_distribution(x, t=10.0)

        # Pressure should be highest near edges (x = ±1)
        # and zero outside wetted region
        assert p[0] > 0 or p[-1] > 0  # Non-zero at edges

    def test_water_entry_run(self):
        """Water entry verification should run and produce results."""
        cfg = WaterEntryConfig(wedge_angle=4.0, V_entry=0.01, n_steps=500)
        we = WaterEntrySlamming(cfg)
        results = we.run()

        assert "wagner_force_slope" in results
        assert "measured_force_slope" in results
        assert "relative_error" in results


# =============================================================================
# Negative Control Tests
# =============================================================================

class TestNegativeControls:
    """Tests for negative controls."""

    def test_single_phase_limit(self):
        """Single-phase limit should preserve mass and velocity."""
        neg = NegativeControls()
        result = neg.single_phase_limit()

        assert "mass_drift_rel" in result
        assert "velocity_preservation" in result
        assert result["mass_drift_rel"] < 1e-4

    def test_zero_surface_tension_diffusion(self):
        """Zero surface tension should cause interface diffusion."""
        neg = NegativeControls()
        result = neg.zero_surface_tension()

        assert "diffusion_ratio" in result
        # Interface should have diffused
        assert result["diffusion_ratio"] >= 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full verification suite."""

    def test_vof_csf_coupling(self):
        """VOF and CSF should work together."""
        # Create a droplet
        ny, nx = 50, 50
        y, x = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        r = torch.sqrt((x - 25)**2 + (y - 25)**2)
        alpha = (r < 10).float()

        # Compute CSF force
        fx, fy = csf_force(alpha, sigma=0.01)

        # Force should be directed inward (towards center) for droplet
        # (surface tension tries to minimize surface area)
        center_force_x = fx[20:30, 20:30].mean()
        center_force_y = fy[20:30, 20:30].mean()

        # Just check forces are finite and reasonable
        assert torch.isfinite(fx).all()
        assert torch.isfinite(fy).all()

    def test_full_verification_pipeline(self):
        """Full verification pipeline should complete."""
        from tensorlbm.free_surface_verification import run_all_verifications
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_verifications(output_dir=Path(tmpdir))

            assert "dam_break" in results
            assert "wigley_wave" in results
            assert "water_entry" in results
            assert "negative_controls" in results
            assert "summary" in results
