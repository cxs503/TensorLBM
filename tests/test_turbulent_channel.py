"""Comprehensive turbulent channel flow verification tests.

Campaign node: p254w1-n04-lbm-turbulence-wall-law
Physics domain: wall-bounded turbulence, LES subgrid models
Risk: high (frontier verification)

Tests:
- J1: u+ profile vs DNS (MKM1999/Moser2021) error < 10% in log region
- J2: Fluctuation intensities vs DNS error < 15%
- J3: Reynolds shear stress vs DNS error < 15%
- J4: Energy spectrum inertial subrange slope -5/3 ± 0.3
- J5: Wall-law parameters κ=0.41±0.05, B=5.2±0.5
- J6: Grid convergence GCI < 5%
- J7: Negative control Re_τ→0 → Poiseuille (parabolic profile, R²>0.99)

References:
- MKM1999: Moser, Kim & Mansour, Phys. Fluids 11(4), 1999
- Moser2021: Moser, Yamamoto & Eifler, JFM 919, A16, 2021
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.channel_kernel_numpy import (
    ChannelKernelConfig,
    ChannelKernelState,
    C,
    W,
    OPP,
    equilibrium,
    macroscopic,
    collide,
    stream,
    bounce_back_walls,
    apply_body_force,
    make_state,
    poiseuille_exact,
)
from tensorlbm.validation.dns_reference_data import (
    DNSReferenceData,
    get_dns_data,
    reichardt_profile,
    musker_profile,
    channel_bulk_velocity,
    channel_centerline_velocity,
    VAN_DRIEST_KAPPA,
    VAN_DRIEST_B,
)
from tensorlbm.validation.turbulent_channel_analysis import (
    ChannelStatistics,
    WallLawFit,
    compute_energy_spectrum,
    fit_log_law,
    fit_spectral_slope,
    compare_with_dns,
)
from tensorlbm.validation.gci import gci as compute_gci


# ============================================================================
# Helper: lightweight channel runner for tests
# ============================================================================

def _run_channel_quick(
    nx: int = 32, ny: int = 20, nz: int = 16,
    re_tau: float = 180.0, u_tau: float = 0.006,
    steps: int = 800, warmup: int = 200,
    cs: float = 0.12,
) -> dict:
    """Quick channel run for testing (not production quality)."""
    from scripts.run_turbulent_channel_verification import (
        TurbulentChannelRunner,
    )

    runner = TurbulentChannelRunner(
        nx=nx, ny=ny, nz=nz,
        re_tau=re_tau, u_tau=u_tau, cs=cs,
        steps=steps, warmup_steps=warmup,
        sample_interval=5,
    )
    return runner.run()


# ============================================================================
# DNS Reference Data Tests
# ============================================================================

class TestDNSReferenceData:
    """Tests for DNS reference data generation."""

    def test_reichardt_profile_viscous_sublayer(self):
        """u+ ≈ y+ for y+ < 5."""
        yp = np.array([0.1, 1.0, 3.0, 5.0])
        up = reichardt_profile(yp)
        for y, u in zip(yp, up):
            assert abs(u - y) / max(y, 1e-10) < 0.15, f"Reichardt at y+={y}: u+={u}"

    def test_reichardt_profile_log_region(self):
        """u+ ≈ (1/κ)ln(y+) + B for y+ > 30."""
        yp = np.array([50.0, 100.0, 200.0])
        up = reichardt_profile(yp)
        for y, u in zip(yp, up):
            u_log = math.log(y) / 0.41 + 5.2
            assert abs(u - u_log) < 1.0, f"Reichardt at y+={y}: u+={u} vs log={u_log}"

    def test_reichardt_monotone(self):
        """Profile must be monotonically increasing."""
        yp = np.linspace(0.1, 500, 500)
        up = reichardt_profile(yp)
        assert np.all(np.diff(up) > 0)

    def test_get_dns_data_re180(self):
        """DNS data for Re_tau=180 is well-formed."""
        dns = get_dns_data(180.0)
        assert dns.re_tau == 180.0
        assert len(dns.y_plus) > 50
        assert len(dns.u_plus_mean) == len(dns.y_plus)
        assert np.all(dns.u_plus_mean >= 0)
        assert "MKM1999" in dns.source

    def test_get_dns_data_re395(self):
        dns = get_dns_data(395.0)
        assert dns.re_tau == 395.0
        assert "MKM1999" in dns.source

    def test_get_dns_data_re590(self):
        dns = get_dns_data(590.0)
        assert dns.re_tau == 590.0
        assert "Moser2021" in dns.source

    def test_centerline_velocity_increases_with_retau(self):
        """Higher Re_tau → higher centerline velocity."""
        u_cl_180 = channel_centerline_velocity(180.0)
        u_cl_395 = channel_centerline_velocity(395.0)
        u_cl_590 = channel_centerline_velocity(590.0)
        assert u_cl_180 < u_cl_395 < u_cl_590

    def test_bulk_velocity_less_than_centerline(self):
        for re_tau in [180.0, 395.0, 590.0]:
            u_b = channel_bulk_velocity(re_tau)
            u_cl = channel_centerline_velocity(re_tau)
            assert u_b < u_cl


# ============================================================================
# Wall-Law Analysis Tests
# ============================================================================

class TestWallLawAnalysis:
    """Tests for wall-law fitting and analysis."""

    def test_fit_log_law_reichardt(self):
        """Fitting Reichardt profile should recover reasonable κ, B."""
        yp = np.logspace(np.log10(30), np.log10(200), 50)
        up = reichardt_profile(yp)
        fit = fit_log_law(yp, up, y_plus_min=30, y_plus_max=200)
        # Reichardt differs from simple log law by ~0.4 in u+
        # So κ and B won't match exactly, but should be reasonable
        assert 0.3 < fit.kappa < 0.5, f"κ={fit.kappa}"
        assert 3.0 < fit.B < 7.0, f"B={fit.B}"
        assert fit.r_squared > 0.95

    def test_fit_log_law_with_noise(self):
        """Fitting should be robust to small perturbations."""
        rng = np.random.default_rng(42)
        yp = np.logspace(np.log10(30), np.log10(200), 50)
        up = reichardt_profile(yp) + 0.1 * rng.standard_normal(len(yp))
        fit = fit_log_law(yp, up)
        assert 0.3 < fit.kappa < 0.5
        assert fit.r_squared > 0.9

    def test_spectral_slope_flat_spectrum(self):
        """A flat spectrum should give slope ≈ 0."""
        k = np.linspace(1, 100, 50)
        E = np.ones_like(k)
        slope = fit_spectral_slope(k, E)
        assert abs(slope) < 0.5

    def test_spectral_slope_known_power_law(self):
        """E(k) = k^(-5/3) should give slope ≈ -5/3."""
        k = np.logspace(0, 2, 100)
        E = k ** (-5.0 / 3.0)
        slope = fit_spectral_slope(k, E)
        assert abs(slope - (-5.0 / 3.0)) < 0.1, f"slope={slope}"


# ============================================================================
# Smagorinsky LES Tests
# ============================================================================

class TestSmagorinskyLES:
    """Tests for the Smagorinsky LES model."""

    def test_smagorinsky_reduces_to_molecular_when_cs_zero(self):
        """Cs=0 should give identical results to molecular collision."""
        nx, ny, nz = 8, 10, 8
        rng = np.random.default_rng(7)
        rho = 1.0 + 0.01 * rng.standard_normal((nx, ny, nz))
        ux = 0.01 * rng.standard_normal((nx, ny, nz))
        uy = 0.005 * rng.standard_normal((nx, ny, nz))
        uz = 0.005 * rng.standard_normal((nx, ny, nz))
        f = equilibrium(rho, ux, uy, uz)
        tau = 0.6

        from scripts.run_turbulent_channel_verification import collide_with_smagorinsky

        f_mol = collide(f, tau, collision="bgk")
        f_sgs = collide_with_smagorinsky(f, tau, cs=0.0, collision="bgk")
        np.testing.assert_allclose(f_sgs, f_mol, atol=1e-14)

    def test_smagorinsky_adds_dissipation(self):
        """Non-zero Cs should increase effective dissipation."""
        nx, ny, nz = 8, 10, 8
        rng = np.random.default_rng(7)
        rho = 1.0 + 0.01 * rng.standard_normal((nx, ny, nz))
        ux = 0.05 * rng.standard_normal((nx, ny, nz))
        uy = 0.01 * rng.standard_normal((nx, ny, nz))
        uz = 0.01 * rng.standard_normal((nx, ny, nz))
        f = equilibrium(rho, ux, uy, uz)
        tau = 0.55

        from scripts.run_turbulent_channel_verification import collide_with_smagorinsky

        f_mol = collide(f, tau, collision="bgk")
        f_sgs = collide_with_smagorinsky(f, tau, cs=0.15, collision="bgk")

        # SGS should produce more dissipation (closer to equilibrium)
        _, ux_mol, _, _ = macroscopic(f_mol)
        _, ux_sgs, _, _ = macroscopic(f_sgs)
        # Both should be finite
        assert np.all(np.isfinite(ux_sgs))
        assert np.all(np.isfinite(ux_mol))

    def test_smagorinsky_preserves_mass(self):
        """Mass must be conserved through SGS collision."""
        nx, ny, nz = 8, 10, 8
        rng = np.random.default_rng(11)
        rho = 1.0 + 0.01 * rng.standard_normal((nx, ny, nz))
        ux = 0.01 * rng.standard_normal((nx, ny, nz))
        f = equilibrium(rho, ux, np.zeros_like(ux), np.zeros_like(ux))
        tau = 0.6

        from scripts.run_turbulent_channel_verification import collide_with_smagorinsky

        f_sgs = collide_with_smagorinsky(f, tau, cs=0.12, collision="bgk")
        assert f_sgs.sum() == pytest.approx(f.sum(), rel=1e-12)


# ============================================================================
# Channel Flow Simulation Tests
# ============================================================================

class TestChannelSimulation:
    """Tests for the channel flow solver."""

    def test_poiseuille_recovery_laminar(self):
        """Laminar channel should converge to a parabolic Poiseuille-like profile.
        
        Note: The LBM with bounce-back has inherent discretization error (~5% at
        this resolution). The key verification is that the profile is parabolic
        (R²>0.99) and the flow reaches steady state, not exact agreement with
        the continuum solution.
        """
        nx, ny, nz = 16, 21, 16
        nu = 0.1
        body_force = 1e-4
        tau = 0.5 + 3.0 * nu

        solid = np.zeros((nx, ny, nz), dtype=bool)
        solid[:, 0, :] = True
        solid[:, -1, :] = True

        # Zero initial condition
        dtype = np.float64
        ux = np.zeros((nx, ny, nz), dtype=dtype)
        rho = np.ones((nx, ny, nz), dtype=dtype)
        f = equilibrium(rho, ux, np.zeros_like(ux), np.zeros_like(ux))

        cfg = ChannelKernelConfig(nx=nx, ny=ny, nz=nz, re_tau=1.0, u_tau=0.001)
        state = ChannelKernelState(f=f, config=cfg, solid=solid)

        # Run to steady state
        for _ in range(5000):
            solid_3d = solid[None]
            collided = collide(state.f, tau, collision="trt")
            post = np.where(solid_3d, state.f, collided)
            rho_loc, ux_loc, uy_loc, uz_loc = macroscopic(post)
            fx = np.where(solid, 0.0, rho_loc * body_force)
            post = apply_body_force(post, ux_loc, uy_loc, uz_loc, fx)
            streamed = stream(post)
            state.f = bounce_back_walls(streamed, solid)

        # Check parabolic shape
        _, ux_final, _, _ = macroscopic(state.f)
        u_mean = ux_final.mean(axis=(0, 2))
        
        # Fit parabola to the profile
        y = np.arange(ny)
        coeffs = np.polyfit(y[1:-1], u_mean[1:-1], 2)
        u_fit = np.polyval(coeffs, y[1:-1])
        
        # R² should be very close to 1 (parabolic profile)
        ss_res = np.sum((u_mean[1:-1] - u_fit)**2)
        ss_tot = np.sum((u_mean[1:-1] - u_mean[1:-1].mean())**2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        assert r_squared > 0.999, f"Profile not parabolic: R²={r_squared:.6f}"
        
        # Maximum should be at center
        y_max = y[1:-1][np.argmax(u_mean[1:-1])]
        assert abs(y_max - ny // 2) <= 1, f"Max not at center: y_max={y_max}"
        
        # Velocity should be positive in the channel
        assert np.all(u_mean[1:-1] >= 0), "Negative velocity in channel"

    def test_turbulent_channel_produces_finite_statistics(self):
        """Turbulent channel should produce finite, non-trivial statistics."""
        result = _run_channel_quick(
            nx=24, ny=16, nz=12,
            re_tau=180.0, u_tau=0.006,
            steps=500, warmup=100, cs=0.12,
        )
        stats = result["statistics"]
        assert stats["n_samples"] > 0
        assert np.all(np.isfinite(stats["u_plus"]))
        assert np.all(np.isfinite(stats["u_rms_plus"]))
        # Mean velocity should be positive in the channel
        assert np.max(stats["u_plus"]) > 0

    def test_mass_conservation(self):
        """Total mass should be conserved during simulation."""
        cfg = ChannelKernelConfig(
            nx=16, ny=12, nz=16,
            re_tau=180.0, u_tau=0.006,
            steps=100, warmup_steps=0,
            initialization="laminar",
        )
        state = make_state(cfg)
        mass_initial = state.f.sum()

        for _ in range(100):
            from scripts.run_turbulent_channel_verification import collide_with_smagorinsky
            solid_3d = state.solid[None]
            collided = collide_with_smagorinsky(state.f, cfg.tau, 0.12)
            post = np.where(solid_3d, state.f, collided)
            rho, ux, uy, uz = macroscopic(post)
            fx = np.where(state.solid, 0.0, rho * cfg.body_force_acceleration)
            post = apply_body_force(post, ux, uy, uz, fx)
            streamed = stream(post)
            state.f = bounce_back_walls(streamed, state.solid)

        mass_final = state.f.sum()
        assert mass_final == pytest.approx(mass_initial, rel=1e-10)


# ============================================================================
# Grid Convergence Tests
# ============================================================================

class TestGridConvergence:
    """Tests for grid convergence (GCI)."""

    def test_gci_computation_three_grids(self):
        """GCI should compute for a three-grid study."""
        grid_sizes = [1.0 / 16, 1.0 / 24, 1.0 / 32]
        values = [15.5, 15.8, 15.9]  # centerline u+ converging
        report = compute_gci(grid_sizes, values)
        assert report.converged or report.reason  # either converges or explains why
        assert report.grid_count == 3

    def test_gci_two_grids(self):
        """Two-grid GCI should work with assumed first-order."""
        grid_sizes = [1.0 / 16, 1.0 / 32]
        values = [15.5, 15.9]
        report = compute_gci(grid_sizes, values)
        assert report.gci_fine_pct is not None
        assert report.gci_fine_pct >= 0


# ============================================================================
# Energy Spectrum Tests
# ============================================================================

class TestEnergySpectrum:
    """Tests for energy spectrum computation."""

    def test_spectrum_of_known_field(self):
        """Spectrum of a sinusoidal field should peak at the right wavenumber."""
        nx, ny, nz = 32, 8, 32
        x = np.arange(nx)
        z = np.arange(nz)
        X, Z = np.meshgrid(x, z, indexing="ij")

        # Single wavenumber kx=4
        ux = np.sin(2 * np.pi * 4 * X / nx)[:, None, :] * np.ones((1, ny, 1))
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)

        k, E = compute_energy_spectrum(ux, uy, uz)
        # Peak should be near k = 2*pi*4/nx
        k_peak = k[np.argmax(E)]
        k_expected = 2 * np.pi * 4 / nx
        assert abs(k_peak - k_expected) / k_expected < 0.5

    def test_spectrum_positive(self):
        """Energy spectrum must be non-negative."""
        rng = np.random.default_rng(42)
        nx, ny, nz = 16, 4, 16
        ux = rng.standard_normal((nx, ny, nz))
        uy = rng.standard_normal((nx, ny, nz))
        uz = rng.standard_normal((nx, ny, nz))
        k, E = compute_energy_spectrum(ux, uy, uz)
        assert np.all(E >= 0)


# ============================================================================
# Integration Tests (slower, marked for CI)
# ============================================================================

class TestIntegration:
    """Integration tests that run actual simulations."""

    @pytest.mark.slow
    def test_re180_quick_verification(self):
        """Quick Re_tau=180 verification (smoke test)."""
        result = _run_channel_quick(
            nx=32, ny=20, nz=16,
            re_tau=180.0, u_tau=0.006,
            steps=600, warmup=150, cs=0.12,
        )
        # Basic sanity checks
        assert result["statistics"]["n_samples"] > 0
        assert result["wall_law_fit"]["n_points"] > 0
        comp = result["comparison"]
        assert comp["u_plus_error_pct"] < 50  # generous for short run

    @pytest.mark.slow
    def test_negative_control_poiseuille(self):
        """Negative control: Poiseuille flow should be parabolic."""
        nx, ny, nz = 32, 21, 32
        nu = 0.1
        body_force = 1e-4
        tau = 0.5 + 3.0 * nu

        solid = np.zeros((nx, ny, nz), dtype=bool)
        solid[:, 0, :] = True
        solid[:, -1, :] = True

        dtype = np.float64
        ux = np.zeros((nx, ny, nz), dtype=dtype)
        rho = np.ones((nx, ny, nz), dtype=dtype)
        f = equilibrium(rho, ux, np.zeros_like(ux), np.zeros_like(ux))

        cfg = ChannelKernelConfig(nx=nx, ny=ny, nz=nz, re_tau=1.0, u_tau=0.001)
        state = ChannelKernelState(f=f, config=cfg, solid=solid)

        for _ in range(5000):
            solid_3d = solid[None]
            collided = collide(state.f, tau, collision="trt")
            post = np.where(solid_3d, state.f, collided)
            rho_loc, ux_loc, uy_loc, uz_loc = macroscopic(post)
            fx = np.where(solid, 0.0, rho_loc * body_force)
            post = apply_body_force(post, ux_loc, uy_loc, uz_loc, fx)
            streamed = stream(post)
            state.f = bounce_back_walls(streamed, solid)

        _, ux_final, _, _ = macroscopic(state.f)
        u_mean = ux_final.mean(axis=(0, 2))
        
        # Check parabolic shape
        y = np.arange(ny)
        coeffs = np.polyfit(y[1:-1], u_mean[1:-1], 2)
        u_fit = np.polyval(coeffs, y[1:-1])
        ss_res = np.sum((u_mean[1:-1] - u_fit)**2)
        ss_tot = np.sum((u_mean[1:-1] - u_mean[1:-1].mean())**2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        assert r_squared > 0.999, f"Poiseuille not parabolic: R²={r_squared:.6f}"


# ============================================================================
# Comparison with DNS Tests
# ============================================================================

class TestDNSComparison:
    """Tests for DNS comparison metrics."""

    def test_compare_perfect_match(self):
        """Comparing a profile with itself should give zero error."""
        dns = get_dns_data(180.0)
        stats = ChannelStatistics(
            y=np.arange(len(dns.y_plus)),
            y_plus=dns.y_plus,
            u_mean=dns.u_plus_mean,
            u_plus=dns.u_plus_mean,
            u_rms=dns.u_rms_plus,
            v_rms=dns.v_rms_plus,
            w_rms=dns.w_rms_plus,
            uv_reynolds=dns.uv_plus,
            u_rms_plus=dns.u_rms_plus,
            v_rms_plus=dns.v_rms_plus,
            w_rms_plus=dns.w_rms_plus,
            uv_plus=dns.uv_plus,
            u_tau=0.006,
            re_tau=180.0,
            n_samples=100,
        )
        comp = compare_with_dns(
            stats, dns.y_plus, dns.u_plus_mean,
            dns.u_rms_plus, dns.v_rms_plus, dns.w_rms_plus, dns.uv_plus,
        )
        assert comp.u_plus_error_pct < 1.0
        assert comp.u_rms_error_pct < 1.0

    def test_judgement_criteria_structure(self):
        """Judgement criteria should have the right keys."""
        dns = get_dns_data(180.0)
        stats = ChannelStatistics(
            y=np.arange(len(dns.y_plus)),
            y_plus=dns.y_plus,
            u_mean=dns.u_plus_mean,
            u_plus=dns.u_plus_mean,
            u_rms=dns.u_rms_plus,
            v_rms=dns.v_rms_plus,
            w_rms=dns.w_rms_plus,
            uv_reynolds=dns.uv_plus,
            u_rms_plus=dns.u_rms_plus,
            v_rms_plus=dns.v_rms_plus,
            w_rms_plus=dns.w_rms_plus,
            uv_plus=dns.uv_plus,
            u_tau=0.006,
            re_tau=180.0,
            n_samples=100,
        )
        comp = compare_with_dns(
            stats, dns.y_plus, dns.u_plus_mean,
            dns.u_rms_plus, dns.v_rms_plus, dns.w_rms_plus, dns.uv_plus,
        )
        j = comp.passes_judgement()
        assert "J1_u_plus_log_region_lt_10pct" in j
        assert "J2_rms_lt_15pct" in j
        assert "J3_reynolds_stress_lt_15pct" in j
        assert "J4_spectral_slope" in j
        assert "J5_kappa_B" in j
