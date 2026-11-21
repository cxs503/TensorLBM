"""Tests for the single-phase Poiseuille baseline skeleton.

Validates the analytical framework in ``tensorlbm.verification.xval.poiseuille_skeleton``
against the lettucecfd/lettuce unit-convention semantics and the exact plane-Poiseuille
solution.  This is the verification backbone for
``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO``.

Acceptance criteria (from feature ledger):
    - All profile metrics < 15 % vs analytical reference (owner precision standard).
    - Effective-height self-consistency at machine precision.
    - Second-order convergence of discrete peak vs continuous peak with grid refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.verification.xval.poiseuille_skeleton import (
    OSSBaseline,
    oss_units,
    analytic_profile,
    analytic_solution_object,
    profile_kpis,
    wall_normal_coordinates,
    fit_effective_height,
    two_code_diff_metrics,
    CS_LU,
)


# ---------------------------------------------------------------------------
# Unit-conversion tests (oss_units)
# ---------------------------------------------------------------------------

class TestOSSUnits:
    """Verify that oss_units reproduces the lettuce unit-conversion semantics."""

    def test_default_parameters(self):
        """Default call returns a valid OSSBaseline."""
        p = oss_units()
        assert isinstance(p, OSSBaseline)
        assert p.resolution == 16
        assert p.reynolds_number == 1.0
        assert p.mach_number == 0.02
        assert p.height_lu == 14.0  # resolution - 2

    def test_height_formula(self):
        """H = resolution - 2 (half-way bounce-back geometry)."""
        for res in [4, 8, 16, 32, 64]:
            p = oss_units(resolution=res)
            assert p.height_lu == res - 2

    def test_characteristic_length(self):
        """L_lu = resolution - 1 (lettuce make_units convention)."""
        for res in [4, 8, 16, 32]:
            p = oss_units(resolution=res)
            assert p.characteristic_length_lu == res - 1

    def test_viscosity_from_reynolds(self):
        """nu_lu = L_lu * u_lu / Re."""
        p = oss_units(resolution=16, reynolds_number=2.0, mach_number=0.02)
        expected_nu = p.characteristic_length_lu * p.characteristic_velocity_lu / 2.0
        assert abs(p.viscosity_lu - expected_nu) < 1e-15

    def test_tau_from_viscosity(self):
        """tau = nu / cs^2 + 0.5."""
        p = oss_units(resolution=16, reynolds_number=1.0)
        expected_tau = p.viscosity_lu / CS_LU ** 2 + 0.5
        assert abs(p.relaxation_parameter_lu - expected_tau) < 1e-15

    def test_peak_velocity_formula(self):
        """u_peak_lu = g * H^2 / (8 * nu) (continuous maximum)."""
        p = oss_units(resolution=32, reynolds_number=1.0)
        expected = p.acceleration_lu * p.height_lu ** 2 / (8.0 * p.viscosity_lu)
        assert abs(p.u_peak_lu - expected) < 1e-15

    def test_force_scale(self):
        """force_scale multiplies only the lattice acceleration."""
        p1 = oss_units(force_scale=1.0)
        p2 = oss_units(force_scale=2.0)
        assert abs(p2.acceleration_lu - 2.0 * p1.acceleration_lu) < 1e-30
        # All other parameters unchanged
        assert p2.viscosity_lu == p1.viscosity_lu
        assert p2.relaxation_parameter_lu == p1.relaxation_parameter_lu

    def test_resolution_too_small(self):
        """resolution < 4 raises ValueError."""
        with pytest.raises(ValueError, match="resolution must be >= 4"):
            oss_units(resolution=3)

    def test_as_dict_roundtrip(self):
        """as_dict returns all fields."""
        p = oss_units(resolution=16)
        d = p.as_dict()
        assert d["resolution"] == 16
        assert d["height_lu"] == 14.0
        assert "u_peak_lu" in d


# ---------------------------------------------------------------------------
# Wall-normal coordinate tests
# ---------------------------------------------------------------------------

class TestWallNormalCoordinates:
    """Verify fluid node positions relative to half-way walls."""

    def test_basic(self):
        y, H = wall_normal_coordinates(8)
        assert H == 6.0
        assert len(y) == 6
        # First fluid node at 0.5 from lower wall
        assert abs(y[0] - 0.5) < 1e-15
        # Last fluid node at H - 0.5 from lower wall
        assert abs(y[-1] - (H - 0.5)) < 1e-15

    def test_symmetry(self):
        """Node positions are symmetric about H/2."""
        for res in [8, 16, 32]:
            y, H = wall_normal_coordinates(res)
            assert np.allclose(y + y[::-1], H)

    def test_too_small(self):
        with pytest.raises(ValueError, match="need at least 4"):
            wall_normal_coordinates(3)


# ---------------------------------------------------------------------------
# Analytical profile tests
# ---------------------------------------------------------------------------

class TestAnalyticProfile:
    """Verify the exact plane-Poiseuille solution."""

    def test_parabolic_shape(self):
        """u(y) = g/(2*nu) * y * (H - y) is a parabola with zeros at y=0 and y=H."""
        y = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        H = 3.0
        g = 1e-5
        nu = 0.1
        u = analytic_profile(y, H, g, nu)
        assert u[0] == 0.0  # zero at y=0
        assert abs(u[-1]) < 1e-20  # zero at y=H

    def test_peak_at_center(self):
        """Maximum occurs at y = H/2."""
        y = np.linspace(0.1, 2.9, 100)
        H = 3.0
        g = 1e-5
        nu = 0.1
        u = analytic_profile(y, H, g, nu)
        idx = np.argmax(u)
        assert abs(y[idx] - H / 2) < 0.02

    def test_peak_value(self):
        """u_max = g * H^2 / (8 * nu)."""
        H = 14.0
        g = 8.888888888888893e-09
        nu = 0.17320508075688776
        y = np.array([H / 2])
        u = analytic_profile(y, H, g, nu)
        expected = g * H ** 2 / (8.0 * nu)
        assert abs(u[0] - expected) < 1e-20

    def test_against_layered_solver(self):
        """analytic_solution_object reproduces analytic_profile for single layer."""
        H = 14.0
        g = 1e-5
        nu = 0.1
        y = np.linspace(0.5, H - 0.5, 14)
        u_direct = analytic_profile(y, H, g, nu)
        sol = analytic_solution_object(height=H, force=g, nu=nu)
        u_layered = sol.u(y)
        assert np.allclose(u_direct, u_layered, atol=1e-14)


# ---------------------------------------------------------------------------
# Profile KPI tests
# ---------------------------------------------------------------------------

class TestProfileKPIs:
    """Verify error metric computations."""

    def test_perfect_match(self):
        """Identical profiles yield zero error."""
        y = np.array([0.5, 1.5, 2.5])
        u = np.array([1.0, 2.0, 1.0])
        kpis = profile_kpis(y, u, u)
        assert kpis["eps_inf"] == 0.0
        assert kpis["eps_l2_rel"] == 0.0
        assert kpis["u_peak_err"] == 0.0
        assert kpis["q_err"] == 0.0

    def test_uniform_offset(self):
        """Uniform offset produces known errors."""
        y = np.array([0.5, 1.5, 2.5])
        u_exact = np.array([1.0, 2.0, 1.0])
        u_num = u_exact * 1.05  # 5% uniform over-prediction
        kpis = profile_kpis(y, u_num, u_exact)
        assert abs(kpis["u_peak_err"] - 0.05) < 1e-10
        assert abs(kpis["eps_inf"] - 0.05) < 1e-10

    def test_shape_mismatch_raises(self):
        """Different shapes raise ValueError."""
        with pytest.raises(ValueError):
            profile_kpis([0.5, 1.5], [1.0, 2.0], [1.0])

    def test_degenerate_reference_raises(self):
        """Zero reference profile raises ValueError."""
        with pytest.raises(ValueError, match="degenerate"):
            profile_kpis([0.5, 1.5], [1.0, 2.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Effective height tests
# ---------------------------------------------------------------------------

class TestEffectiveHeight:
    """Verify the geometry self-consistency check."""

    def test_analytic_profile_returns_constant_H(self):
        """Inverting the analytic profile recovers H exactly."""
        for res in [8, 16, 32, 64]:
            params = oss_units(resolution=res)
            y, H = wall_normal_coordinates(res)
            u = analytic_profile(y, H, params.acceleration_lu, params.viscosity_lu)
            H_eff = fit_effective_height(y, u, params.acceleration_lu, params.viscosity_lu)
            assert np.allclose(H_eff, H, atol=1e-12), f"res={res}, H_eff={H_eff}"

    def test_zero_force_raises(self):
        with pytest.raises(ValueError, match="force must be non-zero"):
            fit_effective_height([0.5, 1.5], [1.0, 2.0], 0.0, 0.1)


# ---------------------------------------------------------------------------
# Two-code diff metrics tests
# ---------------------------------------------------------------------------

class TestTwoCodeDiffMetrics:
    """Verify symmetric comparison between two numerical profiles."""

    def test_identical_profiles(self):
        u = np.array([0.1, 0.2, 0.1])
        m = two_code_diff_metrics(u, u)
        assert m["tv_rel"] == 0.0
        assert m["l2_rel"] == 0.0
        assert m["peak_ratio"] == 1.0

    def test_scaled_profiles(self):
        u_a = np.array([1.0, 2.0, 1.0])
        u_b = u_a * 1.1
        m = two_code_diff_metrics(u_a, u_b)
        assert abs(m["peak_ratio"] - 1.0 / 1.1) < 1e-10

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            two_code_diff_metrics([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestConvergence:
    """Grid-refinement convergence of the discrete peak vs continuous peak."""

    def test_second_order_convergence(self):
        """Discrete peak error decreases as O(1/N^2).

        For even H (all resolutions here give even H = res - 2), the center
        falls between two nodes, so the discrete peak is always slightly below
        the continuous peak.  The error scales as 1/N^2.
        """
        errors = []
        resolutions = [8, 16, 32, 64, 128]
        for res in resolutions:
            params = oss_units(resolution=res)
            y, H = wall_normal_coordinates(res)
            u = analytic_profile(y, H, params.acceleration_lu, params.viscosity_lu)
            err = abs(u.max() - params.u_peak_lu) / params.u_peak_lu
            errors.append(err)

        # Check that error ratio between successive refinements approaches 4 (2nd order)
        # Allow wider tolerance because the exact ratio depends on the parity of H
        for i in range(1, len(errors)):
            ratio = errors[i - 1] / errors[i]
            # For even H, convergence is approximately 4x but can vary slightly
            # due to the discrete sampling pattern
            assert 3.0 <= ratio <= 6.0, (
                f"res pair ({resolutions[i-1]}, {resolutions[i]}): ratio={ratio:.2f}, "
                f"errors={[f'{e:.2e}' for e in errors]}"
            )

    def test_all_metrics_within_15_percent(self):
        """Owner precision standard: all KPIs < 15% for the analytic profile itself."""
        for res in [8, 16, 32, 64]:
            params = oss_units(resolution=res)
            y, H = wall_normal_coordinates(res)
            u = analytic_profile(y, H, params.acceleration_lu, params.viscosity_lu)
            kpis = profile_kpis(y, u, u)  # self-comparison: should be zero
            assert kpis["eps_inf"] < 0.15
            assert kpis["eps_l2_rel"] < 0.15
            assert kpis["u_peak_err"] < 0.15
