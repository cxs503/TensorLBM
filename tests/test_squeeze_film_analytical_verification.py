"""Reynolds squeeze-film analytical module verification.

Background
----------
``tensorlbm.squeeze_film`` (433 lines, 13 public functions) and
``tensorlbm.squeeze_film_reference`` (independent numerical solver via Thomas
algorithm) had **zero dedicated tests** prior to this file.  The only existing
squeeze-film test (``test_squeeze_film_lbm_verification.py``) validates the LBM
solver, not the analytical anchors.

This file provides four independent verification pillars:

A. **Analytical formula correctness**: Reynolds plane-strain and disk pressure/
   force formulas checked against hand-computed values and literature constants.
B. **Analytical vs numerical cross-validation**: closed-form solutions compared
   against the independent finite-volume + tridiagonal solver in
   ``squeeze_film_reference`` (different code path: grid, Thomas algorithm,
   quadrature).  Tolerance: <0.1% for plane, <0.2% for disk.
C. **Grid convergence**: numerical solution error → 0 as n → ∞, with measured
   convergence rate ≥ 1.5 (second-order scheme expected).
D. **Auxiliary function verification**: ``squeeze_drainage_number``,
   ``fit_loglog_slope``, ``lubrication_share``, ``lubrication_onset_gap``,
   ``lubrication_share_contour`` — each checked against closed-form inverses
   or known limits.

Physical / literature basis
---------------------------
* Reynolds, O. (1886) Phil. Trans. R. Soc. Lond. 177, 157–234.
* Stefan, J. (1874) Sitzber. Akad. Wiss. Wien 69, 713.
* Plane strain:  F = 8 μ V a³ L / h³
* Axisymmetric:  F = 3π μ V a⁴ / (2 h³)
* Pressure (plane):  p(x) = 6 μ V (a² − x²) / h³
* Pressure (disk):   p(r) = 3 μ V (a² − r²) / h³
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.squeeze_film import (
    MU_WATER_20C,
    RHO_WATER_20C,
    SqueezeFilmSolution,
    fit_loglog_slope,
    lubrication_onset_gap,
    lubrication_share,
    lubrication_share_contour,
    reynolds_force_disk,
    reynolds_force_numeric,
    reynolds_force_plane,
    reynolds_pressure_disk,
    reynolds_pressure_plane,
    reynolds_velocity_plane,
    squeeze_drainage_number,
)
from tensorlbm.squeeze_film_reference import (
    reynolds_force_numeric as ref_force_numeric,
    reynolds_pressure_numeric,
    solve_tridiag,
)


# ===========================================================================
# A. Analytical formula correctness
# ===========================================================================

class TestAnalyticalFormulas:
    """Pillar A: closed-form Reynolds solutions match hand-computed values."""

    def test_plane_force_hand_computed(self):
        """F = 8 μ V a³ L / h³ — hand check with μ=1e-3, V=0.1, a=0.5, h=1e-3, L=1."""
        mu, a, h, V, L = 1e-3, 0.5, 1e-3, 0.1, 1.0
        expected = 8.0 * mu * V * a**3 * L / h**3
        assert reynolds_force_plane(mu, a, h, V, L) == pytest.approx(expected, rel=1e-12)

    def test_plane_force_stefan_scaling(self):
        """h⁻³ scaling: halving h should multiply force by 8."""
        mu, a, V, L = 1e-3, 0.5, 0.1, 1.0
        f1 = reynolds_force_plane(mu, a, h=2e-3, V=V, length=L)
        f2 = reynolds_force_plane(mu, a, h=1e-3, V=V, length=L)
        assert f2 / f1 == pytest.approx(8.0, rel=1e-12)

    def test_disk_force_hand_computed(self):
        """F = 3π μ V a⁴ / (2 h³) — hand check."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        expected = 3.0 * math.pi * mu * V * a**4 / (2.0 * h**3)
        assert reynolds_force_disk(mu, a, h, V) == pytest.approx(expected, rel=1e-12)

    def test_disk_force_stefan_scaling(self):
        """h⁻³ scaling for disk: halving h → force × 8."""
        mu, a, V = 1e-3, 0.5, 0.1
        f1 = reynolds_force_disk(mu, a, h=2e-3, V=V)
        f2 = reynolds_force_disk(mu, a, h=1e-3, V=V)
        assert f2 / f1 == pytest.approx(8.0, rel=1e-12)

    def test_plane_pressure_parabolic(self):
        """p(x) = 6 μ V (a² − x²) / h³ — parabolic profile, zero at edges."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        x = np.array([0.0, 0.25, 0.5])
        p = reynolds_pressure_plane(x, mu, a, h, V)
        # Centre pressure
        p_center = 6.0 * mu * V * a**2 / h**3
        assert p[0] == pytest.approx(p_center, rel=1e-12)
        # Edge pressure = 0
        assert p[2] == pytest.approx(0.0, abs=1e-15)
        # Mid-point: x=0.25 → p = 6μV(a²−0.0625)/h³
        p_mid = 6.0 * mu * V * (a**2 - 0.25**2) / h**3
        assert p[1] == pytest.approx(p_mid, rel=1e-12)

    def test_plane_pressure_zero_outside(self):
        """Pressure must be zero for |x| > a."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        x_out = np.array([0.6, 1.0, -0.7])
        p = reynolds_pressure_plane(x_out, mu, a, h, V)
        assert np.all(p == 0.0)

    def test_disk_pressure_parabolic(self):
        """p(r) = 3 μ V (a² − r²) / h³ — parabolic, zero at r=a."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        r = np.array([0.0, 0.3, 0.5])
        p = reynolds_pressure_disk(r, mu, a, h, V)
        p_center = 3.0 * mu * V * a**2 / h**3
        assert p[0] == pytest.approx(p_center, rel=1e-12)
        assert p[2] == pytest.approx(0.0, abs=1e-15)

    def test_disk_pressure_zero_outside(self):
        """Pressure must be zero for r > a."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        r_out = np.array([0.6, 1.0])
        p = reynolds_pressure_disk(r_out, mu, a, h, V)
        assert np.all(p == 0.0)

    def test_plane_force_equals_pressure_integral(self):
        """F = L ∫_{-a}^{a} p(x) dx — force must equal pressure integral."""
        mu, a, h, V, L = 1e-3, 0.5, 1e-3, 0.1, 2.0
        x = np.linspace(-a, a, 10001)
        p = reynolds_pressure_plane(x, mu, a, h, V)
        F_integral = float(np.trapezoid(p, x) * L)
        F_closed = reynolds_force_plane(mu, a, h, V, L)
        assert F_integral == pytest.approx(F_closed, rel=1e-6)

    def test_disk_force_equals_pressure_integral(self):
        """F = 2π ∫_0^a p(r) r dr — force must equal pressure integral."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        r = np.linspace(0, a, 10001)
        p = reynolds_pressure_disk(r, mu, a, h, V)
        F_integral = float(2.0 * math.pi * np.trapezoid(p * r, r))
        F_closed = reynolds_force_disk(mu, a, h, V)
        assert F_integral == pytest.approx(F_closed, rel=1e-6)

    def test_plane_disk_force_ratio(self):
        """For same params, F_plane / F_disk = 16 / (3π a) — derived from formulas."""
        mu, a, h, V, L = 1e-3, 0.5, 1e-3, 0.1, 1.0
        Fp = reynolds_force_plane(mu, a, h, V, L)
        Fd = reynolds_force_disk(mu, a, h, V)
        # Fp = 8μVa³L/h³, Fd = 3πμVa⁴/(2h³)
        # Fp/Fd = 16L / (3πa)
        expected_ratio = 16.0 * L / (3.0 * math.pi * a)
        assert Fp / Fd == pytest.approx(expected_ratio, rel=1e-12)


# ===========================================================================
# B. Analytical vs numerical cross-validation
# ===========================================================================

class TestAnalyticalVsNumerical:
    """Pillar B: closed-form vs independent numerical solver (different code path)."""

    @pytest.mark.parametrize("mu,a,h,V", [
        (1e-3, 0.5, 1e-3, 0.1),
        (1e-3, 0.5, 5e-4, 0.05),
        (2e-3, 1.0, 2e-3, 0.2),
        (MU_WATER_20C, 0.3, 5e-4, 0.01),
    ])
    def test_plane_force_cross_validation(self, mu, a, h, V):
        """Plane-strain force: analytical vs numerical < 0.1%."""
        F_analytical = reynolds_force_plane(mu, a, h, V)
        F_numerical = ref_force_numeric("plane", mu, a, h, V)
        err = abs(F_analytical - F_numerical) / F_analytical * 100.0
        assert err < 0.1, f"Plane force err={err:.4f}% (analytical={F_analytical:.6e}, numerical={F_numerical:.6e})"

    @pytest.mark.parametrize("mu,a,h,V", [
        (1e-3, 0.5, 1e-3, 0.1),
        (1e-3, 0.5, 5e-4, 0.05),
        (2e-3, 1.0, 2e-3, 0.2),
        (MU_WATER_20C, 0.3, 5e-4, 0.01),
    ])
    def test_disk_force_cross_validation(self, mu, a, h, V):
        """Disk force: analytical vs numerical < 0.2%."""
        F_analytical = reynolds_force_disk(mu, a, h, V)
        F_numerical = ref_force_numeric("disk", mu, a, h, V)
        err = abs(F_analytical - F_numerical) / F_analytical * 100.0
        assert err < 0.2, f"Disk force err={err:.4f}% (analytical={F_analytical:.6e}, numerical={F_numerical:.6e})"

    def test_plane_pressure_profile_cross_validation(self):
        """Plane pressure profile: pointwise comparison with numerical solution."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        s_num, p_num = reynolds_pressure_numeric("plane", mu, a, h, V, n=4001)
        p_ana = reynolds_pressure_plane(s_num, mu, a, h, V)
        # Max relative error (exclude near-zero edge region)
        mask = p_num > 0.01 * p_num.max()
        rel_err = np.abs(p_ana[mask] - p_num[mask]) / p_num[mask]
        assert rel_err.max() < 1e-4, f"Max pointwise pressure rel err = {rel_err.max():.2e}"

    def test_disk_pressure_profile_cross_validation(self):
        """Disk pressure profile: pointwise comparison with numerical solution.

        The axisymmetric finite-volume scheme has larger discretization error
        near r=a due to the coordinate singularity treatment; we use a relaxed
        5% tolerance for the full profile and 1% for the core region (r < 0.9a).
        """
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        s_num, p_num = reynolds_pressure_numeric("disk", mu, a, h, V, n=4001)
        p_ana = reynolds_pressure_disk(s_num, mu, a, h, V)
        # Core region (r < 0.9a): tighter tolerance
        core_mask = (s_num < 0.9 * a) & (p_num > 0.01 * p_num.max())
        rel_err_core = np.abs(p_ana[core_mask] - p_num[core_mask]) / p_num[core_mask]
        assert rel_err_core.max() < 0.01, f"Core disk pressure rel err = {rel_err_core.max():.2e}"
        # Full profile (excluding near-zero): relaxed tolerance
        mask = p_num > 0.01 * p_num.max()
        rel_err = np.abs(p_ana[mask] - p_num[mask]) / p_num[mask]
        assert rel_err.max() < 0.05, f"Full disk pressure rel err = {rel_err.max():.2e}"

    def test_reynolds_force_numeric_delegates_correctly(self):
        """squeeze_film.reynolds_force_numeric delegates to squeeze_film_reference."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        F1 = reynolds_force_numeric("plane", mu, a, h, V)
        F2 = ref_force_numeric("plane", mu, a, h, V)
        assert F1 == pytest.approx(F2, rel=1e-12)


# ===========================================================================
# C. Grid convergence of numerical solver
# ===========================================================================

class TestGridConvergence:
    """Pillar C: numerical solution converges to analytical as grid refines."""

    def test_plane_force_convergence_rate(self):
        """Plane force error should decrease ≥ O(n⁻¹·⁵) as n increases."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        F_exact = reynolds_force_plane(mu, a, h, V)
        ns = [101, 401, 1601, 6401]
        errors = []
        for n in ns:
            F_num = ref_force_numeric("plane", mu, a, h, V, n=n)
            errors.append(abs(F_exact - F_num) / F_exact)
        # Check monotonic decrease
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Non-monotonic convergence: err[{ns[i]}]={errors[i]:.2e} > err[{ns[i+1]}]={errors[i+1]:.2e}"
            )
        # Estimate convergence rate from last two points
        rate = math.log(errors[-2] / errors[-1]) / math.log(ns[-1] / ns[-2])
        assert rate > 1.5, f"Convergence rate {rate:.2f} < 1.5 (expected ≥ second order)"

    def test_disk_force_convergence_rate(self):
        """Disk force error should decrease as n increases."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        F_exact = reynolds_force_disk(mu, a, h, V)
        ns = [101, 401, 1601, 6401]
        errors = []
        for n in ns:
            F_num = ref_force_numeric("disk", mu, a, h, V, n=n)
            errors.append(abs(F_exact - F_num) / F_exact)
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Non-monotonic convergence: err[{ns[i]}]={errors[i]:.2e} > err[{ns[i+1]}]={errors[i+1]:.2e}"
            )

    def test_fine_grid_accuracy(self):
        """At n=4001, both plane and disk forces should be < 0.1% of analytical."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        for kind, F_exact in [
            ("plane", reynolds_force_plane(mu, a, h, V)),
            ("disk", reynolds_force_disk(mu, a, h, V)),
        ]:
            F_num = ref_force_numeric(kind, mu, a, h, V, n=4001)
            err = abs(F_exact - F_num) / F_exact * 100.0
            assert err < 0.1, f"{kind} force err at n=4001: {err:.4f}%"


# ===========================================================================
# D. Auxiliary function verification
# ===========================================================================

class TestAuxiliaryFunctions:
    """Pillar D: squeeze_drainage_number, fit_loglog_slope, lubrication_* functions."""

    def test_squeeze_drainage_number_formula(self):
        """Re_h = ρ V h / μ — gap-based Reynolds number (inertia control)."""
        mu, a, h, V, rho = 1e-3, 0.5, 1e-3, 0.1, RHO_WATER_20C
        expected = rho * V * h / mu
        assert squeeze_drainage_number(mu, a, h, V, rho) == pytest.approx(expected, rel=1e-12)

    def test_squeeze_drainage_number_scaling(self):
        """Re_h ∝ h: doubling h → Re_h × 2."""
        mu, a, V, rho = 1e-3, 0.5, 0.1, RHO_WATER_20C
        sq1 = squeeze_drainage_number(mu, a, h=1e-3, V=V, rho=rho)
        sq2 = squeeze_drainage_number(mu, a, h=2e-3, V=V, rho=rho)
        assert sq2 / sq1 == pytest.approx(2.0, rel=1e-12)

    def test_squeeze_drainage_number_inertia_criterion(self):
        """For thin-film lubrication, (h/a)² Re_h << 1 should hold."""
        mu, a, h, V = MU_WATER_20C, 0.5, 1e-4, 0.01
        Re_h = squeeze_drainage_number(mu, a, h, V)
        inertia_param = (h / a)**2 * Re_h
        assert inertia_param < 0.01, f"Inertia parameter {inertia_param:.2e} should be << 1 for lubrication"

    def test_fit_loglog_slope(self):
        """fit_loglog_slope recovers exact power-law exponent from dict return."""
        h = np.logspace(-3, 0, 50)
        force = 2.5 * h**(-3.0)  # exact -3 slope
        result = fit_loglog_slope(h, force)
        assert isinstance(result, dict)
        assert result["slope"] == pytest.approx(-3.0, rel=1e-6)
        assert result["r2"] > 0.9999

    def test_fit_loglog_slope_h_cubed(self):
        """h⁻³ law: fit should recover slope ≈ -3."""
        hs = np.array([1e-3, 2e-3, 4e-3, 8e-3])
        forces = np.array([1.0, 1.0/8, 1.0/64, 1.0/512])  # exact h⁻³
        result = fit_loglog_slope(hs, forces)
        assert result["slope"] == pytest.approx(-3.0, rel=1e-10)
        assert result["r2"] == pytest.approx(1.0, rel=1e-12)

    def test_lubrication_share_formula(self):
        """lubrication_share = F_film / (F_dry + F_film)."""
        assert lubrication_share(100.0, 400.0) == pytest.approx(0.2, rel=1e-12)
        assert lubrication_share(0.0, 100.0) == pytest.approx(0.0, rel=1e-12)
        assert lubrication_share(100.0, 0.0) == pytest.approx(1.0, rel=1e-12)

    def test_lubrication_onset_gap_inverse(self):
        """lubrication_onset_gap inverts lubrication_share correctly."""
        mu, a, V, L, R_dry = 1e-3, 0.5, 0.1, 1.0, 1000.0
        target_share = 0.15
        h_star = lubrication_onset_gap(target_share, R_dry, mu, a, V, L)
        # Verify: at h_star, the share should be target_share
        F_film = reynolds_force_plane(mu, a, h_star, V, L)
        actual_share = lubrication_share(F_film, R_dry)
        assert actual_share == pytest.approx(target_share, rel=1e-6)

    def test_lubrication_onset_gap_scaling(self):
        """h* ∝ V^{1/3}: doubling V → h* × 2^{1/3}."""
        mu, a, L, R_dry = 1e-3, 0.5, 1.0, 1000.0
        h1 = lubrication_onset_gap(0.15, R_dry, mu, a, V=0.1, length=L)
        h2 = lubrication_onset_gap(0.15, R_dry, mu, a, V=0.8, length=L)
        assert h2 / h1 == pytest.approx(2.0, rel=1e-6)  # 0.8/0.1 = 8, 8^{1/3} = 2

    def test_lubrication_share_contour_slope(self):
        """Contour h*(V) has log-log slope 1/3."""
        result = lubrication_share_contour(
            share_levels=(0.05, 0.15),
            mu=MU_WATER_20C, a=0.5, length=1.0, dry_force=500.0,
        )
        for level, ctr in result["contours"].items():
            assert abs(ctr["slope_in_loglog"] - 1.0 / 3.0) < 0.01, (
                f"Contour slope for level {level}: {ctr['slope_in_loglog']:.4f} (expected ~0.333)"
            )

    def test_squeeze_film_solution_dataclass(self):
        """SqueezeFilmSolution dataclass computes p_max correctly."""
        mu, a, h, V = 1e-3, 0.5, 1e-3, 0.1
        x = np.linspace(-a, a, 1001)
        p = reynolds_pressure_plane(x, mu, a, h, V)
        sol = SqueezeFilmSolution(s=x, p=p, force=reynolds_force_plane(mu, a, h, V),
                                   mu=mu, a=a, h=h, V=V, kind="plane")
        p_max_expected = 6.0 * mu * V * a**2 / h**3
        assert sol.p_max == pytest.approx(p_max_expected, rel=1e-12)
        assert sol.kind == "plane"


# ===========================================================================
# E. Input validation
# ===========================================================================

class TestInputValidation:
    """Edge cases and error handling."""

    def test_negative_mu_raises(self):
        with pytest.raises(ValueError, match="mu"):
            reynolds_force_plane(mu=-1e-3, a=0.5, h=1e-3, V=0.1)

    def test_zero_h_raises(self):
        with pytest.raises(ValueError, match="h"):
            reynolds_force_plane(mu=1e-3, a=0.5, h=0.0, V=0.1)

    def test_negative_V_raises(self):
        with pytest.raises(ValueError, match="V"):
            reynolds_force_plane(mu=1e-3, a=0.5, h=1e-3, V=-0.1)

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            reynolds_force_plane(mu=1e-3, a=0.5, h=1e-3, V=0.1, length=0.0)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="unknown kind"):
            reynolds_force_numeric("sphere", 1e-3, 0.5, 1e-3, 0.1)

    def test_n_too_small_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            reynolds_pressure_numeric("plane", 1e-3, 0.5, 1e-3, 0.1, n=5)

    def test_share_out_of_range_raises(self):
        with pytest.raises(ValueError, match="share must"):
            lubrication_onset_gap(1.5, 1000.0, 1e-3, 0.5, 0.1)

    def test_tridiag_single_element(self):
        """Thomas algorithm with n=1 should return rhs/diag."""
        result = solve_tridiag(np.array([2.0]), np.array([]), np.array([]), np.array([6.0]))
        assert result[0] == pytest.approx(3.0)


# ===========================================================================
# F. Physical constants
# ===========================================================================

class TestPhysicalConstants:
    """Verify physical constants are reasonable."""

    def test_mu_water_20c(self):
        """Water viscosity at 20°C ≈ 1.002 mPa·s (IAPWS-95)."""
        assert 0.99e-3 < MU_WATER_20C < 1.02e-3

    def test_rho_water_20c(self):
        """Water density at 20°C ≈ 998 kg/m³."""
        assert 997.0 < RHO_WATER_20C < 999.0
