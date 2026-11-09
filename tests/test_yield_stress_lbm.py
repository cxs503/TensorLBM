"""Comprehensive test suite for yield-stress fluid LBM (IN32-N05 campaign).

Tests cover:
(a) Bingham Couette: velocity profile validation
(b) Bingham Poiseuille: full-field validation
(c) Newtonian limit: tau_y -> 0 vs Poiseuille
(d) Regularisation independence: m scan
(e) Yield surface topology: Bn scan
(f) Energy budget
(g) Negative control
(h) Cross-validation
(i) >= 18 pytest items including negative cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream
from tensorlbm.powerlaw import strain_rate_shear_rate_2d
from tensorlbm.yield_stress_rheology import (
    papanastasiou_viscosity,
    herschel_bulkley_viscosity,
    bingham_viscosity,
    bi_viscosity_model,
    effective_relaxation_time,
    shear_rate_from_stress,
    YieldStressRheologyError,
    BinghamParams,
    HerschelBulkleyParams,
)
from tensorlbm.yield_stress_lbm import (
    collide_bingham_bgk,
    collide_bingham_bgk_forced,
    collide_hb_bgk,
    collide_hb_bgk_forced,
    compute_tau_field_bingham,
    compute_tau_field_hb,
    neq_stress_2d,
)
from tensorlbm.yield_stress_analytical import (
    bingham_couette_profile,
    bingham_poiseuille_profile,
    bingham_poiseuille_plug_half_width,
    bingham_poiseuille_max_velocity,
    newtonian_poiseuille_profile,
    newtonian_couette_profile,
    squeeze_film_lubrication_bounds,
)
from tensorlbm.yield_stress_channel import (
    run_bingham_couette,
    run_bingham_poiseuille,
    run_hb_poiseuille,
    compute_lubrication_force,
    compute_yield_surface_topology,
)


# ---------------------------------------------------------------------------
# (a) Bingham Couette: velocity profile validation
# ---------------------------------------------------------------------------

class TestBinghamCouette:
    """Bingham Couette flow validation."""

    def test_couette_l2_error_n8(self):
        """L2 relative error < 15% at N=8."""
        H = 8
        u_wall = 0.05
        tau_y = 0.001
        mu_0 = 0.02
        m = 100.0

        result = run_bingham_couette(
            H, u_wall, tau_y, mu_0, m,
            max_iter=5000, tol=1e-4,
            dtype=torch.float64,
        )
        assert result["l2_error"] < 0.15, (
            f"Couette L2 error {result['l2_error']:.6f} >= 15% at N={H}"
        )

    def test_couette_linear_profile(self):
        """Couette profile is linear (R² > 0.999)."""
        H = 8
        result = run_bingham_couette(
            H, u_wall=0.05, tau_y=0.001, mu_0=0.02, m=100.0,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )
        u_sim = result["u_sim"].numpy()
        y = np.arange(1, H + 1, dtype=np.float64)
        coeffs = np.polyfit(y, u_sim, 1)
        u_fit = np.polyval(coeffs, y)
        ss_res = np.sum((u_sim - u_fit) ** 2)
        ss_tot = np.sum((u_sim - u_sim.mean()) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-30)
        assert r_squared > 0.999, f"Couette R² = {r_squared:.6f} < 0.999"


# ---------------------------------------------------------------------------
# (b) Bingham Poiseuille: full-field validation
# ---------------------------------------------------------------------------

class TestBinghamPoiseuille:
    """Bingham Poiseuille flow validation."""

    def test_poiseuille_runs(self):
        """Poiseuille simulation runs and produces reasonable results."""
        H = 8
        ax = 1e-5
        tau_y = 2e-5
        mu_0 = 0.02
        m = 100.0

        result = run_bingham_poiseuille(
            H, ax, tau_y, mu_0, m,
            max_iter=5000, tol=1e-4,
            dtype=torch.float64,
        )
        assert result["converged"], "Simulation did not converge"
        assert result["u_max_sim"] > 0, "Max velocity should be positive"
        assert result["l2_error"] < 0.5, f"L2 error {result['l2_error']:.4f} too high"


# ---------------------------------------------------------------------------
# (c) Newtonian limit: tau_y -> 0 vs Poiseuille
# ---------------------------------------------------------------------------

class TestNewtonianLimit:
    """Newtonian limit validation."""

    def test_tau_y_zero_vs_newtonian_poiseuille(self):
        """tau_y=0 solution matches Newtonian Poiseuille within 5%."""
        H = 8
        ax = 1e-5
        mu_0 = 0.02
        m = 100.0

        result = run_bingham_poiseuille(
            H, ax, tau_y=0.0, mu_0=mu_0, m=m,
            max_iter=5000, tol=1e-4,
            dtype=torch.float64,
        )

        analytic_u, _ = newtonian_poiseuille_profile(H, ax, mu_0)
        u_sim = result["u_sim"]
        l2 = float(torch.sqrt(((u_sim - analytic_u) ** 2).sum()) / (torch.sqrt((analytic_u ** 2).sum()) + 1e-30))
        assert l2 < 0.05, f"Newtonian limit L2 error {l2:.6f} >= 5%"

    def test_hb_n1_vs_bingham(self):
        """HB with n=1, K=mu_0 matches Bingham within 1e-12."""
        H = 8
        ax = 1e-5
        tau_y = 2e-5
        mu_0 = 0.02
        m = 100.0

        result_bingham = run_bingham_poiseuille(
            H, ax, tau_y, mu_0, m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )
        result_hb = run_hb_poiseuille(
            H, ax, tau_y, mu_0, n=1.0, m=m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )

        diff = float((result_bingham["u_sim"] - result_hb["u_sim"]).abs().max())
        assert diff < 1e-12, f"HB(n=1) vs Bingham max diff {diff:.2e} >= 1e-12"


# ---------------------------------------------------------------------------
# (d) Regularisation independence: m scan
# ---------------------------------------------------------------------------

class TestRegularisationIndependence:
    """Papanastasiou regularisation parameter independence."""

    def test_m_scan_monotonic_error(self):
        """Error decreases monotonically with increasing m."""
        H = 8
        ax = 1e-5
        tau_y = 2e-5
        mu_0 = 0.02

        errors = []
        m_values = [10.0, 50.0, 100.0, 500.0, 1000.0]
        for m in m_values:
            result = run_bingham_poiseuille(
                H, ax, tau_y, mu_0, m,
                max_iter=5000, tol=1e-4, dtype=torch.float64,
            )
            errors.append(result["l2_error"])

        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] * 1.1, (
                f"Error not monotonic: m={m_values[i-1]}→{m_values[i]}, "
                f"error {errors[i-1]:.6f}→{errors[i]:.6f}"
            )


# ---------------------------------------------------------------------------
# (e) Yield surface topology: Bn scan
# ---------------------------------------------------------------------------

class TestYieldSurfaceTopology:
    """Yield surface topology analysis."""

    def test_plug_area_monotonic_with_bn(self):
        """A_plug/A is monotonically non-decreasing with Bn."""
        H = 8
        mu_0 = 0.02
        m = 100.0
        ax = 1e-5

        fractions = []
        bn_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        for bn in bn_values:
            tau_y = bn * mu_0 * 0.01 / H
            result = run_bingham_poiseuille(
                H, ax, tau_y, mu_0, m,
                max_iter=5000, tol=1e-4, dtype=torch.float64,
            )
            _, ux, uy = macroscopic(result["f"])
            topo = compute_yield_surface_topology(ux, uy, tau_y, mu_0)
            fractions.append(topo["plug_area_fraction"])

        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1] - 0.05, (
                f"Plug fraction not monotonic: Bn={bn_values[i-1]}→{bn_values[i]}, "
                f"f={fractions[i-1]:.4f}→{fractions[i]:.4f}"
            )

    def test_critical_bn_exists(self):
        """Critical Bn* where plug transitions exists."""
        H = 8
        mu_0 = 0.02
        m = 100.0
        ax = 1e-5

        has_plug = []
        bn_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        for bn in bn_values:
            tau_y = bn * mu_0 * 0.01 / H
            result = run_bingham_poiseuille(
                H, ax, tau_y, mu_0, m,
                max_iter=5000, tol=1e-4, dtype=torch.float64,
            )
            _, ux, uy = macroscopic(result["f"])
            topo = compute_yield_surface_topology(ux, uy, tau_y, mu_0)
            has_plug.append(topo["n_components"] > 0)

        transitions = sum(1 for i in range(1, len(has_plug)) if has_plug[i] != has_plug[i - 1])
        assert transitions >= 1, (
            f"No critical Bn* found in scan: has_plug={has_plug}"
        )


# ---------------------------------------------------------------------------
# (f) Energy budget
# ---------------------------------------------------------------------------

class TestEnergyBudget:
    """Energy conservation check."""

    def test_viscous_dissipation_equals_work(self):
        """Viscous dissipation integral = boundary/external work, residual < 1e-8."""
        H = 8
        ax = 1e-5
        tau_y = 2e-5
        mu_0 = 0.02
        m = 100.0

        result = run_bingham_poiseuille(
            H, ax, tau_y, mu_0, m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )
        _, ux, uy = macroscopic(result["f"])

        gamma = strain_rate_shear_rate_2d(ux, uy)
        mu_eff = papanastasiou_viscosity(gamma, tau_y, mu_0, m)

        fluid_mask = ~result["wall"][1:-1, :]
        dissipation = float((mu_eff[1:-1, :] * gamma[1:-1, :] ** 2 * fluid_mask).sum())
        work = float((ax * ux[1:-1, :] * fluid_mask).sum())

        if abs(work) > 1e-30:
            residual = abs(dissipation - work) / abs(work)
        else:
            residual = abs(dissipation - work)

        assert residual < 1e-8, (
            f"Energy residual {residual:.2e} >= 1e-8 "
            f"(dissipation={dissipation:.6e}, work={work:.6e})"
        )


# ---------------------------------------------------------------------------
# (g) Negative control
# ---------------------------------------------------------------------------

class TestNegativeControl:
    """Negative control: yield solution must differ from Newtonian at high Bn."""

    def test_newtonian_vs_yield_at_high_bn(self):
        """tau_y=0 Newtonian solution differs from yield solution by > 50% at Bn>=10."""
        H = 8
        ax = 1e-5
        mu_0 = 0.02
        m = 100.0

        result_newton = run_bingham_poiseuille(
            H, ax, tau_y=0.0, mu_0=mu_0, m=m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )

        tau_y = 10.0 * mu_0 * 0.01 / H
        result_yield = run_bingham_poiseuille(
            H, ax, tau_y, mu_0, m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )

        u_newton = result_newton["u_sim"]
        u_yield = result_yield["u_sim"]

        diff = float(((u_newton - u_yield).abs() / (u_newton.abs() + 1e-30)).mean())
        assert diff > 0.50, (
            f"Newtonian vs yield difference {diff:.4f} <= 50% at Bn>=10"
        )

    @pytest.mark.parametrize("bn_target", [10, 20, 50, 100, 200, 500])
    def test_newtonian_vs_yield_multiple_bn(self, bn_target):
        """Six Bn>=10 cases all show > 50% difference."""
        H = 8
        ax = 1e-5
        mu_0 = 0.02
        m = 100.0

        result_newton = run_bingham_poiseuille(
            H, ax, tau_y=0.0, mu_0=mu_0, m=m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )

        tau_y = bn_target * mu_0 * 0.01 / H
        result_yield = run_bingham_poiseuille(
            H, ax, tau_y, mu_0, m,
            max_iter=5000, tol=1e-4, dtype=torch.float64,
        )

        u_newton = result_newton["u_sim"]
        u_yield = result_yield["u_sim"]
        diff = float(((u_newton - u_yield).abs() / (u_newton.abs() + 1e-30)).mean())
        assert diff > 0.50, (
            f"Bn={bn_target}: Newtonian vs yield diff {diff:.4f} <= 50%"
        )


# ---------------------------------------------------------------------------
# (h) Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """LBM vs platform algebraic branch cross-validation."""

    def test_lbm_vs_algebraic_bn_01_to_1(self):
        """LBM lubrication force vs algebraic branch at Bn in [0.1, 1], diff < 15%."""
        H = 8
        mu_0 = 0.02
        m = 100.0
        ax = 1e-5

        for bn in [0.1, 0.3, 0.5, 0.7, 1.0]:
            tau_y = bn * mu_0 * 0.01 / H

            result = run_bingham_poiseuille(
                H, ax, tau_y, mu_0, m,
                max_iter=5000, tol=1e-4, dtype=torch.float64,
            )
            _, ux, uy = macroscopic(result["f"])
            gamma = strain_rate_shear_rate_2d(ux, uy)
            mu_eff = papanastasiou_viscosity(gamma, tau_y, mu_0, m)

            f_lbm = compute_lubrication_force(ux, mu_eff, result["wall"])

            h_lu = float(H)
            U_char = float(ux[1:-1, :].max())
            L_lu = float(H)
            bounds = squeeze_film_lubrication_bounds(h_lu, U_char, L_lu, mu_0, tau_y)

            f_alg = bounds["upper_bound"]

            if f_alg > 1e-30:
                rel_diff = abs(f_lbm - f_alg) / f_alg
            else:
                rel_diff = abs(f_lbm - f_alg)

            assert rel_diff < 0.15, (
                f"Bn={bn}: LBM vs algebraic diff {rel_diff:.4f} >= 15% "
                f"(f_lbm={f_lbm:.6e}, f_alg={f_alg:.6e})"
            )


# ---------------------------------------------------------------------------
# Rheology unit tests
# ---------------------------------------------------------------------------

class TestRheologyUnit:
    """Unit tests for rheology functions."""

    def test_papanastasiou_zero_shear(self):
        """Papanastasiou viscosity at zero shear rate is finite."""
        gamma = torch.tensor([0.0, 1e-15, 1e-12], dtype=torch.float64)
        mu = papanastasiou_viscosity(gamma, tau_y=0.01, mu_0=0.02, m=100.0)
        assert torch.isfinite(mu).all()
        expected = 0.02 + 0.01 * 100.0
        assert abs(mu[0].item() - expected) < 1e-6

    def test_papanastasiou_high_shear(self):
        """Papanastasiou viscosity at high shear → mu_0."""
        gamma = torch.tensor([100.0, 1000.0])
        mu = papanastasiou_viscosity(gamma, tau_y=0.01, mu_0=0.02, m=100.0)
        assert (mu - 0.02).abs().max() < 0.001

    def test_hb_reduces_to_bingham_at_n1(self):
        """HB with n=1, K=mu_0 equals Bingham."""
        gamma = torch.linspace(0.01, 10.0, 100)
        mu_pap = papanastasiou_viscosity(gamma, tau_y=0.01, mu_0=0.02, m=100.0)
        mu_hb = herschel_bulkley_viscosity(gamma, tau_y=0.01, K=0.02, n=1.0, m=100.0)
        assert (mu_pap - mu_hb).abs().max() < 1e-12

    def test_effective_relaxation_time(self):
        """Relaxation time formula tau = 3*nu + 0.5."""
        mu = torch.tensor([0.01, 0.02, 0.1])
        tau = effective_relaxation_time(mu, rho=1.0, tau_min=0.501)
        expected = 3.0 * mu + 0.5
        assert (tau - expected).abs().max() < 1e-12

    def test_bingham_params_validation(self):
        """BinghamParams rejects negative tau_y."""
        with pytest.raises(YieldStressRheologyError):
            BinghamParams(tau_y=-0.01, mu_0=0.02)

    def test_hb_params_validation(self):
        """HerschelBulkleyParams rejects invalid n."""
        with pytest.raises(YieldStressRheologyError):
            HerschelBulkleyParams(tau_y=0.01, K=0.02, n=0.0)
        with pytest.raises(YieldStressRheologyError):
            HerschelBulkleyParams(tau_y=0.01, K=0.02, n=3.0)

    def test_bi_viscosity_model(self):
        """Bi-viscosity model returns mu_plast below yield rate."""
        gamma = torch.tensor([0.001, 0.01, 0.1, 1.0])
        mu = bi_viscosity_model(gamma, tau_y=0.01, mu_0=0.02, mu_ratio=1e4)
        mu_plast = 1e4 * 0.02
        assert mu[0].item() == mu_plast

    def test_neq_stress_2d_shape(self):
        """neq_stress_2d returns correct shapes."""
        f = torch.ones((9, 32, 32), dtype=torch.float64)
        rho, ux, uy, pi_xx, pi_xy, pi_yy = neq_stress_2d(f)
        assert rho.shape == (32, 32)
        assert ux.shape == (32, 32)
        assert pi_xx.shape == (32, 32)

    def test_squeeze_film_bounds(self):
        """Squeeze-film bounds are consistent."""
        bounds = squeeze_film_lubrication_bounds(h=0.01, U=0.1, L=1.0, mu_0=0.001, tau_y=0.01)
        assert bounds["Bn"] > 0
        assert bounds["upper_bound"] >= bounds["newtonian"]
        assert bounds["lower_bound"] >= bounds["newtonian"]

    def test_analytic_poiseuille_plug_width(self):
        """Analytic plug half-width is correct."""
        H = 100
        ax = 1e-4
        tau_y = 0.002
        mu_0 = 0.01
        y_p = bingham_poiseuille_plug_half_width(H, ax, tau_y, mu_0)
        expected = tau_y / ax
        assert abs(y_p - expected) < 1e-10

    def test_analytic_poiseuille_max_velocity(self):
        """Analytic max velocity is positive and finite."""
        H = 100
        ax = 1e-4
        tau_y = 0.002
        mu_0 = 0.01
        u_max = bingham_poiseuille_max_velocity(H, ax, tau_y, mu_0)
        assert u_max > 0
        assert math.isfinite(u_max)

    def test_collide_bingham_bgk_shape(self):
        """Bingham collision preserves shape."""
        f = torch.ones((9, 32, 32), dtype=torch.float64) * 4.0 / 9.0
        f_out = collide_bingham_bgk(f, tau_y=0.01, mu_0=0.02, m=100.0)
        assert f_out.shape == f.shape

    def test_collide_hb_bgk_shape(self):
        """HB collision preserves shape."""
        f = torch.ones((9, 32, 32), dtype=torch.float64) * 4.0 / 9.0
        f_out = collide_hb_bgk(f, tau_y=0.01, K=0.02, n=0.7, m=100.0)
        assert f_out.shape == f.shape

    def test_tau_y_zero_is_newtonian(self):
        """tau_y=0 gives Newtonian viscosity mu_0."""
        gamma = torch.tensor([0.01, 0.1, 1.0, 10.0])
        mu = papanastasiou_viscosity(gamma, tau_y=0.0, mu_0=0.02, m=100.0)
        assert (mu - 0.02).abs().max() < 1e-12


# ---------------------------------------------------------------------------
# (f) Wall bounce-back order: NoDynamics(pre-collision) -> reflect -> stream
# ---------------------------------------------------------------------------

class TestBounceBackOrder:
    """壁面处理顺序的回归闸门（2026-11-09 修复）。

    旧实现是 ``stream`` 之后再反弹：壁面格点里是**流入**分布与周期回卷的混合，
    反弹后仍留下一个非零滑移，Poiseuille 振幅偏高约 20%。正确顺序是先回填碰撞前
    分布（NoDynamics）、再对碰撞后分布反弹、最后 stream（与
    ``tests/test_lbm_verification.py`` 的解析解阶梯同约定）。
    """

    H = 8
    AX = 1e-5
    MU = 0.02
    M = 100.0

    def _run(self, scheme, **kw):
        return run_bingham_poiseuille(
            self.H, self.AX, 0.0, self.MU, self.M,
            max_iter=kw.get("max_iter", 8000), tol=kw.get("tol", 1e-8),
            dtype=torch.float64, bounce_scheme=scheme,
        )

    def test_halfway_scheme_satisfies_newtonian_limit(self):
        r = self._run("halfway")
        assert r["l2_error"] < 0.05, f"halfway L2 {r['l2_error']:.4f} >= 5%"

    def test_post_stream_scheme_violates_no_slip(self):
        """负控：旧顺序必须**失败** 5% 判据，否则本闸门没有鉴别力。"""
        r = self._run("post_stream")
        assert r["l2_error"] > 0.10, (
            f"post_stream L2 {r['l2_error']:.4f} 未超 10%——负控失效，"
            "说明扫描参数已不能暴露该缺陷"
        )

    def test_wall_row_velocity_is_no_slip(self):
        """第一层流体格点速度必须是「半格外推到零」的值，不是滑移值。"""
        r_h = self._run("halfway", max_iter=8000)
        r_l = self._run("post_stream", max_iter=2000, tol=1e-6)
        slip_h = abs(float(r_h["u_sim"][0]) - float(r_h["analytic_u"][0]))
        slip_l = abs(float(r_l["u_sim"][0]) - float(r_l["analytic_u"][0]))
        assert slip_h < slip_l / 3.0, (
            f"halfway wall slip {slip_h:.3e} 未显著小于 post_stream {slip_l:.3e}"
        )

    def test_couette_moving_wall_still_no_slip(self):
        """Couette 走同一修复路径：线性剖面 L2 误差 < 5%。"""
        r = run_bingham_couette(
            self.H, 0.01, 0.0, self.MU, self.M,
            max_iter=8000, tol=1e-8, dtype=torch.float64,
            bounce_scheme="halfway",
        )
        assert r["l2_error"] < 0.05, f"Couette L2 {r['l2_error']:.4f} >= 5%"

    def test_unknown_bounce_scheme_raises(self):
        import pytest as _pytest
        with _pytest.raises(ValueError):
            self._run("nope", max_iter=200, tol=1e-3)
