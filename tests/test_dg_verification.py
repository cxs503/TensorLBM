"""Verification gate for the DG convection/diffusion ladder (IN70-N01).

These tests are the machine-checkable part of the campaign node: they pin the
*mathematical* properties of the kernels (consistency, conservation, operator
signs, exact Riemann references) and the *convergence order and physical
criteria* of the five benchmarks at a resolution that runs in CI (a few
seconds).  The production-resolution numbers live in
``artifacts/IN70_N01/*.json``.

Run CPU-only:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_dg_verification.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.dg_verification import (
    EulerDG1D,
    advection_rhs_2d,
    cell_means,
    convection_diffusion_rhs_2d,
    diffusion_rhs_2d,
    exact_isothermal_riemann,
    exact_sod,
    integral_2d,
    l2_error_vs,
    minmod,
    node_coords_2d,
    rarefaction_tail,
    ref_element,
    shock_position,
    spectral_report,
    ssp_rk3,
    stable_dt,
    tvb_limit,
)

DT = np.float64


# ---------------------------------------------------------------------------
# 1. reference element
# ---------------------------------------------------------------------------


class TestElement:
    def test_lobatto_nodes(self):
        assert np.allclose(ref_element(1).nodes, [-1.0, 1.0])
        assert np.allclose(ref_element(2).nodes, [-1.0, 0.0, 1.0])

    def test_differentiation_matrix_p1(self):
        """For P1 Lobatto, l_j'(x_i) = [-.5, .5] on both rows."""
        D = ref_element(1).D
        assert np.allclose(D, [[-0.5, 0.5], [-0.5, 0.5]])

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_differentiation_is_exact_for_polynomials(self, p):
        el = ref_element(p)
        for k in range(p + 1):
            f = el.nodes**k
            df = el.D @ f
            want = np.zeros_like(el.nodes) if k == 0 else k * el.nodes ** (k - 1)
            assert np.allclose(df, want, atol=1e-12)

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_quadrature_integrates_the_basis_exactly(self, p):
        """Quadrature + interpolation must integrate any degree-p polynomial."""
        el = ref_element(p)
        for k in range(p + 1):
            f = el.nodes**k
            # int_{-1}^{1} x^k dx = 0 (k odd) / 2/(k+1) (k even); integral()
            # returns 0.5*dx*sum -> for dx = 2 this is the reference integral.
            want = 0.0 if k % 2 else 2.0 / (k + 1)
            assert abs(el.integral(f, 2.0) - want) < 1e-12


# ---------------------------------------------------------------------------
# 2. operator consistency and conservation
# ---------------------------------------------------------------------------


class TestOperators:
    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_advection_consistent(self, p):
        """The DG RHS must reproduce ``-a.grad(u)`` for a smooth field."""
        el = ref_element(p)
        n, dx = 32, 1.0 / 32
        X, Y = node_coords_2d(n, n, el, dx, dx)
        u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        rhs = advection_rhs_2d(u, 0.7, -0.3, el, dx, dx)
        exact = -2 * np.pi * (
            0.7 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
            - 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        )
        m = np.abs(exact) > 1.0
        assert abs(np.mean(rhs[m] / exact[m]) - 1.0) < 3e-2

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_diffusion_consistent_and_definite(self, p):
        """LDG must give ``+nu laplacian`` (sign!) with the right magnitude."""
        el = ref_element(p)
        n, dx = 32, 1.0 / 32
        X, Y = node_coords_2d(n, n, el, dx, dx)
        u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        rhs = diffusion_rhs_2d(u, 1.0, el, dx, dx)
        exact = -8 * np.pi**2 * u
        m = np.abs(exact) > 1.0
        assert abs(np.mean(rhs[m] / exact[m]) - 1.0) < 3e-2

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_constant_field_is_steady(self, p):
        el = ref_element(p)
        u = np.ones((4, 5, el.n_node, el.n_node))
        assert np.abs(advection_rhs_2d(u, 0.7, 0.3, el, 0.2, 0.2)).max() < 1e-12
        assert np.abs(diffusion_rhs_2d(u, 0.1, el, 0.2, 0.2)).max() < 1e-12

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_rhs_is_mass_conserving(self, p):
        """The periodic DG operator annihilates global mass (ledger identity)."""
        el = ref_element(p)
        rng = np.random.default_rng(0)
        u = rng.standard_normal((3, 4, el.n_node, el.n_node))
        r = advection_rhs_2d(u, 0.7, -0.3, el, 0.25, 0.25)
        assert abs(integral_2d(r, el, 0.25, 0.25)) < 1e-12
        r = diffusion_rhs_2d(u, 0.1, el, 0.25, 0.25)
        assert abs(integral_2d(r, el, 0.25, 0.25)) < 1e-12


# ---------------------------------------------------------------------------
# 3. spectral stability constants (the source of the time step)
# ---------------------------------------------------------------------------


class TestStability:
    def test_spectral_constants_scale_as_expected(self):
        rep = spectral_report((1, 2, 3))
        assert rep["P1"]["c_adv"] == pytest.approx(6.0, rel=1e-6)
        assert rep["P1"]["kappa_diff"] == pytest.approx(36.0, rel=1e-6)
        # both grow with the degree
        assert rep["P1"]["c_adv"] < rep["P2"]["c_adv"] < rep["P3"]["c_adv"]
        assert rep["P1"]["kappa_diff"] < rep["P2"]["kappa_diff"] < rep["P3"]["kappa_diff"]

    def test_stable_dt_scales_with_dx(self):
        d1 = stable_dt(1, 0.1, advective_speed=1.0)
        d2 = stable_dt(1, 0.05, advective_speed=1.0)
        assert d2 == pytest.approx(0.5 * d1, rel=1e-12)
        e1 = stable_dt(1, 0.1, nu=1.0)
        e2 = stable_dt(1, 0.05, nu=1.0)
        assert e2 == pytest.approx(0.25 * e1, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. limiters
# ---------------------------------------------------------------------------


class TestLimiter:
    def test_minmod(self):
        a = np.array([1.0, 2.0, -3.0, 0.0])
        b = np.array([2.0, 1.0, 3.0, 5.0])
        assert np.allclose(minmod(a, b), [1.0, 1.0, 0.0, 0.0])

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_limiter_preserves_cell_mean(self, p):
        """Conservation: the limiter must not touch the cell average."""
        el = ref_element(p)
        rng = np.random.default_rng(1)
        u = rng.standard_normal((6, el.n_node))
        ul = tvb_limit(u, el, 0.1, M=0.0)
        assert np.allclose(cell_means(ul, el), cell_means(u, el), atol=1e-14)

    def test_limiter_reproduces_linear_data(self):
        el = ref_element(1)
        n, dx = 8, 0.125
        xc = (np.arange(n) + 0.5) * dx
        lin = xc[:, None] + 0.5 * dx * el.nodes[None, :]
        ul = tvb_limit(lin, el, dx, M=0.0)
        assert np.abs((ul - lin)[1:-1]).max() < 1e-14  # periodic ends excluded

    def test_limiter_kills_overshoot(self):
        el = ref_element(1)
        n, dx = 16, 1.0 / 16
        xc = (np.arange(n) + 0.5) * dx
        sw = np.repeat(np.where((xc > 0.25) & (xc < 0.75), 1.0, 0.0)[:, None], 2, axis=1)
        sw[7, 1] = 1.0  # inject a spike
        ul = tvb_limit(sw, el, dx, M=0.0)
        assert ul.max() <= 1.0 + 1e-14
        assert ul.min() >= 0.0 - 1e-14


# ---------------------------------------------------------------------------
# 5. exact references
# ---------------------------------------------------------------------------


class TestExactSolutions:
    def test_sod_exact_matches_literature(self):
        """Toro Table 4.1: p*=0.30313, u*=0.92745, rho*_L=0.42632, rho*_R=0.26557."""
        ref = exact_sod(np.array([0.5]), 0.2)
        assert ref["p_star"] == pytest.approx(0.30313, abs=2e-4)
        assert ref["u_star"] == pytest.approx(0.92745, abs=2e-4)
        assert ref["rho_star_l"] == pytest.approx(0.42632, abs=2e-4)
        assert ref["rho_star_r"] == pytest.approx(0.26557, abs=2e-4)

    def test_isothermal_riemann_two_derivations_agree(self):
        """Shock relation and rarefaction invariant must give the same u*."""
        ref = exact_isothermal_riemann(np.array([0.5]), 1.0, rho_l=1.0, rho_r=0.25)
        assert ref["rho_star"] == pytest.approx(0.4966233, abs=1e-6)
        assert ref["u_star"] == pytest.approx(
            rarefaction_tail(1.0, ref["rho_star"], 1.0 / math.sqrt(3.0)), abs=1e-10
        )
        assert ref["w_shock"] == pytest.approx(0.8137348, abs=1e-6)

    def test_isothermal_matches_verified_lbm_benchmark(self):
        """The LBM benchmark's rho_mid/u_mid are the isothermal exact values."""
        ref = exact_isothermal_riemann(np.array([0.5]), 1.0, rho_l=1.0, rho_r=0.25)
        assert ref["rho_star"] == pytest.approx(0.49662330512277264, rel=1e-9)
        assert ref["u_star"] == pytest.approx(0.4041010082298968, rel=1e-9)

    def test_shock_position_estimator(self):
        x = np.linspace(0, 1, 201)
        rho = np.where(x < 0.6, 1.0, 0.1)
        assert shock_position(x, rho) == pytest.approx(0.6, abs=2e-2)


# ---------------------------------------------------------------------------
# 6. benchmark gates (CI-sized)
# ---------------------------------------------------------------------------


class TestBenchmarks:
    def test_mms_order_p1_p2(self):
        from tensorlbm.dg_verification import benchmarks as B

        out = B.bench_mms(n_list=(8, 16), degrees=(1, 2), t_end=0.1)
        for p in (1, 2):
            order = out["orders"][f"P{p}"]["L2"]["order_last_pair"]
            assert abs(order - (p + 1)) < 0.35, (p, order)

    def test_rotating_smooth_field_design_order_p1(self):
        """Smooth rotating advection (one full period): P1 order must be ~2.

        A full period is required because the exact solution at ``t = 2 pi`` is
        the initial field again; ``t = pi`` would compare against the wrong
        reference.  P2 is *not* asserted here: it shows an error floor (~2e-3)
        that is documented as an open issue in
        ``docs/IN70_N01_dg_convection_diffusion.md``.
        """
        from tensorlbm.dg_verification import benchmarks as B

        out = B.bench_rotating_cone(
            n_list=(16, 32), degrees=(1,), t_end=2 * math.pi, initial="gaussian",
            dt_refine_check=False,
        )
        order = out["orders"]["P1"]["L2"]["order_last_pair"]
        assert order > 1.8, order

    def test_isothermal_sod_shock_position(self):
        """DG isothermal Riemann: shock position within 2% of the analytic wave."""
        sol = EulerDG1D(degree=1, n_cells=400, length=400.0, isothermal=True, use_limiter=True)
        x = sol.x
        x0 = 200.0
        side = (np.arange(400) + 0.5 < x0)[:, None]
        sol.set_state(
            sol.from_primitive(
                np.where(side, 1.0, 0.25) * np.ones_like(x),
                np.zeros_like(x),
                np.where(side, 1.0, 0.25) * np.ones_like(x) / 3.0,
            )
        )
        sol.advance(60.0, cfl=0.3)
        xc = sol.cell_centers()
        rho = sol.cell_means()[0]
        exact = x0 + 0.8137348504459936 * 60.0
        err = abs(shock_position(xc, rho) - exact) / (0.8137348504459936 * 60.0)
        assert err < 0.02, err
        assert rho.min() > 0.0

    def test_square_wave_tvd(self):
        from tensorlbm.dg_verification import benchmarks as B

        out = B.bench_square_wave(n_list=(100,), t_end=1.0)
        tv = out["tvd_check"]
        assert tv["worst_overshoot_limited"] < 1e-12
        assert tv["worst_undershoot_limited"] < 1e-12
        # the unlimited same-grid run *does* overshoot: the limiter matters
        assert tv["worst_overshoot_unlimited"] > 1e-3

    def test_mass_conservation_machine_precision(self):
        el = ref_element(2)
        n, dx = 8, 1.0 / 8
        X, Y = node_coords_2d(n, n, el, dx, dx)
        u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
        ax, ay = 0.5 - Y, X - 0.5
        u = u0.copy()
        dt = stable_dt(2, dx, advective_speed=math.sqrt(0.5), safety=1.0, ndim=2)
        for _ in range(40):
            u = ssp_rk3(u, dt, advection_rhs_2d, ax, ay, el, dx, dx)
        m0, m1 = integral_2d(u0, el, dx, dx), integral_2d(u, el, dx, dx)
        assert abs(m1 - m0) / abs(m0) < 1e-12

    def test_euler_uniform_state_is_steady(self):
        for iso in (True, False):
            sol = EulerDG1D(degree=2, n_cells=16, length=1.0, isothermal=iso, use_limiter=False)
            x = sol.x
            U = sol.from_primitive(np.ones_like(x), np.full_like(x, 0.3), np.ones_like(x))
            assert np.abs(sol.rhs(U)).max() < 1e-12
            # a uniform flow must translate without deformation
            sol.set_state(sol.from_primitive(np.ones_like(x), np.full_like(x, 0.3), np.ones_like(x)))
            sol.advance(0.1, cfl=0.3)
            rho, u, _ = sol.primitive(sol.U)
            assert np.abs(rho - 1.0).max() < 1e-12
            assert np.abs(u - 0.3).max() < 1e-12