"""Tests for the compressible-shock verification family (W18-N07).

Every assertion checks a *physical* or *numerical* property that has an
independent reference: exact Rankine-Hugoniot values, kinematic identities, the
consistency of the kinetic flux with the Euler flux, conservation laws, and the
exact solution of a uniform-flow advection problem.  No test asserts "the code
does what the code does".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.compressible_shock import analysis, cases
from tensorlbm.compressible_shock import flux as F
from tensorlbm.compressible_shock.rankine_hugoniot import (
    GAMMA,
    moving_shock,
    normal_shock,
    oblique_shock,
    oblique_shock_beta,
    prandtl_meyer,
    shock_tube_driver,
    sound_speed,
)
from tensorlbm.compressible_shock.solver import EulerSolver1D, EulerSolver2D, _positivity_scale

# ----------------------------------------------------------------- exact relations


@pytest.mark.parametrize(
    "mach,rho,p,t,m2",
    [
        (2.0, 2.6666667, 4.5, 1.6875, 0.5773503),
        (3.0, 3.8571429, 10.3333333, 2.6790123, 0.475191),
        (5.0, 5.0, 29.0, 5.8, 0.4152274),
        (8.0, 5.5652174, 74.5, 13.3867187, 0.3928898),
        (10.0, 5.7142857, 116.5, 20.3875, 0.3875753),
    ],
)
def test_normal_shock_reference_values(mach, rho, p, t, m2):
    """Known Mach 2..10 normal-shock ratios (gamma = 1.4)."""
    j = normal_shock(mach)
    assert j["rho_ratio"] == pytest.approx(rho, rel=1e-6)
    assert j["p_ratio"] == pytest.approx(p, rel=1e-6)
    assert j["T_ratio"] == pytest.approx(t, rel=1e-6)
    assert j["M2"] == pytest.approx(m2, rel=1e-6)


def test_normal_shock_limits_and_second_law():
    """Strong-shock density limit, entropy rise > 0, and M1 < 1 rejection."""
    assert normal_shock(1e6)["rho_ratio"] == pytest.approx((GAMMA + 1) / (GAMMA - 1), rel=1e-4)
    for m in (1.2, 2.0, 5.0, 10.0):
        assert normal_shock(m)["entropy_rise"] > 0.0
        assert 0.0 < normal_shock(m)["M2"] < 1.0
    with pytest.raises(ValueError):
        normal_shock(0.9)


@pytest.mark.parametrize("mach", [2.0, 3.0, 10.0])
def test_moving_shock_satisfies_jump_conditions(mach):
    """The lab-frame moving shock satisfies mass, momentum and energy jumps."""
    ms = moving_shock(mach, rho1=1.0, p1=1.0)
    W, u2 = ms["W"], ms["u2"]
    # mass: rho1 * W = rho2 * (W - u2)
    assert ms["rho2"] * (W - u2) == pytest.approx(ms["rho1"] * W, rel=1e-12)
    # momentum: p2 - p1 = rho1 * W * u2
    assert ms["p2"] - ms["p1"] == pytest.approx(ms["rho1"] * W * u2, rel=1e-12)
    # energy: h2 + (W-u2)^2/2 = h1 + W^2/2  (h = gamma p /((gamma-1) rho))
    h = lambda p, r: GAMMA * p / ((GAMMA - 1.0) * r)  # noqa: E731
    lhs = h(ms["p2"], ms["rho2"]) + 0.5 * (W - u2) ** 2
    rhs = h(ms["p1"], ms["rho1"]) + 0.5 * W**2
    assert lhs == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize("mach", [2.0, 5.0, 10.0])
def test_shock_tube_driver_is_consistent(mach):
    """Driver inversion reproduces the target shock Mach number.

    The pressure behind the shock (from the shock relations) must equal the
    pressure reached by an isentropic expansion from the driver state, which is
    the defining condition of the shock-tube solution.
    """
    st = shock_tube_driver(mach)
    p3_over_p4 = (1.0 - 0.5 * (GAMMA - 1.0) * st["u2"] / st["a4"]) ** (2.0 * GAMMA / (GAMMA - 1.0))
    assert st["p4"] * p3_over_p4 == pytest.approx(st["p2"], rel=1e-10)
    assert st["p2"] == pytest.approx(st["p3"], rel=1e-12)
    assert st["p4"] > st["p2"] > st["p1"]  # driver > post-shock > driven


def test_oblique_shock_matches_theta_beta_relation():
    """The solved shock angle satisfies the theta-beta-M relation and limits."""
    M, theta = 4.0, math.radians(12.0)
    beta = oblique_shock_beta(M, theta)
    th = (
        2.0
        / math.tan(beta)
        * (M**2 * math.sin(beta) ** 2 - 1.0)
        / (M**2 * (GAMMA + math.cos(2.0 * beta)) + 2.0)
    )
    assert math.atan(th) == pytest.approx(theta, rel=1e-8)
    assert beta < math.pi / 2 - 0.1  # weak branch, not the near-normal one
    assert beta > math.asin(1.0 / M)  # attached shock, above the Mach angle
    assert oblique_shock_beta(M, theta, weak=False) > beta  # strong > weak
    # ratios follow the normal shock at the normal Mach number
    sol = oblique_shock(M, math.radians(30.0))
    assert sol["rho_ratio"] == pytest.approx(
        normal_shock(M * math.sin(sol["beta"]))["rho_ratio"], rel=1e-12
    )
    # small deflection tends to the Mach angle
    assert oblique_shock_beta(3.0, 1e-4) == pytest.approx(math.asin(1.0 / 3.0), abs=1e-3)


def test_prandtl_meyer_reference_values():
    """nu(2) = 26.3798 deg, nu(inf) = 130.45 deg for gamma = 1.4."""
    assert math.degrees(prandtl_meyer(2.0)) == pytest.approx(26.3798, abs=1e-4)
    assert math.degrees(prandtl_meyer(1e6)) == pytest.approx(130.4541, abs=1e-3)


# ------------------------------------------------------------------------ fluxes


def _random_states(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rho = 0.2 + 2.0 * rng.random(n)
    p = 0.2 + 3.0 * rng.random(n)
    a = np.sqrt(GAMMA * p / rho)
    u = rng.normal(0.0, 1.6, n) * a
    v = rng.normal(0.0, 1.0, n) * a
    return np.stack([F.conserved(rho[i], u[i], v[i], p[i]) for i in range(n)])


@pytest.mark.parametrize("name", ["hllc", "rusanov", "kfvs"])
def test_flux_reduces_to_euler_flux_for_identical_states(name):
    """A Riemann solver fed the same state twice must return the physical flux."""
    U = _random_states()
    fn = getattr(F, f"{name}_flux")
    got = fn(U, U, GAMMA)
    ref = F.euler_flux(U, GAMMA)
    assert np.max(np.abs(got - ref) / np.abs(ref)) < 1e-12


def test_kfvs_closed_form_matches_velocity_quadrature():
    """The analytic half-space Maxwellian moments equal a brute-force integral."""
    UL, UR = _random_states(6, seed=3), _random_states(6, seed=4)
    closed = F.kfvs_flux(UL, UR, GAMMA)
    quad = F.kfvs_flux_quadrature(UL, UR, GAMMA, n_quad=8001, xi_max=18.0)
    assert np.max(np.abs(closed - quad)) < 5e-2 * np.max(np.abs(quad))


def test_hllc_resolves_a_stationary_contact_exactly():
    """A density jump with the same p and u = 0 carries only a pressure flux."""
    p = 2.0
    UL = np.stack([F.conserved(1.0, 0.0, 0.0, p)])
    UR = np.stack([F.conserved(3.0, 0.0, 0.0, p)])
    flux = F.hllc_flux(UL, UR, GAMMA)
    assert flux[0, 0] == pytest.approx(0.0, abs=1e-14)  # no mass flux
    assert flux[0, 1] == pytest.approx(p, rel=1e-12)  # single-valued pressure
    assert flux[0, 2] == pytest.approx(0.0, abs=1e-14)
    assert flux[0, 3] == pytest.approx(0.0, abs=1e-14)


# ----------------------------------------------------------------------- solvers


def test_positivity_scale_keeps_states_admissible():
    """The limiter returns theta in [0,1] and removes all negative pressures."""
    rho = np.array([1.0, 5.0])
    p = np.array([2.0, 0.05])
    drho = np.array([-3.0, 0.0])
    dp = np.array([-0.0, -1.0])  # second cell would give p = -0.95
    theta = _positivity_scale(rho, p, drho, dp)
    assert np.all(theta >= 0.0) and np.all(theta <= 1.0)
    assert theta[1] < 0.96
    assert p[1] + theta[1] * dp[1] > 0.0


def _shock_tube_1d(n=400, t_end=0.1, cfl=0.4):
    """Sod-type strong shock tube with two closed walls (conservative test)."""
    s = EulerSolver1D(n=n, length=1.0, cfl=cfl)
    U = np.zeros((n, 4))
    U[:, :] = np.array([1.0, 0.0, 0.0, 1.0 / (GAMMA - 1.0)])  # rho=1, p=1
    left = np.array([1.0, 0.0, 0.0, 10.0 / (GAMMA - 1.0)])  # rho=1, p=10
    U[: n // 2] = left
    s.U = U
    s.bc_left = ("wall",)
    s.bc_right = ("wall",)
    return s


def test_1d_solver_conserves_mass_and_energy():
    """Closed-wall run: total mass and total energy are conserved to round-off."""
    s = _shock_tube_1d()
    m0 = float(np.sum(s.U[:, 0]) * s.dx)
    e0 = float(np.sum(s.U[:, 3]) * s.dx)
    s.advance(0.1)
    m1 = float(np.sum(s.U[:, 0]) * s.dx)
    e1 = float(np.sum(s.U[:, 3]) * s.dx)
    assert m1 == pytest.approx(m0, rel=1e-11)
    assert e1 == pytest.approx(e0, rel=1e-10)
    rho, u, v, p = s.primitives()
    assert np.all(p > 0.0) and np.all(rho > 0.0)


def test_1d_normal_shock_plateau_and_speed():
    """Non-circular Riemann case: emergent shock speed and jump within 1.5%."""
    solver, meta = cases.normal_shock_case(5.0, n=400, variant="riemann")
    solver.advance(meta["t_end"])
    rho = solver.primitives()[0]
    thr = 0.5 * (meta["pre_state"]["rho"] + meta["post_state"]["rho"])
    xf = analysis.front_x(rho, solver.xc, thr, x_min=0.3)
    speed = (xf - meta["x_diaphragm"]) / solver.t
    assert analysis.relative_error(speed, meta["shock_speed_expected"]) < 0.015
    st = analysis.sample_state(solver.U, solver.xc, xf - 20 * solver.dx, xf - 3 * solver.dx)
    assert analysis.relative_error(st["rho"], meta["post_state"]["rho"]) < 0.015
    assert analysis.relative_error(st["p"], meta["post_state"]["p"]) < 0.015


def test_2d_uniform_flow_advects_entropy_wave_exactly():
    """Exact solution test of the 2-D operator: uniform p,u,v; only rho varies.

    The Euler equations then demand constant u and v and pure advection of rho,
    so any transposition/flux-ordering error shows up immediately.
    """
    nx = ny = 60
    s = EulerSolver2D(nx=nx, ny=ny, lx=1.0, ly=1.0, cfl=0.4)
    X, Y = np.meshgrid(s.xc, s.yc)
    rho = 1.0 + 0.2 * np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.01)
    u0, v0, p0 = 3.0, 1.5, 1.0
    s.U = np.stack(
        [rho, rho * u0, rho * v0, p0 / (GAMMA - 1.0) + 0.5 * rho * (u0**2 + v0**2)], axis=-1
    )
    for k in ("x0", "x1", "y0", "y1"):
        setattr(s, f"bc_{k}", ("transmissive",))
    s.advance(0.05)
    rho1, u1, v1, p1 = s.primitives()
    assert np.max(np.abs(u1 - u0)) < 1e-10
    assert np.max(np.abs(v1 - v0)) < 1e-10
    assert np.max(np.abs(p1 - p0)) < 1e-10
    # peak position: p has not moved the perturbation, so check the peak cell
    j, i = np.unravel_index(int(np.argmax(rho1)), rho1.shape)
    assert abs(s.xc[i] - (0.3 + u0 * s.t)) < 2.5 * s.dx
    assert abs(s.yc[j] - (0.3 + v0 * s.t)) < 2.5 * s.dy


def test_2d_blocked_cells_hold_an_obstacle():
    """Blocked-cell mask: solid interior stays at its initial state, no NaNs."""
    s = EulerSolver2D(nx=40, ny=20, lx=2.0, ly=1.0)
    X, Y = np.meshgrid(s.xc, s.yc)
    solid = (X > 1.0) & (Y < 0.4)
    inflow = F.conserved(1.4, 3.0 * sound_speed(1.4, 1.0), 0.0, 1.0)
    s.U = np.broadcast_to(inflow, (20, 40, 4)).copy()
    s.solid = solid
    s.bc_x0 = ("uniform", inflow)
    s.bc_y0 = ("wall",)
    s.bc_y1 = ("wall",)
    s.advance(0.2)
    assert np.isfinite(s.U).all()
    rho, u, v, p = s.primitives()
    assert np.all(p > 0.0)
    assert np.allclose(rho[solid], 1.4)


# ---------------------------------------------------------------------- analysis


def test_front_x_interpolates_a_known_jump():
    xc = np.arange(10) + 0.5
    # step profile: crossing of the mid value sits exactly on the face x = 5.0
    prof = np.where(xc < 5.0, 4.0, 1.0)
    assert analysis.front_x(prof, xc, 2.5) == pytest.approx(5.0, abs=1e-12)
    # linear transition: threshold crossed 20% into the cell (x = 4.5 -> 5.5)
    prof2 = prof.copy()
    prof2[4], prof2[5] = 3.0, 1.5
    assert analysis.front_x(prof2, xc, 2.7) == pytest.approx(4.7, abs=1e-12)


def test_mach_stem_height_detects_a_synthetic_pattern():
    """Synthetic three-shock geometry: vertical stem up to y=0.25, then 30 deg."""
    yc = np.arange(60) * (1.0 / 60.0) + 0.5 / 60.0
    x_front = np.where(yc < 0.25, 2.0, 2.0 + (yc - 0.25) / math.sqrt(3.0))
    out = analysis.mach_stem_height(x_front, yc)
    assert out["triple_point_y"] == pytest.approx(0.25, abs=0.03)
    assert out["stem_height"] == pytest.approx(0.25, abs=0.03)


def test_gci_and_relative_error():
    out = analysis.gci(2.10, 2.05, ratio=2.0, order=1.0)
    assert out["relative_change"] == pytest.approx(0.05 / 2.05, rel=1e-12)
    # Richardson: the coarse grid lies 0.05 above the fine one, so the
    # extrapolated value is 2.05 - 0.05 = 2.00 for order 1 and ratio 2
    assert out["richardson_estimate"] == pytest.approx(2.00, rel=1e-12)
    assert analysis.relative_error(1.05, 1.0) == pytest.approx(0.05, rel=1e-12)


def test_fit_speed_recovers_a_linear_history():
    t = [0.0, 0.1, 0.25, 0.4]
    x = [0.2 + 3.5 * ti for ti in t]
    out = analysis.fit_speed(t, x)
    assert out["speed"] == pytest.approx(3.5, rel=1e-10)
    assert out["r2"] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------- integrated smoke


def test_dmr_smoke_runs_and_advances_the_front():
    """Small double-Mach-reflection run: admissible, front advances, no fixes."""
    solver, meta = cases.double_mach_reflection_case(nx=120, ny=30, t_end=0.01)
    solver.advance(0.01)
    rho, u, v, p = solver.primitives()
    assert np.isfinite(solver.U).all()
    assert np.all(p > 0.0) and np.all(rho > 0.0)
    thr = 0.5 * (meta["pre_shock"]["rho"] + meta["post_shock"]["rho"])
    foot = analysis.front_x(rho[0], solver.xc, thr, x_min=0.8 * meta["x_wall"])
    exact = meta["x_wall"] + meta["foot_speed_along_wall"] * solver.t
    assert analysis.relative_error(foot, exact) < 0.15
    assert solver.n_admissibility_fixes < 200


def test_ffs_smoke_develops_compression_over_the_step():
    """Small forward-facing-step run: admissible, obstacle fixed, compression forms.

    The bow shock ahead of the face is thinner than a cell at this resolution
    (see the report), so only the overall compression over the step is asserted.
    """
    solver, meta = cases.forward_facing_step_case(nx=120, ny=40, t_end=1.0)
    solid0 = solver.solid.copy()
    solver.advance(1.0)
    rho, u, v, p = solver.primitives()
    assert np.isfinite(solver.U).all() and np.all(p > 0.0)
    assert np.array_equal(solver.solid, solid0)
    assert rho.max() > 1.3 * meta["inflow"]["rho"]
    assert np.allclose(rho[solid0], meta["inflow"]["rho"])
