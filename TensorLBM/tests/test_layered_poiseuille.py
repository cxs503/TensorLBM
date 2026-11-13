"""Verification tests for tensorlbm.verification.layered_poiseuille.

Covers (a) the exact reference against closed forms and an independent
finite-difference solve, (b) the TCLB-port boundary-condition algebra, (c) the
TCLB-port integral metrics, and (d) two short end-to-end solver runs.
"""
import numpy as np
import pytest

from tensorlbm.verification.layered_poiseuille import analytic as A
from tensorlbm.verification.layered_poiseuille.metrics import (
    layer_fluxes, profile_errors, track_interface_y, update_my_globals)
from tensorlbm.verification.layered_poiseuille.solver import (
    ChannelConfig, LayeredChannel, equilibrium)


# --------------------------------------------------------------------------- #
# 1. exact reference
# --------------------------------------------------------------------------- #
def test_tclb_parabolic_profile_mean_and_peak():
    """TCLB developedFlow: U = 6 Uavg Y(H-Y)/H^2 -> mean Uavg, peak 1.5 Uavg."""
    y = np.linspace(0.0, 1.0, 4001)
    u = A.parabolic_profile(y, 1.0, 0.03)
    assert np.trapezoid(u, y) == pytest.approx(0.03, rel=1e-6)
    assert u.max() == pytest.approx(1.5 * 0.03, rel=1e-6)


def test_tclb_pipe_profile_mean_and_peak():
    """TCLB developedPipeFlow_X: U = 2 Uavg (1-(r/R)^2) -> mean Uavg, peak 2 Uavg."""
    r = np.linspace(0.0, 1.0, 4001)
    u = A.pipe_profile(r, 1.0, 0.03)
    assert 2.0 * np.trapezoid(u * r, r) == pytest.approx(0.03, rel=1e-6)
    assert u.max() == pytest.approx(2 * 0.03, rel=1e-9)


def test_single_layer_reduces_to_poiseuille():
    H, nu, g = 64.0, 0.1, 1e-5
    sol = A.solve_layered(H, (), (nu,), g)
    y = A.wall_normal_coordinates(66)
    exact = g / (2 * nu) * y * (H - y)
    assert np.allclose(sol.u(y), exact, rtol=0, atol=1e-14)
    assert sol.u_peak() == pytest.approx(g * H * H / (8 * nu), rel=1e-5)
    assert sol.y_peak() == pytest.approx(H / 2, abs=1e-3)


def test_two_layer_velocity_and_stress_continuity():
    H, nu1, nu2, g = 64.0, 0.15, 0.05, 1e-5
    sol = A.solve_layered(H, (32.0,), (nu1, nu2), g)
    eps = 1e-9
    assert sol.u(np.array([32 - eps]))[0] == pytest.approx(sol.u(np.array([32 + eps]))[0], abs=1e-10)
    assert sol.shear_stress(np.array([32 - eps]))[0] == pytest.approx(
        sol.shear_stress(np.array([32 + eps]))[0], rel=1e-12)
    # global momentum balance: g*H = tau_bottom - tau_top
    tau_b = float(sol.shear_stress(np.array([0.0]))[0])
    tau_t = float(sol.shear_stress(np.array([H]))[0])
    assert (tau_b - tau_t) == pytest.approx(g * H, rel=1e-12)


def test_two_layer_matches_independent_finite_difference():
    """Independent FD solve of d/dy(nu du/dy) = -g as a second reference."""
    H, nu1, nu2, g = 64.0, 0.15, 0.05, 1e-5
    sol = A.solve_layered(H, (32.0,), (nu1, nu2), g)
    N = 8001
    y = np.linspace(0.0, H, N)
    dy = y[1] - y[0]
    nu = np.where(y < 32.0, nu1, nu2)
    nuh = 0.5 * (nu[1:] + nu[:-1])
    M = np.zeros((N, N))
    b = np.full(N, -g)
    for j in range(1, N - 1):
        M[j, j - 1] = nuh[j - 1] / dy ** 2
        M[j, j] = -(nuh[j - 1] + nuh[j]) / dy ** 2
        M[j, j + 1] = nuh[j] / dy ** 2
    M[0, 0] = 1.0
    b[0] = 0.0
    M[-1, -1] = 1.0
    b[-1] = 0.0
    u_fd = np.linalg.solve(M, b)
    u_ex = sol.u(y)
    assert np.max(np.abs(u_fd - u_ex)) / np.max(np.abs(u_ex)) < 1e-4


def test_peak_moves_into_the_less_viscous_layer():
    base = A.solve_layered(64.0, (32.0,), (0.1, 0.1), 1e-5)
    layered = A.solve_layered(64.0, (32.0,), (0.15, 0.05), 1e-5)
    assert base.y_peak() == pytest.approx(32.0, abs=1e-3)
    assert layered.y_peak() > 32.0


# --------------------------------------------------------------------------- #
# 2. geometry
# --------------------------------------------------------------------------- #
def test_half_way_bounce_back_geometry():
    """bb places the wall 0.5 dx inside the solid row -> H = ny-2, wall at 0.5."""
    ny = 34
    y = A.wall_normal_coordinates(ny)
    assert y[0] == pytest.approx(0.5)
    assert y[-1] + 0.5 == pytest.approx(ny - 2)


def test_fit_effective_height_recovers_channel_height():
    ny, nu, g = 66, 0.1, 1e-5
    y = A.wall_normal_coordinates(ny)
    H = ny - 2
    u = g / (2 * nu) * y * (H - y)
    assert A.fit_effective_height(y, u, g, nu) == pytest.approx(H, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. TCLB boundary-condition port (window of the solver, tested standalone)
# --------------------------------------------------------------------------- #
EAST, WEST = (1, 5, 7), (2, 6, 8)          # this lattice: c5=(1,1) c6=(-1,1) c7=(1,-1) c8=(-1,-1)


@pytest.mark.parametrize("rho,u,v", [(1.0, 0.02, 0.0), (0.98, -0.03, 0.0), (1.02, 0.05, 0.0)])
def test_zou_he_inlet_algebra_is_exact(rho, u, v):
    c = np.array([equilibrium(np.full((1, 1), rho), np.full((1, 1), u),
                              np.full((1, 1), v))[i, 0, 0] for i in range(9)])
    r = (c[0] + c[3] + c[4] + 2 * (c[2] + c[6] + c[8])) / (1 - u)
    f1 = c[2] + (2 / 3) * r * u
    s = r * u + (c[2] + c[6] + c[8]) - f1
    d = r * v - (c[3] - c[4]) - (c[6] - c[8])
    assert r == pytest.approx(rho, abs=1e-14)
    assert f1 == pytest.approx(c[1], abs=1e-14)
    assert 0.5 * (s + d) == pytest.approx(c[5], abs=1e-14)
    assert 0.5 * (s - d) == pytest.approx(c[7], abs=1e-14)


@pytest.mark.parametrize("rho,u", [(1.0, 0.02), (1.02, 0.05)])
def test_zou_he_pressure_outlet_algebra_is_exact(rho, u):
    c = np.array([equilibrium(np.full((1, 1), rho), np.full((1, 1), u),
                              np.zeros((1, 1)))[i, 0, 0] for i in range(9)])
    ux = (c[0] + c[3] + c[4] + 2 * (c[1] + c[5] + c[7])) / rho - 1
    f2 = c[1] - (2 / 3) * rho * ux
    s = (c[1] + c[5] + c[7]) - rho * ux - f2
    d = -(c[3] - c[4]) - (c[5] - c[7])
    assert ux == pytest.approx(u, abs=1e-14)
    assert f2 == pytest.approx(c[2], abs=1e-14)
    assert 0.5 * (s + d) == pytest.approx(c[6], abs=1e-14)
    assert 0.5 * (s - d) == pytest.approx(c[8], abs=1e-14)


# --------------------------------------------------------------------------- #
# 4. TCLB integral metrics port
# --------------------------------------------------------------------------- #
def test_globals_port_matches_tclb_definitions():
    ux = np.full((4, 3), 0.1)
    uy = np.zeros((4, 3))
    pf = np.zeros((4, 3))
    pf[2:, :] = 1.0                       # top half "liquid"
    g = update_my_globals(ux, uy, pf)
    assert g.total_density == pytest.approx(ux.size * 1.0)
    assert g.liq_total_phase == pytest.approx(6.0)
    assert g.gas_total_phase == pytest.approx(6.0)
    assert g.liq_total_velocity_x == pytest.approx(6 * 0.1)
    assert g.gas_total_velocity_x == pytest.approx(6 * 0.1)
    # TCLB adds rho*|u|^2 (no 1/2 factor): 12 * 0.01
    assert g.kinetic_energy == pytest.approx(12 * 0.01)


def test_density_interpolation_matches_tclb():
    pf = np.array([[0.0, 1.0]])
    g = update_my_globals(np.zeros((1, 2)), np.zeros((1, 2)), pf,
                          density_l=1.0, density_h=2.0)
    assert g.total_density == pytest.approx(3.0)


def test_interface_tracking_linear_interpolation():
    # TCLB convention: the crossing is detected where C < 0.5 and the node below
    # has PhaseF > 0.5 (liquid below the interface)
    pf = np.array([1.0, 0.6, 0.4, 0.0])
    y = np.array([0.5, 1.5, 2.5, 3.5])
    loc = track_interface_y(pf, y)
    assert loc == pytest.approx(2.0)   # TCLB: Y - (C-0.5)/(C-PhaseF_below)


def test_layer_fluxes_same_functional_on_both_sides():
    sol = A.solve_layered(32.0, (16.0,), (0.15, 0.05), 1e-5)
    y = A.wall_normal_coordinates(34)
    u = sol.u(y)
    q = layer_fluxes(u, y, [16.0])
    assert sum(q) == pytest.approx(sol.flow_rate(), rel=2e-3)


def test_profile_errors_zero_for_identical_profiles():
    a = np.linspace(0, 1, 11)
    e = profile_errors(a, a)
    assert e["eps_inf"] == 0.0 and e["eps_l2_rel"] == 0.0


# --------------------------------------------------------------------------- #
# 5. end-to-end solver (short, still below the owner band)
# --------------------------------------------------------------------------- #
def _run(H, nus, fracs, steps=4000, tol=1e-12):
    cfg = ChannelConfig(height=H, nx=8, nus=nus, interface_fracs=fracs, u_peak=0.05,
                        x_boundary="periodic", init_mode="analytic", steps=steps,
                        sample_every=200, tol_steady=tol)
    unit = A.solve_layered(float(H), [f * H for f in fracs], nus, 1.0)
    ch = LayeredChannel(cfg, unit_profile=lambda y: unit.u(y))
    ch.run()
    ref = A.solve_layered(float(H), [f * H for f in fracs], nus, ch.g)
    return ch, ref, ch.profile(), ref.u(ch.y_wall)


def test_solver_single_phase_under_one_percent():
    ch, ref, u, ue = _run(16, (0.1,), ())
    assert profile_errors(u, ue)["eps_inf"] < 0.01


def test_solver_layered_under_three_percent():
    ch, ref, u, ue = _run(16, (0.15, 0.05), (0.5,))
    assert profile_errors(u, ue)["eps_inf"] < 0.03


def test_solver_geometry_check():
    ch, ref, u, ue = _run(16, (0.1,), ())
    h = A.fit_effective_height(ch.y_wall, u, ch.g, 0.1)
    assert h == pytest.approx(16.0, abs=0.05)
