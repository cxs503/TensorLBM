"""Tests for the Brinkman (Darcy-drag) channel closure and its exact reference.

The reference itself is re-derived in ``verify_exact_solution`` (ODE residual,
boundary conditions, Darcy/Stokes limits), so these tests fail if the closed
form and the solver drift apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.brinkman_channel import (
    BrinkmanChannelConfig,
    CX,
    CY,
    _stream,
    exact_brinkman_profile,
    exact_centreline_velocity,
    exact_mean_velocity,
    exact_shape_function,
    exact_wall_shear,
    guo_source_2d,
    run_brinkman_channel,
    sample_exact,
    verify_exact_solution,
)


def test_reference_residuals_are_small():
    chk = verify_exact_solution(accel=1e-5, nu=0.1, permeability=40.96, half_width=32.0)
    assert chk["ode_residual_rel"] < 1e-5
    assert chk["bc_left_rel"] < 1e-12 and chk["bc_right_rel"] < 1e-12
    assert chk["mean_closed_vs_quad_rel"] < 1e-6
    assert chk["stokes_limit_rel"] < 1e-5
    assert chk["darcy_limit_rel"] < 1e-3


def test_shape_function_limits():
    h = 10.0
    # Stokes limit beta -> 0: u_bar = g h^2 / (3 nu)  =>  R ~ beta^2/3
    k_big = 1e8 * h * h
    beta = h / math.sqrt(k_big)
    assert exact_shape_function(k_big, h) == pytest.approx(beta**2 / 3.0, rel=1e-6)
    # Darcy limit beta -> infinity: u_bar -> g K / nu  =>  R -> 1
    assert exact_shape_function(1e-8 * h * h, h) > 0.99


def test_profile_satisfies_ode_and_bcs():
    g, nu, K, h = 1e-5, 0.1, 40.96, 32.0
    y, u = exact_brinkman_profile(g, nu, K, h, n=4001)
    dy = float(y[1] - y[0])
    u_yy = (u[2:] - 2 * u[1:-1] + u[:-2]) / dy**2
    assert np.abs(nu * u_yy - nu / K * u[1:-1] + g).max() < 1e-6 * g
    assert abs(float(u[0])) < 1e-15 and abs(float(u[-1])) < 1e-15
    # monotone, symmetric, centreline closed form
    assert np.all(np.diff(u[len(u) // 2:]) < 0)
    assert u[len(u) // 2] == pytest.approx(
        exact_centreline_velocity(g, nu, K, h), rel=1e-3)
    assert exact_mean_velocity(g, nu, K, h) == pytest.approx(
        float(np.trapezoid(u, y)) / (2 * h), rel=1e-4)
    # wall shear is the derivative of the closed form at y = h
    slope = (u[-1] - u[-2]) / (y[-1] - y[-2])
    assert abs(slope) == pytest.approx(
        exact_wall_shear(g, K, h) / nu, rel=5e-2)


def test_sample_exact_matches_profile_array():
    g, nu, K, h = 1e-5, 0.1, 40.96, 32.0
    y, u = exact_brinkman_profile(g, nu, K, h, n=101)
    for yi, ui in zip(y[::10], u[::10]):
        assert sample_exact(g, nu, K, h, float(yi)) == pytest.approx(float(ui), rel=1e-12)


def test_streaming_gather_equals_nested_rolls():
    rng = np.random.default_rng(0)
    f = rng.random((9, 13, 5))
    ref = np.empty_like(f)
    for q in range(9):
        ref[q] = np.roll(np.roll(f[q], int(CY[q]), axis=0), int(CX[q]), axis=1)
    assert np.array_equal(_stream(f), ref)


def test_guo_source_is_zero_sum():
    rng = np.random.default_rng(1)
    rho = np.ones((8, 3))
    ux, uy = rng.normal(scale=0.01, size=(8, 3)), rng.normal(scale=0.01, size=(8, 3))
    ax, ay = rng.normal(scale=1e-5, size=(8, 3)), rng.normal(scale=1e-5, size=(8, 3))
    S = guo_source_2d(rho, ux, uy, ax, ay, 0.8)
    assert np.abs(S.sum(axis=0)).max() < 1e-18
    # the truncation that drops the 9 (c.u)(c.a) partner must NOT be zero-sum
    from tensorlbm.brinkman_channel import CX as CXa, CY as CYa, W as Wa

    a_dot_c = CXa[:, None, None] * ax[None] + CYa[:, None, None] * ay[None]
    u_dot_a = ux * ax + uy * ay
    S_bad = (1 - 0.5 / 0.8) * Wa[:, None, None] * 3.0 * (a_dot_c - u_dot_a[None])
    assert np.abs(S_bad.sum(axis=0)).max() > 1e-6 * np.abs(ax).max()


def test_config_validation():
    with pytest.raises(ValueError):
        BrinkmanChannelConfig(tau=0.5)
    with pytest.raises(ValueError):
        BrinkmanChannelConfig(permeability=0.0)
    with pytest.raises(ValueError):
        BrinkmanChannelConfig(ny=4)
    cfg = BrinkmanChannelConfig(ny=34, tau=0.8)
    assert cfg.height == 32.0 and cfg.half_width == 16.0
    assert cfg.nu == pytest.approx((0.8 - 0.5) / 3.0)


def test_drag_off_reproduces_poiseuille():
    cfg = BrinkmanChannelConfig(ny=34, nx=2, permeability=1e12, accel=1e-5,
                                n_steps=40000, check_every=500, steady_tol=1e-11)
    r = run_brinkman_channel(cfg, drag=False)
    assert r["reached_steady"]
    assert r["eps_inf"] < 0.01
    assert abs(r["amp_err"]) < 0.01
    assert abs(r["mass_drift"]) < 1e-10


def test_brinkman_channel_green_and_steady():
    H = 16.0
    cfg = BrinkmanChannelConfig(ny=int(H + 2), nx=2, permeability=1e-2 * H * H,
                                accel=1e-5, n_steps=40000, check_every=500,
                                steady_tol=1e-11)
    r = run_brinkman_channel(cfg)
    assert r["reached_steady"]
    assert r["eps_inf"] < 0.03
    assert abs(r["amp_err"]) < 0.01
    assert abs(r["mass_drift"]) < 1e-10
    assert r["x_nonuniformity"] < 1e-10           # profile is x-independent
    u = np.array(r["u_num"])
    assert np.allclose(u, u[::-1], rtol=1e-8, atol=1e-14)  # symmetry
    assert 0.0 < float(u.max()) < 0.05            # incompressible regime


def test_error_decreases_under_refinement():
    eps = []
    for H in (16.0, 32.0):
        cfg = BrinkmanChannelConfig(ny=int(H + 2), nx=2, permeability=1e-2 * H * H,
                                    accel=1e-5, n_steps=60000, check_every=500,
                                    steady_tol=1e-11)
        eps.append(run_brinkman_channel(cfg)["eps_inf"])
    assert eps[1] < eps[0]


def test_shape_function_error_small():
    H = 32.0
    cfg = BrinkmanChannelConfig(ny=int(H + 2), nx=2, permeability=1e-2 * H * H,
                                accel=1e-5, n_steps=60000, check_every=500,
                                steady_tol=1e-11)
    r = run_brinkman_channel(cfg)
    assert abs(r["shape_num"] / r["shape_ex"] - 1.0) < 0.01
