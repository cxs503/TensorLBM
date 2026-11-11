"""Regression tests for node p338w3-n05 (flexural-gravity wave dispersion).

Covers: closed-form limits and derivatives, the two independent solution paths of
the discrete tank (they must agree to round-off), the numerical-vs-analytic
dispersion for representative cells, and the four negative controls.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.wave_ice_dispersion import analytic as A
from tensorlbm.wave_ice_dispersion.campaign import (
    control_rigid_limit,
    control_shallow_limit,
    control_zero_thickness,
    estimate_omega,
    numerical_mode,
    solver_cross_check,
)
from tensorlbm.wave_ice_dispersion.circulant import CirculantTank
from tensorlbm.wave_ice_dispersion.tank import FlexuralWaveTank, TankConfig

DEPTH = 20.0


# --------------------------------------------------------------- closed form
def test_rigidity_scales_with_cube_of_thickness():
    D1 = A.flexural_rigidity(A.E_ICE, 1.0)
    D2 = A.flexural_rigidity(A.E_ICE, 2.0)
    assert D2 / D1 == pytest.approx(8.0, rel=1e-12)
    # explicit value for the registry constants: E = 5 GPa, nu = 0.3, h_i = 1 m
    assert D1 == pytest.approx(5.0e9 / (12.0 * (1.0 - 0.09)), rel=1e-12)


def test_zero_rigidity_reduces_to_free_surface_dispersion():
    k = np.logspace(-3, 1, 40)
    om = A.dispersion_omega(k, DEPTH, 0.0)
    om_fs = A.free_surface_omega(k, DEPTH)
    assert np.allclose(om, om_fs, rtol=1e-14)


def test_shallow_water_limit_is_sqrt_gh():
    c = A.shallow_water_speed(DEPTH)
    assert c == pytest.approx(np.sqrt(A.G * DEPTH), rel=1e-14)
    cp = A.phase_speed(1e-9, DEPTH, A.flexural_rigidity(A.E_ICE, 1.0))
    assert cp == pytest.approx(c, rel=1e-8)


def test_deep_water_limit_and_plate_dominance():
    k = 1.0
    D = A.flexural_rigidity(A.E_ICE, 1.0)
    om = A.dispersion_omega(k, 1e6, D)          # tanh(k h) -> 1
    assert om == pytest.approx(np.sqrt(A.G * k + D * k**5 / A.RHO_W), rel=1e-12)
    # pure plate branch: omega ~ sqrt(D/rho) k^{5/2} -> c_g = (5/2) c_p
    kk = 3.0
    om_p = np.sqrt(D * kk**5 / A.RHO_W)
    cp = om_p / kk
    cg = A.group_speed(kk, 1e6, D)
    assert cg == pytest.approx(2.5 * cp, rel=2e-3)


def test_group_speed_matches_numerical_derivative():
    D = A.flexural_rigidity(A.E_ICE, 0.3)
    k = np.linspace(0.05, 0.8, 4001)
    om = A.dispersion_omega(k, DEPTH, D)
    cg_fd = np.gradient(om, k)
    cg = A.group_speed(k, DEPTH, D)
    assert np.allclose(cg[5:-5], cg_fd[5:-5], rtol=1e-5, atol=1e-6)


def test_rigid_limit_diverges_as_thickness_power_three_halves():
    ctrl = control_rigid_limit(0.5)
    assert ctrl["loglog_slope"] == pytest.approx(1.5, abs=0.02)
    assert ctrl["monotone_increasing"]
    # mass loading still diverges; asymptotically with exponent 1 instead of 3/2
    hs = np.array([16.0, 32.0, 64.0, 128.0, 256.0])
    om_ml = np.array([float(A.dispersion_omega(0.5, DEPTH, A.flexural_rigidity(A.E_ICE, h),
                                               mass_loading_param=A.mass_loading(h)))
                      for h in hs])
    slope_ml = np.polyfit(np.log(hs), np.log(om_ml), 1)[0]
    assert slope_ml == pytest.approx(1.0, abs=0.03)
    assert slope_ml < 1.5                       # massless exponent is 3/2
    assert all(b > a for a, b in zip(om_ml, om_ml[1:]))


def test_mass_loading_reduces_to_massless_when_zero():
    D = A.flexural_rigidity(A.E_ICE, 0.5)
    k = np.array([0.1, 0.3, 0.6])
    a = A.dispersion_omega(k, DEPTH, D, mass_loading_param=0.0)
    b = A.dispersion_omega(k, DEPTH, D)
    assert np.allclose(a, b, rtol=1e-15)
    c = A.dispersion_omega(k, DEPTH, D, mass_loading_param=A.mass_loading(0.5))
    assert np.all(c < b)                        # inertia slows the waves


def test_critical_phase_speed_exists_and_shifts_with_mass_loading():
    D = A.flexural_rigidity(A.E_ICE, 1.0)
    c0 = A.critical_phase_speed(D)
    cm = A.critical_phase_speed(D, mass_loading_param=A.mass_loading(1.0))
    assert 0 < cm < c0                      # inertia lowers the wave speed
    cp = A.phase_speed(np.logspace(-6, 2, 500), 1e3, D)
    assert c0 == pytest.approx(float(np.min(cp)), rel=1e-3)


def test_rigidity_sensitivity_is_positive_and_grows_with_k():
    D = A.flexural_rigidity(A.E_ICE, 1.0)
    s = A.rigidity_sensitivity(np.array([0.1, 0.6]), DEPTH, D)
    assert s[0] > 0 and s[1] > s[0]


# --------------------------------------------------------- discrete operators
def test_two_solution_paths_of_the_same_discrete_system_agree():
    res = solver_cross_check(n=28, h_i=0.5, nx=128, nz=16, nsteps=300)
    assert res["max_rel_field_difference"] < 1e-10


def test_horizontal_symbol_matches_sparse_stencil():
    cfg = TankConfig(nx=64, nz=16)
    tank = FlexuralWaveTank(cfg, 4.0e7)
    cir = CirculantTank(cfg, 4.0e7)
    x = tank.x
    for n in (3, 11):
        f = np.cos(2.0 * np.pi * n * x / cfg.length)
        lhs = tank._d2x(f)
        rhs = np.fft.irfft(-cir.kappa2 * np.fft.rfft(f), n=cfg.nx)
        assert np.allclose(lhs, rhs, atol=1e-10 * np.max(np.abs(lhs)))


def test_dtn_symbol_matches_sparse_solve():
    cfg = TankConfig(nx=64, nz=16)
    tank = FlexuralWaveTank(cfg, 4.0e7)
    cir = CirculantTank(cfg, 4.0e7)
    x = tank.x
    for n in (3, 11):
        phi = np.cos(2.0 * np.pi * n * x / cfg.length)
        w_sparse = tank.surface_velocity(phi)
        w_circ = np.fft.irfft(cir.s_dtn * np.fft.rfft(phi), n=cfg.nx)
        assert np.allclose(w_sparse, w_circ, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------ numerical arm
@pytest.mark.parametrize("h_i", [0.1, 0.5, 1.0])
def test_numerical_dispersion_matches_analytic(h_i):
    cfg = TankConfig(nx=256, nz=32)
    D = A.flexural_rigidity(A.E_ICE, h_i)
    tank = CirculantTank(cfg, D)
    n = 24
    k = 2.0 * np.pi * n / cfg.length
    om = float(A.dispersion_omega(k, cfg.depth, D))
    res = numerical_mode(tank, n, om, n_periods=8)
    assert abs(res["rel_err_total"]) < 0.01
    assert res["fit_residual"] < 0.05           # the signal really is one tone


def test_estimator_recovers_a_known_frequency():
    t = np.arange(0, 2000) * 0.01
    om_true = 3.7
    y = 0.7 * np.cos(om_true * t) + 0.3 * np.sin(om_true * t) + 0.05
    om, resid = estimate_omega(t, y, 4.0)
    assert om == pytest.approx(om_true, rel=1e-6)
    assert resid < 1e-12


# ------------------------------------------------------------------ controls
def test_negative_control_zero_thickness_degenerates_to_free_surface():
    cfg = TankConfig(nx=256, nz=32)
    res = control_zero_thickness(cfg, [8, 24])
    assert res["max_rel_err"] < 1e-2


def test_negative_control_shallow_limit():
    res = control_shallow_limit(DEPTH)
    assert res["max_rel_err_analytic_limit"] < 1e-2
    assert res["pass"]


def test_rigid_limit_and_perturbation_detectability():
    D = A.flexural_rigidity(A.E_ICE, 1.0)
    shift = A.rigidity_sensitivity(np.array([0.6]), DEPTH, D)[0]
    assert shift > 0.01                          # a 10 % D error moves c_p by >1 %