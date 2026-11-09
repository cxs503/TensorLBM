"""Tests for the cross-solver verification bench (node P78-N01).

Fast checks only (the full benchmark is :mod:`scripts.run_cross_solver_verification`):

* the manufactured solution really satisfies the incompressible Navier-Stokes
  equations and its analytic ``curl`` matches finite differences;
* each of the three solvers reproduces the manufactured solution and shows its
  designed order of accuracy on a small grid;
* the conservation ledgers behave (mass flux, energy budget, negative control).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.cross_solver_verification.fem import solve_taylor_hood
from tensorlbm.cross_solver_verification.fvm import solve_psi_omega
from tensorlbm.cross_solver_verification.lbm import run_lbm_cavity, run_lbm_mms
from tensorlbm.cross_solver_verification.ledger import (
    compare_ghia,
    divergence_ledger,
    energy_ledger,
    ghia_reference,
    relative_l2,
)
from tensorlbm.cross_solver_verification.mms import ManufacturedSolution

K = 2.0 * np.pi
NU = 0.002
AMP = 0.01 / (2.0 * K)   # -> B = 0.01, Re_mms = B/(nu k) ~ 0.8
MMS = ManufacturedSolution(nu=NU, amplitude=AMP, pressure_scale=0.005, k=K)


def _exact(n):
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return xx, yy, MMS.velocity(xx, yy)


# --------------------------------------------------------------------- MMS checks
def test_mms_is_divergence_free_and_vanishes_on_walls():
    x = np.linspace(0.0, 1.0, 57)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    u, v = MMS.velocity(xx, yy)
    # homogeneous Dirichlet on all four walls
    for arr in (u, v):
        assert np.abs(arr[0, :]).max() < 1e-12
        assert np.abs(arr[-1, :]).max() < 1e-12
        assert np.abs(arr[:, 0]).max() < 1e-12
        assert np.abs(arr[:, -1]).max() < 1e-12
    # divergence free by construction (stream function), checked by FD
    h = x[1] - x[0]
    dux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
    dvy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)
    assert np.abs(dux + dvy).max() < 1e-9


def test_mms_source_satisfies_momentum():
    """Substitute the closed forms back into the momentum residual (4th-order FD)."""
    h = 2e-4
    x = np.array([0.31, 0.52, 0.77])
    y = np.array([0.23, 0.61, 0.88])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    u, v = MMS.velocity(xx, yy)
    fx, fy = MMS.source(xx, yy)
    p = MMS.pressure(xx, yy)

    def d(fn, axis, xa, ya):
        if axis == 0:
            return (-fn(xa + h, ya) + 8 * fn(xa + h / 2, ya)
                    - 8 * fn(xa - h / 2, ya) + fn(xa - h, ya)) / (6 * h)
        return (-fn(xa, ya + h) + 8 * fn(xa, ya + h / 2)
                - 8 * fn(xa, ya - h / 2) + fn(xa, ya - h)) / (6 * h)

    def ux_(a, b):
        return MMS.velocity(a, b)[0]

    def uy_(a, b):
        return MMS.velocity(a, b)[1]

    def p_(a, b):
        return MMS.pressure(a, b)

    u_x = d(ux_, 0, xx, yy)
    u_y = d(ux_, 1, xx, yy)
    v_x = d(uy_, 0, xx, yy)
    v_y = d(uy_, 1, xx, yy)
    u_xx = d(lambda a, b: d(ux_, 0, a, b), 0, xx, yy)
    u_yy = d(lambda a, b: d(ux_, 1, a, b), 1, xx, yy)
    v_xx = d(lambda a, b: d(uy_, 0, a, b), 0, xx, yy)
    v_yy = d(lambda a, b: d(uy_, 1, a, b), 1, xx, yy)
    p_x = d(p_, 0, xx, yy)
    p_y = d(p_, 1, xx, yy)
    res_x = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) - fx
    res_y = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) - fy
    scale = max(np.abs(fx).max(), np.abs(fy).max())
    assert np.abs(res_x).max() / scale < 1e-4
    assert np.abs(res_y).max() / scale < 1e-4
    # continuity
    assert np.abs(u_x + v_y).max() < 1e-6


def test_mms_curl_source_matches_finite_difference():
    h = 1e-6
    x = np.array([0.17, 0.4, 0.83])
    y = np.array([0.29, 0.55, 0.71])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    fy_p = MMS.source(xx + h, yy)[1]
    fy_m = MMS.source(xx - h, yy)[1]
    fx_p = MMS.source(xx, yy + h)[0]
    fx_m = MMS.source(xx, yy - h)[0]
    fd = (fy_p - fy_m) / (2 * h) - (fx_p - fx_m) / (2 * h)
    an = MMS.curl_source(xx, yy)
    assert np.abs(fd - an).max() < 1e-6 * max(1.0, np.abs(an).max())
    # zero mean is required by the periodic Poisson problem
    xg = (np.arange(64) + 0.5) / 64
    xxg, yyg = np.meshgrid(xg, xg, indexing="xy")
    assert abs(MMS.curl_source(xxg, yyg).mean()) < 1e-12


# ------------------------------------------------------------------ solver orders
def test_lbm_mms_second_order():
    errs = []
    sizes = [16, 32]
    for n in sizes:
        r = run_lbm_mms(MMS, n, max_steps=40000, tol=1e-9, check_every=200)
        _, _, (ue, ve) = _exact(n)
        errs.append(float(np.sqrt(np.mean((r.u - ue) ** 2 + (r.v - ve) ** 2))))
    order = float(np.log2(errs[0] / errs[1]))
    assert 1.7 < order < 2.4, f"LBM observed order {order:.2f} (errors {errs})"


def test_fvm_mms_second_order():
    errs = []
    sizes = [16, 32]
    for n in sizes:
        r = solve_psi_omega(n, NU, mode="periodic", mms=MMS, max_iter=60000, tol=1e-11)
        _, _, (ue, ve) = _exact(n)
        errs.append(float(np.sqrt(np.mean((r.u - ue) ** 2 + (r.v - ve) ** 2))))
    order = float(np.log2(errs[0] / errs[1]))
    # The coarse pair is *superconvergent* for this particular sin^2*sin^2 field
    # (~3.4-4.0); the asymptotic second order is reached on the finer pair
    # (n=64->128 gives 2.30, see cross_solver_verification_report.json C3).
    # The robust, fast claim is therefore "at least second order".
    assert order > 1.9, f"FVM observed order {order:.2f} (errors {errs})"


def test_fem_mms_third_order_ish():
    errs = []
    for n in (16, 32):
        r = solve_taylor_hood(n, NU, bc="periodic", mms=MMS, max_newton=14, tol=1e-11)
        _, _, (ue, ve) = _exact(n)
        errs.append(float(np.sqrt(np.mean((r.u - ue) ** 2 + (r.v - ve) ** 2))))
    order = float(np.log2(errs[0] / errs[1]))
    # Q2 velocity is third order in L2; the coarse-to-medium pair is genuinely
    # above second order -- that is the claim being tested here
    assert order > 2.4, f"FEM observed order {order:.2f} (errors {errs})"


# ----------------------------------------------------------------- cavity physics
def test_fvm_cavity_matches_ghia_re100():
    """Coarse-grid regression for the psi-omega cavity (Woods wall vorticity).

    The measured deviation is ~11 % at n=48 (16 % at n=32, 8 % at n=64), i.e. the
    wall-vorticity closure is only first order in practice -- the 3 % target needs
    n >= 128 and is *not* claimed here.  This test pins the achieved accuracy so a
    regression is visible.
    """
    r = solve_psi_omega(48, 1.0 / 100.0, mode="cavity", u_lid=1.0, max_iter=200000,
                        tol=1e-8, check_every=200, cfl=0.25)
    ref = ghia_reference(100)
    g = compare_ghia(r.u, r.v, 1.0, ref)
    assert g["u_max_dev_pct"] < 14.0, g
    assert g["v_max_dev_pct"] < 6.0, g
    assert float(np.abs(r.u).max()) < 1.2


def test_lbm_cavity_zero_driving_control():
    r = run_lbm_cavity(n=16, re=100.0, u_lid=0.0, max_steps=500, check_every=250)
    assert float(r.speed.max()) < 1e-14


def test_fem_zero_driving_control():
    r = solve_taylor_hood(16, 1.0, bc="walls", u_lid=0.0, max_newton=4, tol=1e-12)
    assert float(r.speed.max()) < 1e-13


def test_streamfunction_velocity_is_divergence_free():
    r = solve_psi_omega(32, 1.0 / 100.0, mode="cavity", u_lid=1.0, max_iter=4000,
                        tol=1e-8, check_every=200)
    led = divergence_ledger(r.u, r.v, 1.0 / 32, 1.0)
    assert led["flux_integral"] < 1e-12
    assert led["max_abs_div_scaled"] < 1e-12


def test_energy_ledger_is_computed_but_does_not_close_at_coarse_grid():
    """The steady budget P = eps does NOT close at n=48 (measured 27-30 % gap).

    The volume dissipation converges quickly and agrees between solvers to a few
    percent, while the lid power needs the wall-normal gradient at the lid, which
    is under-resolved at n<=64 (see the report, criterion C4: 30-61 % gap).  The
    test therefore only pins that both terms are computed, positive and of the
    right order of magnitude -- the gap is reported, not hidden.
    """
    r = solve_psi_omega(48, 1.0 / 100.0, mode="cavity", u_lid=1.0, max_iter=200000,
                        tol=1e-8, check_every=200, cfl=0.25)
    led = energy_ledger(r.u, r.v, 1.0 / 48, 1.0 / 100.0, 1.0)
    assert led["power_in"] > 0.0
    assert led["dissipation"] > 0.0
    assert 0.05 < led["dissipation"] < 0.5
    assert 0.01 < led["power_in"] < 0.5
    assert led["rel_error_pct"] < 60.0, led


def test_relative_l2_sanity():
    a = np.ones((4, 4))
    assert relative_l2(a, a) == 0.0
    assert relative_l2(2 * a, a) == pytest.approx(1.0)