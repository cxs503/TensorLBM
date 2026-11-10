"""Tests for the rarefied / slip-flow verification ladder (P218-N06).

Covers
  * the analytic reference layer (mean free path <-> Kn, Maxwell slip, NS+slip
    flow rate, free-molecular asymptote and its ``6 Kn`` consistency relation),
  * the deterministic linearised-BGK DVM solver (linearity, trivial null state,
    slip-length monotonicity and TMAC scaling, free-molecular plug, Kn -> 0
    degeneration, Knudsen-layer thickness),
  * the independent DSMC solver (free-molecular Couette plateau, no driving ->
    no flow, agreement with the DVM in the transitional regime),
  * the ladder plumbing (Richardson extrapolation, JSON verdict structure).

Grids/particle counts are deliberately small so the whole file runs in well
under two minutes; the production numbers come from ``rarefied.ladder``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.rarefied import knudsen as K
from tensorlbm.rarefied.bgk_slab import LinearizedBGKSlab
from tensorlbm.rarefied.dsmc_slab import run_dsmc_slab
from tensorlbm.rarefied.ladder import richardson


# --------------------------------------------------------------- analytics
def test_mean_free_path_and_kn_roundtrip():
    for kn in (0.001, 0.05, 1.0, 10.0):
        tau = K.tau_from_kn(kn)
        assert K.mean_free_path(tau) == pytest.approx(kn, rel=1e-12)
        assert K.kn_from_tau(tau) == pytest.approx(kn, rel=1e-12)
    # BGK tau factor must be sqrt(2/pi)
    assert K.tau_from_kn(1.0) == pytest.approx(math.sqrt(2.0 / math.pi), rel=1e-14)


def test_maxwell_slip_length_scaling():
    lam = 0.01
    assert K.maxwell_slip_length(1.0, lam) == pytest.approx(lam, rel=1e-12)
    assert K.maxwell_slip_length(0.5, lam) == pytest.approx(3.0 * lam, rel=1e-12)
    assert K.maxwell_slip_length(0.2, lam) == pytest.approx(9.0 * lam, rel=1e-12)
    assert K.maxwell_slip_coefficient(1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        K.maxwell_slip_length(0.0, lam)


def test_ns_slip_flow_rate_matches_profile_integral():
    H, nu, g, ls = 1.0, 0.05, 0.3, 0.02
    y = np.linspace(0.0, H, 200001)
    u = K.ns_slip_poiseuille_profile(y, H, nu, g, ls)
    q_num = np.trapezoid(u, y)
    q_ana = K.ns_slip_flow_rate(H, nu, g, ls)
    assert q_num == pytest.approx(q_ana, rel=1e-8)
    # wall slip velocity equals l_s * du/dy at the wall
    du_dy = g * H / (2.0 * nu)
    assert u[0] == pytest.approx(ls * du_dy, rel=1e-12)


def test_ns_slip_flow_rate_enhancement():
    H, nu, g = 1.0, 0.04, 0.1
    q_ns = K.ns_slip_flow_rate(H, nu, g, 0.0)
    for kn in (0.001, 0.01, 0.1):
        ls = K.maxwell_slip_length(1.0, kn)
        q = K.ns_slip_flow_rate(H, nu, g, ls)
        assert q / q_ns == pytest.approx(1.0 + 6.0 * kn, rel=1e-12)


def test_free_molecular_ratio_is_six_kn():
    """Q_FM / Q_NS = 6 Kn exactly -- the classical FM/slip crossover relation."""
    g = 0.2
    lam = 0.05
    tau = lam * math.sqrt(2.0 / math.pi)
    q_ns = K.ns_slip_flow_rate(1.0, tau, g, 0.0)
    q_fm = K.free_molecular_flow_rate(1.0, g)
    assert q_fm / q_ns == pytest.approx(6.0 * lam, rel=1e-10)
    # the plug velocity is consistent with the flow rate
    assert K.free_molecular_plug_velocity(1.0, g) == pytest.approx(q_fm, rel=1e-12)


# ------------------------------------------------------------------- DVM
def test_dvm_linear_response_and_null_state():
    s = LinearizedBGKSlab(0.1, ny=48, nv=24)
    a = s.solve_poiseuille(1.0)
    b = s.solve_poiseuille(2.0)
    assert b.q == pytest.approx(2.0 * a.q, rel=1e-10)
    null = s.solve(g=0.0, u_walls=(0.0, 0.0))
    assert np.max(np.abs(null.u)) < 1e-12


def test_dvm_slip_length_positive_and_growing_with_kn():
    ls = []
    for kn in (0.02, 0.1, 0.3):
        sol = LinearizedBGKSlab(kn, ny=96, nv=24).solve_couette(1.0)
        ls.append(sol.slip_length())
        assert sol.slip_length() > 0.0
    assert ls[0] < ls[1] < ls[2]
    # l_s must be O(lambda) for a diffuse wall
    for kn, v in zip((0.02, 0.1, 0.3), ls):
        assert 0.8 < v / kn < 1.6


def test_dvm_tmac_scaling_of_slip_length():
    """l_s(sigma)/l_s(1) must approach Maxwell's (2-sigma)/sigma."""
    ref = LinearizedBGKSlab(0.1, sigma=1.0, ny=96, nv=24).solve_couette(1.0).slip_length()
    for sig in (0.5, 0.2):
        ls = LinearizedBGKSlab(0.1, sigma=sig, ny=96, nv=24).solve_couette(1.0).slip_length()
        theory = (2.0 - sig) / sig
        assert ls / ref == pytest.approx(theory, rel=0.20)


def test_dvm_couette_free_molecular_plug():
    sol = LinearizedBGKSlab(30.0, ny=64, nv=24).solve_couette(1.0)
    assert sol.centre_velocity() == pytest.approx(0.5, rel=0.02)
    assert np.max(np.abs(sol.u - 0.5)) / 0.5 < 0.10


def test_dvm_ns_limit_recovery():
    """Kn -> 0 must reproduce Q of the NS + Maxwell-slip asymptote."""
    kn = 0.02
    ny = 128                                   # dy/lambda = 0.39
    fine = 256                                 # dy/lambda = 0.195
    c = LinearizedBGKSlab(kn, ny=ny, nv=24).solve_poiseuille(1.0)
    f = LinearizedBGKSlab(kn, ny=fine, nv=24).solve_poiseuille(1.0)
    q_ext = richardson(c.q, f.q, ny, fine)
    assert q_ext / f.extra["q_ns_slip"] == pytest.approx(1.0, rel=0.08)
    assert f.extra["q_ns"] < f.extra["q_ns_slip"] < f.q


def test_dvm_knudsen_layer_thickness_is_order_lambda():
    diag = LinearizedBGKSlab(0.1, ny=96, nv=24).knudsen_layer_thickness(g=1.0)
    assert 0.4 < diag["kn_layer_ratio_fit"] < 2.5
    assert np.max(np.abs(diag["deviation"])) > 0.0


def test_dvm_arguments_validated():
    with pytest.raises(ValueError):
        LinearizedBGKSlab(0.0)
    with pytest.raises(ValueError):
        LinearizedBGKSlab(0.1, sigma=0.0)
    with pytest.raises(ValueError):
        LinearizedBGKSlab(0.1, ny=4)


# ------------------------------------------------------------------ DSMC
def test_dsmc_no_forcing_no_flow():
    res = run_dsmc_slab(1.0, g=0.0, ncells=24, ppc=30, steps=4000, seed=11)
    assert abs(float(np.mean(res.u))) < 0.02


def test_dsmc_couette_free_molecular_plateau():
    """Kn=100: the diffuse-wall Couette profile is the plug u = U/2."""
    res = run_dsmc_slab(100.0, g=0.0, u_walls=(0.0, 1.0), ncells=24, ppc=40,
                        steps=6000, seed=12)
    assert float(np.mean(res.u)) == pytest.approx(0.5, abs=0.04)
    assert float(np.ptp(res.u)) < 0.15


def test_dsmc_matches_dvm_transitional():
    """Two independent kinetic solvers must agree in the transitional regime."""
    kn = 1.0
    dsmc = run_dsmc_slab(kn, g=0.05, ncells=40, ppc=40, steps=20000, seed=13)
    dvm = LinearizedBGKSlab(kn, ny=64, nv=24).solve_poiseuille(0.05)
    assert dsmc.q == pytest.approx(dvm.q, rel=0.15)
    assert dsmc.q_err == dsmc.q_err  # not NaN


# ------------------------------------------------------------------ ladder
def test_richardson_extrapolation():
    f = lambda n: 2.0 + 3.0 / n
    assert richardson(f(64), f(128), 64, 128) == pytest.approx(2.0, rel=1e-12)


def test_ladder_c4_continuum_verdict_structure(tmp_path):
    from tensorlbm.rarefied.ladder import run_c4

    crit = run_c4(quick=True)
    d = crit.to_dict()
    assert d["key"] == "C4"
    assert d["detail"]["l_s_below_0.01H"] is True
    assert d["detail"]["profile_dev_max_pct_vs_noslip"] < 1.0
    assert d["verdict"] == "PASS"
    assert crit.detail["dvm_trend"]  # the DVM trend evidence is present


def test_maxwell_slip_zero_length_limit_for_specular_wall():
    """sigma -> 0 (specular) must give a diverging slip length, i.e. free slip."""
    big = K.maxwell_slip_length(1e-6, 0.01)
    assert big > 1e3