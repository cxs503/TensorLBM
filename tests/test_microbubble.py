"""Tests for the microbubble drag-reduction module (IN62-N04).

The tests are written to *break* the closures, and every numerical test is
graded against an anchor that is not the code under test:

* modified log law -- clean limit, monotonicity, exact relation ``DR = 1-r^2``;
* void transport   -- finite volume / LBM vs the closed form of the same ODE,
  plus the structural mass ledger;
* two-phase LBM    -- Euler-Euler mixture channel vs the exact two-layer
  solution (quadrature on a fine grid, independent of the lattice);
* boundary layer   -- log-law reproduction by a solver that never uses the
  log law, and agreement of the two independent DR routes;
* power ledger     -- thermodynamics limits and the break-even algebra.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.microbubble import (
    BLParams,
    ChannelConfig,
    CompressorSpec,
    LogLawParams,
    MixtureChannelLBM,
    TransportConfig,
    TransportFV,
    TransportLBM,
    alpha_profile_analytic,
    analytic_supply,
    b_eff,
    boundary_layer,
    drag_reduction,
    drag_reduction_layer,
    drag_reduction_loglaw,
    drag_reduction_re_theta,
    isothermal_work,
    kappa_eff,
    log_law_check,
    log_layer_average,
    mass_ledger,
    multistage_work,
    net_power,
    profitability_threshold,
    two_layer_drag_reduction,
    u_plus,
    u_tau_ratio,
)


# ---------------------------------------------------------------------------
# modified log law
# ---------------------------------------------------------------------------


def test_clean_limit():
    """alpha = 0 must reproduce the universal log law exactly."""
    p = LogLawParams()
    assert kappa_eff(0.0) == pytest.approx(p.kappa)
    assert b_eff(0.0) == pytest.approx(p.b_intercept)
    for yp in (30.0, 100.0, 500.0):
        assert float(u_plus(yp, 0.0)) == pytest.approx(
            math.log(yp) / p.kappa + p.b_intercept, rel=1e-12)
    assert u_tau_ratio(0.0) == pytest.approx(1.0, rel=1e-9)
    assert drag_reduction(0.0) == pytest.approx(0.0, abs=1e-12)


def test_kappa_eff_is_density_weighted_mixing_length():
    for a in (0.0, 0.1, 0.2, 0.3):
        assert kappa_eff(a) == pytest.approx(0.41 * math.sqrt(1.0 - a))


def test_sublayer_continuity():
    """The modified profile must pass through the clean value at y0+ = 11."""
    for a in (0.05, 0.15, 0.3):
        u_mod = float(u_plus(11.0, a))
        u_clean = math.log(11.0) / 0.41 + 5.2
        assert u_mod == pytest.approx(u_clean, rel=1e-12)


def test_drag_reduction_monotone_and_bounded():
    alphas = np.linspace(0.0, 0.3, 13)
    dr = np.array([drag_reduction(float(a)) for a in alphas])
    assert np.all(np.diff(dr) > 0.0)
    assert np.all(dr >= 0.0) and np.all(dr < 0.6)
    for a, d in zip(alphas, dr):
        r = u_tau_ratio(a)
        assert d == pytest.approx(1.0 - (1.0 - a) * r * r, rel=1e-12)


def test_log_layer_average_recovers_uniform_case():
    """A uniform void fraction must give back kappa sqrt(1-alpha)."""
    for a in (0.1, 0.2, 0.3):
        avg = log_layer_average(lambda _y: a, u_tau=0.05, nu=1.5e-5,
                                delta=0.02)
        assert avg["kappa_eff"] == pytest.approx(0.41 * math.sqrt(1.0 - a),
                                                 rel=1e-6)


# ---------------------------------------------------------------------------
# void transport
# ---------------------------------------------------------------------------

LATTICE_CFG = TransportConfig(ny=64, H=32.0, D=0.2, v_sweep=0.00275,
                              alpha_w=0.2)


def test_transport_closed_form_limits():
    """Pe -> 0 gives the linear profile; large Pe flattens it and collapses it at the outflow wall."""
    lin = TransportConfig(ny=16, H=8.0, D=1.0, v_sweep=1e-9, alpha_w=0.2)
    y = np.array([0.0, 2.0, 4.0])
    assert np.allclose(alpha_profile_analytic(lin, y), [0.2, 0.15, 0.1])
    # drift pushes gas towards the outflow wall, so a large Peclet number
    # keeps the layer almost uniform and collapses it near y = H
    steep = TransportConfig(ny=16, H=8.0, D=0.05, v_sweep=0.05, alpha_w=0.2)
    prof = alpha_profile_analytic(steep, np.array([0.0, 4.0, 7.9, 8.0]))
    assert prof[0] == pytest.approx(0.2, rel=1e-9)
    assert prof[1] > 0.19 and prof[2] < 0.05
    assert prof[3] == pytest.approx(0.0, abs=1e-12)
    # d/dy < 0 everywhere (monotone decay towards the outflow boundary)
    ys = np.linspace(0.0, 8.0, 50)
    assert np.all(np.diff(alpha_profile_analytic(steep, ys)) <= 1e-12)


def test_transport_fv_matches_closed_form():
    fv = TransportFV(LATTICE_CFG)
    fv.run(60000)
    ana = alpha_profile_analytic(LATTICE_CFG, fv.y)
    err = np.max(np.abs(fv.profile() - ana)) / LATTICE_CFG.alpha_w
    assert err < 0.01, f"FV deviation {err:.4f}"


def test_transport_mass_ledger_closes():
    fv = TransportFV(LATTICE_CFG)
    fv.run(60000)
    sup = fv.supply()
    assert abs(sup["closure_residual"]) / sup["supply"] < 1e-8
    led = mass_ledger(fv)
    assert abs(led["closure_error"]) < 1e-8
    # closed form of the same budget
    ana = analytic_supply(LATTICE_CFG)
    assert ana["supply"] == pytest.approx(sup["supply"], rel=5e-3)


def test_transport_lbm_agrees_with_closed_form():
    """The advection-diffusion LBM vs the closed form.

    The lattice solve is the same equation; the measured deviation at this
    resolution is a few per cent (documented in the report as the limitation
    of the anti-bounce-back Dirichlet closure for the advective flux).
    """
    lbm = TransportLBM(LATTICE_CFG)
    lbm.run(60000, tol=1e-14)
    ana = alpha_profile_analytic(LATTICE_CFG, lbm.y)
    err = np.max(np.abs(lbm.profile() - ana)) / LATTICE_CFG.alpha_w
    assert err < 0.10, f"LBM deviation {err:.4f}"


# ---------------------------------------------------------------------------
# two-phase mixture LBM vs exact two-layer solution
# ---------------------------------------------------------------------------


def test_lbm_clean_channel_matches_poiseuille():
    cfg = ChannelConfig(ny=32)
    lb = MixtureChannelLBM(cfg)
    lb.run(20000)
    ex = two_layer_drag_reduction(cfg.H, cfg.U_target, cfg.nu_liquid, 0.0, 0.0)
    assert lb.bulk_velocity() == pytest.approx(cfg.U_target, rel=1e-3)
    assert 1.0 - lb.G / ex["G_clean"] == pytest.approx(0.0, abs=2e-3)


@pytest.mark.parametrize("alpha_layer,h_layer", [(0.5, 5.0), (0.2, 10.0)])
def test_lbm_two_layer_drag_reduction(alpha_layer, h_layer):
    cfg = ChannelConfig(ny=32, alpha_layer=alpha_layer, h_layer=h_layer)
    base = MixtureChannelLBM(ChannelConfig(ny=32))
    base.run(20000)
    lb = MixtureChannelLBM(cfg)
    lb.run(20000)
    dr_lbm = 1.0 - lb.G / base.G
    ex = two_layer_drag_reduction(cfg.H, cfg.U_target, cfg.nu_liquid,
                                 alpha_layer, h_layer)
    assert dr_lbm == pytest.approx(ex["DR"], rel=0.05)


def test_two_layer_exact_zero_layer_is_poiseuille():
    ex = two_layer_drag_reduction(30.0, 0.05, 0.1, 0.5, 0.0)
    assert ex["DR"] == pytest.approx(0.0, abs=1e-12)
    u = ex["clean"]["u"]
    y = ex["clean"]["y"]
    H = 30.0
    u_pois = 6.0 * 0.05 * y * (H - y) / H ** 2
    assert np.max(np.abs(u - u_pois)) < 1e-3


# ---------------------------------------------------------------------------
# turbulent boundary layer
# ---------------------------------------------------------------------------


def test_bl_solver_reproduces_log_law():
    res = log_law_check(2000.0)
    assert res["max_abs_error"] < 0.6      # in u+ units (~3 % of u+ ~ 17)
    assert res["mean_error"] == pytest.approx(0.0, abs=0.2)


def test_bl_drag_reduction_monotone():
    dr = []
    for aw in (0.0, 0.1, 0.2, 0.3):
        row = drag_reduction_re_theta(2000.0, alpha_w=aw,
                                      alpha_of_y=(None if aw == 0 else
                                                  (lambda y, a=aw: a)))
        dr.append(row["DR_total"])
    assert dr[0] == 0.0
    assert np.all(np.diff(dr) > 0.0)


def test_two_routes_agree_within_model_gap():
    """Mixing-length route vs profile-aware log law (documented gap ~20 %)."""
    a = 0.2
    res = drag_reduction_re_theta(2000.0, alpha_w=a, alpha_of_y=lambda y: a)
    delta = float(res["clean"]["y"][-1])
    u_tau = float(res["clean"]["u_tau"])
    ana = drag_reduction_layer(a, lambda y: a, u_tau, 1.5e-5, delta)
    gap = abs(res["DR_total"] - ana["DR"]) / ana["DR"]
    assert gap < 0.25, f"two-route gap {gap:.3f}"


def test_boundary_layer_consistency():
    """A boundary layer thick enough to be turbulent reproduces Cf ~ 4.3e-3."""
    bl = boundary_layer(1.0, 0.30, 1.5e-5)
    assert bl["y"][-1] * bl["u_tau"] / 1.5e-5 > 500.0     # turbulent delta+
    assert 0.003 < bl["Cf"] < 0.006              # Cf = 0.00475 measured
    assert 1000.0 < bl["Re_theta"] < 1600.0      # Re_theta = 1238 measured
    assert np.all(np.diff(bl["u"]) > 0.0)


# ---------------------------------------------------------------------------
# power ledger
# ---------------------------------------------------------------------------


def test_compression_work_bounds():
    op = CompressorSpec()
    assert op.pressure_ratio > 1.0
    assert isothermal_work(op) < multistage_work(op)
    single = multistage_work(op, n_stages=1)
    triple = multistage_work(op, n_stages=3)
    assert triple < single              # intercooling saves work


def test_net_power_scaling():
    tau = 80.0
    base = net_power(0.2, 10.0, tau, 1e-3)
    assert base["saved_power_density"] == pytest.approx(0.2 * tau * 10.0)
    more_gas = net_power(0.2, 10.0, tau, 2e-3)
    assert more_gas["net_power_density"] < base["net_power_density"]
    zero_gas = net_power(0.2, 10.0, tau, 0.0)
    assert zero_gas["compressor_power_density"] == pytest.approx(0.0)
    assert zero_gas["net_power_density"] == pytest.approx(
        zero_gas["saved_power_density"])


def test_recirculation_recovers_gas_cost():
    tau = 80.0
    no_rec = net_power(0.2, 10.0, tau, 1e-3, recirculation=0.0)
    full_rec = net_power(0.2, 10.0, tau, 1e-3, recirculation=1.0)
    assert full_rec["compressor_power_density"] == pytest.approx(0.0)
    assert full_rec["net_power_density"] > no_rec["net_power_density"]


def test_profitability_threshold_algebra():
    thr = profitability_threshold(10.0, 80.0, 0.2, 1e-2)
    # a large make-up flow at zero recirculation loses money
    assert 0.0 < thr["flow_ratio"] < 1.0
    assert 0.0 < thr["recirculation_min"] < 1.0
    assert thr["C_q_max"] == pytest.approx(thr["q_break_even"] / 10.0)
    cheap = profitability_threshold(10.0, 80.0, 0.2, 1e-3)
    assert cheap["flow_ratio"] > 1.0            # same DR, ten times less gas
    rich = profitability_threshold(10.0, 80.0, 0.5, 1e-5)
    assert rich["flow_ratio"] > 1.0
    assert rich["recirculation_min"] == pytest.approx(0.0)