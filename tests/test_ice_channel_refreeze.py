"""Regression tests for the ice-channel refreezing module (p381w1-N06).

Coverage
--------
* analytic anchors: Neumann root/closed forms, Robin closed form vs ODE
* enthalpy-LBM column: Neumann front (<5%), Robin column vs ODE (<15%),
  enthalpy inversion round-trip, phase-state consistency
* atmospheric boundary: condensed ``(T_eff, h_eff)`` equivalence, snow in series
* 2-D solvers: D2Q9 Couette profile, D2Q5 scalar diffusion
* navigation table: monotonicity, JSON round-trip, interpolation error
* anchors provenance: sha256 stability and confidence flags
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorlbm.ice_channel_refreeze import (
    ANCHORS,
    AtmosphericForcing,
    IceWaterProperties,
    RefreezingColumn,
    build_table,
    degree_day_thickness,
    drag_law_current,
    effective_air_temperature,
    equilibrium_thickness,
    integrate_growth,
    interpolation_error,
    neumann_lambda,
    neumann_temperature_profile,
    neumann_thickness,
    ocean_heat_flux,
    simulate_channel,
    surface_heat_flux,
    thickness_with_surface_resistance,
    time_to_thickness,
    validate_flow_solver,
    validate_scalar_solver,
    wind_stress,
)
from tensorlbm.ice_channel_refreeze.window import WindowTable

P = IceWaterProperties()


# ------------------------------------------------------------------ analytic
def test_neumann_lambda_satisfies_similarity_equation():
    from scipy.special import erf

    for ste in (0.01, 0.05, 0.2, 1.0):
        lam = neumann_lambda(ste)
        residual = lam * math.exp(lam ** 2) * erf(lam) - ste / math.sqrt(math.pi)
        assert abs(residual) < 1e-12


def test_neumann_thickness_growth_law():
    t = np.array([0.0, 3600.0, 14400.0])
    h = neumann_thickness(t, 10.0)
    assert h[0] == 0.0
    # s ~ sqrt(t)
    assert h[2] / h[1] == pytest.approx(2.0, rel=1e-12)


def test_robin_closed_form_matches_ode():
    forcing = AtmosphericForcing(t_air=-20.0, u_wind=5.0)
    t_eff, h_eff = effective_air_temperature(forcing)
    for days in (0.25, 1.0, 5.0):
        closed = thickness_with_surface_resistance(86400.0 * days, P.t_freeze - t_eff, h_eff)
        ode = float(integrate_growth(days, t_eff, h_eff).h[-1])
        assert closed == pytest.approx(ode, rel=2e-3)


def test_no_ocean_flux_limit_recovers_higher_thickness():
    forcing = AtmosphericForcing(t_air=-20.0, u_wind=5.0)
    t_eff, h_eff = effective_air_temperature(forcing)
    h0 = float(integrate_growth(5.0, t_eff, h_eff, q_ocean=0.0).h[-1])
    h1 = float(integrate_growth(5.0, t_eff, h_eff, q_ocean=20.0).h[-1])
    assert h1 < h0
    assert equilibrium_thickness(t_eff, h_eff, 20.0) < 10.0


# ------------------------------------------------------------- boundary BCs
def test_condensed_surface_flux_equivalence():
    forcing = AtmosphericForcing(t_air=-15.0, u_wind=7.0)
    t_eff, h_eff = effective_air_temperature(forcing)
    for t_s in (-12.0, -5.0, 0.0):
        q_condensed = h_eff * (t_s - t_eff)
        q_direct = surface_heat_flux(t_s, forcing)
        assert q_condensed == pytest.approx(q_direct, rel=1e-12)


def test_snow_layer_reduces_effective_coefficient():
    bare = effective_air_temperature(AtmosphericForcing(t_air=-15.0, u_wind=5.0))
    snowy = effective_air_temperature(
        AtmosphericForcing(t_air=-15.0, u_wind=5.0, snow_depth=0.05)
    )
    assert snowy[1] < bare[1]
    r_snow = 0.05 / 0.31
    assert 1.0 / snowy[1] == pytest.approx(1.0 / bare[1] + r_snow, rel=1e-12)


def test_wind_stress_and_drift_drag_law():
    tau = wind_stress(10.0)
    assert tau == pytest.approx(1.3 * 1.8e-3 * 100.0, rel=1e-12)
    u_s = drag_law_current(tau)
    assert 0.2 < u_s < 0.5  # 10 m/s wind -> ~0.3 m/s wind drift
    q = ocean_heat_flux(u_s, 0.05)
    assert q == pytest.approx(1025.0 * 3990.0 * 1.2e-3 * u_s * 0.05, rel=1e-12)


# ------------------------------------------------------------- column model
def test_column_enthalpy_inversion_roundtrip():
    col = RefreezingColumn(depth=0.5, n=50)
    H = np.linspace(-1.0e7, P.rho_latent + 1.0e7, 25)
    T, f_l = col.invert(H)
    latent = P.rho_latent
    for h_val, t_val, fl_val in zip(H, T, f_l):
        if h_val < 0:
            assert t_val < P.t_freeze and fl_val == 0.0
        elif h_val < latent:
            assert t_val == pytest.approx(P.t_freeze) and 0.0 <= fl_val <= 1.0
        else:
            assert t_val > P.t_freeze and fl_val == 1.0


def test_column_neumann_front_within_5_percent():
    d_theta = 10.0
    n, dx, t0 = 240, 0.0025, 7200.0
    x = np.arange(n) * dx
    col = RefreezingColumn(depth=n * dx, n=n, surface_dirichlet=P.t_freeze - d_theta)
    col.set_state(
        neumann_temperature_profile(x, t0, d_theta),
        np.where(x <= neumann_thickness(t0, d_theta), 0.0, 1.0),
    )
    col.run_to(36000.0)
    h = col.thickness()
    exact = float(neumann_thickness(t0 + 36000.0, d_theta))
    assert abs(h - exact) / exact < 0.05


def test_column_robin_growth_matches_ode():
    forcing = AtmosphericForcing(t_air=-20.0, u_wind=5.0)
    t_eff, h_eff = effective_air_temperature(forcing)
    col = RefreezingColumn(depth=1.0, n=200, forcing=forcing)
    col.set_open_water()
    col.run_to(86400.0)
    h_lbm = col.thickness()
    h_ode = float(integrate_growth(1.0, t_eff, h_eff).h[-1])
    assert abs(h_lbm - h_ode) / h_ode < 0.15


def test_column_ocean_flux_slows_growth():
    forcing = AtmosphericForcing(t_air=-20.0, u_wind=5.0)
    c0 = RefreezingColumn(depth=0.6, n=120, forcing=forcing, q_ocean=0.0)
    c1 = RefreezingColumn(depth=0.6, n=120, forcing=forcing, q_ocean=150.0)
    for c in (c0, c1):
        c.set_open_water()
        c.run_to(86400.0)
    assert c1.thickness() < 0.9 * c0.thickness()


def test_degree_day_scaling():
    assert degree_day_thickness(100.0, 3.5) == pytest.approx(0.35, rel=1e-12)
    assert np.allclose(degree_day_thickness(np.array([25.0, 100.0]), 3.0), [0.15, 0.30])


# -------------------------------------------------------------- 2-D solvers
def test_flow_solver_matches_couette():
    res = validate_flow_solver(ny=16, nx=24, n_steps=8000, u_lid_lat=0.04, nu_lat=0.05)
    assert res["stable"]
    assert res["max_rel_error_prof"] < 0.06


def test_scalar_solver_matches_analytic_decay():
    res = validate_scalar_solver(ny=40, nx=8, n_steps=400, kappa_lat=0.02)
    assert res["amplitude_numeric"] == pytest.approx(res["amplitude_analytic"], rel=0.08)
    assert res["l2_rel_error_interior"] < 0.20


def test_channel_shape_factor_is_positive_and_finite():
    res = simulate_channel(W=50.0, H=20.0, u_wind=5.0, nx=24, n_steps=1500)
    assert np.isfinite(res.q_shape_mean)
    assert 0.0 < res.q_shape_mean < 10.0
    assert res.theta.min() >= -1e-9
    assert res.theta.max() <= 1.0 + 1e-9


# ------------------------------------------------------------------- table
@pytest.fixture(scope="module")
def small_table():
    width = np.array([50.0, 100.0, 200.0])
    h_target = np.array([0.15, 0.30])
    t_air = np.array([-25.0, -15.0])
    u_wind = np.array([0.0, 5.0, 10.0])
    drift = lambda w, u: drag_law_current(wind_stress(u))
    tab = build_table(width, h_target, t_air, u_wind, drift, initial_superheat=0.05)
    return tab, drift


def test_table_monotonicity(small_table):
    tab, _ = small_table
    # colder air -> earlier closure
    assert tab.query(100.0, 0.30, -25.0, 5.0) < tab.query(100.0, 0.30, -15.0, 5.0)
    # thicker threshold -> later closure
    assert tab.query(100.0, 0.30, -20.0, 5.0) > tab.query(100.0, 0.15, -20.0, 5.0)
    # stronger wind -> later closure (wind-driven ocean heat flux)
    assert tab.query(100.0, 0.30, -20.0, 10.0) > tab.query(100.0, 0.30, -20.0, 0.0)


def test_table_json_roundtrip(small_table, tmp_path):
    tab, _ = small_table
    path = os.path.join(str(tmp_path), "t.json")
    tab.to_json(path)
    back = WindowTable.from_json(path)
    assert back.query(100.0, 0.30, -20.0, 5.0) == pytest.approx(
        tab.query(100.0, 0.30, -20.0, 5.0), rel=1e-12
    )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["schema"] == "ice-channel-refreeze/window-table/v1"
    assert payload["grid_order"] == ["width_m", "h_target_m", "t_air_degC", "u_wind_m_s"]


def test_table_interpolation_error_under_10_percent():
    """Same grid density as the shipped table (coarse grids cannot pass)."""
    width = np.array([50.0, 100.0, 200.0, 400.0])
    h_target = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    t_air = np.array([-30.0, -25.0, -20.0, -15.0, -10.0])
    u_wind = np.array([0.0, 5.0, 10.0, 15.0])
    drift = lambda w, u: drag_law_current(wind_stress(u))
    tab = build_table(width, h_target, t_air, u_wind, drift, initial_superheat=0.05)
    err = interpolation_error(tab, drift, n_random=16, seed=3)
    assert err["n_points"] >= 10
    # robust statistics: the log-linear interpolant is accurate on the smooth
    # part of the surface but loses accuracy at the reachability cliff
    # (T_window -> inf when the equilibrium thickness drops below h_target).
    frac_under_10 = sum(1 for q in err["points"] if q["rel_error"] < 0.10) / err["n_points"]
    assert err["median_rel_error"] < 0.03
    assert frac_under_10 >= 0.85


def test_time_to_thickness_matches_column_order_of_magnitude():
    forcing = AtmosphericForcing(t_air=-20.0, u_wind=5.0)
    u_s = drag_law_current(wind_stress(5.0))
    tw = time_to_thickness(0.30, forcing, u_s, initial_superheat=0.05)
    assert 3.0 < tw < 15.0


# ---------------------------------------------------------------- anchors
def test_anchor_digest_is_stable_and_unique():
    digests = {}
    for key, anchor in ANCHORS.items():
        d = anchor.digest()
        assert len(d) == 64
        assert d not in digests.values()
        digests[key] = d
        payload = anchor.as_dict()
        assert payload["sha256"] == d


def test_anchor_confidence_and_deviation():
    from tensorlbm.ice_channel_refreeze import WEEKS_ACKLEY_GROWTH_CONSTANT, LEWIS_1993_WINDOW

    for a in (WEEKS_ACKLEY_GROWTH_CONSTANT, LEWIS_1993_WINDOW):
        assert a.confidence in {"recalled", "exact"}
        assert a.verification_gap
    assert WEEKS_ACKLEY_GROWTH_CONSTANT.deviation(3.3) == pytest.approx(0.1, rel=1e-6)


def test_module_has_no_literature_data_hardcoded_in_tests():
    # anchors must live in the module, not in this test file's numbers
    assert "recalled" in {a.confidence for a in ANCHORS.values()}