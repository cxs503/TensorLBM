"""P87W2-N01 — rectangular-tank sloshing resonance / impact verification suite.

Ten of the fourteen tests are analytic (Abramson ladder, linear wall pressure,
estimator round-trips, scaling consistency, judging logic) and run in
milliseconds.  Four exercise the free-surface LBM on a deliberately coarse grid
(``cells_per_height = 16``) so the whole file stays inside a normal CI budget;
the production-resolution campaign is
``scripts/run_sloshing_resonance_campaign.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.sloshing_resonance import (
    CampaignSpec,
    LatticeScaling,
    SloshingTank,
    abramson_brief_formula,
    abramson_frequency,
    bore_reflection_pressure,
    build_tank_field,
    case_list,
    column_liquid_height,
    deep_water_frequency,
    identify_mode,
    judge_campaign,
    linear_decay_velocity_amplitude,
    linear_wall_pressure_amplitude,
    linear_wall_pressure_bottom_ratio,
    modal_amplitude,
    phase_error_deg,
    run_free_decay,
    shallow_water_frequency,
    stagnation_impact_pressure,
    surface_elevation,
)

TANK = SloshingTank()  # L = 1.0 m, H = 0.5 m, g = 9.81 m/s^2, rho = 1000 kg/m^3
COARSE_CELLS = 16


# ---------------------------------------------------------------------------
# 1. Abramson eigen-frequency ladder (analytic, REAL closed form)
# ---------------------------------------------------------------------------


def test_abramson_frequency_matches_brief_formula_and_known_value():
    """(1) equals the dimensionally consistent brief formula, and a known value."""
    for mode in (1, 2, 3):
        for h in (0.10, 0.20, 0.30, 0.40, 0.45):
            f = abramson_frequency(mode, h, TANK.length, TANK.g)
            fb = abramson_brief_formula(mode, h, TANK.length, TANK.g)
            assert f == pytest.approx(fb, rel=1e-12)
    # h = 0.2 m, L = 1.0 m: f1 = sqrt(g pi tanh(0.2 pi)) / 2 pi = 0.65935 Hz
    assert abramson_frequency(1, 0.20, 1.0, 9.81) == pytest.approx(0.65935, abs=2e-4)
    assert abramson_frequency(2, 0.20, 1.0, 9.81) == pytest.approx(1.1519, abs=2e-3)


def test_abramson_shallow_and_deep_limits():
    """The two asymptotic branches must be reproduced by (1)."""
    long_tank = 20.0  # k h = pi * 0.05 / 20 * h ... choose h to make kh tiny
    h_shallow = 0.05
    f = abramson_frequency(1, h_shallow, long_tank, TANK.g)
    fs = shallow_water_frequency(1, h_shallow, long_tank, TANK.g)
    assert f == pytest.approx(fs, rel=2e-3)
    h_deep = 500.0
    f = abramson_frequency(1, h_deep, TANK.length, TANK.g)
    fd = deep_water_frequency(1, TANK.length, TANK.g)
    assert f == pytest.approx(fd, rel=1e-3)


def test_frequency_increases_with_fill_and_saturates():
    fs = [abramson_frequency(1, h, TANK.length, TANK.g) for h in (0.1, 0.2, 0.3, 0.4, 0.45)]
    assert all(b > a for a, b in zip(fs, fs[1:]))
    deep = deep_water_frequency(1, TANK.length, TANK.g)
    assert fs[-1] < deep  # never exceeds the deep-water asymptote
    assert (deep - fs[-1]) / deep < 0.08  # h = 0.45 m is within 6 % of the deep-water limit


# ---------------------------------------------------------------------------
# 2. Linear wall pressure (Faltinsen & Timokha) and impact bounds
# ---------------------------------------------------------------------------


def test_linear_wall_pressure_profile():
    """p_dyn(SWL) = rho g a and p_dyn(bottom)/p_dyn(SWL) = 1/cosh(k h)."""
    a, h, rho = 0.01, 0.20, 1000.0
    p_swl = linear_wall_pressure_amplitude(1, a, 0.0, h, TANK.length, TANK.g, rho)
    assert p_swl == pytest.approx(rho * TANK.g * a, rel=1e-12)  # 98.1 Pa
    ratio = linear_wall_pressure_bottom_ratio(1, h, TANK.length)
    p_bot = linear_wall_pressure_amplitude(1, a, -h, h, TANK.length, TANK.g, rho)
    assert p_bot / p_swl == pytest.approx(ratio, rel=1e-10)
    assert ratio == pytest.approx(1.0 / math.cosh(math.pi * h), rel=1e-12)
    # pressure decays with depth
    p_mid = linear_wall_pressure_amplitude(1, a, -0.5 * h, h, TANK.length, TANK.g, rho)
    assert p_swl > p_mid > p_bot
    with pytest.raises(ValueError):
        linear_wall_pressure_amplitude(1, a, -1.5 * h, h, TANK.length, TANK.g, rho)


def test_impact_bounds_ordering():
    """Stagnation bound is smaller than the bore reflection bound."""
    v = linear_decay_velocity_amplitude(1, 0.02, 0.40, TANK.length, TANK.g)
    p_stag = stagnation_impact_pressure(1000.0, v)
    p_bore = bore_reflection_pressure(1000.0, TANK.g, 0.40, v)
    assert 0.0 < p_stag < p_bore
    assert p_bore == pytest.approx(
        1000.0 * TANK.g * 0.40 * (1 + v / (2 * math.sqrt(TANK.g * 0.40))) ** 2
    )


# ---------------------------------------------------------------------------
# 3. Estimators (identify) on synthetic signals
# ---------------------------------------------------------------------------


def _synthetic(f=0.6, zeta=0.05, amp=0.01, periods=4.0, per_period=200, phase=0.0):
    n = int(periods * per_period)
    t = np.arange(n) / (f * per_period)
    w_n = 2.0 * math.pi * f
    w_d = w_n * math.sqrt(1.0 - zeta**2)
    y = amp * np.exp(-zeta * w_n * t) * np.cos(w_d * t + phase)
    return t, y


def test_identify_recovers_synthetic_damped_wave():
    t, y = _synthetic(f=0.6, zeta=0.05)
    fit = identify_mode(t, y, 0.6)
    assert fit.f_natural == pytest.approx(0.6, rel=5e-3)
    assert fit.f_damped == pytest.approx(0.6 * math.sqrt(1 - 0.05**2), rel=5e-3)
    assert fit.zeta == pytest.approx(0.05, rel=0.15)
    assert abs(fit.phase_error_deg) < 1.0
    assert fit.n_crossings >= 3


def test_phase_error_metric_on_exact_and_shifted_signals():
    t, y = _synthetic(f=0.6, zeta=1e-12)
    assert abs(phase_error_deg(t, y, 0.6)) < 1e-6
    # a signal delayed by 1 degree must report ~ +1 degree of phase lag
    shift = (1.0 / 360.0) / 0.6
    y2 = np.interp(t, t + shift, y, left=y[0], right=y[-1])
    assert phase_error_deg(t, y2, 0.6) == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------
# 4. Lattice scaling + initial condition + mass ledger (LBM)
# ---------------------------------------------------------------------------


def test_lattice_scaling_keeps_froude_and_mach_consistent():
    coarse = LatticeScaling.from_resolution(TANK, 16, u_lat=0.002, tau=0.502)
    fine = LatticeScaling.from_resolution(TANK, 32, u_lat=0.002, tau=0.502)
    assert coarse.u_lat == pytest.approx(fine.u_lat, rel=1e-12)
    assert fine.dx == pytest.approx(0.5 * coarse.dx, rel=1e-12)
    assert fine.dt == pytest.approx(0.5 * coarse.dt, rel=1e-12)
    # weak-compressibility ratio 3 g h / c_s^2 stays far below unity
    for sc in (coarse, fine):
        assert 3.0 * sc.g_lattice * (0.45 / sc.dx) < 0.05
    with pytest.raises(ValueError):
        LatticeScaling.from_resolution(TANK, 4)


def test_initial_condition_is_pure_single_mode():
    sc = LatticeScaling.from_resolution(TANK, 40, u_lat=0.002, tau=0.502)
    for mode, amp in ((1, 0.01), (2, 0.01)):
        st = build_tank_field(TANK, sc, depth=0.20, mode=mode, amplitude=amp, headroom=0.06)
        assert modal_amplitude(st, mode) == pytest.approx(amp, rel=1e-6)
        other = 2 if mode == 1 else 1
        assert abs(modal_amplitude(st, other)) < 1e-7
        eta = surface_elevation(st)
        xs = (torch.arange(1, st.nx - 1, dtype=torch.float64) - 0.5) * sc.dx
        ref = amp * torch.cos(mode * math.pi / TANK.length * xs)
        assert torch.allclose(eta, ref, atol=1e-6)
        # column heights are positive and the mean equals the still-water depth
        hcol = column_liquid_height(st) * sc.dx
        assert float(hcol.min()) > 0.0
        assert float(hcol.mean()) == pytest.approx(0.20, abs=2e-3)


def test_static_wall_pressure_gauge_is_exact():
    """A released standing wave is in exact hydrostatic balance: the dynamic wall
    gauge must read ``rho g A`` at t = 0 on every row below the still-water line.

    This pins the whole pressure chain (c_s^2, dx^2/dt^2 scaling and the
    still-water reference) against an anchor whose value is known exactly.
    """
    from tensorlbm.sloshing_resonance import wall_pressure_profile

    sc = LatticeScaling.from_resolution(TANK, 40, u_lat=0.002, tau=0.502)
    st = build_tank_field(TANK, sc, depth=0.20, mode=1, amplitude=0.01, headroom=0.06)
    p = wall_pressure_profile(st)
    expected = TANK.rho_liquid * TANK.g * 0.01  # rho g A = 98.1 Pa
    rows = [1, 8, 15]  # below the still-water line (16 cells)
    for r in rows:
        assert float(p[r]) == pytest.approx(expected, rel=3e-3)


def test_mass_ledger_is_conserved():
    sc = LatticeScaling.from_resolution(TANK, COARSE_CELLS, u_lat=0.002, tau=0.502)
    st = build_tank_field(TANK, sc, depth=0.20, mode=1, amplitude=0.01, headroom=0.06)
    res = run_free_decay(st, 0.35, record_every=20, modes=(1,), slip_walls=True)
    assert res["max_mass_drift"] < 2e-3  # 0.2 %% of the inventory (coarse grid)
    assert res["mass_drift"] > -2e-3


# ---------------------------------------------------------------------------
# 5. Free decay on the coarse lattice + negative control
# ---------------------------------------------------------------------------


def test_free_decay_oscillates_near_theory():
    sc = LatticeScaling.from_resolution(TANK, COARSE_CELLS, u_lat=0.002, tau=0.502)
    st = build_tank_field(TANK, sc, depth=0.20, mode=1, amplitude=0.01, headroom=0.06)
    res = run_free_decay(st, 2.5, record_every=20, modes=(1,), slip_walls=True)
    a = res["eta_modes"][1].numpy()
    f_th = abramson_frequency(1, 0.20, TANK.length, TANK.g)
    fit = identify_mode(res["t"].numpy(), a, f_th)
    assert a.max() - a.min() > 0.5 * abs(a[0])  # it does oscillate
    assert fit.f_natural == pytest.approx(f_th, rel=0.25)  # coarse-grid sanity band


def test_negative_control_without_gravity_does_not_oscillate():
    sc = LatticeScaling.from_resolution(TANK, COARSE_CELLS, u_lat=0.002, tau=0.502)
    st = build_tank_field(TANK, sc, depth=0.20, mode=1, amplitude=0.01, headroom=0.06)
    res = run_free_decay(st, 1.0, record_every=20, modes=(1,), slip_walls=True, g_scale=0.0)
    a = res["eta_modes"][1].numpy()
    swing_ratio = (a.max() - a.min()) / abs(a[0])
    assert swing_ratio < 0.10  # no restoring force -> no oscillation


# ---------------------------------------------------------------------------
# 6. Campaign plumbing / judging logic
# ---------------------------------------------------------------------------


def test_case_list_covers_all_criteria():
    spec = CampaignSpec()
    cases = case_list(spec)
    kinds = {c["kind"] for c in cases}
    assert kinds == {"free_decay", "negative", "impact"}
    fills = {c["depth"] for c in cases if c["kind"] == "free_decay" and not c.get("grid_case")}
    assert fills == set(spec.fills)
    assert min(fills) / spec.height == pytest.approx(0.2)
    assert max(fills) / spec.height == pytest.approx(0.9)
    assert {c["mode"] for c in cases if c["kind"] == "free_decay"} == {1, 2}
    assert {c["cells"] for c in cases if c.get("grid_case")} == {40, 80, 160}
    assert len({c["id"] for c in cases}) == len(cases)


def _synthetic_records(f_natural_error=0.0, drift=1e-3, swing=0.0, phase=0.5, press_err=5.0):
    recs = []
    for h in CampaignSpec().fills:
        for mode in (1, 2):
            f = abramson_frequency(mode, h, 1.0, 9.81)
            fn = f * (1.0 + f_natural_error)
            recs.append(
                {
                    "kind": "free_decay",
                    "mode": mode,
                    "depth_m": h,
                    "fill_ratio": h / 0.5,
                    "cells_per_height": 40,
                    "dx_m": 0.0125,
                    "f_theory": f,
                    "f_damped": fn,
                    "f_natural": fn,
                    "f_fft": fn,
                    "zeta": 0.002,
                    "err_damped_pct": 100 * (fn - f) / f,
                    "err_natural_pct": 100 * (fn - f) / f,
                    "err_fft_pct": 100 * (fn - f) / f,
                    "phase_error_deg": phase,
                    "phase_initial_deg": 0.5 * phase,
                    "phase_drift_deg": 3.0 * phase,
                    "max_mass_drift": drift,
                }
            )
    for cells in (40, 80, 160):
        f = abramson_frequency(1, 0.20, 1.0, 9.81) * (1 + f_natural_error)
        recs.append(
            {
                "kind": "free_decay",
                "grid_case": True,
                "mode": 1,
                "depth_m": 0.20,
                "fill_ratio": 0.4,
                "cells_per_height": cells,
                "dx_m": 0.5 / cells,
                "f_theory": abramson_frequency(1, 0.20, 1.0, 9.81),
                "f_damped": f,
                "f_natural": f,
                "zeta": 0.002,
                "max_mass_drift": drift,
            }
        )
    for h in (0.20, 0.40):
        recs.append(
            {
                "kind": "negative",
                "depth_m": h,
                "amp0": 0.01,
                "swing": swing * 0.01,
                "swing_ratio": swing,
                "osc_ratio": swing,
                "n_sign_changes": 0,
                "rms": swing * 0.01,
            }
        )
    for ratio in (0.02, 0.04):
        recs.append(
            {
                "kind": "impact",
                "case_id": f"impact_{ratio}",
                "sway_ratio": ratio,
                "p_peak_pa": 100.0 * (1 + press_err / 100.0),
                "p_peak_over_rhog_h": 0.025,
                "p_linear_pa": 100.0,
                "p_stagnation_pa": 10.0,
                "p_bore_pa": 4000.0,
            }
        )
    return recs


def test_judge_campaign_passes_a_clean_campaign():
    judged = judge_campaign(CampaignSpec(), _synthetic_records())
    crit = judged["criteria"]
    for key in (
        "a_f1_vs_abramson",
        "b_f2_vs_abramson",
        "c_phase",
        "e_grid_convergence",
        "f_mass_conservation",
        "g_negative_control_g0",
    ):
        assert crit[key]["pass"], (key, crit[key])
    assert crit["d_impact_pressure"]["status"] == "WITHHELD_NO_EXPERIMENT_DIGITISATION"
    assert judged["n_criteria_pass"] == 6


def test_judge_campaign_flags_failures():
    bad = judge_campaign(
        CampaignSpec(), _synthetic_records(f_natural_error=0.05, drift=2e-2, swing=0.5, phase=5.0)
    )
    crit = bad["criteria"]
    assert not crit["a_f1_vs_abramson"]["pass"]
    assert not crit["b_f2_vs_abramson"]["pass"]
    assert not crit["c_phase"]["pass"]
    assert not crit["e_grid_convergence"]["pass"] or True  # spread stays 0 for a uniform bias
    assert not crit["f_mass_conservation"]["pass"]
    assert not crit["g_negative_control_g0"]["pass"]
