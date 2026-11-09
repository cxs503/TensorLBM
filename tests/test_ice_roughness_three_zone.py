"""Three-lens cross-adjudication of the ice-underside drag closure (P103W2C-N01).

These tests pin the *analytic* content of
:mod:`tensorlbm.ice_roughness.three_zone` and
:mod:`tensorlbm.ice_roughness.propagation`: the limits of the Colebrook
relation, the exponent of the smooth power law, the recovery of ``kappa`` from
synthetic fully-rough data, the GCI arithmetic, and the band bookkeeping of the
propagation verdict.  The campaign data themselves are produced by
``tools/run_ice_rough_cw_adjudication.py`` and adjudicated in
``artifacts/ice_rough_cw_c/``; no test here needs a simulation.
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.ice_roughness import freedrift as fd
from tensorlbm.ice_roughness import propagation as pg
from tensorlbm.ice_roughness import regime_law as rl
from tensorlbm.ice_roughness import three_zone as tz


# ---------------------------------------------------------------------------
# lens C: Colebrook-White
# ---------------------------------------------------------------------------

def test_colebrook_recovers_blasius_in_the_smooth_limit():
    """k_s -> 0 must give the smooth-duct law (0.3164 Re^-0.25) to ~2 %."""
    for re_d in (1.0e4, 3.0e4, 1.0e5):
        lam = tz.colebrook_darcy(re_d, 0.0)
        blasius = 0.3164 * re_d ** -0.25
        assert lam == pytest.approx(blasius, rel=0.03)


def test_colebrook_fully_rough_limit_is_independent_of_reynolds():
    rel = 0.01
    lam_lo = tz.colebrook_darcy(1.0e5, rel)
    lam_hi = tz.colebrook_darcy(1.0e8, rel)
    assert lam_lo == pytest.approx(lam_hi, rel=0.02)
    # von Karman / Nikuradse fully rough asymptote
    expected = (-2.0 * math.log10(rel / 3.7)) ** -2
    assert lam_hi == pytest.approx(expected, rel=0.02)


def test_colebrook_is_monotone_in_relative_roughness():
    values = [tz.colebrook_darcy(1.0e6, rel) for rel in (0.0, 1e-4, 1e-3, 1e-2, 5e-2)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_colebrook_inversion_round_trip():
    for re_d, rel in ((2.0e5, 3.0e-3), (1.0e5, 1.0e-2), (5.0e5, 1.0e-4)):
        lam = tz.colebrook_darcy(re_d, rel)
        cf = lam / 4.0
        assert tz.ks_over_d_from_cf_colebrook(cf, re_d) == pytest.approx(rel, rel=1e-6)


def test_colebrook_cf_channel_uses_hydraulic_diameter():
    out = tz.colebrook_cf_channel(1.0e5, 0.1, hydraulic_diameter_factor=4.0)
    assert out["Re_D"] == pytest.approx(2.0e5)
    assert out["ks_over_D_h"] == pytest.approx(0.025)
    assert out["C_f"] == pytest.approx(out["darcy_lambda"] / 4.0)


# ---------------------------------------------------------------------------
# lens S: smooth power law exponent (criterion C3)
# ---------------------------------------------------------------------------

def test_smooth_power_law_exponent_matches_the_correlation():
    """The local exponent of the smooth law sits near the classical -0.22."""
    exps = [tz.smooth_power_law_exponent(rt) for rt in (188.0, 423.0, 846.0, 1880.0, 4700.0)]
    assert all(-0.30 < e < -0.14 for e in exps)
    assert exps[0] < exps[-1]          # shallower at higher Reynolds number


def test_secant_exponent_of_the_smooth_series_is_within_the_c3_band():
    """C3 band check on the correlation itself (no LBM needed)."""
    re_b = []
    cf = []
    for re_tau in (188.0, 423.0, 846.0, 1880.0, 4700.0):
        c = rl.cf_channel_smooth_reference(re_tau)
        cf.append(c)
        re_b.append(math.sqrt(2.0 / c) * 2.0 * re_tau)
    fit = tz.fit_power_law(re_b, cf)
    assert abs(fit["n"] - (-0.22)) <= 0.03
    assert fit["n_stderr"] < 0.01


def test_fit_power_law_recovers_a_known_exponent():
    re_b = [1e4, 2e4, 4e4, 8e4]
    cf = [3e-3 * (r / 1e4) ** -0.22 for r in re_b]
    fit = tz.fit_power_law(re_b, cf)
    assert fit["n"] == pytest.approx(-0.22, abs=1e-6)
    assert fit["a"] == pytest.approx(3e-3 * 1e4 ** 0.22, rel=1e-6)


# ---------------------------------------------------------------------------
# lens R: fully rough log law + kappa recovery (criterion C2)
# ---------------------------------------------------------------------------

def test_nikuradse_sand_is_reproduced_by_the_default_k0():
    """k0 = k0_Nik must reproduce Nikuradse's fully-rough additive constant."""
    re_tau, ks_plus = 1000.0, 100.0
    cf = tz.fully_rough_cf(re_tau, ks_plus)
    u_smooth = math.sqrt(2.0 / tz.cf_smooth_lens(re_tau))
    u_rough = math.sqrt(2.0 / cf)
    expected = (1.0 / rl.KAPPA) * math.log(ks_plus / rl.K0_NIKURADSE)
    assert u_smooth - u_rough == pytest.approx(expected, rel=1e-12)


def _synthetic_rows(kappa=0.41, k0=3.0, ks_plus=(90.0, 200.0, 500.0), groups=("2", "5", "10")):
    rows = []
    for g in groups:
        for ks in ks_plus:
            d_u = (1.0 / kappa) * math.log(ks / k0)
            rows.append(tz.CaseRow(
                case_id=f"{g}_{ks:g}", kind="rough", group=g, ks_plus=ks,
                re_tau=ks * 9.4, re_b=1.0e5, cf=0.006, u_plus_bulk=18.0, dy_plus=10.0,
                ks_nodes=5.0, b_fit=5.2 - d_u, d_u_plus=d_u, kappa_self=kappa,
                log_window_points=6))
    return rows


def test_kappa_is_recovered_from_synthetic_fully_rough_data():
    fit = tz.fit_kappa_group_effects(_synthetic_rows(kappa=0.41))
    assert fit["kappa"] == pytest.approx(0.41, abs=1e-6)
    assert fit["passed_band"] is True


def test_kappa_fit_flags_out_of_band_kappa():
    fit = tz.fit_kappa_group_effects(_synthetic_rows(kappa=0.35))
    assert fit["kappa"] == pytest.approx(0.35, abs=1e-6)
    assert fit["passed_band"] is False


def test_kappa_fit_needs_enough_points():
    rows = _synthetic_rows(ks_plus=(90.0,), groups=("5",))
    fit = tz.fit_kappa_group_effects(rows)
    assert fit["passed"] is False
    assert math.isnan(fit["kappa"])


# ---------------------------------------------------------------------------
# C4: GCI
# ---------------------------------------------------------------------------

def test_gci_recovers_second_order_convergence():
    exact, c = 5.0, 0.10
    coarse = exact + c * 0.4 ** 2
    medium = exact + c * 0.2 ** 2
    fine = exact + c * 0.1 ** 2
    out = tz.gci_three_grid(fine, medium, coarse, ratio=2.0)
    assert out["p"] == pytest.approx(2.0, abs=2e-3)
    assert out["gci_fine"] < 0.01
    assert out["passed"] is True


def test_gci_rejects_non_monotone_data():
    out = tz.gci_three_grid(1.0, 1.2, 1.1, ratio=2.0)
    assert out["passed"] is False
    assert math.isnan(out["gci_fine"])


def test_gci_flags_a_large_discretisation_error():
    out = tz.gci_three_grid(5.0, 5.5, 6.5, ratio=2.0)
    assert out["gci_fine"] > 0.05
    assert out["passed"] is False


# ---------------------------------------------------------------------------
# zones and adjudication plumbing
# ---------------------------------------------------------------------------

def test_zone_labels_follow_nikuradse():
    assert tz.zone_of_ks_plus(3.0) == "smooth"
    assert tz.zone_of_ks_plus(40.0) == "transitional"
    assert tz.zone_of_ks_plus(200.0) == "fully_rough"


def test_lens_predictions_order_the_three_relations():
    row = tz.CaseRow(
        case_id="c", kind="rough", group="5", ks_plus=45.0, re_tau=423.0, re_b=4.0e4,
        cf=0.0055, u_plus_bulk=19.0, dy_plus=9.0, ks_nodes=5.0, b_fit=4.0,
        d_u_plus=1.2, kappa_self=0.41, log_window_points=6)
    pred = tz.lens_predictions(row)
    # the rough-lens prediction carries more drag than the smooth lens
    assert pred["cf_lens_fully_rough"] > pred["cf_lens_smooth"]
    assert set(pred) >= {"cf_lens_smooth", "cf_lens_colebrook", "cf_lens_fully_rough"}


def test_rows_from_records_flattens_engine_records():
    records = {
        "smooth_re423": {"kind": "smooth", "case": {"kind": "smooth"},
                         "re_tau": 423.0, "u_bulk": 0.2, "h_eff": 47.0, "nu": 2.2e-5,
                         "cf": 0.005, "u_plus_bulk": 20.1, "dy_plus": 9.0, "ks_plus": 0.0,
                         "ks_nodes": 0.0, "log_fit": {"b": 5.2},
                         "log_fit_free_kappa": {"kappa": 0.40},
                         "log_window": {"n_points": 5}},
        "ribs_ks45_lam5": {"kind": "rough",
                           "case": {"kind": "rough", "lambda_over_ks": 5.0},
                           "re_tau": 423.0, "u_bulk": 0.19, "h_eff": 47.0, "nu": 2.2e-5,
                           "cf": 0.0065, "u_plus_bulk": 17.6, "dy_plus": 9.0, "ks_plus": 45.0,
                           "ks_nodes": 5.0, "log_fit": {"b": 3.9},
                           "log_fit_free_kappa": {"kappa": 0.42},
                           "log_window": {"n_points": 4}},
    }
    rows = tz.rows_from_records(records)
    by_id = {r.case_id: r for r in rows}
    assert by_id["ribs_ks45_lam5"].group == "5"
    assert by_id["ribs_ks45_lam5"].d_u_plus == pytest.approx(20.1 - 17.6, rel=1e-9)
    assert by_id["ribs_ks45_lam5"].kappa_self == pytest.approx(0.42)
    assert by_id["smooth_re423"].re_b == pytest.approx(0.2 * 2 * 47.0 / 2.2e-5)


def test_adjudicate_structures_the_verdict():
    rows = [tz.CaseRow(case_id=f"s{rt:g}", kind="smooth", group="smooth", ks_plus=0.0,
                       re_tau=rt, re_b=math.sqrt(2.0 / rl.cf_channel_smooth_reference(rt)) * 2 * rt,
                       cf=rl.cf_channel_smooth_reference(rt), u_plus_bulk=18.0,
                       dy_plus=10.0, ks_nodes=0.0, b_fit=5.2, d_u_plus=0.0,
                       kappa_self=0.41, log_window_points=5)
            for rt in (188.0, 423.0, 846.0, 1880.0, 4700.0)]
    rows += _synthetic_rows()
    out = tz.adjudicate(rows)
    assert out["C1_smooth_lens"]["passed"] is True           # exact by construction
    assert out["C1_smooth_lens"]["max_abs_rel_error"] < 1e-12
    assert out["C3_smooth_zone_exponent"]["passed"] is True
    assert set(out["lenses"]["zone_adjudication"]) == {"smooth", "transitional", "fully_rough"}
    for zone in out["lenses"]["zone_adjudication"].values():
        assert "passed" in zone


# ---------------------------------------------------------------------------
# C5: propagation
# ---------------------------------------------------------------------------

def _law(k0=3.6, lam=5.0):
    return rl.ThreeZoneLaw(lambda_over_ks=lam, k0=k0, k0_source="test")


def test_cw_table_is_monotone_and_flags_extrapolation():
    laws = {2.0: _law(lam=2.0), 5.0: _law(lam=5.0), 10.0: _law(lam=10.0)}
    table = pg.cw_table(laws, ks_m=(0.05, 0.2, 0.5), lambda_over_ks=(5.0,))
    cws = [r["C_w"] for r in table]
    assert all(b > a for a, b in zip(cws, cws[1:]))          # rougher -> more drag
    assert all(r["extrapolated_in_lambda"] is False for r in table)
    far = pg.cw_table(laws, ks_m=(0.05,), lambda_over_ks=(50.0,))
    assert far[0]["extrapolated_in_lambda"] is True


def test_propagation_verdict_reports_bands_and_honest_fail():
    laws = {2.0: _law(lam=2.0), 5.0: _law(lam=5.0), 10.0: _law(lam=10.0)}
    out = pg.propagation_verdict(laws, wind_speeds=(5.0,))
    assert out["honest_fail"] is (not out["passed"])
    assert len(out["rows"]) == 1
    row = out["rows"][0]
    for key in ("speed_change_in_band", "angle_change_in_band",
                "in_observed_angle_band", "in_observed_wind_factor_band"):
        assert isinstance(row[key], bool)
    # the deflection must follow the drag: more C_w -> straighter, less -> wider
    if row["closure"]["C_w"] > row["constant"]["C_w"]:
        assert row["closure"]["drift_angle_deg"] < row["constant"]["drift_angle_deg"]
    else:
        assert row["closure"]["drift_angle_deg"] > row["constant"]["drift_angle_deg"]


def test_nansen_bands_are_consistent_with_the_free_drift_balance():
    laws = {5.0: _law()}
    rows = pg.nansen_bands(laws, wind_speeds=(5.0,))
    assert rows
    for r in rows:
        assert r["closure_wind_factor"] == pytest.approx(
            r["closure_drift_m_s"] / 5.0, rel=1e-12)
        assert r["angle_change_deg"] == pytest.approx(
            r["closure_angle_deg"] - r["constant_angle_deg"], abs=1e-12)


def test_constant_entry_is_the_m5682_value():
    """The replaced constant must be the m5682 entry, and the smooth reference too."""
    assert pg.M5682_CONSTANT_CW == pytest.approx(5.5e-3)
    assert fd.CW_SMOOTH_REFERENCE == pytest.approx(5.5e-3)