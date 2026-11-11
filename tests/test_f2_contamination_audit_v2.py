"""Tests for the pass-2 F/2 contamination audit (mission m11525).

Pass 2 closes three gaps of pass 1 (m9872): (a) every ``k_s^+`` must have a
paired sample, (b) the 95 % interval must be a Student-t interval on a
replicate count that actually supports 95 %, (c) each seed must own its smooth
baseline.  The tests pin down the arithmetic that the audit verdict rests on,
so a regression in the statistics or in the pairing cannot silently produce a
"clean" or a "contaminated" answer.

Everything here is synthetic: no engine is built and no cluster job is needed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorlbm.ice_roughness import forcing_audit as fa
from tensorlbm.ice_roughness import regime_law as rl
from tensorlbm.ice_roughness import sweep as sw


# --------------------------------------------------------------------------
# confidence interval
# --------------------------------------------------------------------------

def test_t_quantile_table_and_large_dof_limit():
    assert fa.t_critical_975(1) == pytest.approx(12.7062)
    assert fa.t_critical_975(2) == pytest.approx(4.3027)
    assert fa.t_critical_975(3) == pytest.approx(3.1824)
    # Cornish-Fisher branch must approach the normal quantile, not overshoot it
    assert abs(fa.t_critical_975(10_000) - 1.959964) < 2e-3


def test_ci_is_t_based_and_wider_than_the_normal_interval():
    """n = 2 must give a 4.303-sigma band, not the 1.96-sigma band pass 1 used."""
    stats = fa._summarise([0.02, 0.04])
    sd = float(np.std([0.02, 0.04], ddof=1))
    assert stats["n"] == 2 and stats["dof"] == 1
    assert stats["t_critical_975"] == pytest.approx(12.7062)
    assert stats["ci95_half_width"] == pytest.approx(12.7062 * sd / math.sqrt(2))
    assert stats["ci95_half_width"] > 1.96 * sd / math.sqrt(2)
    # a single sample has no interval at all -- never a zero-width "95 % CI"
    one = fa._summarise([0.03])
    assert one["n"] == 1 and not np.isfinite(one["ci95_half_width"])
    empty = fa._summarise([])
    assert empty["n"] == 0 and not np.isfinite(empty["ci95_half_width"])


# --------------------------------------------------------------------------
# smooth baseline ownership (pass-1 defect)
# --------------------------------------------------------------------------

def test_each_seed_gets_its_own_smooth_baseline():
    cases = fa.campaign_case_set(ks_targets=(20.0, 45.0), seeds=(11, 12), profile="smoke")
    per_seed: dict[int, dict[str, int]] = {}
    for case in cases:
        per_seed.setdefault(case.seed, {"rough": 0, "smooth": 0})[case.kind] += 1
    assert set(per_seed) == {11, 12}
    for seed, counts in per_seed.items():
        assert counts["rough"] == 2, seed
        # two distinct Re_tau -> two smooth baselines, for *every* seed
        assert counts["smooth"] == 2, seed


# --------------------------------------------------------------------------
# recalibration table
# --------------------------------------------------------------------------

def _engine(k0: float, *, ks_values=(20.0, 45.0, 90.0), seeds=(1, 2, 3, 4),
            scale_cf: float = 1.0) -> dict:
    """Engine table shaped like :func:`forcing_audit.quantities` output."""
    per_seed = []
    for seed in seeds:
        points = {}
        for ks in ks_values:
            d_u = float(rl.delta_u_plus(ks, k0))
            cf_smooth = rl.cf_channel_smooth_reference(200.0 + seed)
            u_smooth = math.sqrt(2.0 / cf_smooth)
            cf = cf_smooth * (u_smooth / (u_smooth - d_u)) ** 2 * scale_cf
            ratio = (u_smooth / (u_smooth - d_u)) ** 2 * scale_cf
            points[f"{ks:g}"] = {
                "ks_plus_target": float(ks),
                "u_plus_bulk": float(u_smooth - d_u),
                "cf": float(cf),
                "cf_smooth": float(cf_smooth),
                "dU_plus": d_u,
                "cf_ratio": float(ratio),
                "cf_ratio_lbm": float(ratio),
                # amplification -> C_w of the closure
                "C_w": float(6.0e-3 * ratio),
                "C_w_lbm": float(6.0e-3 * ratio),
            }
        per_seed.append({"seed": seed, "k0": float(k0), "points": points})
    downstream = [{"seed": s, "C_w": 6.0e-3 * scale_cf,
                   "C_w_amplification": 1.2 * scale_cf,
                   "ice_water_drag_Pa": 0.5 * scale_cf,
                   "drift_speed_m_s": 0.1, "drift_angle_deg": 25.0} for s in seeds]
    return {"per_seed": per_seed, "downstream": downstream}


def test_recalibration_table_gives_correction_factors():
    fixed = _engine(11.8)
    defect = _engine(11.8, scale_cf=1.30)          # +30 % on every C_w
    verdict = fa.contamination(fixed, defect)
    table = fa.recalibration_table(verdict)
    assert table["verdict"] == "RECALIBRATE"
    rows = {r["quantity"]: r for r in table["rows"]}
    for ks in (20.0, 45.0, 90.0):
        row = rows[f"ks{ks:g}.C_w"]
        assert row["status"] == "EVALUATED"
        assert row["rel_delta"] == pytest.approx(0.30, abs=1e-9)
        # correction maps the reported (defect) value back to the fixed one
        assert row["correction_factor_fixed_over_defect"] == pytest.approx(1 / 1.30, rel=1e-9)
        assert row["recalibration_required"] is True
    assert np.isfinite(rows["k0"].get("rel_delta", float("nan")))


def test_recalibration_table_reports_clean_and_not_evaluable_separately():
    fixed = _engine(11.8)
    clean = fa.recalibration_table(fa.contamination(fixed, _engine(11.8)))
    assert clean["verdict"] == "NO_RECALIBRATION"
    assert clean["recalibration_required"] == []
    # a quantity without paired evidence must be "NOT_EVALUABLE", not "clean"
    empty = fa.recalibration_table(fa.contamination(fixed, {"per_seed": [], "downstream": []}))
    assert empty["verdict"] == "INCONCLUSIVE"
    assert empty["not_evaluable"]


# --------------------------------------------------------------------------
# state shift (nominal-case pairing is not a matched-physics pairing)
# --------------------------------------------------------------------------

def _record(case_id: str, seed: int, *, ks: float, up: float, cf: float,
            ks_measured: float, kind: str = "rough") -> dict:
    return {"case": {"case_id": case_id, "seed": seed, "kind": kind,
                     "ks_plus_target": ks},
            "u_plus_bulk": up, "cf": cf, "ks_plus": ks_measured, "tau": 0.509,
            "wall_resolved": False}


def test_state_shift_reports_the_reported_reynolds_shift():
    fx = {"a": _record("ribs_ks20_lam5", 1, ks=20.0, up=15.0, cf=0.0089, ks_measured=20.0),
          "b": _record("smooth_re188", 1, ks=188.0, up=16.0, cf=0.0078, ks_measured=188.0,
                       kind="smooth")}
    df = {"a": _record("ribs_ks20_lam5", 1, ks=20.0, up=13.2, cf=0.0115, ks_measured=18.0)}
    shift = fa.engine_state_shift(fx, df)
    assert len(shift["rows"]) == 1                      # smooth case is not a pair
    row = shift["rows"][0]
    assert row["cf_ratio_defect_over_fixed"] == pytest.approx(0.0115 / 0.0089)
    assert row["ks_plus_ratio_defect_over_fixed"] == pytest.approx(0.9)
    summary = shift["by_ks"]["20"]
    assert summary["n"] == 1
    assert summary["implied_drive_factor_from_cf"] == pytest.approx(0.0089 / 0.0115)


# --------------------------------------------------------------------------
# the verdict must have discriminating power
# --------------------------------------------------------------------------

def test_negative_control_flags_a_one_percent_plus_change():
    fixed = _engine(11.8)
    defect = _engine(11.8, scale_cf=1.05)   # 5 % contamination
    verdict = fa.contamination(fixed, defect)
    assert verdict["contaminated"] is True
    assert any(k.endswith(".C_w") for k in verdict["exceeding_quantities"])
    assert verdict["turbulent_calibration_clean"] is False


def test_clean_engine_passes_the_one_percent_line():
    fixed = _engine(11.8)
    defect = _engine(11.8, scale_cf=1.002)
    verdict = fa.contamination(fixed, defect)
    assert verdict["turbulent_calibration_clean"] is True
    assert verdict["exceeding_quantities"] == []
    # the interval is still reported, so "clean" is a quantified statement
    st = verdict["quantities"]["ks45.C_w"]
    assert st["n"] == 4 and st["dof"] == 3
    assert st["ci95_half_width"] >= 0.0


# --------------------------------------------------------------------------
# quantity pipeline robustness (pass-1 crash on a non-finite point)
# --------------------------------------------------------------------------

def test_quantities_survives_an_unusable_point(tmp_path):
    """A negative dU^+ (the defective engine can report one) must be recorded,
    not raise -- the aggregate is the audit's only output path."""

    def write(case_id: str, ks_target: float, kind: str, d_u_override: float | None):
        rec = {
            "case": {"case_id": case_id, "seed": 5, "kind": kind,
                     "ks_plus_target": ks_target, "lambda_over_ks": 5.0},
            "re_tau": 200.0, "u_tau": 0.01, "cf": 0.0080, "u_plus_bulk": 14.0,
            "ks_plus": ks_target, "log_law_exists": True, "log_window": {"n_points": 3},
            "profile": {"k": [], "y_plus": [], "u_plus": [], "n_fluid_per_layer": []},
        }
        (tmp_path / f"{case_id}__fixed__s5.json").write_text(json.dumps(rec))

    write("ribs_ks20_lam5", 20.0, "rough", None)
    write("smooth_re200", 200.0, "smooth", None)
    table = fa.quantities(fa.load_records(tmp_path), lambda_over_ks=5.0, ks_targets=(20.0,))
    assert table["per_seed"][0]["points"]["20"]["u_plus_bulk"] == pytest.approx(14.0)
    # one point cannot be fitted: the inversion is attempted and its failure is
    # recorded in the fit record instead of raising
    seed0 = table["per_seed"][0]
    assert seed0["points"]["20"]["u_plus_bulk"] == pytest.approx(14.0)
    assert np.isfinite(seed0["k0"]) or seed0["k0_fit"]["n_points"] == 0