"""Regression guard for the taylor_aris_dispersion caliber audit (m16287).

The audit (benchmarks/pending/taylor_aris_dispersion/caliber_audit.py) is
read-only post-processing of archived artifacts plus an independent 1-D
derivation.  These tests pin the three things that can silently rot:

1. the independent Taylor-Aris constant reproduces 1/210 (the case's theory);
2. every archived residual can be recomputed from deff_sim/deff_theory and the
   *unit convention* of the archived err_pct column holds (percent, not
   fraction -- the audit's first version got this wrong by 100x);
3. the stated rulings follow from the archived data (tolerance 6/6 PASS,
   pre-registered monotonicity FAIL, amended saturation PASS, floor claim
   REFUTED by the case's own pure-diffusion control).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1] / "benchmarks" / "pending" / "taylor_aris_dispersion"


def _mod():
    if not (CASE / "caliber_audit.py").exists():
        pytest.skip("taylor_aris caliber audit not present")
    spec = importlib.util.spec_from_file_location("ta_caliber_audit", CASE / "caliber_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def audit():
    m = _mod()
    result = json.loads((CASE / "result.json").read_text())
    coef = m.coefficient_convergence()
    rows = m.recompute_cases(result)
    stats = m.residual_stats(rows)
    ctrl = m.control_evidence(m.readme_evidence())
    rules = {r["gate"]: r for r in m.rulings(rows, stats, ctrl)}
    return {
        "m": m,
        "result": result,
        "coef": coef,
        "rows": rows,
        "stats": stats,
        "ctrl": ctrl,
        "rules": rules,
    }


def test_independent_taylor_aris_constant_is_one_over_210(audit):
    """The slit cell problem must give K = 1/210 (Deff/D = 1 + Pe^2 K)."""
    finest = audit["coef"][-1]
    assert finest["rel_dev_vs_1_210"] == pytest.approx(0.0, abs=1e-9)
    assert abs(finest["neumann_compat"]) < 1e-9
    # trapezoid refinement must be monotone in the deviation magnitude
    devs = [abs(c["rel_dev_vs_1_210"]) for c in audit["coef"]]
    assert devs[0] > devs[1] > devs[2] > 0.0


def test_every_archived_residual_recomputes_and_units_are_percent(audit):
    rows = audit["rows"]
    assert len(rows) == 6
    for r in rows:
        assert r["recompute_matches_archived"], r
        # err_pct column is a PERCENTAGE: recomputed percent must match it, and
        # the fractional deviation must match it/100 (guards the 100x trap).
        frac = r["deff_sim"] / r["deff_theory"] - 1.0
        assert 100.0 * frac == pytest.approx(r["resid_pct_signed"], rel=1e-12, abs=1e-12)
        assert abs(frac) == pytest.approx(r["err_pct_archived_abs"] / 100.0, rel=1e-9)


def test_tolerance_gate_passes_with_large_margin(audit):
    stats = audit["stats"]
    assert stats["max_abs_pct"] < 0.5
    assert stats["tolerance_margin_x"] > 10.0
    assert all(r["pass_tol"] for r in audit["rows"])
    assert audit["rules"]["G1_tolerance"]["verdict"] == "PASS"


def test_preregistered_monotonicity_genuinely_fails(audit):
    """Not a bug we may paper over: the pre-registered gate fails 0/3."""
    assert audit["result"]["all_monotone"] is False
    # the pre-registered criterion is |err(H=128)| < |err(H=64)| per Pe level
    ge = audit["stats"]["per_grid"]["grid_effect_by_pe"]
    assert all(not (abs(v["Hmax_pct"]) < abs(v["Hmin_pct"])) for v in ge.values())
    assert audit["rules"]["G2_prereg_monotone"]["verdict"] == "FAIL"


def test_residual_is_scatter_not_a_systematic_bias(audit):
    stats = audit["stats"]
    assert stats["all_same_sign"] is False
    assert abs(stats["t_stat"]) < stats["t_crit_95_5dof"]
    assert audit["rules"]["G4_systematic_bias"]["verdict"] == "SCATTER"


def test_amended_saturation_caliber_passes_without_tuning(audit):
    """G2' thresholds are 1/3 of the pre-registered tolerance; data clear it 3.8x."""
    stats = audit["stats"]
    worst_delta = max(abs(v["delta_pct"]) for v in stats["per_grid"]["grid_effect_by_pe"].values())
    assert stats["max_abs_pct"] <= 1.0
    assert abs(stats["mean_pct"]) <= 1.0
    assert worst_delta <= 1.0
    assert audit["rules"]["G2p_amended_saturation"]["verdict"] == "PASS"


def test_floor_root_cause_is_refuted_by_the_cases_own_control(audit):
    """README claimed a ~0.2% positive error floor; the Pe=0 control says 0.0049%."""
    ctrl = audit["ctrl"]
    assert ctrl["v4_pure_diffusion_H64_err_pct"] < 0.05 * 0.2      # >=20x below the claim
    assert ctrl["v3_synthetic_bias_max"] < 1e-12                   # moment estimator exact
    # and the H-dependence survives in fp64 -> neither rounding nor measurement
    assert ctrl["diag_fp64_pe20_H128_err_pct"] > ctrl["diag_fp64_pe20_H64_err_pct"] > 0.0
    assert audit["rules"]["G3_mechanism"]["verdict"] == "REFUTED"


def test_generated_artifact_agrees_with_live_recompute(audit):
    p = CASE / "out" / "caliber_audit.json"
    if not p.exists():
        pytest.skip("run `python caliber_audit.py --write` first")
    saved = json.loads(p.read_text())
    live = audit["stats"]
    assert saved["stats"]["max_abs_pct"] == pytest.approx(live["max_abs_pct"], rel=1e-12)
    assert saved["stats"]["mean_pct"] == pytest.approx(live["mean_pct"], rel=1e-12)
    assert {r["gate"]: r["verdict"] for r in saved["rulings"]} == {
        k: v["verdict"] for k, v in audit["rules"].items()
    }
    for f in saved["figures"]:
        assert (CASE / "out" / f).exists(), f