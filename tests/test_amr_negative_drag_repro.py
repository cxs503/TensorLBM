"""Unit tests for the AMR negative-drag reproduction harness (paper-exp-lbm-suboff-tau-4).

The tests pin the two things that must never drift once the criteria are
frozen: (a) the arm table's Reynolds matching arithmetic, and (b) the verdict
logic (reproduction tolerance, falsification arm count, honest NOT_EVALUATED
when arms are missing).  No solver is run here — the physics arms are the
artifact's job; these tests protect the bookkeeping.

Run:  PYTHONPATH=src python -m pytest tests/test_amr_negative_drag_repro.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "benchmarks/verified/suboff_sail_resistance/run_amr_negative_drag_repro.py"


@pytest.fixture(scope="module")
def repro():
    spec = importlib.util.spec_from_file_location("amr_negative_drag_repro", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["amr_negative_drag_repro"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Arm table invariants
# --------------------------------------------------------------------------- #
def test_historical_ladder_is_verbatim(repro):
    """The four historical rungs and their D_QW values appear as recorded."""
    taus = [rung["tau"] for rung in repro.HISTORICAL_LADDER]
    assert taus == [0.50036, 1.0, 1.5, 4.0]
    dqw = [rung["D_QW"] for rung in repro.HISTORICAL_LADDER]
    assert dqw[0] == pytest.approx(-3.20e-3)
    assert dqw[-1] == pytest.approx(4.66e-4)
    # one sign flip only, and the record quotes carry line numbers
    signs = [rung["sign"] for rung in repro.HISTORICAL_LADDER]
    assert signs == ["negative", "negative", "positive", "positive"]
    assert all("line" in quote and "text" in quote for quote in repro.HISTORICAL_QUOTES)


def test_matched_arms_are_exactly_reynolds_matched(repro):
    """Each matched arm's (u_lb, tau) pair must reproduce RE_MATCH bit-for-bit."""
    matched = [a for a in repro.arm_definitions() if a["protocol"] == "matched"]
    assert len(matched) >= repro.FALSIFY_MIN_ARMS
    for arm in matched:
        nu = (arm["tau_target"] - 0.5) / 3.0
        re_implied = arm["u_lb"] * repro.HULL_LEN_COARSE / nu
        assert re_implied == pytest.approx(arm["re_target"], rel=1e-12)
        # tau stays well inside the stable BGK window
        assert 0.5 < arm["tau_target"] < 1.0


def test_replay_arms_hold_speed_and_walk_the_ladder(repro):
    replay = [a for a in repro.arm_definitions() if a["protocol"] == "replay"]
    assert [a["tau_target"] for a in replay] == [r["tau"] for r in repro.HISTORICAL_LADDER]
    assert all(a["u_lb"] == repro.REPLAY_U_LB for a in replay)
    assert all(a["steps"] > 0 for a in replay)


def test_step_budget_respects_floor_and_cap(repro):
    assert repro._steps_for(1e-3) == repro.MAX_STEPS
    assert repro._steps_for(1.0) == repro.MIN_STEPS
    assert repro._steps_for(0.05, viscous_floor=True) == repro.MIN_STEPS


def test_matched_re_infeasibility_flags_the_historical_ladder(repro):
    infeas = repro.matched_re_infeasibility()
    rows = {row["tau"]: row for row in infeas["rows"]}
    # near-singular production point: speed needed is below the safe floor
    assert rows[0.50036]["u_lb_required"] < repro.U_MIN_SAFE
    # super-viscous rungs: speed needed blows past the Mach window
    assert rows[4.0]["u_lb_required"] > repro.U_MAX_SAFE
    assert infeas["feasible_taus"] == []
    assert len(infeas["rows"]) == len(repro.HISTORICAL_LADDER)


# --------------------------------------------------------------------------- #
# Verdict logic
# --------------------------------------------------------------------------- #
def _fake_result(arm, *, D_QW, mem=1.0, status="ok", re_matched=True):
    re_lb = arm["u_lb"] * 56.0 / ((arm["tau_target"] - 0.5) / 3.0)
    if arm["protocol"] == "matched":
        re_lb = arm["re_target"]
    return {
        "arm": arm["arm"],
        "protocol": arm["protocol"],
        "status": status,
        "u_lb": arm["u_lb"],
        "reynolds_coarse": re_lb,
        "re_matched": re_matched,
        "steps_completed": arm["steps"],
        "flow_passes": 1.25,
        "window": {
            "mean": {
                "D_QW": D_QW,
                "mem_fx": mem,
                "mem_bgsub_fx": mem,
                "ct": 0.5,
                "ct_bgsub": 0.5,
                "rho_fwd": 1.0 + D_QW / 2,
                "rho_mid": 1.0,
                "rho_aft": 1.0 - D_QW / 2,
            },
            "drift_pct": {"D_QW": 0.1, "mem_bgsub_fx": 0.1, "ct_bgsub": 0.1},
        },
    }


def _all_results(repro, *, prod_D_QW, matched_D_QW=1e-4, matched_mem=1.0):
    results = {}
    for arm in repro.arm_definitions():
        if arm["arm"].startswith("replay_tau_0.50036"):
            dqw = prod_D_QW
        elif arm["protocol"] == "matched":
            dqw = matched_D_QW
        else:
            dqw = 1e-5
        mem = matched_mem if arm["protocol"] == "matched" else 5.0
        results[arm["arm"]] = _fake_result(arm, D_QW=dqw, mem=mem)
    return results


def test_reproduction_pass_uses_frozen_tolerance(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    results = _all_results(repro, prod_D_QW=anchor * (1.0 - 0.5 * repro.REPRO_TOL))
    ev = repro.evaluate(results)
    assert ev["R1_reproduction"]["status"] == "PASS"
    assert ev["R1_reproduction"]["tolerance_rel"] == repro.REPRO_TOL
    assert ev["R1_reproduction"]["abs_relative_deviation_pct"] <= 10.0


def test_reproduction_fails_when_sign_flips(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    results = _all_results(repro, prod_D_QW=-anchor)  # same magnitude, wrong sign
    ev = repro.evaluate(results)
    assert ev["R1_reproduction"]["sign_match"] is False
    assert ev["R1_reproduction"]["status"] == "FAIL"


def test_reproduction_fails_outside_tolerance(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    results = _all_results(repro, prod_D_QW=anchor * (1.0 + 2 * repro.REPRO_TOL))
    ev = repro.evaluate(results)
    assert ev["R1_reproduction"]["status"] == "FAIL"
    assert ev["R1_reproduction"]["magnitude_match"] is False


def test_falsified_needs_three_positive_matched_arms(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    # pathology reproduces in sign but not magnitude -> R1 FAIL, F2 FALSIFIED
    results = _all_results(repro, prod_D_QW=anchor * 5.0, matched_mem=+2.0)
    ev = repro.evaluate(results)
    assert ev["F2_falsification"]["usable_matched_arms"] >= repro.FALSIFY_MIN_ARMS
    assert ev["F2_falsification"]["positive_drag_arms"] == ev["F2_falsification"]["usable_matched_arms"]
    assert ev["F2_falsification"]["status"] == "FALSIFIED"
    assert ev["overall"].startswith("FALSIFIED")


def test_negative_matched_drag_prevents_falsification(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    results = _all_results(repro, prod_D_QW=anchor, matched_mem=-1.0)
    ev = repro.evaluate(results)
    assert ev["F2_falsification"]["status"] != "FALSIFIED"


def test_missing_arms_are_reported_not_guessed(repro):
    ev = repro.evaluate({})
    assert ev["R1_reproduction"]["status"] == "NOT_EVALUATED"
    assert ev["F2_falsification"]["status"] == "NOT_EVALUATED"
    assert "HONEST-FAIL" in ev["overall"]


def test_usable_arms_require_ok_status_and_re_match(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    results = _all_results(repro, prod_D_QW=anchor)
    # break two matched arms: one not Re-matched, one diverged
    matched = [a for a in repro.arm_definitions() if a["protocol"] == "matched"]
    results[matched[0]["arm"]] = _fake_result(matched[0], D_QW=1e-4, re_matched=False)
    results[matched[1]["arm"]] = _fake_result(matched[1], D_QW=1e-4, status="diverged")
    ev = repro.evaluate(results)
    f2 = ev["F2_falsification"]
    assert f2["usable_matched_arms"] == 1
    assert f2["status"] != "FALSIFIED"


def test_conclusion_text_reports_both_branches(repro):
    anchor = repro.HISTORICAL_LADDER[0]["D_QW"]
    results = _all_results(repro, prod_D_QW=anchor, matched_mem=+1.0)
    ev = repro.evaluate(results)
    text = repro._conclusion_text(ev)
    assert "R1" in text and "F2" in text
    assert "Matched-Re reachability" in text