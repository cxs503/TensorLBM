"""CASE-BACKWARD-STEP-LBM verdict gate -- asserts the acceptance *conditions*, not the code.

Round 2 of this case could not be judged because no converged point existed.  The gate
implemented in ``benchmarks/case_pool/backward_step_armaly/verdict.py`` decomposes the
anchor into five independent conditions (error / steady / caliper / mass / lattice) so a
point may not pass by accident.  These tests pin the gate's own logic on synthetic
artefacts (always run) and re-judge the committed ladder artefacts (skipped when the
artefacts are absent, e.g. on a clean checkout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "case_pool" / "backward_step_armaly"))

import verdict as V  # noqa: E402

ART = REPO / "artifacts" / "case_backward_step_armaly"


def _run(tag="synthetic_re100", x1=3.02, ext=2.80, steady=None, hist=None, mass=1e-6,
         re=100.0, hi=52, s=49, tau=0.712, mach=0.26, steps=50000):
    hist = hist if hist is not None else [
        {"step": steps - 8000, "x1_over_s_instant": 3.01, "max_speed": 1.0, "mean_rho": 1.0, "wall_time_s": 1.0},
        {"step": steps - 4000, "x1_over_s_instant": 3.02, "max_speed": 1.0, "mean_rho": 1.0, "wall_time_s": 1.0},
        {"step": steps, "x1_over_s_instant": 3.02, "max_speed": 1.0, "mean_rho": 1.0, "wall_time_s": 1.0},
    ]
    return {
        "config": {"re": re, "hi": hi, "s": s, "tag": tag},
        "derived": {"tau": tau, "mach_inlet": mach, "expansion_ratio": (hi + s) / hi,
                    "expansion_ratio_deviation_pct": 0.0},
        "result": {"x1_over_s": x1, "x1_over_s_reverse_extent": ext, "x1_over_s_row1": x1,
                   "x1_over_H": x1 / (hi + s), "steps_run": steps, "steady_at_step": steady,
                   "n_averaged_samples": 25, "average_window_steps": [25000, steps],
                   "mass_flux_rel_error": mass},
        "convergence_history": hist,
    }


REFS = {"adopted_numeric_reference": {"points": {
    "100": {"x1_over_s": 3.05}, "200": {"x1_over_s": 5.0}, "400": {"x1_over_s": 8.0}}}}


def test_converged_point_passes_all_conditions():
    row = V.judge_run("synthetic_re100", _run(), REFS)
    assert row["checks"] == {"error": True, "steady": True, "caliper": True,
                             "mass": True, "lattice": True}
    assert row["pass"] is True
    assert abs(row["error_pct"] - 100.0 * (3.02 - 3.05) / 3.05) < 1e-9


def test_statistically_quiet_but_unsteady_point_is_rejected():
    """The round-2 failure mode: a plausible X1/s from a run that never reached steady state."""
    drifting = [{"step": 10000 + 4000 * i, "x1_over_s_instant": 3.0 + 0.25 * i,
                 "max_speed": 1.0, "mean_rho": 1.0, "wall_time_s": 1.0} for i in range(4)]
    row = V.judge_run("synthetic_re100", _run(hist=drifting), REFS)
    assert row["checks"]["steady"] is False
    assert row["pass"] is False
    assert row["note_missing"] == ["steady"]


def test_error_beyond_tolerance_is_rejected():
    row = V.judge_run("synthetic_re400", _run(tag="synthetic_re400", re=400.0, x1=6.0), REFS)
    assert row["checks"]["error"] is False
    assert row["pass"] is False


def test_caliper_disagreement_is_rejected():
    """reversed-flow extent beyond the shear reattachment point => inconsistent calipers."""
    row = V.judge_run("synthetic_re100", _run(ext=4.2), REFS)
    assert row["checks"]["caliper"] is False


def test_mass_leak_and_lattice_window_are_rejected():
    row = V.judge_run("synthetic_re100", _run(mass=5e-3, tau=0.51, mach=0.45), REFS)
    assert row["checks"]["mass"] is False
    assert row["checks"]["lattice"] is False


def test_self_stop_counts_as_steady_without_history():
    row = V.judge_run("synthetic_re100", _run(steady=41000, hist=[]), REFS)
    assert row["checks"]["steady"] is True


@pytest.mark.skipif(not ART.exists() or not any(ART.glob("*/result.json")),
                    reason="ladder artefacts not committed in this checkout")
def test_committed_ladder_verdict_is_self_consistent():
    v = V.evaluate(REPO)
    assert v["n_runs"] >= 1
    for r in v["runs"]:
        # a point may only be reported as passing when every condition holds
        assert r["pass"] == all(r["checks"].values())
        if r["pass"]:
            assert abs(r["error_pct"]) <= v["tolerance_pct"]
            assert r["steadiness"]["steady_at_step"] is not None or \
                r["steadiness"]["drift_pct"] <= V.STEADY_DRIFT_PCT
    (ART / "verdict.md").read_text() if (ART / "verdict.md").exists() else None