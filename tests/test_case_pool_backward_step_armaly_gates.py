"""Gate-provenance tests for the backward-facing-step pool case (CASE-BACKSTEP-FVM-RE100).

Why these exist
---------------
Round 3 of the case shipped ``verdict.py`` with a hardcoded ``TOLERANCE_PCT = 15.0``
while the machine referee row registered for the anchor
(``tensor_benchmark_cases``, anchor_slug ``CASE-BACKSTEP-FVM-RE100``) tolerates 10 %.
A point 12 % away from the reference would therefore have been called PASS by the
verdict and FAIL by the referee -- the verdict was not the referee's verdict.

These tests pin the contract on synthetic artefacts only (they need no cluster run):
the tolerance must come from ``refs.json`` and a point that fails the registered gate
must be reported as failing.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "case_pool" / "backward_step_armaly"
REFS = json.loads((CASE / "refs.json").read_text())

sys.path.insert(0, str(CASE))
verdict = importlib.import_module("verdict")


def _run(x1: float, *, re: float = 100.0, steady: bool = True, mass: float = 1e-4,
         drift_pct: float = 0.0) -> dict:
    """Minimal synthetic result.json payload accepted by verdict.judge_run."""
    hi = 26
    s = 25
    return {
        "config": {"re": re, "hi": hi, "s": s, "u_bar": 0.1},
        "derived": {"tau": 0.656, "mach_inlet": 0.0173,
                    "expansion_ratio": (hi + s) / hi,
                    "expansion_ratio_deviation_pct": 0.99},
        "result": {"x1_over_s": x1, "x1_over_s_reverse_extent": x1 - 0.2,
                   "x1_over_s_row1": x1, "mass_flux_rel_error": mass,
                   "steps_run": 88000, "steady_at_step": 88000 if steady else None,
                   "average_window_steps": 40000, "n_averaged_samples": 200},
        "convergence_history": [{"step": 88000 - 200 * (2 - i),
                                 "x1_over_s_instant": x1 * (1.0 + drift_pct / 100.0 * i)}
                                for i in range(3)],
    }


def test_primary_tolerance_is_read_from_refs_not_hardcoded():
    tols = verdict.tolerances(REFS)
    assert tols["primary_pct"] == REFS["adopted_numeric_reference"]["ledger_tolerance_pct"] == 10.0
    assert "ledger_tolerance_pct" in tols["primary_source"]
    # the registered cross-check anchor at Re=100 carries its own tolerance
    assert tols["cross_check"][100.0]["value"] == pytest.approx(2.878)
    assert tols["cross_check"][100.0]["tolerance_pct"] == pytest.approx(5.0)


def test_twelve_percent_point_fails_the_registered_gate():
    """12 % off the Armaly anchor: inside the old hardcoded 15 %, outside the 10 % gate."""
    x1_ref = float(REFS["adopted_numeric_reference"]["points"]["100"]["x1_over_s"])
    row = verdict.judge_run("synthetic_12pct", _run(1.12 * x1_ref), REFS)
    assert abs(row["error_pct"]) == pytest.approx(12.0, abs=1e-9)
    assert row["checks"]["error"] is False
    assert row["pass"] is False
    assert "error" in row["note_missing"]


def test_cross_check_anchor_is_enforced_independently():
    """2.92 is inside 10 % of Armaly 3.05 but the 2.878 cross-check must be reported too."""
    ok = verdict.judge_run("synthetic_ok", _run(2.920), REFS)
    assert ok["cross_check"]["value"] == pytest.approx(2.878)
    assert ok["checks"]["error"] is True
    assert ok["checks"]["cross_check"] is True
    assert ok["pass"] is True

    # 3.05 (the Armaly anchor itself) is +5.98 % from the 2-D numerical anchor -> cross-check FAIL
    exp = verdict.judge_run("synthetic_armaly_exact", _run(3.05), REFS)
    assert exp["checks"]["error"] is True           # 0 % from the experimental anchor
    assert exp["checks"]["cross_check"] is False    # but the 2-D anchor disagrees by ~6 %
    assert "cross_check" in exp["note_missing"]


def test_transient_point_never_passes():
    # not self-stopped and still drifting 5 % over the last three checks -> transient
    row = verdict.judge_run("synthetic_transient",
                            _run(2.920, steady=False, drift_pct=5.0), REFS)
    assert row["checks"]["steady"] is False
    assert row["pass"] is False


def test_evaluate_is_data_driven_and_all_judged_pass_tracks_the_gate(tmp_path):
    """End-to-end over a synthetic repo layout: one good point -> PASS, one 12 % point -> FAIL."""
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "case_pool").symlink_to(REPO / "benchmarks" / "case_pool")
    art = tmp_path / "artifacts" / "case_backward_step_armaly"

    art.mkdir(parents=True)
    (art / "h_good").mkdir()
    (art / "h_good" / "result.json").write_text(json.dumps(_run(2.920)))
    v = verdict.evaluate(repo=tmp_path)
    assert v["tolerance_pct"] == 10.0
    assert v["n_runs"] == 1 and v["all_judged_pass"] is True
    assert v["max_abs_error_pct"] <= 10.0

    (art / "h_twelve").mkdir()
    x1_ref = float(REFS["adopted_numeric_reference"]["points"]["100"]["x1_over_s"])
    (art / "h_twelve" / "result.json").write_text(json.dumps(_run(1.12 * x1_ref)))
    v2 = verdict.evaluate(repo=tmp_path)
    assert v2["n_runs"] == 2 and v2["all_judged_pass"] is False
    assert v2["n_pass"] == 1
    bad = next(r for r in v2["runs"] if r["tag"] == "h_twelve")
    assert bad["checks"]["error"] is False