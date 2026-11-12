"""Tests for the TensorLBM validation matrix (framework + provenance + CI case).

These tests are cheap by design: the whole point of the matrix's CI tier is that
the framework, the reference provenance gate and at least one complete case can
be exercised inside a normal test run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MATRIX_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "validation_matrix"
REPO_ROOT = MATRIX_DIR.parents[1]
sys.path.insert(0, str(MATRIX_DIR))

import criteria  # noqa: E402
import reference_audit  # noqa: E402
import runner  # noqa: E402


# --------------------------------------------------------------------------- #
# acceptance rules
# --------------------------------------------------------------------------- #
def test_pct_error_sign_and_value():
    assert criteria.pct_error(1.03, 1.0) == pytest.approx(3.0)
    assert criteria.pct_error(0.97, 1.0) == pytest.approx(-3.0)
    with pytest.raises(criteria.CriterionError):
        criteria.pct_error(1.0, 0.0)


def test_judge_metric_gate_and_kinds():
    ok = criteria.judge_metric(1.02, 1.0, 3.0)
    assert ok["pass"] and ok["err_pct"] == pytest.approx(2.0)
    bad = criteria.judge_metric(1.05, 1.0, 3.0)
    assert not bad["pass"]
    # error_pct keeps the sign but gates on magnitude
    assert criteria.judge_metric(-2.9, None, 3.0, kind="error_pct")["pass"]
    assert not criteria.judge_metric(-3.1, None, 3.0, kind="error_pct")["pass"]
    with pytest.raises(criteria.CriterionError):
        criteria.judge_metric(1.0, None, 3.0, kind="value")


def test_convergence_trend_span_and_monotone():
    good = criteria.convergence_trend([4.0, 2.0, 1.0])
    assert good["pass"] and good["monotone"] and good["require_monotone"]
    drifting = criteria.convergence_trend([1.0, 2.0, 3.0])
    assert not drifting["pass"]  # monotone violation + span 2 <= 3
    noisy_but_stable = criteria.convergence_trend([2.0, 1.0], max_span_pct=3.0)
    assert noisy_but_stable["pass"] and not noisy_but_stable["require_monotone"]
    assert not criteria.convergence_trend([1.0, 6.0], max_span_pct=3.0)["pass"]
    with pytest.raises(criteria.CriterionError):
        criteria.convergence_trend([1.0])


def test_per_level_gating_allows_informational_coarse_levels():
    spec = {
        "name": "err",
        "kind": "error_pct",
        "tol_pct": 3.0,
        "tol_pct_by_level": [None, 3.0],
        "levels": ["coarse", "fine"],
    }
    verdict = criteria.judge_case(
        {"err": [-3.17, -1.15]}, [spec], [{"metric": "err", "max_span_pct": 3.0}],
        levels=["coarse", "fine"],
    )
    assert verdict["status"] == "passed"
    assert verdict["metrics"][0]["levels"][0]["gated"] is False
    # the same numbers with a uniform 3% gate must fail
    spec2 = {k: v for k, v in spec.items() if k != "tol_pct_by_level"}
    verdict2 = criteria.judge_case(
        {"err": [-3.17, -1.15]}, [spec2], [], levels=["coarse", "fine"]
    )
    assert verdict2["status"] == "failed"


def test_declared_negative_result_is_never_upgraded():
    spec = {"name": "err", "kind": "error_pct", "tol_pct": 3.0,
            "levels": ["a", "b"]}
    failed = criteria.judge_case({"err": [1.0, 0.5]}, [spec], [],
                                 levels=["a", "b"],
                                 declared_status="failed", blocker="root cause")
    assert failed["status"] == "failed" and "root cause" in failed["reason"]
    pending = criteria.judge_case({"err": [1.0, 0.5]}, [spec], [],
                                  levels=["a", "b"],
                                  declared_status="pending", blocker="withheld")
    assert pending["status"] == "pending"
    with pytest.raises(criteria.CriterionError):
        criteria.judge_case({"err": [1.0]}, [spec], [], levels=["a"],
                            declared_status="bogus")


# --------------------------------------------------------------------------- #
# matrix schema + reference provenance
# --------------------------------------------------------------------------- #
def test_matrix_schema_covers_nine_slots():
    matrix = runner.load_matrix()
    slots = {s["slot"] for s in matrix["slots"]}
    assert slots == {
        "cavity", "cylinder", "sphere", "channel", "taylor_green",
        "static_droplet", "rayleigh_taylor", "dam_break", "ship",
    }
    ids = [c["id"] for c in matrix["cases"]]
    assert len(ids) == len(set(ids))
    for case in matrix["cases"]:
        assert case["slot"] in slots
        assert case["title"] and case["reference_basis"] and case["scope"]
        assert case.get("refs"), f"{case['id']}: no reference anchor"
        measure = case.get("measure") or {}
        if "metrics" in measure:
            assert len(measure["levels"]) >= 2, f"{case['id']}: needs >= 2 levels"
            for spec in measure["metrics"]:
                assert len(spec["paths"]) == len(measure["levels"])
                if spec.get("kind", "value") == "value":
                    assert spec.get("reference") is not None
        # every case must be runnable from a declared command
        assert (case.get("repro") or {}).get("cmd")


def test_reference_provenance_gate_is_clean():
    report = reference_audit.audit_all()
    assert report["n_references"] >= 10
    bad = {r["id"]: r["reasons"] for r in report["references"] if not r["pass"]}
    assert not bad, bad


def test_every_repro_command_points_at_a_repo_file():
    matrix = runner.load_matrix()
    missing = []
    for case in matrix["cases"]:
        cmd = case["repro"]["cmd"]
        py = [tok for tok in cmd.split() if tok.endswith(".py")]
        assert py, f"{case['id']}: repro command has no script"
        for tok in py:
            if not (REPO_ROOT / tok).exists():
                missing.append((case["id"], tok))
    assert not missing, missing


def test_status_files_match_fresh_evaluation():
    """The committed three-level table must agree with a fresh machine verdict."""
    matrix = runner.load_matrix()
    mismatched = []
    for case in matrix["cases"]:
        fresh = runner.evaluate_case(case)
        status = fresh["status"]
        folder = ("verified" if status == "passed" else "pending")
        path = REPO_ROOT / "benchmarks" / folder / case["id"] / "validation_status.json"
        if not path.exists():
            mismatched.append((case["id"], "missing status file"))
            continue
        stored = json.loads(path.read_text())
        if stored.get("status") != status:
            mismatched.append((case["id"], f"stored {stored.get('status')} != fresh {status}"))
        assert stored.get("evidence_chain"), f"{case['id']}: no evidence chain"
    assert not mismatched, mismatched


# --------------------------------------------------------------------------- #
# one complete CI-tier case, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("TLBM_SKIP_MATRIX_CI") == "1",
                    reason="matrix CI case explicitly skipped")
def test_dam_break_ci_case_runs_and_is_judged(tmp_path):
    case_dir = MATRIX_DIR / "cases" / "dam_break_ritter_ci"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(case_dir / "run.py"), "--nx", "100,200",
         "--ladder", "", "--out", str(tmp_path)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads((tmp_path / "result.json").read_text())
    levels = payload["levels"]
    assert [lv["level"] for lv in levels] == ["nx=100", "nx=200"]
    for lv in levels:
        m = lv["metrics"]
        assert m["front_err_pct"] < 15.0
        assert m["mass_drift_pct"] < 1e-6  # flux-form conservation
    # the front law must converge in the right direction
    assert abs(levels[1]["metrics"]["front_err_pct"]) < abs(
        levels[0]["metrics"]["front_err_pct"]
    )
    # the matrix judge must consume these numbers through the same code path
    case = runner.find_case(runner.load_matrix(), "dam_break_ritter_ci")
    measure = case["measure"]
    values = {
        spec["name"]: [
            float(levels[i]["metrics"][spec["name"]])
            for i in range(len(levels))
        ]
        for spec in measure["metrics"]
    }
    verdict = criteria.judge_case(
        values, measure["metrics"], measure["convergence"],
        levels=measure["levels"],
    )
    assert verdict["status"] in {"passed", "failed"}
    assert verdict["metrics"] and verdict["convergence"]