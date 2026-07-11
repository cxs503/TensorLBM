"""Unit tests for fail-closed benchmark regression-gate manifests."""
from __future__ import annotations

import json

from tensorlbm.regression_gate import evaluate_regression_gate, write_regression_manifest


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_status(*, metrics=None, numerical_failure=None):
    return {
        "state": "PASSED",
        "requested_steps": 10,
        "completed_steps": 10,
        "numerical_failure": numerical_failure,
        "metrics": {"pass": True, "residual": 0.01} if metrics is None else metrics,
    }


def _complete_case(tmp_path, name="case"):
    case_dir = tmp_path / name
    case_dir.mkdir()
    (case_dir / "history.csv").write_text("step,value\n1,1.0\n", encoding="utf-8")
    _write_json(case_dir / "run_status.json", _completed_status())
    return case_dir


def test_gate_passes_only_with_completed_finite_status_artifacts_and_physics(tmp_path):
    _complete_case(tmp_path)
    report = evaluate_regression_gate(
        tmp_path,
        {
            "case": {
                "run_dir": "case",
                "required_artifacts": ["history.csv"],
                "physics": {"pass": True, "metrics": {"amplitude": 0.1}},
            },
        },
    )

    assert report["pass"] is True
    case = report["cases"][0]
    assert case["completion"]["pass"] is True
    assert case["artifacts"]["pass"] is True
    assert case["numerics"]["pass"] is True
    assert case["physics"]["pass"] is True


def test_gate_fails_closed_for_incomplete_or_nonfinite_or_missing_artifact(tmp_path):
    case_dir = _complete_case(tmp_path)
    _write_json(case_dir / "run_status.json", _completed_status(metrics={"pass": True, "residual": None}))
    report = evaluate_regression_gate(
        tmp_path,
        {
            "case": {
                "run_dir": "case",
                "required_artifacts": ["history.csv", "missing.png"],
                "physics": {"pass": True},
            },
        },
    )

    case = report["cases"][0]
    assert report["pass"] is False
    assert case["artifacts"]["pass"] is False
    assert case["numerics"]["pass"] is False


def test_gate_rejects_false_physics_and_partial_completion(tmp_path):
    case_dir = _complete_case(tmp_path)
    _write_json(case_dir / "run_status.json", {
        **_completed_status(), "completed_steps": 9,
    })
    report = evaluate_regression_gate(
        tmp_path,
        {"case": {"run_dir": "case", "physics": {"pass": False}}},
    )

    case = report["cases"][0]
    assert report["pass"] is False
    assert case["completion"]["pass"] is False
    assert case["physics"]["pass"] is False


def test_gate_loads_shared_physics_report_from_artifacts_root(tmp_path):
    _complete_case(tmp_path)
    _write_json(
        tmp_path / "kpis.json",
        {"cases": [{"case": "case", "pass": True, "metrics": {"amplitude": 0.1}}]},
    )
    report = evaluate_regression_gate(
        tmp_path,
        {"case": {"run_dir": "case", "physics": {"report": "kpis.json", "case": "case"}}},
    )

    assert report["pass"] is True


def test_gate_uses_completed_run_status_metrics_as_explicit_physics_result(tmp_path):
    _complete_case(tmp_path)
    report = evaluate_regression_gate(
        tmp_path,
        {
            "case": {
                "run_dir": "case",
                "required_artifacts": ["history.csv"],
                "physics": {"status_metrics": True},
            },
        },
    )

    physics = report["cases"][0]["physics"]
    assert report["pass"] is True
    assert physics["source"] == "status_metrics"
    assert physics["reported_pass"] is True


def test_gate_does_not_turn_failed_checkpoint_status_metrics_into_a_pass(tmp_path):
    case_dir = _complete_case(tmp_path)
    _write_json(case_dir / "run_status.json", {
        **_completed_status(metrics={"pass": False, "amplitude": 0.0}),
        "state": "FAILED",
    })
    report = evaluate_regression_gate(
        tmp_path,
        {"case": {"run_dir": "case", "physics": {"status_metrics": True}}},
    )

    case = report["cases"][0]
    assert report["pass"] is False
    assert case["completion"]["pass"] is False
    assert case["physics"]["pass"] is False
    assert case["physics"]["reported_pass"] is False


def test_gate_rejects_missing_or_invalid_status_metrics_physics_configuration(tmp_path):
    _complete_case(tmp_path)
    missing = evaluate_regression_gate(
        tmp_path,
        {"case": {"run_dir": "case", "physics": {"status_metrics": False}}},
    )
    invalid = evaluate_regression_gate(
        tmp_path,
        {"case": {"run_dir": "case", "physics": {"status_metrics": "yes"}}},
    )

    assert missing["pass"] is False
    assert "must be true" in missing["cases"][0]["physics"]["reason"]
    assert invalid["pass"] is False
    assert "must be true" in invalid["cases"][0]["physics"]["reason"]


def test_gate_rejects_path_escape_and_writes_json_manifest_atomically(tmp_path):
    _complete_case(tmp_path)
    report = evaluate_regression_gate(
        tmp_path,
        {"case": {"run_dir": "../outside", "physics": {"pass": True}}},
    )
    manifest = tmp_path / "gate.json"
    write_regression_manifest(manifest, report)

    assert report["pass"] is False
    assert "escapes artifacts root" in report["cases"][0]["errors"][0]
    persisted = json.loads(manifest.read_text(encoding="utf-8"))
    assert persisted == report
