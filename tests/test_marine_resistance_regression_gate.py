"""Fail-closed tests for canonical marine resistance KPI artifacts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensorlbm.regression_gate import evaluate_marine_resistance_gate


def _artifact() -> dict:
    return {
        "kind": "marine_resistance_kpi",
        "schema_version": 1,
        "case": "suboff_full",
        "completion": {"state": "COMPLETED", "requested_steps": 1000, "completed_steps": 1000},
        "preflight": {
            "pass": True,
            "checks": {
                "geometry_resolved": {"pass": True},
                "low_mach": {"pass": True},
                "outlet_clearance": {"pass": True},
            },
        },
        "numerics": {"pass": True, "rho_min": 0.98, "rho_max": 1.02, "nan_count": 0},
        "conservation": {"pass": True, "mass_relative_drift": 2.0e-5, "momentum_relative_drift": 4.0e-5},
        "resistance": {"pass": True, "coefficient": 0.0041, "reference_coefficient": 0.004, "relative_error_pct": 2.5},
        "physics": {"pass": True},
    }


def _spec() -> dict:
    return {
        "suboff_full": {
            "artifact": "suboff_full/marine_resistance_kpi.json",
            "max_relative_error_pct": 3.0,
            "max_mass_relative_drift": 1.0e-4,
            "max_momentum_relative_drift": 1.0e-4,
            "required_preflight_checks": ["geometry_resolved", "low_mach", "outlet_clearance"],
        }
    }


def _write(root: Path, artifact: dict | None = None) -> None:
    destination = root / "suboff_full" / "marine_resistance_kpi.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(artifact or _artifact()), encoding="utf-8")


def test_marine_resistance_gate_accepts_complete_canonical_artifact(tmp_path):
    _write(tmp_path)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    assert report["pass"] is True
    row = report["cases"][0]
    assert row["completion"]["pass"] is True
    assert row["preflight"]["pass"] is True
    assert row["numerics"]["pass"] is True
    assert row["conservation"]["pass"] is True
    assert row["resistance"]["pass"] is True
    assert row["physics"]["pass"] is True


def test_marine_resistance_gate_recomputes_relative_error_from_coefficients(tmp_path):
    artifact = _artifact()
    artifact["resistance"]["relative_error_pct"] = 0.0
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    row = report["cases"][0]
    assert report["pass"] is False
    assert row["resistance"]["pass"] is False
    assert row["resistance"]["relative_error_pct"] == pytest.approx(2.5)
    assert row["resistance"]["reported_relative_error_pct"] == 0.0
    assert "resistance relative_error_pct contradicts coefficients" in row["errors"]


def test_marine_resistance_gate_uses_recomputed_error_for_its_limit(tmp_path):
    artifact = _artifact()
    artifact["resistance"]["relative_error_pct"] = 0.0
    spec = _spec()
    spec["suboff_full"]["max_relative_error_pct"] = 1.0
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, spec)

    row = report["cases"][0]
    assert row["resistance"]["pass"] is False
    assert row["resistance"]["relative_error_pct"] == pytest.approx(2.5)


def test_marine_resistance_gate_rejects_nonzero_report_at_zero_error_tolerance_boundary(tmp_path):
    artifact = _artifact()
    artifact["resistance"]["coefficient"] = artifact["resistance"]["reference_coefficient"]
    artifact["resistance"]["relative_error_pct"] = 1.0e-13
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    row = report["cases"][0]
    assert report["pass"] is False
    assert row["resistance"]["relative_error_pct"] == 0.0
    assert row["resistance"]["pass"] is False
    assert "resistance relative_error_pct contradicts coefficients" in row["errors"]


@pytest.mark.parametrize("reference", [0.0, float("nan"), float("inf")])
def test_marine_resistance_gate_fail_closes_for_zero_or_nonfinite_reference(tmp_path, reference):
    artifact = _artifact()
    artifact["resistance"]["reference_coefficient"] = reference
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    row = report["cases"][0]
    assert report["pass"] is False
    assert row["resistance"]["pass"] is False
    assert row["physics"]["pass"] is False


def test_marine_resistance_gate_accepts_exact_relative_error_at_limit(tmp_path):
    artifact = _artifact()
    artifact["resistance"]["relative_error_pct"] = abs(artifact["resistance"]["coefficient"] - artifact["resistance"]["reference_coefficient"]) / artifact["resistance"]["reference_coefficient"] * 100.0
    spec = _spec()
    spec["suboff_full"]["max_relative_error_pct"] = artifact["resistance"]["relative_error_pct"]
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, spec)

    assert report["pass"] is True
    assert report["cases"][0]["resistance"]["pass"] is True


def test_marine_resistance_gate_rejects_recomputed_error_just_above_limit(tmp_path):
    artifact = _artifact()
    artifact["resistance"]["coefficient"] = 0.004100000000000001
    artifact["resistance"]["relative_error_pct"] = abs(0.004100000000000001 - 0.004) / 0.004 * 100.0
    spec = _spec()
    spec["suboff_full"]["max_relative_error_pct"] = 2.5
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, spec)

    assert report["pass"] is False
    assert report["cases"][0]["resistance"]["pass"] is False


@pytest.mark.parametrize(
    ("path", "value", "section"),
    [
        (("completion", "completed_steps"), 999, "completion"),
        (("preflight", "checks", "low_mach", "pass"), False, "preflight"),
        (("conservation", "mass_relative_drift"), 2.0e-4, "conservation"),
        (("conservation", "momentum_relative_drift"), float("inf"), "conservation"),
        (("resistance", "relative_error_pct"), 3.1, "resistance"),
    ],
)
def test_marine_resistance_gate_rejects_each_required_evidence_failure(tmp_path, path, value, section):
    artifact = _artifact()
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    row = report["cases"][0]
    assert report["pass"] is False
    assert row[section]["pass"] is False


def test_numeric_completion_and_finite_resistance_do_not_imply_physics_pass(tmp_path):
    artifact = _artifact()
    artifact["physics"]["pass"] = False
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    row = report["cases"][0]
    assert row["completion"]["pass"] is True
    assert row["resistance"]["pass"] is True
    assert row["physics"]["pass"] is False
    assert report["pass"] is False


def test_marine_resistance_gate_requires_every_reported_preflight_check_to_pass(tmp_path):
    artifact = _artifact()
    artifact["preflight"]["checks"]["unrequested_warning"] = {"pass": False}
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    assert report["pass"] is False
    assert report["cases"][0]["preflight"]["pass"] is False


def test_marine_resistance_gate_refuses_generic_or_wrong_version_artifacts(tmp_path):
    artifact = _artifact()
    artifact["kind"] = "benchmark_result"
    artifact["schema_version"] = 99
    _write(tmp_path, artifact)

    report = evaluate_marine_resistance_gate(tmp_path, _spec())

    assert report["pass"] is False
    assert "artifact kind is not marine_resistance_kpi" in report["cases"][0]["errors"]
    assert "unsupported marine resistance artifact schema_version" in report["cases"][0]["errors"]


def test_marine_resistance_gate_rejects_escaped_artifact_and_bad_limits(tmp_path):
    outside = tmp_path.parent / "marine_resistance_kpi.json"
    outside.write_text(json.dumps(_artifact()), encoding="utf-8")
    spec = copy.deepcopy(_spec())
    spec["suboff_full"]["artifact"] = str(outside)
    spec["suboff_full"]["max_mass_relative_drift"] = -1.0

    report = evaluate_marine_resistance_gate(tmp_path, spec)

    assert report["pass"] is False
    assert any("max_mass_relative_drift" in error for error in report["cases"][0]["errors"])
