"""Fail-closed binding coverage for the SUBOFF marine resistance gate."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tensorlbm.marine_reference_manifest import build_marine_reference_manifest
from tensorlbm.marine_resistance_artifact import build_marine_resistance_artifact
from tensorlbm.marine_run_provenance import build_marine_run_provenance


ROOT = Path(__file__).parents[1]


def _observation() -> dict[str, Any]:
    return {
        "schema": "suboff-resistance-runtime-observation-v1",
        "case": "suboff_runtime",
        "runner": "test.runner",
        "completion": {"state": "COMPLETED", "requested_steps": 10, "completed_steps": 10},
        "resistance": {"coefficient": 0.0042, "basis": "test", "status": "measured"},
        "preflight": {"pass": True, "checks": {"config": {"pass": True}}},
        "numerics": {"pass": True, "finite": True},
        "conservation": {"status": "withheld", "pass": False},
        "physics": {"status": "withheld", "pass": False},
    }


def _artifact() -> dict[str, Any]:
    observation = _observation()
    provenance = build_marine_run_provenance(observation, runner="test.runner")
    reference = build_marine_reference_manifest(case="suboff_runtime", coefficient=0.004, source="test reference")
    return build_marine_resistance_artifact(observation, provenance, reference)


def _run_gate(tmp_path: Path, artifact: dict[str, Any]) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    artifact_path = tmp_path / "suboff_runtime" / "marine_resistance_kpi.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    manifest_path = tmp_path / "gate.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps({"gate": "marine_resistance", "cases": {
        "suboff_runtime": {"artifact": "suboff_runtime/marine_resistance_kpi.json",
        "max_relative_error_pct": 10.0, "max_mass_relative_drift": 1.0,
        "max_momentum_relative_drift": 1.0, "required_preflight_checks": []},
    }}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_benchmark_gate.py", "--artifacts", str(tmp_path),
         "--manifest", str(manifest_path), "--report", str(report_path)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return completed, json.loads(report_path.read_text(encoding="utf-8"))


def test_marine_gate_fail_closes_without_or_with_forged_bound_evidence(tmp_path):
    pristine = _artifact()
    mutations = {
        "missing_binding": lambda item: item.pop("binding"),
        "observation": lambda item: item["evidence"]["observation"]["resistance"].update(coefficient=0.0041),
        "provenance": lambda item: item["evidence"]["provenance"].update(runner="forged.runner"),
        "reference": lambda item: item["evidence"]["reference"].update(coefficient=0.0041),
    }
    for label, mutate in mutations.items():
        artifact = copy.deepcopy(pristine)
        mutate(artifact)
        completed, report = _run_gate(tmp_path / label, artifact)
        assert completed.returncode == 1
        assert report["pass"] is False
        assert report["cases"][0]["pass"] is False
        assert report["cases"][0]["errors"]


def test_marine_gate_preserves_legacy_v1_contract(tmp_path):
    artifact = {
        "kind": "marine_resistance_kpi", "schema_version": 1, "case": "suboff_runtime",
        "completion": {"state": "COMPLETED", "requested_steps": 10, "completed_steps": 10},
        "preflight": {"pass": True, "checks": {"config": {"pass": True}}},
        "numerics": {"pass": True, "finite": True},
        "conservation": {"pass": True, "mass_relative_drift": 0.0, "momentum_relative_drift": 0.0},
        "resistance": {"pass": True, "coefficient": 0.0042, "reference_coefficient": 0.004,
                       "relative_error_pct": 5.0},
        "physics": {"pass": True},
    }
    completed, report = _run_gate(tmp_path, artifact)
    assert completed.returncode == 0
    assert report["pass"] is True


def test_marine_gate_rejects_forged_top_level_passes_despite_valid_binding(tmp_path):
    artifact = _artifact()
    artifact.update({
        "completion": {"state": "COMPLETED", "requested_steps": 10, "completed_steps": 10},
        "preflight": {"pass": True, "checks": {"config": {"pass": True}}},
        "numerics": {"pass": True, "finite": True},
        "conservation": {"pass": True, "mass_relative_drift": 0.0, "momentum_relative_drift": 0.0},
        "resistance": {"pass": True, "coefficient": 0.0042, "reference_coefficient": 0.004,
                       "relative_error_pct": 5.0},
        "physics": {"pass": True},
    })
    completed, report = _run_gate(tmp_path, artifact)
    assert completed.returncode == 1
    assert report["pass"] is False
    row = report["cases"][0]
    assert row["completion"]["pass"] is True
    assert row["preflight"]["pass"] is True
    assert row["numerics"]["pass"] is True
    assert row["conservation"]["pass"] is False
    assert row["physics"]["pass"] is False


def test_real_style_small_runner_artifact_remains_withheld_and_fails_gate(tmp_path):
    completed, report = _run_gate(tmp_path, _artifact())
    assert completed.returncode == 1
    assert report["pass"] is False
    # Binding is valid, but no physical assertion is manufactured from a small run.
    assert report["cases"][0]["completion"]["pass"] is True
    assert report["cases"][0]["preflight"]["pass"] is True
    assert report["cases"][0]["numerics"]["pass"] is True
    assert report["cases"][0]["conservation"]["pass"] is False
