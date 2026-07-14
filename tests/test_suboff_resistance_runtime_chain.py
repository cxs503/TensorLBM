"""Runtime-chain coverage for fail-closed SUBOFF resistance evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tensorlbm.marine_reference_manifest import build_marine_reference_manifest
from tensorlbm.marine_resistance_artifact import build_marine_resistance_artifact
from tensorlbm.marine_run_provenance import build_marine_run_provenance
import tensorlbm.suboff_resistance as suboff


def test_real_runner_measures_hash_bound_mass_conservation_and_does_not_promote_physics():
    config = suboff.SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
        conservation_max_relative_mass_drift=1.0e-12,
    )
    observation = cast(dict[str, Any], suboff.run_suboff_resistance_runtime(config))

    conservation = observation["conservation"]
    assert conservation["status"] == "measured"
    assert conservation["pass"] is False
    assert conservation["initial_lattice_mass"] > 0.0
    assert conservation["final_lattice_mass"] > 0.0
    assert conservation["max_abs_mass_drift"] >= 0.0
    assert conservation["max_relative_mass_drift"] > 1.0e-12
    assert conservation["mass_sample_count"] == 11
    assert conservation["sampled_step_count"] == 10
    assert conservation["max_relative_mass_drift_limit"] == pytest.approx(1.0e-12)
    assert observation["physics"]["pass"] is False


def test_real_runner_records_three_hash_bound_grid_levels_with_order_and_conservation_attribution():
    config = suboff.SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=80.0, max_iterations=3,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
        target_error_pct=0.01,
    )

    observation = cast(dict[str, Any], suboff.run_suboff_resistance_runtime(config))

    numerics = cast(dict[str, Any], observation["numerics"])
    levels = cast(list[dict[str, Any]], numerics["refinement_levels"])
    assert numerics["status"] == "measured"
    assert numerics["refinement_kind"] == "grid"
    assert numerics["required_levels"] == 3
    assert len(levels) == 3
    assert len({tuple(sorted(level["grid"].items())) for level in levels}) == 3
    assert all(level["evidence_sha256"] for level in levels)
    assert all(level["completion"]["completed_steps"] == 10 for level in levels)
    assert all(level["finite"]["pass"] is True for level in levels)
    assert isinstance(numerics["coefficient_change_pct"], float)
    assert len(numerics["coefficient_changes_pct"]) == 2
    assert numerics["observed_order"] is not None
    assert numerics["monotonicity"]["status"] == "measured"
    # The true three-level campaign has complete evidence and reports its
    # measured non-converged result; it must not be promoted by conservation.
    assert numerics["convergence"]["pass"] is False
    assert numerics["pass"] is False
    assert observation["conservation"]["pass"] is False
    attribution = observation["conservation"]["source_attribution"]
    assert attribution["status"] == "measured"
    assert attribution["dominant_channel"] in {"mass", "momentum", "balanced"}
    assert attribution["mass"]["max_relative_drift"] == observation["conservation"]["max_relative_mass_drift"]
    assert attribution["momentum"]["max_relative_drift"] == observation["conservation"]["max_relative_momentum_drift"]


def test_real_runner_observation_binds_to_withheld_canonical_artifact(monkeypatch):
    calls: list[object] = []

    def runner(config):
        calls.append(config)
        return {"simulated": {"cd": 0.0042}, "iterations": [{
            "grid": {"nx": 36, "ny": 32, "nz": 32},
            "runtime_evidence": {"requested_steps": 10, "completed_steps": 10,
                "finite_population_checks": 10, "finite_density_checks": 10,
                "all_populations_finite": True, "all_densities_finite": True,
                "density_min": 0.99, "density_max": 1.01},
        }]}

    monkeypatch.setattr(suboff, "run_suboff_resistance_benchmark", runner)
    config = suboff.SuboffResistanceBenchmarkConfig(lbm_steps=10, lbm_warmup_steps=0)
    observation = cast(dict[str, Any], suboff.run_suboff_resistance_runtime(config))
    provenance = build_marine_run_provenance(observation, runner=observation["runner"])
    reference = build_marine_reference_manifest(
        case="suboff_runtime", coefficient=0.004, source="test reference",
    )
    artifact = build_marine_resistance_artifact(observation, provenance, reference)

    assert calls == [config]
    assert observation["completion"]["completed_steps"] == 10
    assert artifact["binding"]["reference_sha256"] == reference["sha256"]
    assert artifact["resistance"]["relative_error_pct"] == pytest.approx(5.0)
    assert artifact["preflight"]["pass"] is True
    assert artifact["preflight"]["checks"]["config"]["pass"] is True
    assert artifact["preflight"]["checks"]["domain"]["pass"] is True
    assert artifact["preflight"]["checks"]["mach"]["pass"] is True
    # A one-level mocked runner lacks refinement evidence and therefore cannot
    # be upgraded to a numerical PASS.
    assert artifact["numerics"]["pass"] is False
    assert artifact["numerics"]["finite_population_checks"] == 10
    assert artifact["numerics"]["finite_density_checks"] == 10
    assert artifact["conservation"]["pass"] is False
    assert artifact["physics"]["pass"] is False


def test_gate_cli_reports_withheld_evidence_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(suboff, "run_suboff_resistance_benchmark", lambda config: {"simulated": {"cd": 0.0042}, "iterations": [{
        "grid": {"nx": 36, "ny": 32, "nz": 32},
        "runtime_evidence": {"requested_steps": 10, "completed_steps": 10,
            "finite_population_checks": 10, "finite_density_checks": 10,
            "all_populations_finite": True, "all_densities_finite": True,
            "density_min": 0.99, "density_max": 1.01},
    }]})
    observation = cast(dict[str, Any], suboff.run_suboff_resistance_runtime(
        suboff.SuboffResistanceBenchmarkConfig(lbm_steps=10, lbm_warmup_steps=0)
    ))
    provenance = build_marine_run_provenance(observation, runner=observation["runner"])
    reference = build_marine_reference_manifest(case="suboff_runtime", coefficient=0.004, source="test reference")
    artifact = build_marine_resistance_artifact(observation, provenance, reference)
    artifact_path = tmp_path / "suboff_runtime" / "marine_resistance_kpi.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    manifest = {
        "gate": "marine_resistance",
        "cases": {"suboff_runtime": {"artifact": "suboff_runtime/marine_resistance_kpi.json",
        "max_relative_error_pct": 10.0, "max_mass_relative_drift": 1.0, "max_momentum_relative_drift": 1.0,
        "required_preflight_checks": []}},
    }
    manifest_path = tmp_path / "gate.json"
    report_path = tmp_path / "gate-report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    import subprocess
    completed = subprocess.run(
        ["python", "scripts/evaluate_benchmark_gate.py", "--artifacts", str(tmp_path),
         "--manifest", str(manifest_path), "--report", str(report_path)],
        cwd=Path(__file__).parents[1], text=True, capture_output=True, check=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert report["gate"] == "marine_resistance"
    assert report["pass"] is False
    assert report["cases"][0]["completion"]["pass"] is True
    assert report["cases"][0]["preflight"]["pass"] is True
    assert report["cases"][0]["numerics"]["pass"] is False
    assert report["cases"][0]["physics"]["pass"] is False
