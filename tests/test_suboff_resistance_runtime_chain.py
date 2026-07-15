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


def test_d3q19_face_integrated_momentum_flux_has_outward_face_signs():
    """The two x control faces have opposite outward normals at rest."""
    rho = suboff.torch.ones((2, 3, 4), dtype=suboff.torch.float32)
    zero = suboff.torch.zeros_like(rho)
    f = suboff.equilibrium3d(rho, zero, zero, zero)

    flux = suboff.d3q19_x_face_momentum_flux(f)

    # Pi_xx = rho / 3 per cell.  The inlet outward normal is -x and the
    # outlet outward normal is +x; the transverse components vanish.
    assert flux["inlet_outward"][0] == pytest.approx(-2.0)
    assert flux["outlet_outward"][0] == pytest.approx(2.0)
    assert flux["net_outward"] == pytest.approx([0.0, 0.0, 0.0])


def test_runtime_budget_reports_measured_face_flux_separately_from_bc_delta():
    config = suboff.SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, momentum_budget_diagnostic=True,
        momentum_budget_interval=1,
    )

    observation = cast(dict[str, Any], suboff.run_suboff_resistance_runtime(config))
    budget = observation["conservation"]["source_attribution"]["momentum"]["operator_budget"]
    boundary_flux = budget["boundary_flux"]

    assert boundary_flux["status"] == "measured"
    assert boundary_flux["kind"] == "face_integrated_population_momentum_flux"
    assert len(boundary_flux["samples"]) == 10
    assert "inlet_boundary" in budget["samples"][0]  # BC population delta
    assert "face_flux" in budget["samples"][0]       # measured face transport
    assert boundary_flux["closure"]["status"] == "withheld"
    assert boundary_flux["closure"]["reason"] == "face_flux_is_not_a_bc_population_delta"


def test_full_operator_diagnostic_binds_same_time_control_volume_evidence_without_false_closure():
    observation = cast(dict[str, Any], suboff.run_suboff_resistance_runtime(
        suboff.SuboffResistanceBenchmarkConfig(
            base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
            lbm_steps=10, lbm_warmup_steps=0, momentum_budget_diagnostic=True,
            momentum_budget_interval=1,
        )
    ))
    budget = observation["conservation"]["source_attribution"]["momentum"]["operator_budget"]
    evidence = budget["same_time_control_volume"]

    assert evidence["status"] == "measured"
    assert evidence["coverage"] == "full_per_step"
    assert evidence["control_volume"] == (
        "entire retained D3Q19 lattice population domain; x-normal inlet and outlet faces only"
    )
    assert len(evidence["samples"]) == 10
    first = evidence["samples"][0]
    assert first["step"] == 1
    assert first["time_interval"] == {"start": "retained_state[0]", "end": "retained_state[1]"}
    assert first["sample_phase"] == budget["boundary_flux"]["sampling_state"]
    assert first["measured_x_face_transport"]["value"] == budget["boundary_flux"]["samples"][0]["net_outward"]
    assert first["operator_state_deltas"]["values"]["inlet_boundary"] == budget["samples"][0]["inlet_boundary"]
    assert first["operator_state_deltas"]["meaning"].endswith("not face-flux terms")
    residual = first["control_volume_residual"]
    assert residual["status"] == "withheld"
    assert residual["value"] is None
    assert residual["missing_terms"] == [
        "streaming_face_crossing_term",
        "wall_control_volume_boundary_term",
        "solid_control_volume_boundary_term",
    ]
    assert evidence["closure"]["status"] == "withheld"
    assert "face_flux_is_not_a_population_delta" in evidence["closure"]["reason"]
    # The evidence itself remains JSON-machine-readable with the withheld terms.
    assert json.loads(json.dumps(evidence))["samples"][0]["control_volume_residual"] == residual


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
