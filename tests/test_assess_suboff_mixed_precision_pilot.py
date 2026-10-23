from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / ("assess_suboff_mixed_precision_pilot.py")
)
SPEC = importlib.util.spec_from_file_location("mixed_pilot_assessor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


TAUS = [0.5000162, 0.5000324, 0.5000648, 0.5001296]


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    result = tmp_path / "pilot.json"
    result.write_text(
        json.dumps(
            {
                "schema": "tensorlbm-suboff-nested-amr-smoke-v3",
                "configuration": {
                    "collision_model": "natural_kbc",
                    "resolved_reynolds": 1_000_000,
                    "tau_by_level": TAUS,
                    "population_storage_dtype": "float32",
                    "natural_kbc_compute_dtype": "float64",
                    "maximum_health_speed": 0.3,
                    "maximum_positivity_limited_fraction": 0.0,
                    "maximum_reflux_applied_correction_fraction": 0.001,
                },
                "runtime": {
                    "root_steps_advanced": 180,
                    "seconds_per_root_step": 8.0,
                },
                "planning": {"measured_peak_allocated_gib": 18.0},
                "result": {
                    "finite": True,
                    "minimum_observed_population": 0.005,
                    "maximum_observed_speed": 0.08,
                    "maximum_positivity_limited_fraction": 0.0,
                    "maximum_reflux_applied_correction_fraction_by_interface": [1e-6],
                    "collision_execution": {
                        "storage_dtype_policy": "preserve_input",
                        "compute_dtype": "float64",
                        "collision_calls": 100,
                        "input_signatures": [{"dtype": "torch.float32"}],
                        "torch_dynamo_process_unique_graphs": 4,
                    },
                },
                "acceptance": {
                    "population_health_target_met": True,
                    "target_reynolds_reached": True,
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "precision.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "tensorlbm-suboff-collision-viscosity-precision-evidence-v1",
                "audit_configuration": {
                    "taus_by_level": TAUS,
                    "storage_dtype": "float32",
                },
                "float32_storage_float64_collision_compute": {
                    "status": "admitted",
                    "admitted_by_level": [True, True, True, True],
                },
            }
        ),
        encoding="utf-8",
    )
    return result, evidence


def test_admits_bounded_mixed_precision_runtime_path(tmp_path: Path) -> None:
    result, evidence = _inputs(tmp_path)
    assessment = MODULE.assess(result, evidence)
    assert assessment["status"] == "runtime_path_admitted"
    assert assessment["acceptance"]["long_run_launch_admitted"] is True
    assert assessment["physical_validation"] is False
    assert assessment["acceptance"]["resistance_accuracy_assessed"] is False


def test_rejects_native_float32_collision_identity(tmp_path: Path) -> None:
    result, evidence = _inputs(tmp_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["result"]["collision_execution"]["compute_dtype"] = "storage"
    result.write_text(json.dumps(payload), encoding="utf-8")
    assessment = MODULE.assess(result, evidence)
    assert assessment["status"] == "runtime_path_rejected"
    assert assessment["acceptance"]["float32_storage_float64_collision_identity"] is False


def test_rejects_unbounded_runtime_or_nonpositive_population(tmp_path: Path) -> None:
    result, evidence = _inputs(tmp_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["runtime"]["seconds_per_root_step"] = 16.0
    payload["result"]["minimum_observed_population"] = 0.0
    result.write_text(json.dumps(payload), encoding="utf-8")
    assessment = MODULE.assess(result, evidence)
    assert assessment["acceptance"]["runtime_and_memory_within_registered_limits"] is False
    assert assessment["acceptance"]["finite_positive_bounded_flow"] is False
    assert assessment["acceptance"]["long_run_launch_admitted"] is False
