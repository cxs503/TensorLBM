#!/usr/bin/env python3
"""Admit a bounded mixed-precision SUBOFF run as a runtime path only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def assess(
    result_path: Path,
    precision_evidence_path: Path,
    *,
    maximum_seconds_per_root_step: float = 15.0,
    maximum_peak_allocated_gib: float = 22.0,
) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "tensorlbm-suboff-nested-amr-smoke-v3":
        raise ValueError("unsupported SUBOFF result schema")
    evidence = json.loads(precision_evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema") != (
        "tensorlbm-suboff-collision-viscosity-precision-evidence-v1"
    ):
        raise ValueError("unsupported collision precision evidence schema")
    if maximum_seconds_per_root_step <= 0.0:
        raise ValueError("maximum seconds per root step must be positive")
    if maximum_peak_allocated_gib <= 0.0:
        raise ValueError("maximum peak allocation must be positive")

    configuration = result["configuration"]
    runtime = result["runtime"]
    planning = result["planning"]
    flow = result["result"]
    acceptance = result["acceptance"]
    execution = flow["collision_execution"]
    measured_peak = planning.get("measured_peak_allocated_gib")
    seconds_per_step = runtime.get("seconds_per_root_step")
    minimum_population = flow.get("minimum_observed_population")
    maximum_speed = flow.get("maximum_observed_speed")
    maximum_reflux = flow.get(
        "maximum_reflux_applied_correction_fraction_by_interface", []
    )
    signatures = execution.get("input_signatures", [])

    scheduled_taus = [float(value) for value in configuration["tau_by_level"]]
    evidence_taus = [
        float(value) for value in evidence["audit_configuration"]["taus_by_level"]
    ]
    tau_identity = len(scheduled_taus) == len(evidence_taus) and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(scheduled_taus, evidence_taus, strict=True)
    )
    mixed_certificate = evidence["float32_storage_float64_collision_compute"]
    viscosity_certificate_admitted = (
        mixed_certificate.get("status") == "admitted"
        and all(mixed_certificate.get("admitted_by_level", []))
        and len(mixed_certificate.get("admitted_by_level", []))
        == len(scheduled_taus)
    )
    precision_identity = (
        evidence["audit_configuration"].get("storage_dtype") == "float32"
        and configuration.get("population_storage_dtype") == "float32"
        and configuration.get("natural_kbc_compute_dtype") == "float64"
        and execution.get("storage_dtype_policy") == "preserve_input"
        and execution.get("compute_dtype") == "float64"
        and bool(signatures)
        and all(item.get("dtype") == "torch.float32" for item in signatures)
    )
    numerical_health = (
        flow.get("finite") is True
        and _finite_number(minimum_population)
        and float(minimum_population) > 0.0
        and _finite_number(maximum_speed)
        and float(maximum_speed) <= float(configuration["maximum_health_speed"])
        and float(flow.get("maximum_positivity_limited_fraction", math.inf))
        <= float(configuration["maximum_positivity_limited_fraction"])
        and bool(maximum_reflux)
        and max(float(value) for value in maximum_reflux)
        <= float(configuration["maximum_reflux_applied_correction_fraction"])
        and acceptance.get("population_health_target_met") is True
        and acceptance.get("target_reynolds_reached") is True
    )
    bounded_runtime = (
        _finite_number(seconds_per_step)
        and 0.0 < float(seconds_per_step) <= maximum_seconds_per_root_step
        and _finite_number(measured_peak)
        and 0.0 < float(measured_peak) <= maximum_peak_allocated_gib
        and int(runtime.get("root_steps_advanced", 0)) > 0
        and int(execution.get("collision_calls", 0)) > 0
    )
    admitted = (
        configuration.get("collision_model") == "natural_kbc"
        and float(configuration.get("resolved_reynolds", math.nan)) == 1_000_000.0
        and tau_identity
        and viscosity_certificate_admitted
        and precision_identity
        and numerical_health
        and bounded_runtime
    )
    return {
        "schema": "tensorlbm-suboff-mixed-precision-runtime-pilot-v1",
        "status": "runtime_path_admitted" if admitted else "runtime_path_rejected",
        "physical_validation": False,
        "sources": {
            "pilot_result": str(result_path),
            "pilot_result_sha256": _sha256(result_path),
            "collision_precision_evidence": str(precision_evidence_path),
            "collision_precision_evidence_sha256": _sha256(
                precision_evidence_path
            ),
        },
        "observations": {
            "root_steps_advanced": runtime.get("root_steps_advanced"),
            "seconds_per_root_step": seconds_per_step,
            "measured_peak_allocated_gib": measured_peak,
            "minimum_observed_population": minimum_population,
            "maximum_observed_speed": maximum_speed,
            "maximum_reflux_applied_correction_fraction": (
                max(float(value) for value in maximum_reflux)
                if maximum_reflux else None
            ),
            "collision_calls": execution.get("collision_calls"),
            "torch_dynamo_process_unique_graphs": execution.get(
                "torch_dynamo_process_unique_graphs"
            ),
        },
        "acceptance": {
            "tau_schedule_matches_viscosity_certificate": tau_identity,
            "all_level_viscosities_certified": viscosity_certificate_admitted,
            "float32_storage_float64_collision_identity": precision_identity,
            "finite_positive_bounded_flow": numerical_health,
            "runtime_and_memory_within_registered_limits": bounded_runtime,
            "long_run_launch_admitted": admitted,
            "resistance_accuracy_assessed": False,
            "time_convergence_assessed": False,
            "grid_convergence_assessed": False,
            "physical_validation": False,
        },
        "limits": {
            "maximum_seconds_per_root_step": maximum_seconds_per_root_step,
            "maximum_peak_allocated_gib": maximum_peak_allocated_gib,
        },
        "prohibition": (
            "This pilot admits only the precision, health, memory and throughput "
            "path for a longer run; it cannot validate resistance accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("precision_evidence", type=Path)
    parser.add_argument("--maximum-seconds-per-root-step", type=float, default=15.0)
    parser.add_argument("--maximum-peak-allocated-gib", type=float, default=22.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assessment = assess(
        args.result,
        args.precision_evidence,
        maximum_seconds_per_root_step=args.maximum_seconds_per_root_step,
        maximum_peak_allocated_gib=args.maximum_peak_allocated_gib,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(assessment, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(assessment, indent=2, allow_nan=False))
    if not assessment["acceptance"]["long_run_launch_admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
