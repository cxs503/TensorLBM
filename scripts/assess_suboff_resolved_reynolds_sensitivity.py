#!/usr/bin/env python3
"""Assess a configuration-matched SUBOFF resolved-Reynolds sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

IDENTITY_FIELDS = (
    "hull_type",
    "speed_knots",
    "nx",
    "ny",
    "nz",
    "hull_length",
    "center_x_fraction",
    "outer_wall_margin",
    "outer_wake_cells",
    "inner_wall_margin",
    "inner_wake_cells",
    "deep_wall_margin",
    "deep_wake_cells",
    "cv_margin",
    "aux_cv_margins",
    "steps",
    "warmup_steps",
    "statistics_window_steps",
    "ramp_steps",
    "wall_normal_ramp_steps",
    "wall_shear_ramp_steps",
    "lattice_speed",
    "resolved_reynolds_start",
    "viscosity_ramp_start_step",
    "viscosity_ramp_end_step",
    "collision_model",
    "compile_natural_kbc",
    "natural_kbc_compute_dtype",
    "wall_law",
    "stress_exchange_distance",
    "sponge_width",
    "sponge_strength",
    "sponge_inlet",
    "far_field_mode",
    "ghost_interpolation",
    "reflux_correction_stencil",
    "interface_filter_width",
    "interface_filter_strength",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_trend(values: list[float]) -> str:
    differences = [
        right - left for left, right in zip(values, values[1:], strict=False)
    ]
    if all(value < 0.0 for value in differences):
        return "strictly_decreasing"
    if all(value > 0.0 for value in differences):
        return "strictly_increasing"
    return "non_monotonic"


def assess(
    paths: list[Path],
    *,
    viscosity_schedule_paths: list[Path] | None = None,
) -> dict:
    if len(paths) < 3:
        raise ValueError("resolved-Reynolds sensitivity requires at least 3 cases")
    cases = []
    identity = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "tensorlbm-suboff-nested-amr-smoke-v3":
            raise ValueError(f"unsupported SUBOFF result schema: {path}")
        configuration = dict(payload["configuration"])
        configuration.setdefault("natural_kbc_compute_dtype", "storage")
        configuration.setdefault("population_storage_dtype", "float32")
        missing = [field for field in IDENTITY_FIELDS if field not in configuration]
        if missing:
            raise ValueError(f"configuration lacks identity fields {missing}: {path}")
        current_identity = {
            field: configuration[field] for field in IDENTITY_FIELDS
        }
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"configuration identity mismatch: {path}")
        statistics = payload["result"].get("statistics")
        if not statistics:
            raise ValueError(f"result lacks force statistics: {path}")
        acceptance = payload["acceptance"]
        values = {
            "mean_resistance_n": float(statistics["mean_resistance_n"]),
            "mean_link_plus_wall_stress_n": float(
                statistics["mean_bfl_plus_wall_stress_n"]
            ),
            "mean_link_impulse_n": float(statistics["mean_bfl_pressure_n"]),
            "mean_wall_shear_n": float(statistics["mean_wall_shear_n"]),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite force statistic: {path}")
        cases.append({
            "path": str(path),
            "sha256": _sha256(path),
            "resolved_reynolds": float(configuration["resolved_reynolds"]),
            "tau_by_level": [float(value) for value in configuration["tau_by_level"]],
            "population_storage_dtype": configuration["population_storage_dtype"],
            "natural_kbc_compute_dtype": configuration[
                "natural_kbc_compute_dtype"
            ],
            **values,
            "sampling_convective_times": float(
                statistics["sampling_convective_times"]
            ),
            "integration_smoke_admitted": bool(
                acceptance["integration_smoke_admitted"]
            ),
            "population_health_target_met": bool(
                acceptance["population_health_target_met"]
            ),
            "nested_control_volume_target_met": bool(
                acceptance["nested_control_volume_target_met"]
            ),
            "stationarity_target_met": bool(
                acceptance["stationarity_target_met"]
            ),
        })
    cases.sort(key=lambda item: item["resolved_reynolds"])
    reynolds = [item["resolved_reynolds"] for item in cases]
    if len(set(reynolds)) != len(reynolds) or any(
        right <= left
        for left, right in zip(reynolds, reynolds[1:], strict=False)
    ):
        raise ValueError("resolved Reynolds numbers must be strictly increasing")
    component_fields = (
        "mean_resistance_n",
        "mean_link_plus_wall_stress_n",
        "mean_link_impulse_n",
        "mean_wall_shear_n",
    )
    trends = {}
    for field in component_fields:
        values = [item[field] for item in cases]
        adjacent_changes_pct = [
            (right / left - 1.0) * 100.0
            for left, right in zip(values, values[1:], strict=False)
        ]
        trends[field] = {
            "values": values,
            "trend": _strict_trend(values),
            "adjacent_changes_pct": adjacent_changes_pct,
            "first_to_last_change_pct": (values[-1] / values[0] - 1.0) * 100.0,
        }
    health_quality = all(
        item["integration_smoke_admitted"]
        and item["population_health_target_met"]
        and item["nested_control_volume_target_met"]
        for item in cases
    )
    viscosity_certificates = []
    viscosity_certified = False
    if viscosity_schedule_paths is not None:
        if len(viscosity_schedule_paths) != len(cases):
            raise ValueError("provide one viscosity schedule per Reynolds case")
        for case, path in zip(cases, viscosity_schedule_paths, strict=True):
            schedule = json.loads(path.read_text(encoding="utf-8"))
            if schedule.get("schema") != "tensorlbm-collision-viscosity-schedule-v1":
                raise ValueError(f"unsupported viscosity schedule schema: {path}")
            scheduled_taus = [float(value) for value in schedule["taus"]]
            if len(scheduled_taus) != len(case["tau_by_level"]) or any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, expected in zip(
                    scheduled_taus, case["tau_by_level"], strict=True,
                )
            ):
                raise ValueError(f"viscosity schedule tau mismatch: {path}")
            if schedule.get("dtype") != case["population_storage_dtype"]:
                raise ValueError(f"viscosity schedule storage dtype mismatch: {path}")
            if schedule.get("natural_kbc_compute_dtype", "storage") != case[
                "natural_kbc_compute_dtype"
            ]:
                raise ValueError(f"viscosity schedule compute dtype mismatch: {path}")
            admitted = bool(
                schedule.get("acceptance", {}).get(
                    "all_levels_recover_configured_viscosity",
                )
            )
            viscosity_certificates.append({
                "path": str(path),
                "sha256": _sha256(path),
                "resolved_reynolds": case["resolved_reynolds"],
                "admitted": admitted,
            })
        viscosity_certified = all(
            item["admitted"] for item in viscosity_certificates
        )
    causal_quality = health_quality and viscosity_certified
    status = "causal_sequence_admitted" if causal_quality else (
        "rejected_uncertified_collision_viscosity"
        if not viscosity_certified else "rejected"
    )
    return {
        "schema": "tensorlbm-suboff-resolved-reynolds-sensitivity-v2",
        "status": status,
        "physical_validation": False,
        "configuration_identity": identity,
        "cases": cases,
        "component_trends": trends,
        "collision_viscosity_certificates": viscosity_certificates,
        "acceptance": {
            "configuration_identity_admitted": True,
            "strictly_increasing_resolved_reynolds": True,
            "all_integration_health_gates_met": health_quality,
            "all_collision_viscosity_schedules_admitted": (
                viscosity_certified
            ),
            "all_stationarity_gates_met": all(
                item["stationarity_target_met"] for item in cases
            ),
            "causal_sensitivity_admitted": causal_quality,
            "time_convergence_assessed": False,
            "grid_convergence_assessed": False,
            "continuum_reynolds_extrapolation_admitted": False,
            "physical_validation": False,
        },
        "prohibition": (
            "The sequence diagnoses collision-viscosity sensitivity only, "
            "and only when every configured tau has an admitted recovered-"
            "viscosity certificate. It must not be extrapolated or used as "
            "an empirical force correction toward the experiment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--viscosity-schedule",
        action="append",
        type=Path,
        dest="viscosity_schedules",
        help="one recovered-viscosity schedule per result, in increasing Re order",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(
        args.results,
        viscosity_schedule_paths=args.viscosity_schedules,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
