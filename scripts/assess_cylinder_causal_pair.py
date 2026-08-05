#!/usr/bin/env python3
"""Assess one controlled intervention between two cylinder results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PHYSICAL_IDENTITY_FIELDS = (
    "shape_zyx",
    "radius",
    "center_x_fraction",
    "reynolds",
    "sponge_width",
    "sponge_strength",
    "sponge_inlet",
    "cv_margin",
    "far_field_mode",
    "periodic_axes",
    "link_force_frame",
    "minimum_shedding_cycles",
    "domain_clearance_diameters",
)
TEMPORAL_FIELDS = (
    "steps",
    "warmup_steps",
    "ramp_steps",
    "statistics_window_steps_resolved",
)
HENDERSON_RE100_CD = (
    2.5818 / 100.0**0.4369
    + 1.4114
    - 0.2668 * 100.0**0.1648 * math.exp(-3.375e-3 * 100.0)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tensorlbm-cylinder-bfl-control-volume-v4":
        raise ValueError(f"unsupported cylinder result schema: {path}")
    return payload


def _convective_extents(configuration: dict) -> dict[str, float]:
    diameter = 2.0 * float(configuration["radius"])
    speed = float(configuration["lattice_speed"])
    return {
        field: float(configuration[field]) * speed / diameter
        for field in TEMPORAL_FIELDS
    }


def _case(path: Path, payload: dict) -> dict:
    configuration = payload["configuration"]
    result = payload["result"]
    acceptance = payload["acceptance"]
    cd = float(result["cd_control_volume"])
    values = {
        "cd_control_volume": cd,
        "cd_bfl_link": float(result["cd_bfl_link"]),
        "strouhal": float(result["strouhal"]),
        "observer_difference_pct": float(result["observer_difference_pct"]),
        "shedding_cycles_observed": float(result["shedding_cycles_observed"]),
        "stationarity_range_pct": float(
            result["drag_stationarity"]["relative_range_pct"]
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite cylinder observable: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "lattice_speed": float(configuration["lattice_speed"]),
        "lattice_mach": float(configuration["lattice_speed"]) * math.sqrt(3.0),
        "tau": float(configuration["tau"]),
        "collision_model": configuration["collision_model"],
        "convective_extents": _convective_extents(configuration),
        **values,
        "cd_reference": HENDERSON_RE100_CD,
        "cd_reference_error_pct": abs(cd - HENDERSON_RE100_CD)
        / HENDERSON_RE100_CD
        * 100.0,
        "numerical_quality_admitted": bool(
            acceptance["numerical_quality_admitted"]
        ),
        "stationarity_target_met": bool(acceptance["stationarity_target_met"]),
        "force_observer_target_met": bool(
            acceptance["force_observer_target_met"]
        ),
        "cycle_target_met": bool(acceptance["cycle_target_met"]),
        "domain_reference_target_met": bool(
            acceptance["domain_reference_target_met"]
        ),
    }


def assess(baseline_path: Path, candidate_path: Path, intervention: str) -> dict:
    if intervention not in {"collision_model", "lattice_mach"}:
        raise ValueError("intervention must be collision_model or lattice_mach")
    baseline_payload = _load(baseline_path)
    candidate_payload = _load(candidate_path)
    baseline_config = baseline_payload["configuration"]
    candidate_config = candidate_payload["configuration"]
    missing = [
        field
        for field in (*PHYSICAL_IDENTITY_FIELDS, *TEMPORAL_FIELDS, "tau")
        if field not in baseline_config or field not in candidate_config
    ]
    if missing:
        raise ValueError(f"configuration lacks required fields: {sorted(set(missing))}")
    mismatches = {
        field: [baseline_config[field], candidate_config[field]]
        for field in PHYSICAL_IDENTITY_FIELDS
        if baseline_config[field] != candidate_config[field]
    }
    if mismatches:
        raise ValueError(f"physical configuration identity mismatch: {mismatches}")

    baseline = _case(baseline_path, baseline_payload)
    candidate = _case(candidate_path, candidate_payload)
    temporal_relative_errors = {
        field: abs(
            candidate["convective_extents"][field]
            / baseline["convective_extents"][field]
            - 1.0
        )
        for field in TEMPORAL_FIELDS
    }
    temporal_equivalent = max(temporal_relative_errors.values()) <= 1.0e-12

    if intervention == "collision_model":
        if baseline["collision_model"] == candidate["collision_model"]:
            raise ValueError("collision_model intervention did not change collision")
        if baseline["lattice_speed"] != candidate["lattice_speed"]:
            raise ValueError("collision_model intervention changed lattice speed")
        if any(
            baseline_config[field] != candidate_config[field]
            for field in TEMPORAL_FIELDS
        ):
            raise ValueError("collision_model intervention changed time window")
    else:
        if baseline["collision_model"] != candidate["collision_model"]:
            raise ValueError("lattice_mach intervention changed collision model")
        if not candidate["lattice_mach"] < baseline["lattice_mach"]:
            raise ValueError("candidate lattice Mach must be lower than baseline")
        if not temporal_equivalent:
            raise ValueError("lattice_mach cases lack convective-time equivalence")

    health_fields = (
        "numerical_quality_admitted",
        "stationarity_target_met",
        "force_observer_target_met",
        "cycle_target_met",
        "domain_reference_target_met",
    )
    causal_quality = temporal_equivalent and all(
        case[field] for case in (baseline, candidate) for field in health_fields
    )
    cd_change_pct = (
        candidate["cd_control_volume"] / baseline["cd_control_volume"] - 1.0
    ) * 100.0
    strouhal_change_pct = (
        candidate["strouhal"] / baseline["strouhal"] - 1.0
    ) * 100.0
    return {
        "schema": "tensorlbm-cylinder-causal-pair-v1",
        "status": "causal_pair_admitted" if causal_quality else "rejected",
        "physical_validation": False,
        "intervention": intervention,
        "physical_configuration_identity": {
            field: baseline_config[field] for field in PHYSICAL_IDENTITY_FIELDS
        },
        "baseline": baseline,
        "candidate": candidate,
        "effect": {
            "cd_change_pct": cd_change_pct,
            "strouhal_change_pct": strouhal_change_pct,
            "cd_reference_error_change_percentage_points": (
                candidate["cd_reference_error_pct"]
                - baseline["cd_reference_error_pct"]
            ),
        },
        "acceptance": {
            "physical_configuration_identity_admitted": True,
            "convective_time_equivalence_admitted": temporal_equivalent,
            "convective_time_relative_errors": temporal_relative_errors,
            "both_numerically_healthy": causal_quality,
            "causal_pair_admitted": causal_quality,
            "grid_convergence_assessed": False,
            "physical_validation": False,
        },
        "prohibition": (
            "This pair isolates one declared numerical intervention. It must "
            "not be selected by reference proximity, extrapolated, or used "
            "as an empirical correction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--intervention", choices=("collision_model", "lattice_mach"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.baseline, args.candidate, args.intervention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
