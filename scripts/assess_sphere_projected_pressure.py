#!/usr/bin/env python3
"""Assess paired projected-pressure and control-volume sphere histories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from tensorlbm.force_convergence import assess_force_stationarity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(result_path: Path, checkpoint_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "tensorlbm-sphere-bfl-control-volume-v3":
        raise ValueError("unsupported sphere result schema")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configuration = result["configuration"]
    if int(state["step"]) != int(configuration["steps"]):
        raise ValueError("sphere checkpoint is not co-terminal with result")
    state_configuration = state.get("configuration", {})
    for field in (
        "shape_zyx", "radius", "reynolds", "lattice_speed",
        "collision_model", "far_field_mode",
    ):
        if state_configuration.get(field) != configuration.get(field):
            raise ValueError(f"checkpoint/result identity mismatch: {field}")

    statistics_window = int(configuration["statistics_window_steps_resolved"])
    start_step = int(configuration["steps"]) - statistics_window + 1
    samples = [
        sample for sample in state.get("projected_pressure_samples", [])
        if int(sample["step"]) >= start_step
    ]
    projected = [float(sample["pressure_force_x"]) for sample in samples]
    primary = [
        float(sample["paired_control_volume_force_x"]) for sample in samples
    ]
    finite = bool(samples) and all(
        math.isfinite(value) for value in (*projected, *primary)
    )
    projected_stationarity = assess_force_stationarity(
        projected,
        block_size=max(1, len(projected) // 8),
    )
    projected_mean = (
        sum(projected) / len(projected) if projected else math.nan
    )
    primary_mean = sum(primary) / len(primary) if primary else math.nan
    scale = max(abs(primary_mean), 1.0e-30)
    differences = [
        candidate - reference
        for candidate, reference in zip(projected, primary, strict=True)
    ]
    mean_difference_pct = (
        abs(projected_mean - primary_mean) / scale * 100.0
        if finite else math.inf
    )
    rms_difference_pct = (
        math.sqrt(sum(value * value for value in differences) / len(differences))
        / scale * 100.0
        if finite else math.inf
    )
    maximum_difference_pct = (
        max(abs(value) for value in differences) / scale * 100.0
        if finite else math.inf
    )
    usable_fractions = [
        float(sample["diagnostics"]["usable_links"])
        / max(float(sample["diagnostics"]["requested_links"]), 1.0)
        for sample in samples
    ]
    minimum_usable_fraction = min(usable_fractions, default=0.0)
    maximum_fallback_cells = max(
        (int(sample["diagnostics"]["fallback_cells"]) for sample in samples),
        default=0,
    )
    registered_gates = {
        "minimum_samples": 32,
        "maximum_stationarity_pct": 1.0,
        "maximum_mean_difference_pct": 5.0,
        "maximum_rms_difference_pct": 5.0,
        "minimum_usable_link_fraction": 1.0,
        "maximum_fallback_cells": 0,
    }
    baseline_admitted = result["acceptance"].get("admitted") is True
    candidate = (
        baseline_admitted
        and finite
        and len(samples) >= registered_gates["minimum_samples"]
        and projected_stationarity.meets(
            registered_gates["maximum_stationarity_pct"],
        )
        and mean_difference_pct
        <= registered_gates["maximum_mean_difference_pct"]
        and rms_difference_pct
        <= registered_gates["maximum_rms_difference_pct"]
        and minimum_usable_fraction
        >= registered_gates["minimum_usable_link_fraction"]
        and maximum_fallback_cells
        <= registered_gates["maximum_fallback_cells"]
    )
    return {
        "schema": "tensorlbm-sphere-projected-pressure-assessment-v1",
        "status": (
            "single_grid_candidate_convergence_required"
            if candidate else "rejected"
        ),
        "physical_validation": False,
        "configuration": {
            "radius_cells": configuration["radius"],
            "reynolds": configuration["reynolds"],
            "statistics_window_steps": statistics_window,
            "sample_interval_steps": configuration[
                "projected_pressure_interval"
            ],
            "reconstruction": configuration[
                "projected_pressure_reconstruction"
            ],
        },
        "registered_gates": registered_gates,
        "observations": {
            "samples": len(samples),
            "finite": finite,
            "projected_pressure_mean_force": projected_mean,
            "paired_control_volume_mean_force": primary_mean,
            "mean_difference_pct": mean_difference_pct,
            "rms_pair_difference_pct": rms_difference_pct,
            "maximum_pair_difference_pct": maximum_difference_pct,
            "projected_force_stationarity": projected_stationarity.to_dict(),
            "minimum_usable_link_fraction": minimum_usable_fraction,
            "maximum_fallback_cells": maximum_fallback_cells,
        },
        "acceptance": {
            "baseline_sphere_admitted": baseline_admitted,
            "sample_count_target_met": (
                len(samples) >= registered_gates["minimum_samples"]
            ),
            "projected_stationarity_target_met": (
                projected_stationarity.meets(
                    registered_gates["maximum_stationarity_pct"],
                )
            ),
            "mean_pair_target_met": (
                mean_difference_pct
                <= registered_gates["maximum_mean_difference_pct"]
            ),
            "rms_pair_target_met": (
                rms_difference_pct
                <= registered_gates["maximum_rms_difference_pct"]
            ),
            "coverage_target_met": (
                minimum_usable_fraction == 1.0
                and maximum_fallback_cells == 0
            ),
            "single_grid_candidate": candidate,
            "grid_convergence_assessed": False,
            "used_for_production_force": False,
        },
        "artifacts": {
            "result": str(result_path),
            "result_sha256": _sha256(result_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "prohibition": (
            "Passing this paired single-grid diagnostic cannot promote the "
            "projected-pressure observer. R9/R12/R15 convergence is required; "
            "a failed gate rejects it without substituting another force."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess(args.result, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
