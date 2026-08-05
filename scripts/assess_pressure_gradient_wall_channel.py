#!/usr/bin/env python3
"""Assess ODE wall shear against a wall-resolved pressure-driven channel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.pressure_gradient_wall_model import (  # noqa: E402
    solve_pressure_gradient_equilibrium_wall_shear,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("run_directory", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--minimum-y-plus", type=float, default=30.0)
    result.add_argument("--maximum-y-plus", type=float, default=80.0)
    result.add_argument("--maximum-mean-u-tau-error-pct", type=float, default=5.0)
    result.add_argument(
        "--maximum-recent-speed-range-fraction",
        type=float,
        default=0.01,
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assess(
    run_directory: Path,
    *,
    minimum_y_plus: float = 30.0,
    maximum_y_plus: float = 80.0,
    maximum_mean_u_tau_error_pct: float = 5.0,
    maximum_recent_speed_range_fraction: float = 0.01,
) -> dict:
    if not 0.0 < minimum_y_plus < maximum_y_plus:
        raise ValueError("invalid y+ assessment interval")
    if maximum_mean_u_tau_error_pct <= 0.0:
        raise ValueError("maximum mean u_tau error must be positive")
    if maximum_recent_speed_range_fraction <= 0.0:
        raise ValueError("maximum recent speed range fraction must be positive")
    metadata_path = run_directory / "run_metadata.json"
    profile_path = run_directory / "velocity_profile.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    derived = metadata["derived"]
    expected_u_tau = float(config["u_tau"])
    nu = float(derived["nu"])
    body_force = float(derived["body_force"])
    height = int(derived["H"])
    speed = []
    distance = []
    actual_y_plus = []
    with profile_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            y = float(row["y"])
            wall_distance = min(y - 0.5, height + 0.5 - y)
            y_plus = wall_distance * expected_u_tau / nu
            if minimum_y_plus <= y_plus <= maximum_y_plus:
                distance.append(wall_distance)
                actual_y_plus.append(y_plus)
                speed.append(float(row["u_plus"]) * expected_u_tau)
    if not speed:
        raise ValueError("channel profile has no samples in requested y+ interval")
    speed_tensor = torch.tensor(speed, dtype=torch.float64)
    distance_tensor = torch.tensor(distance, dtype=torch.float64)
    model_results = {}
    for model in ("van_driest", "duprat"):
        result = solve_pressure_gradient_equilibrium_wall_shear(
            speed_tensor,
            distance_tensor,
            -body_force,
            nu,
            pressure_gradient_magnitude_acceleration=body_force,
            eddy_viscosity_model=model,
        )
        prediction = result.friction_velocity
        relative_error = (prediction - expected_u_tau) / expected_u_tau
        mean_error_pct = float(relative_error.mean().item() * 100.0)
        rms_error_pct = float(
            torch.sqrt(relative_error.square().mean()).item() * 100.0,
        )
        model_results[model] = {
            "attached_samples": int(result.attached.sum().item()),
            "sample_count": int(prediction.numel()),
            "minimum_predicted_u_tau": float(prediction.min().item()),
            "mean_predicted_u_tau": float(prediction.mean().item()),
            "maximum_predicted_u_tau": float(prediction.max().item()),
            "mean_u_tau_error_pct": mean_error_pct,
            "rms_u_tau_error_pct": rms_error_pct,
            "maximum_absolute_u_tau_error_pct": float(
                relative_error.abs().max().item() * 100.0,
            ),
            "mean_error_target_met": (
                abs(mean_error_pct) <= maximum_mean_u_tau_error_pct
            ),
        }
    diagnostics = metadata.get("diagnostics", [])
    recent_speeds = [float(item["max_speed"]) for item in diagnostics[-3:]]
    recent_range_fraction = (
        (max(recent_speeds) - min(recent_speeds))
        / max(abs(sum(recent_speeds) / len(recent_speeds)), 1.0e-30)
        if len(recent_speeds) >= 2 else None
    )
    reference_stationary = (
        recent_range_fraction is not None
        and recent_range_fraction <= maximum_recent_speed_range_fraction
    )
    for model_result in model_results.values():
        model_result["admitted"] = bool(
            reference_stationary and model_result["mean_error_target_met"]
        )
    return {
        "schema": "tensorlbm-pressure-gradient-wall-channel-assessment-v1",
        "status": "canonical_diagnostic_only",
        "physical_validation": False,
        "source": {
            "run_directory": str(run_directory),
            "metadata_sha256": _sha256(metadata_path),
            "profile_sha256": _sha256(profile_path),
        },
        "configuration": {
            "re_tau": float(config["re_tau"]),
            "expected_u_tau": expected_u_tau,
            "nu": nu,
            "body_force_acceleration": body_force,
            "channel_height_cells": height,
            "y_plus_interval": [minimum_y_plus, maximum_y_plus],
            "sample_y_plus": actual_y_plus,
            "maximum_mean_u_tau_error_pct": maximum_mean_u_tau_error_pct,
            "maximum_recent_speed_range_fraction": (
                maximum_recent_speed_range_fraction
            ),
        },
        "wall_resolved_reference": {
            "averaging_samples": int(metadata["averaging_samples"]),
            "log_law_rms_error": metadata.get("log_law_rms_error"),
            "recent_max_speed_range_fraction": recent_range_fraction,
            "stationarity_target_met": reference_stationary,
            "target_wall_shear_definition": (
                "exact fully-developed channel momentum balance: "
                "u_tau^2 = body_force * H / 2"
            ),
        },
        "models": model_results,
        "production_force_changed": False,
    }


def main() -> None:
    args = parser().parse_args()
    result = assess(
        args.run_directory,
        minimum_y_plus=args.minimum_y_plus,
        maximum_y_plus=args.maximum_y_plus,
        maximum_mean_u_tau_error_pct=args.maximum_mean_u_tau_error_pct,
        maximum_recent_speed_range_fraction=(
            args.maximum_recent_speed_range_fraction
        ),
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
