#!/usr/bin/env python3
"""Read-only wall audit for a finite flat-plate checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.flat_plate_wall_model import _halfway_links  # noqa: E402
from tensorlbm.d3q19 import macroscopic3d  # noqa: E402
from tensorlbm.pressure_gradient_wall_model import (  # noqa: E402
    solve_pressure_gradient_equilibrium_wall_shear,
)
from tensorlbm.spalding_wall_model import (  # noqa: E402
    sample_wall_exchange_velocity,
)
from tensorlbm.wall_checkpoint_diagnostics import (  # noqa: E402
    diagnose_bfl_wall_exchange_state,
)
from tensorlbm.wall_model import physical_wall_lattice_viscosity  # noqa: E402
from tensorlbm.wall_pressure_gradient import (  # noqa: E402
    sample_wall_tangential_pressure_gradient,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("checkpoint", type=Path)
    result.add_argument("--device", default="cpu")
    result.add_argument("--output", type=Path)
    result.add_argument("--y-plus-lower-bound", type=float, default=30.0)
    result.add_argument("--y-plus-upper-bound", type=float, default=1000.0)
    result.add_argument(
        "--minimum-y-plus-in-range-fraction",
        type=float,
        default=0.9,
    )
    return result


def inspect_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    y_plus_lower_bound: float = 30.0,
    y_plus_upper_bound: float = 1000.0,
    minimum_y_plus_in_range_fraction: float = 0.9,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    schema = state.get("schema")
    if not isinstance(schema, str) or not schema.startswith(
        "tensorlbm-flat-plate-checkpoint-v",
    ):
        raise ValueError("not a flat-plate checkpoint")
    configuration = state.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint lacks a configuration mapping")
    populations = state.get("populations")
    if not isinstance(populations, torch.Tensor):
        raise ValueError("checkpoint lacks a population tensor")

    target = torch.device(device)
    populations = populations.to(device=target)
    shape = tuple(int(value) for value in configuration["shape_zyx"])
    if tuple(populations.shape) != (19, *shape):
        raise ValueError("population shape disagrees with configuration")
    plate_length = int(configuration["plate_length"])
    x0 = int(shape[2] * float(configuration["plate_start_fraction"]))
    x1 = x0 + plate_length
    plate_y = shape[1] // 2
    solid = torch.zeros(shape, dtype=torch.bool, device=target)
    solid[:, plate_y, x0:x1] = True
    near = torch.zeros_like(solid)
    near[:, plate_y - 1, x0:x1] = True
    near[:, plate_y + 1, x0:x1] = True
    normal_x = torch.zeros(shape, device=target)
    normal_y = torch.zeros(shape, device=target)
    normal_z = torch.zeros(shape, device=target)
    normal_y[:, plate_y - 1, x0:x1] = -1.0
    normal_y[:, plate_y + 1, x0:x1] = 1.0
    bfl_mask, bfl_q = _halfway_links(solid)
    wall_nu = physical_wall_lattice_viscosity(
        float(configuration["lattice_speed"]),
        plate_length,
        float(configuration["reynolds"]),
    )
    exchange_distance = configuration.get("stress_exchange_distance")
    diagnostics = diagnose_bfl_wall_exchange_state(
        populations,
        solid,
        bfl_mask,
        bfl_q,
        wall_nu,
        wall_law=str(configuration["wall_law"]),
        near_mask=near,
        stress_exchange_distance=(
            float(exchange_distance) if exchange_distance is not None else 0.5
        ),
        wall_normals=(normal_x, normal_y, normal_z),
        y_plus_lower_bound=y_plus_lower_bound,
        y_plus_upper_bound=y_plus_upper_bound,
        minimum_y_plus_in_range_fraction=(minimum_y_plus_in_range_fraction),
        pressure_gradient_periodic_axes=(0,),
    )
    density, velocity_x, velocity_y, velocity_z = macroscopic3d(populations)
    requested_exchange_distance = (
        float(exchange_distance) if exchange_distance is not None else 0.5
    )
    samples = sample_wall_exchange_velocity(
        (velocity_x, velocity_y, velocity_z),
        bfl_mask,
        bfl_q,
        (normal_x, normal_y, normal_z),
        exchange_distance=requested_exchange_distance,
        boundary_mask=near,
        fluid_mask=~solid,
    )
    normal_velocity = (
        samples.velocity_x * samples.normal_x
        + samples.velocity_y * samples.normal_y
        + samples.velocity_z * samples.normal_z
    )
    tangent = torch.stack((
        samples.velocity_x - normal_velocity * samples.normal_x,
        samples.velocity_y - normal_velocity * samples.normal_y,
        samples.velocity_z - normal_velocity * samples.normal_z,
    ), dim=1)
    speed = torch.linalg.vector_norm(tangent, dim=1)
    tangent_direction = tangent / speed[:, None].clamp_min(
        torch.finfo(speed.dtype).tiny,
    )
    gradient = sample_wall_tangential_pressure_gradient(
        (density - 1.0) / 3.0,
        solid,
        samples.boundary,
        (normal_x, normal_y, normal_z),
        periodic_axes=(0,),
    )
    active_density = density[samples.boundary]
    density_floor = active_density.clamp_min(
        torch.finfo(active_density.dtype).tiny,
    )
    signed_acceleration = (
        gradient.vector * tangent_direction
    ).sum(dim=1) / density_floor
    magnitude_acceleration = gradient.magnitude / density_floor
    signed_acceleration = torch.where(
        gradient.valid,
        signed_acceleration,
        torch.full_like(signed_acceleration, torch.nan),
    )
    magnitude_acceleration = torch.where(
        gradient.valid,
        magnitude_acceleration,
        torch.full_like(magnitude_acceleration, torch.nan),
    )
    indices = samples.boundary.nonzero(as_tuple=False)
    central = (
        (indices[:, 2] >= x0 + 0.1 * plate_length)
        & (indices[:, 2] < x1 - 0.1 * plate_length)
    )
    central_production_shear_x = sum(
        float(item["signed_shear_x_sum_lu"])
        for item in (diagnostics.wall_shear_axial_profile or [])
        if item["normalized_x_lower"] >= 0.1
        and item["normalized_x_upper"] <= 0.9
    )
    candidate_audit = {
        "production_reference": {
            "wall_law": str(configuration["wall_law"]),
            "shear_force_x_lu": float(diagnostics.shear_force[0]),
            "central_10_90_shear_force_x_lu": central_production_shear_x,
        },
    }
    for model in ("van_driest", "duprat"):
        candidate = solve_pressure_gradient_equilibrium_wall_shear(
            speed,
            samples.y2,
            signed_acceleration,
            wall_nu,
            pressure_gradient_magnitude_acceleration=magnitude_acceleration,
            eddy_viscosity_model=model,
        )
        shear_x = (
            candidate.shear_stress_over_density * tangent_direction[:, 0]
        )
        central_area = central.sum().clamp_min(1)
        candidate_audit[model] = {
            "scope": "diagnostic_only_not_a_force_correction",
            "requested_nodes": int(speed.numel()),
            "valid_gradient_nodes": gradient.valid_nodes,
            "attached_fraction": float(
                candidate.attached.to(dtype=torch.float64).mean().item(),
            ),
            "separated_fraction": float(
                candidate.separated.to(dtype=torch.float64).mean().item(),
            ),
            "central_10_90_attached_fraction": float(
                candidate.attached[central].sum().div(central_area).item(),
            ),
            "central_10_90_separated_fraction": float(
                candidate.separated[central].sum().div(central_area).item(),
            ),
            "shear_force_x_lu": float(shear_x.sum().item()),
            "central_10_90_shear_force_x_lu": float(
                shear_x[central].sum().item(),
            ),
            "shear_force_vs_production_wall_law_ratio": (
                float(shear_x.sum().item())
                / max(abs(float(diagnostics.shear_force[0])), 1.0e-30)
            ),
            "central_10_90_shear_vs_production_wall_law_ratio": (
                float(shear_x[central].sum().item())
                / max(abs(central_production_shear_x), 1.0e-30)
            ),
            "maximum_attached_speed_residual_lu": (
                float(candidate.residual[candidate.attached].abs().max().item())
                if bool(candidate.attached.any()) else None
            ),
        }
    return {
        "schema": "tensorlbm-flat-plate-wall-checkpoint-audit-v1",
        "status": "diagnostic_only",
        "physical_validation": False,
        "source_path": str(path),
        "source_schema": schema,
        "source_step": int(state["step"]),
        "device": str(target),
        "population_shape": list(populations.shape),
        "plate_length_cells": plate_length,
        "wall_lattice_viscosity": wall_nu,
        "wall_exchange": asdict(diagnostics),
        "pressure_gradient_ode_candidate": candidate_audit,
        "population_state_advanced": False,
    }


def main() -> None:
    args = parser().parse_args()
    result = inspect_checkpoint(
        args.checkpoint,
        device=args.device,
        y_plus_lower_bound=args.y_plus_lower_bound,
        y_plus_upper_bound=args.y_plus_upper_bound,
        minimum_y_plus_in_range_fraction=(args.minimum_y_plus_in_range_fraction),
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
