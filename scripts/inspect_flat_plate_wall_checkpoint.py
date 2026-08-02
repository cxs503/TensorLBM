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
from tensorlbm.wall_checkpoint_diagnostics import (  # noqa: E402
    diagnose_bfl_wall_exchange_state,
)
from tensorlbm.wall_model import physical_wall_lattice_viscosity  # noqa: E402


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
    )
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
