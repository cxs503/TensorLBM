#!/usr/bin/env python3
"""Read-only audit of SUBOFF surface-pressure observers at a checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorlbm.d3q19 import macroscopic3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_3d,
    integrate_bfl_projected_pressure,
    reconstruct_bfl_wall_pressure,
)
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_static_amr import (
    build_fine_suboff_mask,
    plan_suboff_static_amr,
)
from tensorlbm.surface_area_weights import bfl_surface_area_weights

MODEL_LENGTH_M = 4.3561


def _latest(values: torch.Tensor) -> float | None:
    return float(values[-1].item()) if values.numel() else None


def audit_checkpoint(path: Path, *, device: torch.device) -> dict[str, object]:
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("schema") != "tensorlbm-suboff-static-amr-checkpoint-v8":
        raise ValueError("checkpoint is not SUBOFF static-AMR schema v8")
    cfg = state.get("configuration")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint has no configuration dictionary")
    if cfg.get("hull_type") != "bare_hull":
        raise ValueError("the q-aware audit currently supports bare_hull only")
    nz, ny, nx = (int(value) for value in cfg["coarse_shape_zyx"])
    hull_length = float(cfg["hull_length"])
    center = (
        nx * float(cfg["center_x_fraction"]),
        ny / 2.0,
        nz / 2.0,
    )
    geometry_config = SuboffConfig()
    coarse_solid, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=hull_length,
        config=geometry_config, device=device,
    )
    plan = plan_suboff_static_amr(
        coarse_solid,
        coarse_hull_length=hull_length,
        wall_margin=int(cfg["wall_margin"]),
        wake_cells=int(cfg["wake_cells"]),
    )
    fine_solid, fine_geometry = build_fine_suboff_mask(
        plan, hull_type="bare_hull", coarse_center=center,
        config=geometry_config, device=device,
    )
    populations = state["fine_populations"].to(device=device)
    ghost = 1
    if populations.shape[1:] != tuple(size + 2 * ghost for size in fine_solid.shape):
        raise ValueError("checkpoint fine populations do not match reconstructed plan")
    fine_solid_g = torch.zeros(
        populations.shape[1:], dtype=torch.bool, device=device,
    )
    fine_solid_g[ghost:-ghost, ghost:-ghost, ghost:-ghost] = fine_solid
    fine_center = (
        center[0] * 2.0 - plan.box.x0 * 2.0 + ghost,
        center[1] * 2.0 - plan.box.y0 * 2.0 + ghost,
        center[2] * 2.0 - plan.box.z0 * 2.0 + ghost,
    )
    bfl_mask, bfl_q = compute_q_suboff(
        populations.shape[3], populations.shape[2], populations.shape[1],
        *fine_center, hull_length * 2.0,
        hull_type="bare_hull", config=geometry_config, device=device,
        solid_mask=fine_solid_g,
    )
    near = get_near_wall_3d(fine_solid_g)
    surface = SurfaceMesh.from_suboff(
        fine_solid_g, near, *fine_center,
        hull_length * 2.0, hull_length / 8.57,
        config=geometry_config,
    )
    area, area_diagnostics = bfl_surface_area_weights(
        bfl_mask, (surface.nx_n, surface.ny_n, surface.nz_n),
        reference_area=float(fine_geometry["wetted_area_lu2"]),
        boundary_mask=near,
    )
    surface.dA = area
    rho, _, _, _ = macroscopic3d(populations)
    pressure = (rho - 1.0) / 3.0
    inlet_width = 5
    inlet = (~fine_solid_g).clone()
    inlet[:, :, inlet_width:] = False
    p0 = pressure[inlet].mean()
    corrected = pressure - p0
    wall_pressure, reconstruction = reconstruct_bfl_wall_pressure(
        corrected, surface, bfl_mask, bfl_q, solid=fine_solid_g,
    )
    q_aware_force_lu = float(
        -(wall_pressure * surface.nx_n * surface.dA * near).sum().item(),
    )
    projected_force_lu, projected_diagnostics = integrate_bfl_projected_pressure(
        corrected, bfl_mask, bfl_q, solid=fine_solid_g,
    )
    speed_knots = float(cfg["speed_knots"])
    speed_mps = speed_knots * 0.514444
    dx_m = MODEL_LENGTH_M / (2.0 * hull_length)
    scale = (
        float(cfg["rho_water"]) * dx_m**2
        * (speed_mps / float(cfg["lattice_speed"])) ** 2
    )
    old = {
        mode: drag_pressure_integration(
            populations, surface, 1.0, extrap=mode,
            p0_method="inlet", solid=fine_solid_g,
        )[0] * scale
        for mode in ("none", "linear", "quadratic")
    }
    pressure_history = state["pressure_history"]
    wall_history = state["wall_shear_history"]
    cv_history = state["force_history"]
    bfl_history = state["bfl_total_history"]
    latest_wall = _latest(wall_history)
    q_aware_total = (
        q_aware_force_lu * scale + latest_wall
        if latest_wall is not None else None
    )
    projected_pressure_n = projected_force_lu[0] * scale
    projected_total = (
        projected_pressure_n + latest_wall
        if latest_wall is not None else None
    )
    latest_cv = _latest(cv_history)
    difference_pct = (
        abs(q_aware_total - latest_cv) / max(abs(latest_cv), 1.0e-30) * 100.0
        if q_aware_total is not None and latest_cv is not None else None
    )
    return {
        "schema": "tensorlbm-suboff-surface-pressure-checkpoint-audit-v1",
        "checkpoint": str(path),
        "checkpoint_step": int(state["step"]),
        "configuration": {
            "hull_type": cfg["hull_type"],
            "coarse_hull_length_cells": hull_length,
            "fine_hull_length_cells": 2.0 * hull_length,
            "pressure_reference": "inlet",
            "reconstruction": "linkwise_bfl_quadratic_actual_q",
        },
        "instantaneous_force_n": {
            "control_volume": latest_cv,
            "bfl_pressure": _latest(pressure_history),
            "wall_shear": latest_wall,
            "bfl_plus_wall_shear": _latest(bfl_history),
            "surface_pressure_old": old,
            "surface_pressure_bfl_q_aware": q_aware_force_lu * scale,
            "surface_bfl_q_aware_plus_wall_shear": q_aware_total,
            "q_aware_total_vs_control_volume_difference_pct": difference_pct,
            "surface_pressure_bfl_projected": projected_pressure_n,
            "surface_bfl_projected_plus_wall_shear": projected_total,
            "projected_total_vs_control_volume_difference_pct": (
                abs(projected_total - latest_cv)
                / max(abs(latest_cv), 1.0e-30) * 100.0
                if projected_total is not None and latest_cv is not None
                else None
            ),
        },
        "reconstruction_coverage": reconstruction.__dict__,
        "projected_reconstruction_coverage": projected_diagnostics.__dict__,
        "surface_area": area_diagnostics.__dict__,
        "finite": all(
            math.isfinite(value)
            for value in (
                q_aware_force_lu,
                *old.values(),
                *(value for value in (latest_cv, latest_wall) if value is not None),
            )
        ),
        "claim_boundary": (
            "A single checkpoint compares co-temporal observers only; it is "
            "not a time-averaged resistance or validation result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_checkpoint(args.checkpoint, device=torch.device(args.device))
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
