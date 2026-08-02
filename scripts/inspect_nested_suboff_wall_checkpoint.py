#!/usr/bin/env python3
"""Read-only wall-exchange audit of a nested SUBOFF checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "examples"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from suboff_experimental_resistance import (  # noqa: E402
    MODEL_LENGTH_M,
    experimental_point,
    force_scale_newton,
)

from tensorlbm.drag_pressure import (  # noqa: E402
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_3d,
    integrate_bfl_projected_pressure,
)
from tensorlbm.interpolated_bc_suboff import (  # noqa: E402
    compute_q_suboff,
    refine_q_suboff_appendages,
)
from tensorlbm.static_block_amr import BoxRegion  # noqa: E402
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask  # noqa: E402
from tensorlbm.suboff_static_amr import (  # noqa: E402
    SuboffNestedStaticAMRPlan,
    SuboffStaticAMRPlan,
    build_fine_suboff_mask,
    build_nested_fine_suboff_mask,
    plan_nested_suboff_static_amr,
    plan_suboff_static_amr,
)
from tensorlbm.surface_area_weights import bfl_surface_area_weights  # noqa: E402
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


def _box(value: object, label: str) -> BoxRegion:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint configuration lacks {label}")
    return BoxRegion(
        **{
            key: int(value[key])
            for key in (
                "x0",
                "x1",
                "y0",
                "y1",
                "z0",
                "z1",
            )
        }
    )


def _infer_outer_plan(
    solid: torch.Tensor,
    length: float,
    expected: BoxRegion,
) -> SuboffStaticAMRPlan:
    indices = solid.nonzero(as_tuple=False)
    minima = [int(indices[:, axis].min()) for axis in range(3)]
    maxima = [int(indices[:, axis].max()) + 1 for axis in range(3)]
    wall_margin = minima[1] - expected.y0
    wake_cells = expected.x1 - maxima[2] - wall_margin
    plan = plan_suboff_static_amr(
        solid,
        coarse_hull_length=length,
        wall_margin=wall_margin,
        wake_cells=wake_cells,
    )
    if plan.box != expected:
        raise ValueError("stored outer refinement box cannot be reconstructed")
    return plan


def _infer_nested_plan(
    outer: SuboffStaticAMRPlan | SuboffNestedStaticAMRPlan,
    solid: torch.Tensor,
    expected: BoxRegion,
) -> SuboffNestedStaticAMRPlan:
    indices = solid.nonzero(as_tuple=False)
    ghost = 1
    minima = [int(indices[:, axis].min()) + ghost for axis in range(3)]
    maxima = [int(indices[:, axis].max()) + 1 + ghost for axis in range(3)]
    wall_margin = minima[1] - expected.y0
    wake_cells = expected.x1 - maxima[2] - wall_margin
    plan = plan_nested_suboff_static_amr(
        outer,
        solid,
        wall_margin=wall_margin,
        wake_cells=wake_cells,
    )
    if plan.box_in_outer_allocated_coordinates != expected:
        raise ValueError("stored nested refinement box cannot be reconstructed")
    return plan


def inspect_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    y_plus_lower_bound: float = 30.0,
    y_plus_upper_bound: float = 1000.0,
    minimum_y_plus_in_range_fraction: float = 0.9,
) -> dict:
    """Rebuild exact CAD and diagnose the frozen finest population field."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    schema = state.get("schema")
    if not isinstance(schema, str) or not schema.startswith(
        "tensorlbm-suboff-nested-amr-smoke-checkpoint-v",
    ):
        raise ValueError("not a nested SUBOFF checkpoint")
    configuration = state.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint lacks a configuration mapping")
    populations = state.get("level_populations")
    if not isinstance(populations, list) or len(populations) < 2:
        raise ValueError("checkpoint must contain at least two hierarchy levels")

    shape = tuple(int(value) for value in configuration["shape_zyx"])
    length = float(configuration["hull_length"])
    center_fraction = float(configuration["center_x_fraction"])
    center = (shape[2] * center_fraction, shape[1] / 2.0, shape[0] / 2.0)
    hull_type = str(configuration["hull_type"])
    geometry_config = SuboffConfig()
    coarse_solid, _ = build_suboff_mask(
        hull_type,
        shape[2],
        shape[1],
        shape[0],
        cx=center[0],
        cy=center[1],
        cz=center[2],
        length=length,
        config=geometry_config,
        device="cpu",
    )
    outer = _infer_outer_plan(
        coarse_solid,
        length,
        _box(configuration.get("outer_box"), "outer_box"),
    )
    parent_solid, _ = build_fine_suboff_mask(
        outer,
        hull_type=hull_type,
        coarse_center=center,
        config=geometry_config,
        device="cpu",
    )
    nested_boxes = [_box(configuration.get("inner_box"), "inner_box")]
    if "deep_box" in configuration:
        nested_boxes.append(_box(configuration["deep_box"], "deep_box"))
    plan = outer
    geometry = None
    for expected in nested_boxes:
        plan = _infer_nested_plan(plan, parent_solid, expected)
        parent_solid, geometry = build_nested_fine_suboff_mask(
            plan,
            hull_type=hull_type,
            coarse_center=center,
            config=geometry_config,
            device="cpu",
        )
    if len(populations) != 2 + len(nested_boxes):
        raise ValueError("checkpoint level count disagrees with refinement boxes")
    if geometry is None:
        raise RuntimeError("nested geometry was not constructed")

    target = torch.device(device)
    finest = populations[-1].to(device=target)
    solid = F.pad(parent_solid, (1, 1, 1, 1, 1, 1)).to(device=target)
    if tuple(finest.shape[1:]) != tuple(solid.shape):
        raise ValueError("reconstructed finest solid does not match populations")
    finest_center = (
        float(geometry["cx"]) + 1.0,
        float(geometry["cy"]) + 1.0,
        float(geometry["cz"]) + 1.0,
    )
    finest_length = float(plan.effective_hull_length_cells)
    nz, ny, nx = solid.shape
    bfl_mask, bfl_q = compute_q_suboff(
        nx,
        ny,
        nz,
        *finest_center,
        finest_length,
        hull_type=hull_type,
        config=geometry_config,
        device=target,
        solid_mask=solid,
    )
    bare_solid = None
    if hull_type == "full":
        bare_solid, _ = build_suboff_mask(
            "bare_hull",
            nx,
            ny,
            nz,
            cx=finest_center[0],
            cy=finest_center[1],
            cz=finest_center[2],
            length=finest_length,
            config=geometry_config,
            device=target,
        )
        bfl_q, _ = refine_q_suboff_appendages(
            bfl_mask,
            bfl_q,
            solid,
            bare_solid,
            center=finest_center,
            length=finest_length,
            inplace=True,
        )
    near = get_near_wall_3d(solid)
    if hull_type == "bare_hull":
        surface = SurfaceMesh.from_suboff(
            solid,
            near,
            *finest_center,
            finest_length,
            finest_length / (2.0 * 8.57),
            config=geometry_config,
        )
        area_weight, area_diagnostics = bfl_surface_area_weights(
            bfl_mask,
            (surface.nx_n, surface.ny_n, surface.nz_n),
            reference_area=float(geometry["wetted_area_lu2"]),
            boundary_mask=near,
        )
    else:
        surface = SurfaceMesh.from_gradient(solid, near)
        assert bare_solid is not None
        bare_near = get_near_wall_3d(bare_solid)
        bare_surface = SurfaceMesh.from_gradient(bare_solid, bare_near)
        bare_mask, _ = compute_q_suboff(
            nx,
            ny,
            nz,
            *finest_center,
            finest_length,
            hull_type="bare_hull",
            config=geometry_config,
            device=target,
            solid_mask=bare_solid,
        )
        _, bare_area = bfl_surface_area_weights(
            bare_mask,
            (bare_surface.nx_n, bare_surface.ny_n, bare_surface.nz_n),
            reference_area=float(geometry["wetted_area_lu2"]),
            boundary_mask=bare_near,
        )
        area_weight, area_diagnostics = bfl_surface_area_weights(
            bfl_mask,
            (surface.nx_n, surface.ny_n, surface.nz_n),
            calibration_factor=bare_area.calibration_factor,
            boundary_mask=near,
        )

    point = experimental_point(hull_type, float(configuration["speed_knots"]))
    physical_reynolds = point.speed_mps * MODEL_LENGTH_M / float(configuration["nu_water"])
    wall_nu = physical_wall_lattice_viscosity(
        float(configuration["lattice_speed"]),
        finest_length,
        physical_reynolds,
    )
    diagnostics = diagnose_bfl_wall_exchange_state(
        finest,
        solid,
        bfl_mask,
        bfl_q,
        wall_nu,
        wall_law=str(configuration["wall_law"]),
        near_mask=near,
        stress_exchange_distance=float(configuration["stress_exchange_distance"]),
        wall_normals=(surface.nx_n, surface.ny_n, surface.nz_n),
        area_weight=area_weight,
        y_plus_lower_bound=y_plus_lower_bound,
        y_plus_upper_bound=y_plus_upper_bound,
        minimum_y_plus_in_range_fraction=minimum_y_plus_in_range_fraction,
    )
    scale = force_scale_newton(
        rho_water=float(configuration["rho_water"]),
        dx_m=MODEL_LENGTH_M / finest_length,
        speed_mps=point.speed_mps,
        lattice_speed=float(configuration["lattice_speed"]),
    )
    step_records = state.get("step_records", [])
    latest_record = step_records[-1] if step_records else None
    latest_cv = float(latest_record["cv_resistance_n"]) if isinstance(latest_record, dict) else None
    latest_wall_shear = (
        float(latest_record["wall_shear_n"]) if isinstance(latest_record, dict) else None
    )
    latest_bfl_pressure = (
        float(latest_record["bfl_pressure_n"])
        if isinstance(latest_record, dict)
        else None
    )
    unit_area = near.to(device=target, dtype=area_weight.dtype)
    area_policies = {
        "unit_node_area": unit_area,
        "calibrated_bfl_area": area_weight,
    }
    pressure_observers = {}
    surface_closure = {}
    for area_policy, local_area in area_policies.items():
        surface.dA = local_area
        weighted_normals = tuple(
            component.to(device=target, dtype=local_area.dtype) * local_area
            for component in (surface.nx_n, surface.ny_n, surface.nz_n)
        )
        surface_closure[area_policy] = {
            "area_sum_lu2": float(local_area.sum().item()),
            "signed_normal_area_sum_lu2": [
                float(component.sum().item()) for component in weighted_normals
            ],
            "absolute_normal_projection_sum_lu2": [
                float(component.abs().sum().item()) for component in weighted_normals
            ],
        }
        policy_observers = {}
        for pressure_reference in ("near_wall", "far_field", "inlet"):
            for reconstruction in ("none", "linear", "quadratic", "bfl_quadratic"):
                pressure_n = (
                    drag_pressure_integration(
                        finest,
                        surface,
                        1.0,
                        extrap=reconstruction,
                        p0_method=pressure_reference,
                        solid=solid,
                        fluid_boundary_mask=bfl_mask,
                        q_field=bfl_q,
                    )[0]
                    * scale
                )
                total_n = (
                    pressure_n + latest_wall_shear
                    if latest_wall_shear is not None
                    else None
                )
                policy_observers[f"{pressure_reference}:{reconstruction}"] = {
                    "pressure_resistance_n": pressure_n,
                    "pressure_plus_wall_shear_n": total_n,
                    "total_vs_control_volume_difference_pct": (
                        abs(total_n - latest_cv)
                        / max(abs(latest_cv), 1.0e-30)
                        * 100.0
                        if total_n is not None and latest_cv is not None
                        else None
                    ),
                }
        pressure_observers[area_policy] = policy_observers
    pressure_lu = (finest.sum(dim=0) - 1.0) / 3.0
    projected_pressure_observers = {}
    for reconstruction in ("local", "linear", "quadratic"):
        projected_force_lu, projected_diagnostics = (
            integrate_bfl_projected_pressure(
                pressure_lu,
                bfl_mask,
                bfl_q,
                solid=solid,
                reconstruction=reconstruction,
            )
        )
        projected_pressure_n = projected_force_lu[0] * scale
        projected_total_n = (
            projected_pressure_n + latest_wall_shear
            if latest_wall_shear is not None
            else None
        )
        projected_pressure_observers[reconstruction] = {
            "pressure_resistance_n": projected_pressure_n,
            "pressure_plus_wall_shear_n": projected_total_n,
            "total_vs_control_volume_difference_pct": (
                abs(projected_total_n - latest_cv)
                / max(abs(latest_cv), 1.0e-30)
                * 100.0
                if projected_total_n is not None and latest_cv is not None
                else None
            ),
            "runtime_momentum_exchange_difference_pct": (
                abs(projected_pressure_n - latest_bfl_pressure)
                / max(abs(latest_bfl_pressure), 1.0e-30)
                * 100.0
                if latest_bfl_pressure is not None
                else None
            ),
            "diagnostics": asdict(projected_diagnostics),
        }
    return {
        "schema": "tensorlbm-nested-suboff-wall-checkpoint-audit-v2",
        "status": "diagnostic_only",
        "physical_validation": False,
        "source_path": str(path),
        "source_schema": schema,
        "source_step": int(state["step"]),
        "device": str(target),
        "hull_type": hull_type,
        "level_count": len(populations),
        "finest_population_shape": list(finest.shape),
        "finest_hull_length_cells": finest_length,
        "physical_reynolds": physical_reynolds,
        "wall_lattice_viscosity": wall_nu,
        "stress_exchange_distance_cells": float(
            configuration["stress_exchange_distance"],
        ),
        "surface_area_weighting": asdict(area_diagnostics),
        "wall_exchange": asdict(diagnostics),
        "instantaneous_surface_pressure_audit": {
            "control_volume_resistance_n": latest_cv,
            "wall_shear_resistance_n": latest_wall_shear,
            "runtime_bfl_momentum_exchange_pressure_n": latest_bfl_pressure,
            "surface_closure": surface_closure,
            "observers": pressure_observers,
            "projected_bfl_pressure": projected_pressure_observers,
            "selection_policy": (
                "direct observer sensitivity only; no selection by experimental resistance"
            ),
        },
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
