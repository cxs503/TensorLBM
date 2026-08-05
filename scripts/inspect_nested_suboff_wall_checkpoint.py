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
from tensorlbm.d3q19 import macroscopic3d  # noqa: E402
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
from tensorlbm.spalding_wall_model import (  # noqa: E402
    sample_wall_exchange_velocity,
)
from tensorlbm.pressure_gradient_wall_model import (  # noqa: E402
    solve_pressure_gradient_equilibrium_wall_shear,
)
from tensorlbm.two_point_wall_diagnostics import (  # noqa: E402
    estimate_two_point_log_slope_friction_velocity,
    summarize_two_point_log_slope,
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
    density, velocity_x, velocity_y, velocity_z = macroscopic3d(finest)
    exchange_distance = float(configuration["stress_exchange_distance"])
    inner_samples = sample_wall_exchange_velocity(
        (velocity_x, velocity_y, velocity_z),
        bfl_mask,
        bfl_q,
        (surface.nx_n, surface.ny_n, surface.nz_n),
        exchange_distance=exchange_distance,
        boundary_mask=near,
        fluid_mask=~solid,
    )
    outer_samples = sample_wall_exchange_velocity(
        (velocity_x, velocity_y, velocity_z),
        bfl_mask,
        bfl_q,
        (surface.nx_n, surface.ny_n, surface.nz_n),
        exchange_distance=2.0 * exchange_distance,
        boundary_mask=near,
        fluid_mask=~solid,
    )

    def dense_tangential_speed_and_distance(samples):
        normal_velocity = (
            samples.velocity_x * samples.normal_x
            + samples.velocity_y * samples.normal_y
            + samples.velocity_z * samples.normal_z
        )
        tangent_x = samples.velocity_x - normal_velocity * samples.normal_x
        tangent_y = samples.velocity_y - normal_velocity * samples.normal_y
        tangent_z = samples.velocity_z - normal_velocity * samples.normal_z
        speed = torch.sqrt(
            tangent_x.square() + tangent_y.square() + tangent_z.square(),
        )
        dense_speed = torch.full(
            solid.shape,
            torch.nan,
            device=target,
            dtype=finest.dtype,
        )
        dense_distance = torch.full_like(dense_speed, torch.nan)
        dense_tangent_x = torch.full_like(dense_speed, torch.nan)
        dense_speed[samples.boundary] = speed
        dense_distance[samples.boundary] = samples.y2
        dense_tangent_x[samples.boundary] = (
            tangent_x / speed.clamp_min(torch.finfo(speed.dtype).tiny)
        )
        return dense_speed, dense_distance, dense_tangent_x

    inner_speed, inner_distance, inner_tangent_x = (
        dense_tangential_speed_and_distance(
            inner_samples,
        )
    )
    outer_speed, outer_distance, _ = dense_tangential_speed_and_distance(
        outer_samples,
    )
    common_exchange = inner_samples.boundary & outer_samples.boundary
    two_point_u_tau, two_point_valid = (
        estimate_two_point_log_slope_friction_velocity(
            inner_speed[common_exchange],
            outer_speed[common_exchange],
            inner_distance[common_exchange],
            outer_distance[common_exchange],
        )
    )
    two_point_summary = summarize_two_point_log_slope(
        two_point_u_tau,
        two_point_valid,
    )
    common_area = area_weight[common_exchange]
    common_tangent_x = inner_tangent_x[common_exchange]
    two_point_shear_x_lu = (
        two_point_u_tau[two_point_valid].square()
        * common_tangent_x[two_point_valid]
        * common_area[two_point_valid]
    ).sum()
    two_point_area_coverage_fraction = (
        common_area[two_point_valid].sum()
        / common_area.sum().clamp_min(torch.finfo(common_area.dtype).tiny)
    )
    two_point_diagnostic = {
        "scope": "diagnostic_only_not_a_force_correction",
        "inner_requested_distance_cells": exchange_distance,
        "outer_requested_distance_cells": 2.0 * exchange_distance,
        "common_sample_nodes": int(common_exchange.sum().item()),
        "inner_distance_mean_cells": float(
            inner_distance[common_exchange].mean().item(),
        ),
        "outer_distance_mean_cells": float(
            outer_distance[common_exchange].mean().item(),
        ),
        "inner_tangential_speed_mean_lu": float(
            inner_speed[common_exchange].mean().item(),
        ),
        "outer_tangential_speed_mean_lu": float(
            outer_speed[common_exchange].mean().item(),
        ),
        "valid_area_fraction": float(two_point_area_coverage_fraction.item()),
        "covered_shear_force_x_lu": float(two_point_shear_x_lu.item()),
        "friction_velocity": asdict(two_point_summary),
    }
    scale = force_scale_newton(
        rho_water=float(configuration["rho_water"]),
        dx_m=MODEL_LENGTH_M / finest_length,
        speed_mps=point.speed_mps,
        lattice_speed=float(configuration["lattice_speed"]),
    )
    two_point_diagnostic["covered_shear_force_x_n"] = (
        float(two_point_shear_x_lu.item()) * scale
    )

    # Evaluate the pressure-gradient ODE candidate on the same frozen field.
    # This is a sensitivity observer only: it neither changes populations nor
    # replaces the production Musker traction.  The candidate deliberately
    # reports nodes requiring reverse shear as separated until the published
    # pressure-gradient velocity-scale extension is independently implemented.
    pressure_gradient_samples = sample_wall_tangential_pressure_gradient(
        (density - 1.0) / 3.0,
        solid,
        inner_samples.boundary,
        (surface.nx_n, surface.ny_n, surface.nz_n),
    )
    normal_velocity = (
        inner_samples.velocity_x * inner_samples.normal_x
        + inner_samples.velocity_y * inner_samples.normal_y
        + inner_samples.velocity_z * inner_samples.normal_z
    )
    tangent = torch.stack((
        inner_samples.velocity_x - normal_velocity * inner_samples.normal_x,
        inner_samples.velocity_y - normal_velocity * inner_samples.normal_y,
        inner_samples.velocity_z - normal_velocity * inner_samples.normal_z,
    ), dim=1)
    candidate_speed = torch.linalg.vector_norm(tangent, dim=1)
    tangent_direction = tangent / candidate_speed[:, None].clamp_min(
        torch.finfo(candidate_speed.dtype).tiny,
    )
    active_density = density[inner_samples.boundary]
    signed_pressure_acceleration = (
        pressure_gradient_samples.vector * tangent_direction
    ).sum(dim=1) / active_density.clamp_min(
        torch.finfo(active_density.dtype).tiny,
    )
    signed_pressure_acceleration = torch.where(
        pressure_gradient_samples.valid,
        signed_pressure_acceleration,
        torch.full_like(signed_pressure_acceleration, torch.nan),
    )
    pressure_acceleration_magnitude = (
        pressure_gradient_samples.magnitude
        / active_density.clamp_min(torch.finfo(active_density.dtype).tiny)
    )
    pressure_acceleration_magnitude = torch.where(
        pressure_gradient_samples.valid,
        pressure_acceleration_magnitude,
        torch.full_like(pressure_acceleration_magnitude, torch.nan),
    )
    candidate_area = area_weight[inner_samples.boundary]
    candidate_tangent_x = tangent_direction[:, 0]
    total_candidate_area = candidate_area.sum().clamp_min(
        torch.finfo(candidate_area.dtype).tiny,
    )
    candidate_diagnostic = {}
    for eddy_viscosity_model in ("van_driest", "duprat"):
        candidate = solve_pressure_gradient_equilibrium_wall_shear(
            candidate_speed,
            inner_samples.y2,
            signed_pressure_acceleration,
            wall_nu,
            pressure_gradient_magnitude_acceleration=(
                pressure_acceleration_magnitude
            ),
            eddy_viscosity_model=eddy_viscosity_model,
        )
        attached_area_fraction = float(
            (candidate_area * candidate.attached).sum().div(
                total_candidate_area,
            ).item(),
        )
        separated_area_fraction = float(
            (candidate_area * candidate.separated).sum().div(
                total_candidate_area,
            ).item(),
        )
        candidate_shear_x_lu = (
            candidate.shear_stress_over_density
            * candidate_tangent_x
            * candidate_area
        ).sum()
        attached_friction_velocity = candidate.friction_velocity[
            candidate.attached
        ]
        candidate_diagnostic[eddy_viscosity_model] = {
            "scope": "diagnostic_only_not_a_force_correction",
            "scheme": (
                "attached_pressure_gradient_equilibrium_ode_"
                f"{eddy_viscosity_model}_v1"
            ),
            "requested_nodes": int(candidate_speed.numel()),
            "finite_gradient_nodes": pressure_gradient_samples.valid_nodes,
            "attached_nodes": int(candidate.attached.sum().item()),
            "separated_nodes": int(candidate.separated.sum().item()),
            "attached_area_fraction": attached_area_fraction,
            "separated_area_fraction": separated_area_fraction,
            "unresolved_area_fraction": max(
                0.0, 1.0 - attached_area_fraction - separated_area_fraction,
            ),
            "attached_shear_force_x_lu": float(candidate_shear_x_lu.item()),
            "attached_shear_force_x_n": (
                float(candidate_shear_x_lu.item()) * scale
            ),
            "shear_force_vs_production_wall_law_ratio": (
                float(candidate_shear_x_lu.item())
                / max(abs(float(diagnostics.shear_force[0])), 1.0e-30)
            ),
            "maximum_attached_speed_residual_lu": (
                float(candidate.residual[candidate.attached].abs().max().item())
                if bool(candidate.attached.any()) else None
            ),
            "mean_attached_friction_velocity_lu": (
                float(attached_friction_velocity.mean().item())
                if attached_friction_velocity.numel() else None
            ),
            "reverse_shear_force_modelled": False,
        }
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
        "schema": "tensorlbm-nested-suboff-wall-checkpoint-audit-v3",
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
        "two_point_log_slope_diagnostic": two_point_diagnostic,
        "pressure_gradient_ode_candidate": candidate_diagnostic,
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
