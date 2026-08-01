#!/usr/bin/env python3
"""Three-level SUBOFF wall/refinement integration smoke test.

This runner validates allocation, deepest-level geometry/force ownership and
two conservative AMR interfaces.  It is intentionally not a resistance
validation run and never promotes a short trajectory by reference proximity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from suboff_experimental_resistance import (
    MODEL_LENGTH_M,
    experimental_point,
    force_scale_newton,
    smooth_ramp_factor,
)

from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.checkpoint_io import atomic_torch_save
from tensorlbm.control_volume_force import (
    box_control_volume,
    fluid_momentum_change,
    observe_control_volume_force,
)
from tensorlbm.cuda_memory_budget import require_cuda_memory_budget
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_3d,
)
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.population_health import inspect_population_health
from tensorlbm.population_positivity import (
    PositivityDiagnostics,
    limit_nonequilibrium_for_positivity,
)
from tensorlbm.solver3d import stream3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_static_amr import (
    apply_suboff_appendage_halfway_links,
    assess_suboff_geometry_resolution,
    build_fine_suboff_mask,
    build_nested_fine_suboff_mask,
    plan_nested_suboff_static_amr,
    plan_suboff_static_amr,
)
from tensorlbm.surface_area_weights import bfl_surface_area_weights
from tensorlbm.wall_model import (
    bfl_wall_function_3d,
    physical_wall_lattice_viscosity,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--hull-type", choices=("bare_hull", "full"), default="bare_hull",
    )
    result.add_argument("--speed-knots", type=float, default=5.92)
    result.add_argument("--nx", type=int, default=600)
    result.add_argument("--ny", type=int, default=120)
    result.add_argument("--nz", type=int, default=120)
    result.add_argument("--hull-length", type=float, default=120.0)
    result.add_argument("--center-x-fraction", type=float, default=0.3)
    result.add_argument("--outer-wall-margin", type=int, default=8)
    result.add_argument("--outer-wake-cells", type=int, default=100)
    result.add_argument("--inner-wall-margin", type=int, default=4)
    result.add_argument("--inner-wake-cells", type=int, default=8)
    result.add_argument("--cv-margin", type=int, default=4)
    result.add_argument("--aux-cv-margins", default="2,6")
    result.add_argument("--surface-force-interval", type=int, default=50)
    result.add_argument("--steps", type=int, default=2)
    result.add_argument("--warmup-steps", type=int, default=0)
    result.add_argument("--statistics-window-steps", type=int, default=0)
    result.add_argument("--ramp-steps", type=int, default=0)
    result.add_argument("--report-interval", type=int, default=1)
    result.add_argument("--wall-diagnostic-interval", type=int, default=1)
    result.add_argument(
        "--health-interval", type=int, default=0,
        help="root-step cadence for per-level population/rho/speed diagnostics; 0 disables",
    )
    result.add_argument(
        "--regularize-restriction",
        action="store_true",
        help="filter fine-to-coarse transfer to resolved second-order stress",
    )
    result.add_argument("--minimum-convective-times", type=float, default=8.0)
    result.add_argument(
        "--minimum-statistics-convective-times", type=float, default=5.0,
    )
    result.add_argument("--lattice-speed", type=float, default=0.06)
    result.add_argument("--resolved-reynolds", type=float, default=100000.0)
    result.add_argument("--rho-water", type=float, default=998.2)
    result.add_argument("--nu-water", type=float, default=1.004e-6)
    result.add_argument("--cs-smag", type=float, default=0.05)
    result.add_argument("--wall-law", choices=("musker", "reichardt", "log"), default="musker")
    result.add_argument("--stress-exchange-distance", type=float, default=1.0)
    result.add_argument("--sponge-width", type=int, default=24)
    result.add_argument("--sponge-strength", type=float, default=0.3)
    result.add_argument(
        "--memory-bytes-per-cell",
        type=float,
        default=943.0,
        help="explicit peak-memory model coefficient; recorded in output",
    )
    result.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--checkpoint-interval", type=int, default=0)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--preflight-only", action="store_true")
    return result


def run(args: argparse.Namespace) -> dict:
    if min(args.nx, args.ny, args.nz, args.hull_length, args.steps) <= 0:
        raise ValueError("grid, hull length and steps must be positive")
    if args.stress_exchange_distance <= 0.0:
        raise ValueError("stress exchange distance must be positive")
    if args.memory_bytes_per_cell <= 0.0:
        raise ValueError("memory bytes per cell must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint interval must be non-negative")
    if args.health_interval < 0:
        raise ValueError("health interval must be non-negative")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup steps must lie in [0, steps)")
    if not 0 <= args.statistics_window_steps <= args.steps - args.warmup_steps:
        raise ValueError("statistics window exceeds the post-warmup trajectory")
    if min(
        args.report_interval,
        args.wall_diagnostic_interval,
        args.surface_force_interval,
        args.minimum_convective_times,
        args.minimum_statistics_convective_times,
    ) <= 0:
        raise ValueError("report/diagnostic intervals and duration targets must be positive")
    auxiliary_margins = tuple(
        int(value.strip())
        for value in args.aux_cv_margins.split(",")
        if value.strip()
    )
    if len(auxiliary_margins) != 2 or len({args.cv_margin, *auxiliary_margins}) != 3:
        raise ValueError("primary and two auxiliary CV margins must be distinct")
    if min(auxiliary_margins) < 1:
        raise ValueError("auxiliary CV margins must be positive")
    if args.resume and args.checkpoint is None:
        raise ValueError("resume requires a checkpoint path")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    point = experimental_point(args.hull_type, args.speed_knots)
    shape = (args.nz, args.ny, args.nx)
    center = (
        args.nx * args.center_x_fraction,
        args.ny / 2.0,
        args.nz / 2.0,
    )
    geometry_config = SuboffConfig()
    coarse_solid, _ = build_suboff_mask(
        args.hull_type, args.nx, args.ny, args.nz,
        cx=center[0], cy=center[1], cz=center[2],
        length=args.hull_length, config=geometry_config, device=device,
    )
    outer_plan = plan_suboff_static_amr(
        coarse_solid,
        coarse_hull_length=args.hull_length,
        wall_margin=args.outer_wall_margin,
        wake_cells=args.outer_wake_cells,
    )
    outer_solid, _ = build_fine_suboff_mask(
        outer_plan,
        hull_type=args.hull_type,
        coarse_center=center,
        config=geometry_config,
        device=device,
    )
    nested_plan = plan_nested_suboff_static_amr(
        outer_plan,
        outer_solid,
        wall_margin=args.inner_wall_margin,
        wake_cells=args.inner_wake_cells,
    )
    nested_solid, nested_geometry = build_nested_fine_suboff_mask(
        nested_plan,
        hull_type=args.hull_type,
        coarse_center=center,
        config=geometry_config,
        device=device,
    )
    estimated_peak_gib = nested_plan.estimated_peak_gib(
        args.memory_bytes_per_cell,
    )
    memory_budget = require_cuda_memory_budget(
        device,
        estimated_peak_gib=estimated_peak_gib,
        reserve_gib=1.0,
        label="SUBOFF nested static-AMR smoke",
    )
    planning = {
        "outer_box": vars(outer_plan.box),
        "inner_box_in_outer_allocated_coordinates": vars(
            nested_plan.box_in_outer_allocated_coordinates,
        ),
        "outer_fine_shape": list(outer_plan.fine_physical_shape),
        "nested_fine_shape": list(nested_plan.fine_physical_shape),
        "total_allocated_cells": nested_plan.total_allocated_cells,
        "uniform_finest_cells": nested_plan.uniform_finest_cells,
        "cell_saving_fraction": nested_plan.cell_saving_fraction,
        "memory_estimate_bytes_per_cell": args.memory_bytes_per_cell,
        "estimated_peak_gib": estimated_peak_gib,
        "cuda_memory_preflight": (
            memory_budget.to_dict() if memory_budget is not None else None
        ),
    }
    if args.preflight_only:
        return {
            "schema": "tensorlbm-suboff-nested-amr-smoke-v3",
            "status": "preflight_only",
            "physical_validation": False,
            "planning": planning,
        }

    physical_re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    collision_re = args.resolved_reynolds
    nu_coarse = args.lattice_speed * args.hull_length / collision_re
    tau_coarse = 0.5 + 3.0 * nu_coarse
    outer_amr_config = StaticBlockAMRConfig(
        outer_plan.box,
        tau_coarse=tau_coarse,
        regularize_restriction=args.regularize_restriction,
    )
    inner_amr_config = StaticBlockAMRConfig(
        nested_plan.box_in_outer_allocated_coordinates,
        tau_coarse=outer_amr_config.tau_fine,
        regularize_restriction=args.regularize_restriction,
    )
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    hierarchy = NestedStaticBlockAMR3D(
        equilibrium3d(rho, ux, zero, zero, device=device),
        (outer_amr_config, inner_amr_config),
        fine_solids=(None, nested_solid),
    )
    checkpoint_signature = {
        "schema_version": 3,
        "hull_type": args.hull_type,
        "shape_zyx": list(shape),
        "speed_knots": args.speed_knots,
        "hull_length": args.hull_length,
        "center_x_fraction": args.center_x_fraction,
        "outer_box": vars(outer_plan.box),
        "inner_box": vars(nested_plan.box_in_outer_allocated_coordinates),
        "cv_margin": args.cv_margin,
        "auxiliary_cv_margins": list(auxiliary_margins),
        "surface_force_interval": args.surface_force_interval,
        "ramp_steps": args.ramp_steps,
        "lattice_speed": args.lattice_speed,
        "resolved_reynolds": args.resolved_reynolds,
        "rho_water": args.rho_water,
        "nu_water": args.nu_water,
        "cs_smag": args.cs_smag,
        "wall_law": args.wall_law,
        "stress_exchange_distance": args.stress_exchange_distance,
        "wall_diagnostic_interval": args.wall_diagnostic_interval,
        "sponge_width": args.sponge_width,
        "sponge_strength": args.sponge_strength,
        "far_field_mode": args.far_field_mode,
        "regularize_restriction": args.regularize_restriction,
    }
    finest_solid = hierarchy.interfaces[-1].fine_solid_with_ghost
    assert finest_solid is not None
    finest_solid_q = finest_solid.unsqueeze(0).expand_as(hierarchy.finest_f)
    finest_geometry = nested_geometry
    finest_center = (
        float(finest_geometry["cx"]) + inner_amr_config.ghost,
        float(finest_geometry["cy"]) + inner_amr_config.ghost,
        float(finest_geometry["cz"]) + inner_amr_config.ghost,
    )
    nz_f, ny_f, nx_f = finest_solid.shape
    finest_length = nested_plan.effective_hull_length_cells
    bfl_mask, bfl_q = compute_q_suboff(
        nx_f, ny_f, nz_f, *finest_center, finest_length,
        hull_type=args.hull_type, config=geometry_config, device=device,
        solid_mask=finest_solid,
    )
    appendage_halfway_links = 0
    if args.hull_type == "full":
        appendage_halfway_links = apply_suboff_appendage_halfway_links(
            finest_solid,
            bfl_mask,
            bfl_q,
            center=finest_center,
            length=finest_length,
            config=geometry_config,
        )
    near = get_near_wall_3d(finest_solid)
    bare_solid = None
    with_sail_solid = None
    if args.hull_type == "bare_hull":
        surface = SurfaceMesh.from_suboff(
            finest_solid,
            near,
            *finest_center,
            finest_length,
            finest_length / (2.0 * 8.57),
            config=geometry_config,
        )
        area_weight, area_diagnostics = bfl_surface_area_weights(
            bfl_mask,
            (surface.nx_n, surface.ny_n, surface.nz_n),
            reference_area=float(finest_geometry["wetted_area_lu2"]),
            boundary_mask=near,
        )
    else:
        surface = SurfaceMesh.from_gradient(finest_solid, near)
        bare_solid, _ = build_suboff_mask(
            "bare_hull", nx_f, ny_f, nz_f,
            cx=finest_center[0], cy=finest_center[1], cz=finest_center[2],
            length=finest_length, config=geometry_config, device=device,
        )
        with_sail_solid, _ = build_suboff_mask(
            "with_sail", nx_f, ny_f, nz_f,
            cx=finest_center[0], cy=finest_center[1], cz=finest_center[2],
            length=finest_length, config=geometry_config, device=device,
        )
        bare_near = get_near_wall_3d(bare_solid)
        bare_surface = SurfaceMesh.from_gradient(bare_solid, bare_near)
        bare_bfl_mask, _ = compute_q_suboff(
            nx_f, ny_f, nz_f, *finest_center, finest_length,
            hull_type="bare_hull", config=geometry_config, device=device,
            solid_mask=bare_solid,
        )
        _, bare_area_diagnostics = bfl_surface_area_weights(
            bare_bfl_mask,
            (bare_surface.nx_n, bare_surface.ny_n, bare_surface.nz_n),
            reference_area=float(finest_geometry["wetted_area_lu2"]),
            boundary_mask=bare_near,
        )
        area_weight, area_diagnostics = bfl_surface_area_weights(
            bfl_mask,
            (surface.nx_n, surface.ny_n, surface.nz_n),
            calibration_factor=bare_area_diagnostics.calibration_factor,
            boundary_mask=near,
        )
    indices = finest_solid.nonzero(as_tuple=False)
    z_min, y_min, x_min = (
        int(indices[:, axis].min().item()) for axis in range(3)
    )
    z_max, y_max, x_max = (
        int(indices[:, axis].max().item()) + 1 for axis in range(3)
    )
    def build_control_volume(margin: int) -> torch.Tensor:
        bounds = (
            x_min - margin, x_max + margin, nx_f,
            y_min - margin, y_max + margin, ny_f,
            z_min - margin, z_max + margin, nz_f,
        )
        for lower, upper, size in zip(
            bounds[0::3], bounds[1::3], bounds[2::3], strict=True,
        ):
            if lower <= 1 or upper >= size - 1:
                raise ValueError(
                    f"control-volume margin {margin} reaches the nested interface",
                )
        return box_control_volume(
            finest_solid.shape,
            x0=x_min - margin, x1=x_max + margin,
            y0=y_min - margin, y1=y_max + margin,
            z0=z_min - margin, z1=z_max + margin,
            device=device,
        )

    control_volume = build_control_volume(args.cv_margin)
    auxiliary_control_volumes = {
        margin: build_control_volume(margin) for margin in auxiliary_margins
    }
    sponge = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=("x+", "y-", "y+", "z-", "z+"),
    )
    wall_nu = physical_wall_lattice_viscosity(
        args.lattice_speed, finest_length, physical_re,
    )
    scale = force_scale_newton(
        rho_water=args.rho_water,
        dx_m=MODEL_LENGTH_M / finest_length,
        speed_mps=point.speed_mps,
        lattice_speed=args.lattice_speed,
    )
    current_step = 0
    resumed_from_step = 0
    resumed_legacy_v2_checkpoint = False
    force_samples: list[dict] = []
    step_records: list[dict] = []
    maximum_limiter_fraction = 0.0
    maximum_reflux_residual = [0.0, 0.0]
    maximum_reflux_limited_directions = [0, 0]
    maximum_rejected_fraction = 0.0
    health_records: list[dict] = []

    if args.resume:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        stored_configuration = state.get("configuration")
        legacy_v2_signature = dict(checkpoint_signature)
        legacy_v2_signature["schema_version"] = 2
        legacy_v2_signature.pop("hull_type")
        legacy_v2_without_filter = dict(legacy_v2_signature)
        legacy_v2_without_filter.pop("regularize_restriction")
        resumed_legacy_v2_checkpoint = (
            args.hull_type == "bare_hull"
            and not args.regularize_restriction
            and stored_configuration in (
                legacy_v2_signature,
                legacy_v2_without_filter,
            )
        )
        if (
            stored_configuration != checkpoint_signature
            and not resumed_legacy_v2_checkpoint
        ):
            raise ValueError("checkpoint configuration does not match nested smoke")
        current_step = int(state["step"])
        resumed_from_step = current_step
        if current_step >= args.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")
        hierarchy.restore_level_populations([
            populations.to(device=device)
            for populations in state["level_populations"]
        ])
        step_records = list(state["step_records"])
        maximum_limiter_fraction = float(state["maximum_limiter_fraction"])
        maximum_reflux_residual = [
            float(value) for value in state["maximum_reflux_residual"]
        ]
        maximum_reflux_limited_directions = [
            int(value) for value in state["maximum_reflux_limited_directions"]
        ]
        maximum_rejected_fraction = float(state["maximum_rejected_fraction"])
        health_records = list(state.get("health_records", []))

    def save_checkpoint(step: int) -> None:
        assert args.checkpoint is not None
        atomic_torch_save({
            "schema": "tensorlbm-suboff-nested-amr-smoke-checkpoint-v3",
            "configuration": checkpoint_signature,
            "step": step,
            "level_populations": [
                level.detach().cpu() for level in hierarchy.level_populations
            ],
            "step_records": step_records,
            "maximum_limiter_fraction": maximum_limiter_fraction,
            "maximum_reflux_residual": maximum_reflux_residual,
            "maximum_reflux_limited_directions": (
                maximum_reflux_limited_directions
            ),
            "maximum_rejected_fraction": maximum_rejected_fraction,
            "health_records": health_records,
        }, args.checkpoint)

    def require_finite_limiter(
        diagnostic: PositivityDiagnostics, *, level: int, stage: str,
    ) -> None:
        values = (
            diagnostic.minimum_population_before,
            diagnostic.minimum_population_after,
            diagnostic.minimum_alpha,
        )
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(
                "non-finite population detected "
                f"at root_step={current_step}, level={level}, stage={stage}; "
                f"minimum_before={diagnostic.minimum_population_before}, "
                f"minimum_after={diagnostic.minimum_population_after}, "
                f"minimum_alpha={diagnostic.minimum_alpha}",
            )

    def collide(state: torch.Tensor, tau: float, level: int) -> torch.Tensor:
        nonlocal maximum_limiter_fraction
        post = collide_cumulant_d3q19(state, tau=tau, C_s=args.cs_smag)
        post, diagnostic = limit_nonequilibrium_for_positivity(post)
        require_finite_limiter(diagnostic, level=level, stage="post_collision")
        maximum_limiter_fraction = max(
            maximum_limiter_fraction, diagnostic.limited_fraction,
        )
        return post

    def advance(
        state: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        nonlocal maximum_limiter_fraction, maximum_rejected_fraction
        del substep
        before = state
        collided = collide(state, tau, level)
        if level < 2:
            out = stream3d(collided)
            if level == 0:
                if args.far_field_mode == "non_equilibrium_extrapolation":
                    out = non_equilibrium_far_field_bc_3d(
                        out, u_in=args.lattice_speed,
                    )
                else:
                    out = far_field_bc_3d(out, u_in=args.lattice_speed)
                out = apply_equilibrium_difference_sponge(
                    out,
                    sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
                if args.far_field_mode == "non_equilibrium_extrapolation":
                    out = non_equilibrium_far_field_bc_3d(
                        out, u_in=args.lattice_speed,
                    )
                else:
                    out = far_field_bc_3d(out, u_in=args.lattice_speed)
            return AMRAdvanceResult(out, collided)

        post_collision = torch.where(finest_solid_q, before, collided)
        out = stream3d(post_collision)
        activation = smooth_ramp_factor(current_step, args.ramp_steps)
        collect_wall_diagnostics = (
            current_step % args.wall_diagnostic_interval == 0
        )
        wall_result = bfl_wall_function_3d(
            out,
            post_collision,
            finest_solid,
            wall_nu,
            bfl_mask,
            bfl_q,
            wall_law=args.wall_law,
            near_mask=near,
            bfl_wall_mode="wall_model_slip",
            wall_activation=activation,
            stress_exchange_distance=args.stress_exchange_distance,
            wall_normals=(surface.nx_n, surface.ny_n, surface.nz_n),
            area_weight=area_weight,
            return_wall_diagnostics=collect_wall_diagnostics,
        )
        if collect_wall_diagnostics:
            out, friction, pressure, diagnostics = wall_result
            maximum_rejected_fraction = max(
                maximum_rejected_fraction, diagnostics.rejected_fraction,
            )
            mean_y_plus = diagnostics.y_plus_mean
            minimum_y_plus = diagnostics.y_plus_min
            maximum_y_plus = diagnostics.y_plus_max
            mean_wall_distance = diagnostics.wall_distance_mean
        else:
            out, friction, pressure = wall_result
            mean_y_plus = None
            minimum_y_plus = None
            maximum_y_plus = None
            mean_wall_distance = None
        before_positivity = out
        out, positivity = limit_nonequilibrium_for_positivity(out)
        require_finite_limiter(positivity, level=level, stage="post_wall")
        maximum_limiter_fraction = max(
            maximum_limiter_fraction, positivity.limited_fraction,
        )
        cv_force = float(observe_control_volume_force(
            before,
            out,
            post_collision,
            control_volume,
            solid=finest_solid,
        ).force_on_body[0])
        collision_source = float(fluid_momentum_change(
            before,
            post_collision,
            control_volume,
            solid=finest_solid,
        )[0])
        positivity_source = float(fluid_momentum_change(
            before_positivity,
            out,
            control_volume,
            solid=finest_solid,
        )[0])
        auxiliary_forces = (
            {
                margin: float(observe_control_volume_force(
                    before,
                    out,
                    post_collision,
                    auxiliary_cv,
                    solid=finest_solid,
                ).force_on_body[0])
                for margin, auxiliary_cv in auxiliary_control_volumes.items()
            }
            if current_step % args.surface_force_interval == 0 else {}
        )
        force_samples.append({
            "cv": cv_force,
            "bfl": pressure + friction,
            "pressure": pressure,
            "friction": friction,
            "source": collision_source + positivity_source,
            "y_plus": mean_y_plus,
            "y_plus_min": minimum_y_plus,
            "y_plus_max": maximum_y_plus,
            "wall_distance": mean_wall_distance,
            "auxiliary": auxiliary_forces,
        })
        return AMRAdvanceResult(out, post_collision)

    start_step = current_step
    for step in range(start_step + 1, args.steps + 1):
        current_step = step
        force_samples.clear()
        ledgers = hierarchy.step(advance)
        if args.health_interval and current_step % args.health_interval == 0:
            level_health = [
                inspect_population_health(populations).to_dict()
                for populations in hierarchy.level_populations
            ]
            health_records.append({"step": current_step, "levels": level_health})
            print(
                "nested health "
                + json.dumps(health_records[-1], separators=(",", ":")),
                flush=True,
            )
            if not all(record["finite"] for record in level_health):
                raise FloatingPointError(
                    f"non-finite hierarchy state at root_step={current_step}",
                )
        if len(force_samples) != 4:
            raise RuntimeError("three-level hierarchy must emit four finest force samples")
        for index, ledger in enumerate(ledgers):
            maximum_reflux_residual[index] = max(
                maximum_reflux_residual[index],
                float(ledger.residual.abs().max()),
            )
            maximum_reflux_limited_directions[index] = max(
                maximum_reflux_limited_directions[index],
                ledger.limited_directions,
            )
        cv_mean = sum(item["cv"] for item in force_samples) / 4.0
        bfl_mean = sum(item["bfl"] for item in force_samples) / 4.0
        pressure_mean = sum(item["pressure"] for item in force_samples) / 4.0
        friction_mean = sum(item["friction"] for item in force_samples) / 4.0
        source_mean = sum(item["source"] for item in force_samples) / 4.0
        auxiliary_means = (
            {
                margin: sum(
                    item["auxiliary"][margin] for item in force_samples
                ) / 4.0
                for margin in auxiliary_margins
            }
            if current_step % args.surface_force_interval == 0 else None
        )
        corrected = cv_mean + source_mean
        y_plus_samples = [
            item["y_plus"] for item in force_samples
            if item["y_plus"] is not None
        ]
        y_plus_minima = [
            item["y_plus_min"] for item in force_samples
            if item["y_plus_min"] is not None
        ]
        y_plus_maxima = [
            item["y_plus_max"] for item in force_samples
            if item["y_plus_max"] is not None
        ]
        wall_distances = [
            item["wall_distance"] for item in force_samples
            if item["wall_distance"] is not None
        ]
        record = {
            "step": current_step,
            "cv_resistance_n": cv_mean * scale,
            "bfl_plus_wall_stress_n": bfl_mean * scale,
            "bfl_pressure_n": pressure_mean * scale,
            "wall_shear_n": friction_mean * scale,
            "numerical_source_n": source_mean * scale,
            "source_corrected_cv_n": corrected * scale,
            "auxiliary_cv_n": (
                {
                    str(margin): value * scale
                    for margin, value in auxiliary_means.items()
                }
                if auxiliary_means is not None else None
            ),
            "raw_observer_difference_pct": (
                abs(cv_mean - bfl_mean) / max(abs(cv_mean), 1.0e-30) * 100.0
            ),
            "source_corrected_observer_difference_pct": (
                abs(corrected - bfl_mean) / max(abs(corrected), 1.0e-30) * 100.0
            ),
            "mean_y_plus": (
                sum(y_plus_samples) / len(y_plus_samples)
                if y_plus_samples else None
            ),
            "minimum_y_plus": min(y_plus_minima) if y_plus_minima else None,
            "maximum_y_plus": max(y_plus_maxima) if y_plus_maxima else None,
            "mean_wall_distance_cells": (
                sum(wall_distances) / len(wall_distances)
                if wall_distances else None
            ),
            "wall_fully_activated": current_step >= args.ramp_steps,
            "surface_pressure_plus_wall_stress_n": None,
        }
        if current_step % args.surface_force_interval == 0:
            surface_pressure = drag_pressure_integration(
                hierarchy.finest_f,
                surface,
                1.0,
                extrap="none",
                p0_method="near_wall",
                solid=finest_solid,
            )[0]
            record["surface_pressure_plus_wall_stress_n"] = (
                (surface_pressure + friction_mean) * scale
            )
        step_records.append(record)
        if current_step % args.report_interval == 0 or current_step == args.steps:
            print(
                f"nested smoke step={current_step}/{args.steps} "
                f"Rt={step_records[-1]['cv_resistance_n']:.3f} N "
                f"closure={step_records[-1]['source_corrected_observer_difference_pct']:.5f}%",
                flush=True,
            )
        if (
            args.checkpoint is not None
            and args.checkpoint_interval > 0
            and current_step % args.checkpoint_interval == 0
        ):
            save_checkpoint(current_step)

    eligible_records = [
        record for record in step_records if record["wall_fully_activated"]
    ]
    maximum_corrected_difference = (
        max(
            record["source_corrected_observer_difference_pct"]
            for record in eligible_records
        )
        if eligible_records else None
    )
    finite = all(
        bool(torch.isfinite(level).all()) for level in hierarchy.level_populations
    )
    geometry_resolution = assess_suboff_geometry_resolution(
        finest_solid,
        hull_type=args.hull_type,
        fine_hull_length_cells=finest_length,
        center_yz=(finest_center[1], finest_center[2]),
        bare_hull=bare_solid,
        with_sail=with_sail_solid,
        appendage_halfway_links=appendage_halfway_links,
    )
    admitted = (
        finite
        and maximum_corrected_difference is not None
        and maximum_corrected_difference <= 0.1
        and max(maximum_reflux_residual) <= 1.0e-6
        and max(maximum_reflux_limited_directions) == 0
        and maximum_limiter_fraction <= 1.0e-3
        and maximum_rejected_fraction <= 0.01
    )
    post_warmup_records = [
        record for record in eligible_records
        if record["step"] > args.warmup_steps
    ]
    statistics_window_steps = (
        args.statistics_window_steps or len(post_warmup_records)
    )
    selected_records = post_warmup_records[-statistics_window_steps:]
    total_convective_times = (
        args.steps * args.lattice_speed / args.hull_length
    )
    sampling_convective_times = (
        len(selected_records) * args.lattice_speed / args.hull_length
    )
    duration_acceptable = (
        total_convective_times >= args.minimum_convective_times
        and sampling_convective_times
        >= args.minimum_statistics_convective_times
    )
    force_stationarity = None
    mean_resistance = None
    mean_bfl = None
    mean_source = None
    reference_error_pct = None
    wall_records = [
        record for record in selected_records
        if record["mean_y_plus"] is not None
    ]
    if selected_records:
        cv_values = [record["cv_resistance_n"] for record in selected_records]
        mean_resistance = sum(cv_values) / len(cv_values)
        mean_bfl = sum(
            record["bfl_plus_wall_stress_n"] for record in selected_records
        ) / len(selected_records)
        mean_source = sum(
            record["numerical_source_n"] for record in selected_records
        ) / len(selected_records)
        reference_error_pct = (
            abs(mean_resistance - point.resistance_n)
            / point.resistance_n
            * 100.0
        )
        if len(cv_values) >= 4:
            force_stationarity = assess_force_stationarity(
                cv_values,
                block_size=max(1, len(cv_values) // 8),
            )
    stationarity_acceptable = (
        force_stationarity is not None and force_stationarity.meets(1.0)
    )
    auxiliary_cv_difference_pct = None
    nested_cv_acceptable = False
    surface_observer_difference_pct = None
    surface_observer_acceptable = False
    if selected_records and mean_resistance is not None:
        auxiliary_records = [
            record for record in selected_records
            if record["auxiliary_cv_n"] is not None
        ]
        auxiliary_means_n = {
            str(margin): sum(
                record["auxiliary_cv_n"][str(margin)]
                for record in auxiliary_records
            ) / len(auxiliary_records)
            for margin in auxiliary_margins
        } if auxiliary_records else {}
        if auxiliary_records:
            paired_primary_mean = sum(
                record["cv_resistance_n"] for record in auxiliary_records
            ) / len(auxiliary_records)
            auxiliary_cv_difference_pct = {
                margin: abs(value - paired_primary_mean)
                / max(abs(paired_primary_mean), 1.0e-30) * 100.0
                for margin, value in auxiliary_means_n.items()
            }
            nested_cv_acceptable = (
                max(auxiliary_cv_difference_pct.values()) <= 1.0
            )
        surface_records = [
            record for record in selected_records
            if record["surface_pressure_plus_wall_stress_n"] is not None
        ]
        if surface_records:
            surface_mean = sum(
                record["surface_pressure_plus_wall_stress_n"]
                for record in surface_records
            ) / len(surface_records)
            paired_cv_mean = sum(
                record["cv_resistance_n"] for record in surface_records
            ) / len(surface_records)
            surface_observer_difference_pct = (
                abs(surface_mean - paired_cv_mean)
                / max(abs(paired_cv_mean), 1.0e-30)
                * 100.0
            )
            surface_observer_acceptable = surface_observer_difference_pct <= 5.0
    single_grid_candidate = (
        admitted
        and duration_acceptable
        and stationarity_acceptable
        and nested_cv_acceptable
        and surface_observer_acceptable
        and reference_error_pct is not None
        and reference_error_pct <= 5.0
        and geometry_resolution.absolute_reference_resolved
    )
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda" else None
    )
    return {
        "schema": "tensorlbm-suboff-nested-amr-smoke-v3",
        "status": (
            "single_grid_candidate"
            if single_grid_candidate else (
                "integration_smoke_pass" if admitted else "integration_smoke_fail"
            )
        ),
        "physical_validation": False,
        "configuration": vars(args) | {
            "output": str(args.output) if args.output else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "physical_reynolds": physical_re,
            "finest_wall_nu": wall_nu,
            "finest_hull_length_cells": finest_length,
            "tau_by_level": [
                outer_amr_config.tau_coarse,
                outer_amr_config.tau_fine,
                inner_amr_config.tau_fine,
            ],
            "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_interval": args.checkpoint_interval,
            "resumed_from_step": resumed_from_step,
            "resumed_legacy_v2_checkpoint": resumed_legacy_v2_checkpoint,
        },
        "planning": planning | {"measured_peak_allocated_gib": peak_gib},
        "geometry": nested_geometry | {
            "resolution": geometry_resolution.to_dict(),
            "area_weighting": vars(area_diagnostics),
            "appendage_halfway_links": appendage_halfway_links,
            "geometry_owner_level": 2,
            "force_owner_level": 2,
        },
        "result": {
            "steps": step_records,
            "maximum_source_corrected_observer_difference_pct": (
                maximum_corrected_difference
            ),
            "maximum_reflux_residual_by_interface": maximum_reflux_residual,
            "maximum_reflux_limited_directions_by_interface": (
                maximum_reflux_limited_directions
            ),
            "maximum_positivity_limited_fraction": maximum_limiter_fraction,
            "maximum_wall_sample_rejected_fraction": maximum_rejected_fraction,
            "finite": finite,
            "population_health": health_records,
            "statistics": {
                "warmup_steps": args.warmup_steps,
                "statistics_window_steps_requested": args.statistics_window_steps,
                "statistics_window_steps_resolved": len(selected_records),
                "total_convective_times": total_convective_times,
                "sampling_convective_times": sampling_convective_times,
                "mean_resistance_n": mean_resistance,
                "mean_bfl_plus_wall_stress_n": mean_bfl,
                "mean_numerical_source_n": mean_source,
                "experimental_resistance_n": point.resistance_n,
                "reference_error_pct": reference_error_pct,
                "force_stationarity": (
                    force_stationarity.to_dict()
                    if force_stationarity is not None else None
                ),
                "auxiliary_cv_difference_pct": auxiliary_cv_difference_pct,
                "surface_observer_difference_pct": (
                    surface_observer_difference_pct
                ),
                "wall_exchange": {
                    "samples": len(wall_records),
                    "mean_distance_cells": (
                        sum(record["mean_wall_distance_cells"] for record in wall_records)
                        / len(wall_records)
                        if wall_records else None
                    ),
                    "minimum_y_plus": (
                        min(record["minimum_y_plus"] for record in wall_records)
                        if wall_records else None
                    ),
                    "mean_y_plus": (
                        sum(record["mean_y_plus"] for record in wall_records)
                        / len(wall_records)
                        if wall_records else None
                    ),
                    "maximum_y_plus": (
                        max(record["maximum_y_plus"] for record in wall_records)
                        if wall_records else None
                    ),
                },
            },
        },
        "acceptance": {
            "integration_smoke_admitted": admitted,
            "fully_activated_steps_assessed": len(eligible_records),
            "duration_target_met": duration_acceptable,
            "stationarity_target_met": stationarity_acceptable,
            "nested_control_volume_target_met": nested_cv_acceptable,
            "surface_observer_target_met": surface_observer_acceptable,
            "reference_error_target_met": (
                reference_error_pct is not None and reference_error_pct <= 5.0
            ),
            "single_grid_candidate": single_grid_candidate,
            "resistance_accuracy_assessed": False,
            "time_convergence_assessed": False,
            "grid_convergence_assessed": False,
        },
    }


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
