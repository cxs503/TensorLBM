#!/usr/bin/env python3
"""Recursive three- or four-level SUBOFF wall/refinement integration runner.

This runner validates allocation, deepest-level geometry/force ownership and
every conservative AMR interface.  It is intentionally not a resistance
validation claim and never promotes a short trajectory by reference proximity.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from suboff_experimental_resistance import (
    MODEL_LENGTH_M,
    experimental_point,
    force_scale_newton,
    smooth_ramp_factor,
)

from tensorlbm.amr_interface_filter import (
    assess_interface_filter_control_volume_clearance,
)
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.checkpoint_io import atomic_torch_save
from tensorlbm.chunked_collision import (
    NaturalKBCCollisionExecutor,
    collide_in_z_chunks,
)
from tensorlbm.control_volume_force import (
    box_control_volume,
    fluid_momentum_change,
    observe_control_volume_force,
)
from tensorlbm.cuda_memory_budget import (
    plan_hierarchy_device_memory,
    require_cuda_memory_budget,
    require_cuda_runtime_reserve,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import WEIGHT_PRECISION_SCHEME, equilibrium3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_3d,
    integrate_bfl_projected_pressure,
)
from tensorlbm.entropic_kbc import (
    collide_kbc_d3q19,
)
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.hydrodynamics import ittc57_friction_coefficient
from tensorlbm.interpolated_bc_suboff import (
    SUBOFF_APPENDAGE_LINK_SCHEME,
    compute_q_suboff,
    refine_q_suboff_appendages,
)
from tensorlbm.kinetic_flux_register import conserved_population_moments
from tensorlbm.open_boundary_audit import audit_open_boundary_history
from tensorlbm.population_health import inspect_population_health
from tensorlbm.population_positivity import (
    PositivityDiagnostics,
    limit_nonequilibrium_for_positivity,
)
from tensorlbm.resistance_component_audit import audit_resistance_components
from tensorlbm.solver3d import stream3d
from tensorlbm.spalding_wall_model import (
    assess_wall_exchange_interface_clearance,
)
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)
from tensorlbm.subcycled_force import UniformSubcycleAverager
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_static_amr import (
    assess_suboff_geometry_resolution,
    build_fine_suboff_mask,
    build_nested_fine_suboff_mask,
    count_suboff_appendage_boundary_links,
    plan_nested_suboff_static_amr,
    plan_suboff_static_amr,
)
from tensorlbm.surface_area_weights import bfl_surface_area_weights
from tensorlbm.viscosity_continuation import ResolvedReynoldsContinuation
from tensorlbm.wall_exchange_yplus import (
    aggregate_wall_exchange_yplus_summaries,
)
from tensorlbm.wall_model import (
    WALL_TRACTION_SOURCE_SCHEME,
    bfl_wall_function_3d,
    physical_wall_lattice_viscosity,
)
from tensorlbm.wall_pressure_gradient import (
    aggregate_wall_pressure_gradient_summaries,
)
from tensorlbm.yplus_guide import (
    estimate_bfl_exchange_yplus_bounds,
    estimate_exchange_yplus,
    grid_quality_metrics,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--level-devices",
        default="",
        help=(
            "optional comma-separated owner device for every hierarchy level; "
            "for four levels, for example cuda:0,cuda:0,cuda:1,cuda:2"
        ),
    )
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
    result.add_argument(
        "--deep-wall-margin",
        type=int,
        default=0,
        help=(
            "optional third 2:1 interface around the hull; 0 keeps the "
            "three-level hierarchy, values >=2 create four levels"
        ),
    )
    result.add_argument("--deep-wake-cells", type=int, default=0)
    result.add_argument("--cv-margin", type=int, default=4)
    result.add_argument("--aux-cv-margins", default="2,6")
    result.add_argument("--surface-force-interval", type=int, default=50)
    result.add_argument(
        "--enable-rejected-surface-pressure-diagnostic",
        action="store_true",
        help=(
            "opt in to the non-conservative density-pressure surface observer; "
            "it is retained only for forensic comparison and never gates acceptance"
        ),
    )
    result.add_argument(
        "--enable-projected-bfl-pressure-diagnostic",
        action="store_true",
        help=(
            "record an independent finite-volume pressure observer on axial "
            "BFL crossing faces; diagnostic only until canonical validation"
        ),
    )
    result.add_argument(
        "--projected-bfl-pressure-reconstruction",
        choices=("local", "linear", "quadratic"),
        default="linear",
        help="wall-pressure reconstruction order for the projected BFL observer",
    )
    result.add_argument("--steps", type=int, default=2)
    result.add_argument("--warmup-steps", type=int, default=0)
    result.add_argument("--statistics-window-steps", type=int, default=0)
    result.add_argument("--ramp-steps", type=int, default=0)
    result.add_argument(
        "--wall-normal-ramp-steps",
        type=int,
        default=-1,
        help="BFL no-penetration ramp; -1 reuses --ramp-steps",
    )
    result.add_argument(
        "--wall-shear-ramp-steps",
        type=int,
        default=-1,
        help="wall-law traction ramp; -1 reuses --ramp-steps",
    )
    result.add_argument("--report-interval", type=int, default=1)
    result.add_argument("--wall-diagnostic-interval", type=int, default=1)
    result.add_argument(
        "--health-interval", type=int, default=0,
        help="root-step cadence for per-level population/rho/speed diagnostics; 0 disables",
    )
    result.add_argument(
        "--maximum-health-speed",
        type=float,
        default=0.3,
        help="fail-closed peak lattice-speed limit at health cadence",
    )
    result.add_argument(
        "--minimum-health-population",
        type=float,
        default=1.0e-8,
        help="fail-closed minimum population required at health cadence",
    )
    result.add_argument(
        "--maximum-positivity-limited-fraction",
        type=float,
        default=1.0e-6,
        help="largest production-admissible fraction limited at any step",
    )
    result.add_argument(
        "--maximum-reflux-applied-correction-fraction",
        type=float,
        default=1.0e-3,
        help="largest admissible per-direction interface inventory correction",
    )
    result.add_argument(
        "--regularize-restriction",
        action="store_true",
        help="filter fine-to-coarse transfer to resolved second-order stress",
    )
    result.add_argument(
        "--regularize-prolongation",
        action="store_true",
        help="filter coarse-to-fine ghost transfer to resolved second-order stress",
    )
    result.add_argument(
        "--ghost-interpolation",
        choices=("injection", "trilinear"),
        default="injection",
        help="coarse-to-fine ghost-shell spatial interpolation",
    )
    result.add_argument(
        "--reflux-correction-stencil",
        choices=("exterior_cells", "crossing_links"),
        default="exterior_cells",
        help="where the conserved stream-register correction may be applied",
    )
    result.add_argument(
        "--enforce-transfer-positivity",
        action="store_true",
        help="limit fine-to-coarse populations before parent replacement",
    )
    result.add_argument(
        "--interface-filter-width",
        type=int,
        default=0,
        help="physical fine cells damped next to every AMR interface; 0 disables",
    )
    result.add_argument(
        "--interface-filter-strength",
        type=float,
        default=0.0,
        help="maximum moment-preserving non-equilibrium damping in [0,1]",
    )
    result.add_argument("--minimum-convective-times", type=float, default=8.0)
    result.add_argument(
        "--minimum-target-reynolds-convective-times",
        type=float,
        default=7.5,
        help="minimum trajectory duration after collision Re reaches its target",
    )
    result.add_argument(
        "--minimum-statistics-convective-times", type=float, default=5.0,
    )
    result.add_argument("--lattice-speed", type=float, default=0.06)
    result.add_argument("--resolved-reynolds", type=float, default=100000.0)
    result.add_argument(
        "--resolved-reynolds-start",
        type=float,
        default=0.0,
        help="positive startup Reynolds; 0 uses --resolved-reynolds",
    )
    result.add_argument("--viscosity-ramp-start-step", type=int, default=0)
    result.add_argument("--viscosity-ramp-end-step", type=int, default=0)
    result.add_argument("--rho-water", type=float, default=998.2)
    result.add_argument("--nu-water", type=float, default=1.004e-6)
    result.add_argument("--cs-smag", type=float, default=0.05)
    result.add_argument("--wale-cw", type=float, default=0.5)
    result.add_argument("--vreman-cv", type=float, default=0.025)
    result.add_argument(
        "--collision-model",
        choices=(
            "cumulant_smagorinsky",
            "cumulant_wale",
            "cumulant_vreman",
            "entropic_kbc",
            "natural_kbc",
        ),
        default="cumulant_smagorinsky",
    )
    result.add_argument("--kbc-max-iterations", type=int, default=12)
    result.add_argument(
        "--collision-chunk-cells",
        type=int,
        default=0,
        help=(
            "bounded-memory z-slab size for cell-local KBC collision; "
            "0 keeps whole-level collision"
        ),
    )
    result.add_argument(
        "--compile-natural-kbc",
        action="store_true",
        help=(
            "fuse natural-KBC cell-local work while passing tau as a tensor "
            "so viscosity ramps reuse one dynamic graph"
        ),
    )
    result.add_argument(
        "--natural-kbc-compute-dtype",
        choices=("storage", "float64"),
        default="storage",
        help=(
            "compute natural-KBC collision in storage precision or float64; "
            "the latter casts the collided slabs back to storage precision"
        ),
    )
    result.add_argument(
        "--population-storage-dtype",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "population storage precision for every AMR level; float64 is a "
            "high-Re precision path and requires an explicit memory budget"
        ),
    )
    result.add_argument(
        "--wall-force-direction-chunk",
        type=int,
        default=4,
        help="D3Q19 directions per bounded-memory Guo wall-source chunk",
    )
    result.add_argument(
        "--low-memory-wall-macroscopic",
        action="store_true",
        help="use paired-direction D3Q19 moments inside the wall kernel",
    )
    result.add_argument(
        "--omega-bulk",
        type=float,
        default=1.0,
        help="cumulant bulk-mode relaxation rate in (0,2]",
    )
    result.add_argument("--wall-law", choices=("musker", "reichardt", "log"), default="musker")
    result.add_argument(
        "--disable-wall-stress",
        action="store_true",
        help="diagnostic only: retain BFL impermeability but omit wall-stress forcing",
    )
    result.add_argument("--stress-exchange-distance", type=float, default=1.0)
    result.add_argument(
        "--wall-exchange-distance-over-length-target",
        type=float,
        default=3.0 / 256.0,
        help="validation-family exchange-height ratio held fixed across grids",
    )
    result.add_argument(
        "--wall-model-y-plus-lower-bound", type=float, default=30.0,
    )
    result.add_argument(
        "--wall-model-y-plus-upper-bound", type=float, default=1000.0,
    )
    result.add_argument(
        "--minimum-wall-model-y-plus-in-range-fraction",
        type=float,
        default=0.9,
    )
    result.add_argument("--sponge-width", type=int, default=24)
    result.add_argument("--sponge-strength", type=float, default=0.3)
    result.add_argument(
        "--sponge-inlet",
        action="store_true",
        help="include the upstream x- face in equilibrium-difference damping",
    )
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
    invocation_started = time.perf_counter()
    if min(args.nx, args.ny, args.nz, args.hull_length, args.steps) <= 0:
        raise ValueError("grid, hull length and steps must be positive")
    if args.stress_exchange_distance <= 0.0:
        raise ValueError("stress exchange distance must be positive")
    if args.wall_exchange_distance_over_length_target <= 0.0:
        raise ValueError("wall-exchange distance ratio target must be positive")
    if not (
        0.0 <= args.wall_model_y_plus_lower_bound
        < args.wall_model_y_plus_upper_bound
    ):
        raise ValueError("wall-model y+ bounds must be ordered and non-negative")
    if not 0.0 <= args.minimum_wall_model_y_plus_in_range_fraction <= 1.0:
        raise ValueError("minimum wall-model y+ in-range fraction must lie in [0,1]")
    if args.deep_wall_margin != 0 and args.deep_wall_margin < 2:
        raise ValueError("deep wall margin must be 0 or at least two")
    if args.deep_wake_cells < 0 or (
        args.deep_wall_margin == 0 and args.deep_wake_cells != 0
    ):
        raise ValueError("deep wake cells require an enabled deep block")
    if args.memory_bytes_per_cell <= 0.0:
        raise ValueError("memory bytes per cell must be positive")
    if args.kbc_max_iterations < 2:
        raise ValueError("KBC maximum iterations must be at least two")
    if args.collision_chunk_cells < 0:
        raise ValueError("collision chunk cells must be non-negative")
    if args.compile_natural_kbc and args.collision_model != "natural_kbc":
        raise ValueError("compiled natural KBC requires --collision-model natural_kbc")
    if (
        args.natural_kbc_compute_dtype != "storage"
        and args.collision_model != "natural_kbc"
    ):
        raise ValueError(
            "float64 natural-KBC compute requires --collision-model natural_kbc",
        )
    if not 1 <= args.wall_force_direction_chunk <= 19:
        raise ValueError("wall force direction chunk must lie in [1,19]")
    if not 0.0 < args.omega_bulk <= 2.0:
        raise ValueError("bulk relaxation rate must lie in (0,2]")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint interval must be non-negative")
    if args.health_interval < 0:
        raise ValueError("health interval must be non-negative")
    if args.ramp_steps < 0 or min(
        args.wall_normal_ramp_steps,
        args.wall_shear_ramp_steps,
    ) < -1:
        raise ValueError("wall ramp steps must be non-negative or -1")
    wall_normal_ramp_steps = (
        args.ramp_steps
        if args.wall_normal_ramp_steps == -1 else args.wall_normal_ramp_steps
    )
    wall_shear_ramp_steps = (
        args.ramp_steps
        if args.wall_shear_ramp_steps == -1 else args.wall_shear_ramp_steps
    )
    if not args.lattice_speed < args.maximum_health_speed < 1.0:
        raise ValueError("maximum health speed must lie between inlet speed and one")
    if not 0.0 <= args.cs_smag <= 0.3:
        raise ValueError("cs_smag must lie in [0,0.3]")
    if not 0.0 <= args.wale_cw <= 1.0:
        raise ValueError("wale_cw must lie in [0,1]")
    if not 0.0 <= args.vreman_cv <= 0.2:
        raise ValueError("vreman_cv must lie in [0,0.2]")
    if not 0.0 <= args.minimum_health_population < 1.0:
        raise ValueError("minimum health population must lie in [0,1)")
    if not 0.0 <= args.maximum_positivity_limited_fraction <= 1.0:
        raise ValueError("maximum positivity-limited fraction must lie in [0,1]")
    if not 0.0 < args.maximum_reflux_applied_correction_fraction <= 0.2:
        raise ValueError(
            "maximum reflux applied correction fraction must lie in (0,0.2]",
        )
    if (args.interface_filter_width == 0) != (
        args.interface_filter_strength == 0.0
    ):
        raise ValueError(
            "interface filter width and strength must both be zero or positive",
        )
    resolved_reynolds_start = (
        args.resolved_reynolds
        if args.resolved_reynolds_start == 0.0
        else args.resolved_reynolds_start
    )
    if (
        resolved_reynolds_start != args.resolved_reynolds
        and not (
            0 <= args.viscosity_ramp_start_step
            < args.viscosity_ramp_end_step
            <= args.steps
        )
    ):
        raise ValueError(
            "non-constant viscosity continuation needs "
            "0 <= ramp start < ramp end <= steps",
        )
    continuation = ResolvedReynoldsContinuation(
        resolved_reynolds_start,
        args.resolved_reynolds,
        args.viscosity_ramp_start_step,
        args.viscosity_ramp_end_step,
    )
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup steps must lie in [0, steps)")
    if not 0 <= args.statistics_window_steps <= args.steps - args.warmup_steps:
        raise ValueError("statistics window exceeds the post-warmup trajectory")
    if min(
        args.report_interval,
        args.wall_diagnostic_interval,
        args.surface_force_interval,
        args.minimum_convective_times,
        args.minimum_target_reynolds_convective_times,
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
    refinement_plans = [outer_plan, nested_plan]
    planning_solids = [outer_solid, nested_solid]
    finest_geometry = nested_geometry
    if args.deep_wall_margin:
        deep_plan = plan_nested_suboff_static_amr(
            nested_plan,
            nested_solid,
            wall_margin=args.deep_wall_margin,
            wake_cells=args.deep_wake_cells,
        )
        deep_solid, deep_geometry = build_nested_fine_suboff_mask(
            deep_plan,
            hull_type=args.hull_type,
            coarse_center=center,
            config=geometry_config,
            device=device,
        )
        refinement_plans.append(deep_plan)
        planning_solids.append(deep_solid)
        finest_geometry = deep_geometry
    finest_plan = refinement_plans[-1]
    finest_planning_solid = planning_solids[-1]
    wall_exchange_interface_clearance = (
        assess_wall_exchange_interface_clearance(
            exchange_distance_cells=args.stress_exchange_distance,
            available_buffer_cells=finest_plan.wall_buffer_finest_cells,
        )
    )
    if not wall_exchange_interface_clearance.admitted:
        raise ValueError(
            "wall exchange requires "
            f"{wall_exchange_interface_clearance.required_buffer_cells} "
            "finest cells including trilinear support, but the refinement "
            f"block provides {finest_plan.wall_buffer_finest_cells}",
        )
    refinement_depth = len(refinement_plans)
    level_count = refinement_depth + 1
    if args.level_devices:
        level_devices = tuple(
            torch.device(value.strip())
            for value in args.level_devices.split(",")
            if value.strip()
        )
        if len(level_devices) != level_count:
            raise ValueError(
                "level-devices must contain exactly one device per hierarchy level",
            )
        if level_devices[0] != device:
            raise ValueError("the first level device must equal --device")
    else:
        level_devices = (device,) * level_count
    for level_device in dict.fromkeys(level_devices):
        if level_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(level_device)
    finest_device = level_devices[-1]
    force_averager = UniformSubcycleAverager(refinement_depth)
    physical_re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    initial_tau_by_level = continuation.tau_by_level(
        0,
        lattice_speed=args.lattice_speed,
        root_hull_length=args.hull_length,
        levels=level_count,
    )
    refinement_boxes = [
        outer_plan.box,
        *[
            plan.box_in_outer_allocated_coordinates
            for plan in refinement_plans[1:]
        ],
    ]
    amr_configs = []
    for interface_index, box in enumerate(refinement_boxes):
        amr_configs.append(StaticBlockAMRConfig(
            box,
            tau_coarse=initial_tau_by_level[interface_index],
            regularize_restriction=args.regularize_restriction,
            regularize_prolongation=args.regularize_prolongation,
            reflux_correction_stencil=args.reflux_correction_stencil,
            ghost_interpolation=args.ghost_interpolation,
            enforce_transfer_positivity=args.enforce_transfer_positivity,
            interface_filter_width=args.interface_filter_width,
            interface_filter_strength=args.interface_filter_strength,
        ))

    # Derive force-observer geometry before allocating the hierarchy so that
    # preflight mode catches a CV whose radius-one streaming flux stencil
    # would sample the interface filter.  Fine masks omit the AMR ghost shell.
    nested_indices = finest_planning_solid.nonzero(as_tuple=False)
    ghost = amr_configs[-1].ghost
    z_min, y_min, x_min = (
        int(nested_indices[:, axis].min().item()) + ghost for axis in range(3)
    )
    z_max, y_max, x_max = (
        int(nested_indices[:, axis].max().item()) + 1 + ghost
        for axis in range(3)
    )
    finest_shape = tuple(
        int(size) + 2 * ghost for size in finest_planning_solid.shape
    )

    def control_volume_bounds(margin: int) -> tuple[int, int, int, int, int, int]:
        return (
            x_min - margin, x_max + margin,
            y_min - margin, y_max + margin,
            z_min - margin, z_max + margin,
        )

    control_volume_bounds_by_margin = {
        margin: control_volume_bounds(margin)
        for margin in (args.cv_margin, *auxiliary_margins)
    }
    control_volume_clearance = []
    for role, margin in (
        ("primary", args.cv_margin),
        ("auxiliary", auxiliary_margins[0]),
        ("auxiliary", auxiliary_margins[1]),
    ):
        bounds = control_volume_bounds_by_margin[margin]
        assessment = assess_interface_filter_control_volume_clearance(
            finest_shape,
            bounds_xyz=bounds,
            ghost=ghost,
            filter_width=args.interface_filter_width,
        )
        record = {
            "role": role,
            "margin_cells": margin,
            "bounds_xyz_half_open": list(bounds),
        } | assessment.to_dict()
        control_volume_clearance.append(record)
        if not assessment.flux_stencil_outside_filter:
            required = args.interface_filter_width + (
                assessment.required_streaming_source_guard_cells
            )
            raise ValueError(
                f"control-volume margin {margin} leaves only "
                f"{assessment.minimum_physical_interface_clearance_cells} "
                "cells from the physical AMR interface; its streaming flux "
                f"stencil requires at least {required} cells to remain "
                "outside the interface filter",
            )
    estimated_peak_gib = finest_plan.estimated_peak_gib(
        args.memory_bytes_per_cell,
    )
    estimated_exchange_y_plus = estimate_exchange_yplus(
        physical_reynolds=physical_re,
        characteristic_length_cells=finest_plan.effective_hull_length_cells,
        exchange_distance_cells=args.stress_exchange_distance,
    )
    estimated_bfl_exchange_y_plus_bounds = estimate_bfl_exchange_yplus_bounds(
        physical_reynolds=physical_re,
        characteristic_length_cells=finest_plan.effective_hull_length_cells,
        requested_exchange_distance_cells=args.stress_exchange_distance,
    )
    root_external_domain_quality = grid_quality_metrics(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        hull_length=args.hull_length,
        u_in=args.lattice_speed,
        re=physical_re,
    )
    device_memory_plan = plan_hierarchy_device_memory(
        finest_plan.allocated_cells_by_level,
        level_devices,
        bytes_per_cell=args.memory_bytes_per_cell,
    )
    device_memory_preflight = []
    for allocation in device_memory_plan:
        budget = require_cuda_memory_budget(
            torch.device(allocation.device),
            estimated_peak_gib=allocation.estimated_peak_gib,
            reserve_gib=1.0,
            label=(
                "SUBOFF nested static-AMR levels "
                + ",".join(str(value) for value in allocation.level_indices)
            ),
        )
        device_memory_preflight.append(
            allocation.to_dict()
            | {"cuda_budget": budget.to_dict() if budget is not None else None},
        )
    planning = {
        "outer_box": vars(outer_plan.box),
        "inner_box_in_outer_allocated_coordinates": vars(
            nested_plan.box_in_outer_allocated_coordinates,
        ),
        "refinement_boxes_in_parent_allocated_coordinates": [
            vars(box) for box in refinement_boxes
        ],
        "refinement_depth": refinement_depth,
        "level_count": level_count,
        "force_samples_per_root_step": force_averager.expected_samples,
        "collision_chunk_cells": args.collision_chunk_cells,
        "compile_natural_kbc": args.compile_natural_kbc,
        "natural_kbc_compute_dtype": args.natural_kbc_compute_dtype,
        "population_storage_dtype": args.population_storage_dtype,
        "d3q19_weight_precision_scheme": WEIGHT_PRECISION_SCHEME,
        "wall_force_direction_chunk": args.wall_force_direction_chunk,
        "low_memory_wall_macroscopic": args.low_memory_wall_macroscopic,
        "outer_fine_shape": list(outer_plan.fine_physical_shape),
        "nested_fine_shape": list(nested_plan.fine_physical_shape),
        "fine_physical_shapes_by_level": [
            list(plan.fine_physical_shape) for plan in refinement_plans
        ],
        "allocated_cells_by_level": list(finest_plan.allocated_cells_by_level),
        "wall_buffer_parent_cells": finest_plan.wall_buffer_parent_cells,
        "wall_buffer_finest_cells": finest_plan.wall_buffer_finest_cells,
        "wall_exchange_interface_clearance": (
            wall_exchange_interface_clearance.to_dict()
        ),
        "downstream_buffer_parent_cells": (
            finest_plan.downstream_buffer_parent_cells
        ),
        "downstream_buffer_finest_cells": (
            finest_plan.downstream_buffer_finest_cells
        ),
        "total_allocated_cells": finest_plan.total_allocated_cells,
        "uniform_finest_cells": finest_plan.uniform_finest_cells,
        "cell_saving_fraction": finest_plan.cell_saving_fraction,
        "memory_estimate_bytes_per_cell": args.memory_bytes_per_cell,
        "estimated_peak_gib": estimated_peak_gib,
        "level_devices": [str(value) for value in level_devices],
        "device_memory_preflight": device_memory_preflight,
        "stress_exchange_distance_cells": args.stress_exchange_distance,
        "stress_exchange_distance_over_finest_length": (
            args.stress_exchange_distance
            / finest_plan.effective_hull_length_cells
        ),
        "estimated_exchange_y_plus": estimated_exchange_y_plus,
        "estimated_bfl_exchange_y_plus_bounds": (
            estimated_bfl_exchange_y_plus_bounds
        ),
        "root_external_domain_quality": root_external_domain_quality,
        "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
        "appendage_link_scheme": (
            SUBOFF_APPENDAGE_LINK_SCHEME
            if args.hull_type == "full"
            else "analytic_axisymmetric_bisection_v1"
        ),
        "cuda_memory_preflight": (
            device_memory_preflight[0]["cuda_budget"]
            if len(device_memory_preflight) == 1 else None
        ),
        "control_volume_interface_clearance": {
            "interface_filter_width_cells": args.interface_filter_width,
            "streaming_stencil_radius_cells": 1,
            "all_flux_stencils_outside_filter": True,
            "volumes": control_volume_clearance,
        },
    }
    if args.preflight_only:
        planning_bare_solid = None
        planning_with_sail_solid = None
        planning_appendage_links = 0
        if args.hull_type == "full":
            planning_bare_solid, _ = build_nested_fine_suboff_mask(
                finest_plan,
                hull_type="bare_hull",
                coarse_center=center,
                config=geometry_config,
                device=device,
            )
            planning_with_sail_solid, _ = build_nested_fine_suboff_mask(
                finest_plan,
                hull_type="with_sail",
                coarse_center=center,
                config=geometry_config,
                device=device,
            )
            planning_appendage_links = count_suboff_appendage_boundary_links(
                finest_planning_solid,
                planning_bare_solid,
            )
        planning_resolution = assess_suboff_geometry_resolution(
            finest_planning_solid,
            hull_type=args.hull_type,
            fine_hull_length_cells=(
                finest_plan.effective_hull_length_cells
            ),
            center_yz=(
                float(finest_geometry["cy"]),
                float(finest_geometry["cz"]),
            ),
            bare_hull=planning_bare_solid,
            with_sail=planning_with_sail_solid,
            appendage_halfway_links=planning_appendage_links,
        )
        planning_resolution_output = planning_resolution.to_dict()
        if args.hull_type == "full":
            planning_resolution_output["appendage_boundary_links"] = (
                planning_appendage_links
            )
            planning_resolution_output["appendage_halfway_links"] = 0
            planning_resolution_output["appendage_link_scheme"] = (
                SUBOFF_APPENDAGE_LINK_SCHEME
            )
        planning["geometry_resolution"] = planning_resolution_output
        return {
            "schema": "tensorlbm-suboff-nested-amr-smoke-v3",
            "status": "preflight_only",
            "physical_validation": False,
            "planning": planning,
        }

    finest_planning_solid = finest_planning_solid.to(device=finest_device)
    population_dtype = getattr(torch, args.population_storage_dtype)
    rho = torch.ones(shape, device=device, dtype=population_dtype)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    hierarchy = NestedStaticBlockAMR3D(
        equilibrium3d(rho, ux, zero, zero, device=device),
        tuple(amr_configs),
        fine_solids=(None,) * (refinement_depth - 1) + (
            finest_planning_solid,
        ),
        fine_devices=level_devices[1:],
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
        "projected_bfl_pressure_diagnostic": (
            args.enable_projected_bfl_pressure_diagnostic
        ),
        "projected_bfl_pressure_reconstruction": (
            args.projected_bfl_pressure_reconstruction
        ),
        "force_samples_per_root_step": force_averager.expected_samples,
        "ramp_steps": args.ramp_steps,
        "wall_normal_ramp_steps": wall_normal_ramp_steps,
        "wall_shear_ramp_steps": wall_shear_ramp_steps,
        "lattice_speed": args.lattice_speed,
        "resolved_reynolds": args.resolved_reynolds,
        "resolved_reynolds_start": resolved_reynolds_start,
        "viscosity_ramp_start_step": args.viscosity_ramp_start_step,
        "viscosity_ramp_end_step": args.viscosity_ramp_end_step,
        "rho_water": args.rho_water,
        "nu_water": args.nu_water,
        "cs_smag": args.cs_smag,
        "wale_cw": args.wale_cw,
        "vreman_cv": args.vreman_cv,
        "collision_model": args.collision_model,
        "kbc_max_iterations": args.kbc_max_iterations,
        "collision_chunk_cells": args.collision_chunk_cells,
        "compile_natural_kbc": args.compile_natural_kbc,
        "natural_kbc_compute_dtype": args.natural_kbc_compute_dtype,
        "population_storage_dtype": args.population_storage_dtype,
        "d3q19_weight_precision_scheme": WEIGHT_PRECISION_SCHEME,
        "omega_bulk": args.omega_bulk,
        "wall_law": args.wall_law,
        "wall_stress_enabled": not args.disable_wall_stress,
        "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
        "stress_exchange_distance": args.stress_exchange_distance,
        "wall_exchange_distance_over_length_target": (
            args.wall_exchange_distance_over_length_target
        ),
        "wall_model_y_plus_lower_bound": args.wall_model_y_plus_lower_bound,
        "wall_model_y_plus_upper_bound": args.wall_model_y_plus_upper_bound,
        "minimum_wall_model_y_plus_in_range_fraction": (
            args.minimum_wall_model_y_plus_in_range_fraction
        ),
        "wall_diagnostic_interval": args.wall_diagnostic_interval,
        "sponge_width": args.sponge_width,
        "sponge_strength": args.sponge_strength,
        "sponge_inlet": args.sponge_inlet,
        "far_field_mode": args.far_field_mode,
        "regularize_restriction": args.regularize_restriction,
        "regularize_prolongation": args.regularize_prolongation,
        "reflux_correction_stencil": args.reflux_correction_stencil,
        "ghost_interpolation": args.ghost_interpolation,
        "enforce_transfer_positivity": args.enforce_transfer_positivity,
        "interface_filter_width": args.interface_filter_width,
        "interface_filter_strength": args.interface_filter_strength,
        "minimum_health_population": args.minimum_health_population,
        "maximum_positivity_limited_fraction": (
            args.maximum_positivity_limited_fraction
        ),
        "maximum_reflux_applied_correction_fraction": (
            args.maximum_reflux_applied_correction_fraction
        ),
    }
    if refinement_depth > 2:
        checkpoint_signature["refinement_depth"] = refinement_depth
        checkpoint_signature["deep_box"] = vars(refinement_boxes[-1])
    if args.hull_type == "full":
        checkpoint_signature["appendage_link_scheme"] = (
            SUBOFF_APPENDAGE_LINK_SCHEME
        )
    finest_solid = hierarchy.interfaces[-1].fine_solid_with_ghost
    assert finest_solid is not None
    finest_solid_q = finest_solid.unsqueeze(0).expand_as(hierarchy.finest_f)
    finest_center = (
        float(finest_geometry["cx"]) + amr_configs[-1].ghost,
        float(finest_geometry["cy"]) + amr_configs[-1].ghost,
        float(finest_geometry["cz"]) + amr_configs[-1].ghost,
    )
    nz_f, ny_f, nx_f = finest_solid.shape
    finest_length = finest_plan.effective_hull_length_cells
    bfl_mask, bfl_q = compute_q_suboff(
        nx_f, ny_f, nz_f, *finest_center, finest_length,
        hull_type=args.hull_type, config=geometry_config, device=finest_device,
        solid_mask=finest_solid,
    )
    appendage_boundary_links = 0
    appendage_link_diagnostics = None
    bare_solid = None
    if args.hull_type == "full":
        bare_solid, _ = build_suboff_mask(
            "bare_hull", nx_f, ny_f, nz_f,
            cx=finest_center[0], cy=finest_center[1], cz=finest_center[2],
            length=finest_length, config=geometry_config, device=finest_device,
        )
        bfl_q, appendage_link_diagnostics = refine_q_suboff_appendages(
            bfl_mask,
            bfl_q,
            finest_solid,
            bare_solid,
            center=finest_center,
            length=finest_length,
            inplace=True,
        )
        appendage_boundary_links = appendage_link_diagnostics.target_links
    near = get_near_wall_3d(finest_solid)
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
        assert bare_solid is not None
        with_sail_solid, _ = build_suboff_mask(
            "with_sail", nx_f, ny_f, nz_f,
            cx=finest_center[0], cy=finest_center[1], cz=finest_center[2],
            length=finest_length, config=geometry_config, device=finest_device,
        )
        bare_near = get_near_wall_3d(bare_solid)
        bare_surface = SurfaceMesh.from_gradient(bare_solid, bare_near)
        bare_bfl_mask, _ = compute_q_suboff(
            nx_f, ny_f, nz_f, *finest_center, finest_length,
            hull_type="bare_hull", config=geometry_config, device=finest_device,
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
    def build_control_volume(margin: int) -> torch.Tensor:
        x0, x1, y0, y1, z0, z1 = control_volume_bounds_by_margin[margin]
        return box_control_volume(
            finest_solid.shape,
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            z0=z0, z1=z1,
            device=finest_device,
        )

    control_volume = build_control_volume(args.cv_margin)
    auxiliary_control_volumes = {
        margin: build_control_volume(margin) for margin in auxiliary_margins
    }
    sponge_faces = ["x+", "y-", "y+", "z-", "z+"]
    if args.sponge_inlet:
        sponge_faces.insert(0, "x-")
    sponge = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=tuple(sponge_faces),
    )
    persistent_allocated_gib_by_device = {
        str(level_device): torch.cuda.memory_allocated(level_device) / 2**30
        for level_device in dict.fromkeys(level_devices)
        if level_device.type == "cuda"
    }
    planning["cuda_persistent_allocated_gib_by_device"] = (
        persistent_allocated_gib_by_device
    )
    runtime_memory_reserves = []
    for allocation in device_memory_plan:
        reserve = require_cuda_runtime_reserve(
            torch.device(allocation.device),
            required_reserve_gib=1.0,
            label=(
                "SUBOFF nested static-AMR levels "
                + ",".join(str(value) for value in allocation.level_indices)
            ),
        )
        runtime_memory_reserves.append({
            "device": allocation.device,
            "level_indices": list(allocation.level_indices),
            "cuda_reserve": reserve.to_dict() if reserve is not None else None,
        })
    planning["cuda_runtime_reserve_after_persistent_allocation_by_device"] = (
        runtime_memory_reserves
    )
    planning["cuda_runtime_reserve_after_persistent_allocation"] = (
        runtime_memory_reserves[0]["cuda_reserve"]
        if len(runtime_memory_reserves) == 1 else None
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
    resumed_legacy_v3_checkpoint = False
    resumed_pre_gradient_sgs_checkpoint = False
    resumed_pre_inlet_sponge_checkpoint = False
    resumed_pre_collision_chunk_checkpoint = False
    resumed_pre_y_plus_distribution_checkpoint = False
    force_samples: list[dict] = []
    open_boundary_diagnostics: list[dict] = []
    step_records: list[dict] = []
    maximum_limiter_fraction = 0.0
    maximum_reflux_residual = [0.0] * refinement_depth
    maximum_reflux_limited_directions = [0] * refinement_depth
    maximum_reflux_applied_correction_fraction = [0.0] * refinement_depth
    maximum_transfer_limited_fraction = [0.0] * refinement_depth
    minimum_transfer_alpha = [1.0] * refinement_depth
    maximum_raw_mass_mismatch = [0.0] * refinement_depth
    maximum_raw_momentum_mismatch = [0.0] * refinement_depth
    maximum_rejected_fraction = 0.0
    health_records: list[dict] = []

    if args.resume:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        stored_configuration = state.get("configuration")
        if isinstance(stored_configuration, dict):
            stored_configuration = dict(stored_configuration)
            stored_configuration.setdefault("compile_natural_kbc", False)
            stored_configuration.setdefault(
                "natural_kbc_compute_dtype", "storage",
            )
            stored_configuration.setdefault("population_storage_dtype", "float32")
        pre_collision_chunk_signature = dict(checkpoint_signature)
        pre_collision_chunk_signature.pop("collision_chunk_cells")
        resumed_pre_collision_chunk_checkpoint = (
            stored_configuration == pre_collision_chunk_signature
        )
        pre_y_plus_distribution_signature = dict(checkpoint_signature)
        pre_y_plus_distribution_signature.pop(
            "wall_exchange_distance_over_length_target",
        )
        pre_y_plus_distribution_signature.pop("wall_model_y_plus_lower_bound")
        pre_y_plus_distribution_signature.pop("wall_model_y_plus_upper_bound")
        pre_y_plus_distribution_signature.pop(
            "minimum_wall_model_y_plus_in_range_fraction",
        )
        resumed_pre_y_plus_distribution_checkpoint = (
            stored_configuration == pre_y_plus_distribution_signature
        )
        pre_inlet_sponge_signature = dict(checkpoint_signature)
        pre_inlet_sponge_signature.pop("sponge_inlet")
        resumed_pre_inlet_sponge_checkpoint = (
            not args.sponge_inlet
            and stored_configuration == pre_inlet_sponge_signature
        )
        pre_gradient_sgs_signature = dict(checkpoint_signature)
        pre_gradient_sgs_signature.pop("wale_cw")
        pre_gradient_sgs_signature.pop("vreman_cv")
        pre_gradient_sgs_signature.pop("sponge_inlet")
        resumed_pre_gradient_sgs_checkpoint = (
            args.collision_model not in {"cumulant_wale", "cumulant_vreman"}
            and not args.sponge_inlet
            and stored_configuration == pre_gradient_sgs_signature
        )
        legacy_v3_signature = dict(checkpoint_signature)
        legacy_v3_signature.pop("wale_cw")
        legacy_v3_signature.pop("vreman_cv")
        legacy_v3_signature.pop("regularize_restriction")
        legacy_v3_signature.pop("regularize_prolongation")
        legacy_v3_signature.pop("reflux_correction_stencil")
        legacy_v3_signature.pop("ghost_interpolation")
        legacy_v3_signature.pop("enforce_transfer_positivity")
        legacy_v3_signature.pop("interface_filter_width")
        legacy_v3_signature.pop("interface_filter_strength")
        legacy_v3_signature.pop("wall_stress_enabled")
        legacy_v3_signature.pop("collision_model")
        legacy_v3_signature.pop("kbc_max_iterations")
        legacy_v3_signature.pop("omega_bulk")
        legacy_v3_signature.pop("resolved_reynolds_start")
        legacy_v3_signature.pop("viscosity_ramp_start_step")
        legacy_v3_signature.pop("viscosity_ramp_end_step")
        legacy_v3_signature.pop("wall_normal_ramp_steps")
        legacy_v3_signature.pop("wall_shear_ramp_steps")
        legacy_v3_signature.pop("minimum_health_population")
        legacy_v3_signature.pop("maximum_positivity_limited_fraction")
        legacy_v3_signature.pop("maximum_reflux_applied_correction_fraction")
        legacy_v3_signature.pop("sponge_inlet")
        resumed_legacy_v3_checkpoint = (
            not args.regularize_restriction
            and not args.regularize_prolongation
            and args.reflux_correction_stencil == "exterior_cells"
            and args.ghost_interpolation == "injection"
            and not args.enforce_transfer_positivity
            and args.interface_filter_width == 0
            and args.interface_filter_strength == 0.0
            and not args.disable_wall_stress
            and args.collision_model == "cumulant_smagorinsky"
            and resolved_reynolds_start == args.resolved_reynolds
            and wall_normal_ramp_steps == args.ramp_steps
            and wall_shear_ramp_steps == args.ramp_steps
            and args.minimum_health_population == 1.0e-8
            and args.maximum_positivity_limited_fraction == 1.0e-6
            and args.maximum_reflux_applied_correction_fraction == 1.0e-3
            and not args.sponge_inlet
            and stored_configuration == legacy_v3_signature
        )
        legacy_v2_signature = dict(checkpoint_signature)
        legacy_v2_signature["schema_version"] = 2
        legacy_v2_signature.pop("hull_type")
        legacy_v2_without_new_transfer = dict(legacy_v2_signature)
        legacy_v2_without_new_transfer.pop("wale_cw")
        legacy_v2_without_new_transfer.pop("vreman_cv")
        legacy_v2_without_new_transfer.pop("regularize_restriction")
        legacy_v2_without_new_transfer.pop("regularize_prolongation")
        legacy_v2_without_new_transfer.pop("reflux_correction_stencil")
        legacy_v2_without_new_transfer.pop("ghost_interpolation")
        legacy_v2_without_new_transfer.pop("enforce_transfer_positivity")
        legacy_v2_without_new_transfer.pop("interface_filter_width")
        legacy_v2_without_new_transfer.pop("interface_filter_strength")
        legacy_v2_without_new_transfer.pop("wall_stress_enabled")
        legacy_v2_without_new_transfer.pop("collision_model")
        legacy_v2_without_new_transfer.pop("kbc_max_iterations")
        legacy_v2_without_new_transfer.pop("omega_bulk")
        legacy_v2_without_new_transfer.pop("resolved_reynolds_start")
        legacy_v2_without_new_transfer.pop("viscosity_ramp_start_step")
        legacy_v2_without_new_transfer.pop("viscosity_ramp_end_step")
        legacy_v2_without_new_transfer.pop("wall_normal_ramp_steps")
        legacy_v2_without_new_transfer.pop("wall_shear_ramp_steps")
        legacy_v2_without_new_transfer.pop("minimum_health_population")
        legacy_v2_without_new_transfer.pop("maximum_positivity_limited_fraction")
        legacy_v2_without_new_transfer.pop(
            "maximum_reflux_applied_correction_fraction",
        )
        legacy_v2_without_new_transfer.pop("sponge_inlet")
        resumed_legacy_v2_checkpoint = (
            args.hull_type == "bare_hull"
            and not args.regularize_restriction
            and not args.regularize_prolongation
            and args.reflux_correction_stencil == "exterior_cells"
            and args.ghost_interpolation == "injection"
            and not args.enforce_transfer_positivity
            and args.interface_filter_width == 0
            and args.interface_filter_strength == 0.0
            and not args.disable_wall_stress
            and args.collision_model == "cumulant_smagorinsky"
            and resolved_reynolds_start == args.resolved_reynolds
            and wall_normal_ramp_steps == args.ramp_steps
            and wall_shear_ramp_steps == args.ramp_steps
            and args.minimum_health_population == 1.0e-8
            and args.maximum_positivity_limited_fraction == 1.0e-6
            and args.maximum_reflux_applied_correction_fraction == 1.0e-3
            and not args.sponge_inlet
            and stored_configuration == legacy_v2_without_new_transfer
        )
        if (
            stored_configuration != checkpoint_signature
            and not resumed_pre_collision_chunk_checkpoint
            and not resumed_pre_y_plus_distribution_checkpoint
            and not resumed_pre_inlet_sponge_checkpoint
            and not resumed_pre_gradient_sgs_checkpoint
            and not resumed_legacy_v3_checkpoint
            and not resumed_legacy_v2_checkpoint
        ):
            raise ValueError("checkpoint configuration does not match nested smoke")
        current_step = int(state["step"])
        resumed_from_step = current_step
        if current_step >= args.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")
        hierarchy.restore_level_populations([
            populations.to(device=template.device, dtype=template.dtype)
            for populations, template in zip(
                state["level_populations"],
                hierarchy.level_populations,
                strict=True,
            )
        ])
        step_records = list(state["step_records"])
        maximum_limiter_fraction = float(state["maximum_limiter_fraction"])
        maximum_reflux_residual = [
            float(value) for value in state["maximum_reflux_residual"]
        ]
        maximum_reflux_limited_directions = [
            int(value) for value in state["maximum_reflux_limited_directions"]
        ]
        maximum_reflux_applied_correction_fraction = [
            float(value) for value in state.get(
                "maximum_reflux_applied_correction_fraction",
                (0.0,) * refinement_depth,
            )
        ]
        maximum_rejected_fraction = float(state["maximum_rejected_fraction"])
        maximum_transfer_limited_fraction = [
            float(value) for value in state.get(
                "maximum_transfer_limited_fraction", (0.0,) * refinement_depth,
            )
        ]
        minimum_transfer_alpha = [
            float(value) for value in state.get(
                "minimum_transfer_alpha", (1.0,) * refinement_depth,
            )
        ]
        maximum_raw_mass_mismatch = [
            float(value) for value in state.get(
                "maximum_raw_mass_mismatch", (0.0,) * refinement_depth,
            )
        ]
        maximum_raw_momentum_mismatch = [
            float(value) for value in state.get(
                "maximum_raw_momentum_mismatch", (0.0,) * refinement_depth,
            )
        ]
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
            "level_solid_masks": [
                None,
                *[
                    None if interface.fine_solid_with_ghost is None else (
                        interface.fine_solid_with_ghost.detach().cpu()
                    )
                    for interface in hierarchy.interfaces
                ],
            ],
            "step_records": step_records,
            "maximum_limiter_fraction": maximum_limiter_fraction,
            "maximum_reflux_residual": maximum_reflux_residual,
            "maximum_reflux_limited_directions": (
                maximum_reflux_limited_directions
            ),
            "maximum_reflux_applied_correction_fraction": (
                maximum_reflux_applied_correction_fraction
            ),
            "maximum_rejected_fraction": maximum_rejected_fraction,
            "maximum_transfer_limited_fraction": maximum_transfer_limited_fraction,
            "minimum_transfer_alpha": minimum_transfer_alpha,
            "maximum_raw_mass_mismatch": maximum_raw_mass_mismatch,
            "maximum_raw_momentum_mismatch": maximum_raw_momentum_mismatch,
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

    natural_kbc_executor = NaturalKBCCollisionExecutor(
        compile_enabled=args.compile_natural_kbc,
        compute_dtype=args.natural_kbc_compute_dtype,
    )

    def collide(state: torch.Tensor, tau: float, level: int) -> torch.Tensor:
        nonlocal maximum_limiter_fraction
        if args.collision_model == "entropic_kbc":
            if args.collision_chunk_cells:
                post = collide_in_z_chunks(
                    state,
                    lambda slab: collide_kbc_d3q19(
                        slab,
                        tau=tau,
                        max_iter=args.kbc_max_iterations,
                    ),
                    chunk_cells=args.collision_chunk_cells,
                )
            else:
                post = collide_kbc_d3q19(
                    state,
                    tau=tau,
                    max_iter=args.kbc_max_iterations,
                )
        elif args.collision_model == "natural_kbc":
            if args.collision_chunk_cells:
                post = collide_in_z_chunks(
                    state,
                    lambda slab: natural_kbc_executor(slab, tau),
                    chunk_cells=args.collision_chunk_cells,
                )
            else:
                post = natural_kbc_executor(state, tau)
        else:
            sgs_coefficients = {
                "cumulant_smagorinsky": {"C_s": args.cs_smag},
                "cumulant_wale": {
                    "C_w": args.wale_cw,
                    "solid_mask": (
                        hierarchy.interfaces[level - 1].fine_solid_with_ghost
                        if level > 0 else None
                    ),
                },
                "cumulant_vreman": {
                    "C_v": args.vreman_cv,
                    "solid_mask": (
                        hierarchy.interfaces[level - 1].fine_solid_with_ghost
                        if level > 0 else None
                    ),
                },
            }[args.collision_model]
            post = collide_cumulant_d3q19(
                state, tau=tau, omega_b=args.omega_bulk, **sgs_coefficients,
            )
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
        if level < refinement_depth:
            out = stream3d(collided)
            if level == 0:
                collect_open_boundary_diagnostics = (
                    bool(args.health_interval)
                    and current_step % args.health_interval == 0
                    and args.far_field_mode == "non_equilibrium_extrapolation"
                )
                if args.far_field_mode == "non_equilibrium_extrapolation":
                    boundary_result = non_equilibrium_far_field_bc_3d(
                        out,
                        u_in=args.lattice_speed,
                        return_diagnostics=collect_open_boundary_diagnostics,
                    )
                    if collect_open_boundary_diagnostics:
                        out, boundary_diagnostic = boundary_result
                        open_boundary_diagnostics.append({
                            "stage": "post_stream_pre_sponge",
                            **asdict(boundary_diagnostic),
                        })
                    else:
                        out = boundary_result
                else:
                    out = far_field_bc_3d(out, u_in=args.lattice_speed)
                out = apply_equilibrium_difference_sponge(
                    out,
                    sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
                if args.far_field_mode == "non_equilibrium_extrapolation":
                    boundary_result = non_equilibrium_far_field_bc_3d(
                        out,
                        u_in=args.lattice_speed,
                        return_diagnostics=collect_open_boundary_diagnostics,
                    )
                    if collect_open_boundary_diagnostics:
                        out, boundary_diagnostic = boundary_result
                        open_boundary_diagnostics.append({
                            "stage": "post_sponge",
                            **asdict(boundary_diagnostic),
                        })
                    else:
                        out = boundary_result
                else:
                    out = far_field_bc_3d(out, u_in=args.lattice_speed)
            return AMRAdvanceResult(out, collided)

        post_collision = torch.where(finest_solid_q, before, collided)
        out = stream3d(post_collision)
        normal_activation = smooth_ramp_factor(
            current_step, wall_normal_ramp_steps,
        )
        shear_activation = smooth_ramp_factor(
            current_step, wall_shear_ramp_steps,
        )
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
            wall_activation=1.0,
            wall_normal_activation=normal_activation,
            wall_shear_activation=shear_activation,
            stress_exchange_distance=args.stress_exchange_distance,
            wall_normals=(surface.nx_n, surface.ny_n, surface.nz_n),
            area_weight=area_weight,
            apply_wall_stress=not args.disable_wall_stress,
            return_wall_diagnostics=collect_wall_diagnostics,
            guo_direction_chunk_size=args.wall_force_direction_chunk,
            use_low_memory_macroscopic=args.low_memory_wall_macroscopic,
            y_plus_lower_bound=args.wall_model_y_plus_lower_bound,
            y_plus_upper_bound=args.wall_model_y_plus_upper_bound,
            minimum_y_plus_in_range_fraction=(
                args.minimum_wall_model_y_plus_in_range_fraction
            ),
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
            y_plus_summary = diagnostics.y_plus_summary
            pressure_gradient_parameter_mean = (
                diagnostics.pressure_gradient_parameter_mean
            )
            pressure_gradient_parameter_p95 = (
                diagnostics.pressure_gradient_parameter_p95
            )
            pressure_gradient_parameter_max = (
                diagnostics.pressure_gradient_parameter_max
            )
            pressure_gradient_summary = diagnostics.pressure_gradient_summary
            wall_shear_axial_profile = diagnostics.wall_shear_axial_profile
            link_force_decomposition = diagnostics.link_force_decomposition
        else:
            out, friction, pressure = wall_result
            mean_y_plus = None
            minimum_y_plus = None
            maximum_y_plus = None
            mean_wall_distance = None
            y_plus_summary = None
            pressure_gradient_parameter_mean = None
            pressure_gradient_parameter_p95 = None
            pressure_gradient_parameter_max = None
            pressure_gradient_summary = None
            wall_shear_axial_profile = None
            link_force_decomposition = None
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
            "y_plus_summary": y_plus_summary,
            "pressure_gradient_parameter_mean": (
                pressure_gradient_parameter_mean
            ),
            "pressure_gradient_parameter_p95": pressure_gradient_parameter_p95,
            "pressure_gradient_parameter_max": pressure_gradient_parameter_max,
            "pressure_gradient_summary": pressure_gradient_summary,
            "wall_shear_axial_profile": wall_shear_axial_profile,
            "link_force_decomposition": link_force_decomposition,
            "auxiliary": auxiliary_forces,
        })
        return AMRAdvanceResult(out, post_collision)

    start_step = current_step
    for step in range(start_step + 1, args.steps + 1):
        current_step = step
        force_samples.clear()
        open_boundary_diagnostics.clear()
        instantaneous_reynolds = continuation.reynolds_at(current_step)
        instantaneous_tau_by_level = continuation.tau_by_level(
            current_step,
            lattice_speed=args.lattice_speed,
            root_hull_length=args.hull_length,
            levels=level_count,
        )
        ledgers = hierarchy.step(
            advance,
            tau_by_level=instantaneous_tau_by_level,
        )
        raw_mismatch_moments = []
        for ledger in ledgers:
            if ledger.raw_kinetic_mismatch is None:
                raise RuntimeError("nested reflux ledger omitted raw kinetic mismatch")
            raw_mass, raw_momentum = conserved_population_moments(
                ledger.raw_kinetic_mismatch,
            )
            raw_mismatch_moments.append((
                abs(float(raw_mass)),
                float(torch.linalg.vector_norm(raw_momentum)),
            ))
        if args.health_interval and current_step % args.health_interval == 0:
            level_health = [
                inspect_population_health(populations).to_dict()
                for populations in hierarchy.level_populations
            ]
            diagnostic_force_samples = [
                sample for sample in force_samples
                if sample["y_plus"] is not None
            ]
            diagnostic_y_plus_summaries = [
                sample["y_plus_summary"] for sample in force_samples
                if sample["y_plus_summary"] is not None
            ]
            diagnostic_y_plus_aggregate = (
                aggregate_wall_exchange_yplus_summaries(
                    diagnostic_y_plus_summaries,
                ).to_dict()
                if diagnostic_y_plus_summaries else None
            )
            pressure_gradient_samples = [
                sample for sample in force_samples
                if sample["pressure_gradient_parameter_mean"] is not None
            ]
            pressure_gradient_summaries = [
                sample["pressure_gradient_summary"] for sample in force_samples
                if sample["pressure_gradient_summary"] is not None
            ]
            pressure_gradient_aggregate = (
                aggregate_wall_pressure_gradient_summaries(
                    pressure_gradient_summaries,
                ).to_dict()
                if pressure_gradient_summaries else None
            )
            wall_exchange_health = {
                "force_samples_observed": len(force_samples),
                "force_samples_expected": force_averager.expected_samples,
                "diagnostic_samples": len(diagnostic_force_samples),
                "mean_distance_cells": (
                    sum(
                        sample["wall_distance"]
                        for sample in diagnostic_force_samples
                    ) / len(diagnostic_force_samples)
                    if diagnostic_force_samples else None
                ),
                "minimum_y_plus": (
                    min(
                        sample["y_plus_min"]
                        for sample in diagnostic_force_samples
                    )
                    if diagnostic_force_samples else None
                ),
                "mean_y_plus": (
                    sum(
                        sample["y_plus"]
                        for sample in diagnostic_force_samples
                    ) / len(diagnostic_force_samples)
                    if diagnostic_force_samples else None
                ),
                "maximum_y_plus": (
                    max(
                        sample["y_plus_max"]
                        for sample in diagnostic_force_samples
                    )
                    if diagnostic_force_samples else None
                ),
                "y_plus_distribution": diagnostic_y_plus_aggregate,
                "pressure_gradient_parameter": {
                    "observations": len(pressure_gradient_samples),
                    "mean": (
                        sum(
                            sample["pressure_gradient_parameter_mean"]
                            for sample in pressure_gradient_samples
                        ) / len(pressure_gradient_samples)
                        if pressure_gradient_samples else None
                    ),
                    "maximum_p95": (
                        max(
                            sample["pressure_gradient_parameter_p95"]
                            for sample in pressure_gradient_samples
                        ) if pressure_gradient_samples else None
                    ),
                    "maximum": (
                        max(
                            sample["pressure_gradient_parameter_max"]
                            for sample in pressure_gradient_samples
                        ) if pressure_gradient_samples else None
                    ),
                    "distribution": pressure_gradient_aggregate,
                },
            }
            interface_health = [
                {
                    "maximum_reflux_residual": float(ledger.residual.abs().max()),
                    "reflux_limited_directions": ledger.limited_directions,
                    "maximum_applied_correction_fraction": (
                        ledger.maximum_applied_correction_fraction
                    ),
                    "restriction_limited_fraction": (
                        ledger.restriction_limited_fraction
                    ),
                    "restriction_minimum_alpha": ledger.restriction_minimum_alpha,
                    "prolongation_limited_fraction": (
                        ledger.prolongation_limited_fraction
                    ),
                    "prolongation_minimum_alpha": (
                        ledger.prolongation_minimum_alpha
                    ),
                    "raw_mass_mismatch": raw_mismatch_moments[index][0],
                    "raw_momentum_mismatch_norm": raw_mismatch_moments[index][1],
                }
                for index, ledger in enumerate(ledgers)
            ]
            finest_peak_index = level_health[-1]["maximum_speed_index_zyx"]
            finest_peak_context = None
            if finest_peak_index is not None:
                peak_z, peak_y, peak_x = finest_peak_index
                peak_links = bfl_mask[:, peak_z, peak_y, peak_x]
                peak_q = bfl_q[:, peak_z, peak_y, peak_x][peak_links]
                finest_peak_context = {
                    "near_wall": bool(near[peak_z, peak_y, peak_x]),
                    "solid": bool(finest_solid[peak_z, peak_y, peak_x]),
                    "cells_from_allocated_boundary": min(
                        peak_z,
                        peak_y,
                        peak_x,
                        nz_f - 1 - peak_z,
                        ny_f - 1 - peak_y,
                        nx_f - 1 - peak_x,
                    ),
                    "body_bbox_relative_zyx": [
                        peak_z - z_min,
                        peak_y - y_min,
                        peak_x - x_min,
                    ],
                    "bfl_link_count": int(peak_links.sum()),
                    "minimum_bfl_q": (
                        float(peak_q.min()) if peak_q.numel() else None
                    ),
                    "maximum_bfl_q": (
                        float(peak_q.max()) if peak_q.numel() else None
                    ),
                }
            health_records.append({
                "step": current_step,
                "collision_resolved_reynolds": instantaneous_reynolds,
                "collision_tau_by_level": list(instantaneous_tau_by_level),
                "wall_normal_activation": smooth_ramp_factor(
                    current_step, wall_normal_ramp_steps,
                ),
                "wall_shear_activation": smooth_ramp_factor(
                    current_step, wall_shear_ramp_steps,
                ),
                "target_reynolds_reached": math.isclose(
                    instantaneous_reynolds,
                    args.resolved_reynolds,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                ),
                "maximum_collision_limited_fraction": maximum_limiter_fraction,
                "maximum_wall_sample_rejected_fraction": maximum_rejected_fraction,
                "wall_exchange": wall_exchange_health,
                "open_boundary_population_delta": {
                    "stages": open_boundary_diagnostics,
                    "mass_delta": sum(
                        record["mass_delta"]
                        for record in open_boundary_diagnostics
                    ),
                    "momentum_delta": [
                        sum(
                            record["momentum_delta"][axis]
                            for record in open_boundary_diagnostics
                        )
                        for axis in range(3)
                    ],
                    "finite": all(
                        record["finite"] for record in open_boundary_diagnostics
                    ),
                } if open_boundary_diagnostics else None,
                "levels": level_health,
                "interfaces": interface_health,
                "finest_peak_speed_context": finest_peak_context,
            })
            print(
                "nested health "
                + json.dumps(health_records[-1], separators=(",", ":")),
                flush=True,
            )
            if not all(record["finite"] for record in level_health):
                raise FloatingPointError(
                    f"non-finite hierarchy state at root_step={current_step}",
                )
            peak_speed = max(
                float(record["maximum_speed"])
                for record in level_health
                if record["maximum_speed"] is not None
            )
            if peak_speed > args.maximum_health_speed:
                raise FloatingPointError(
                    "hierarchy exceeded the weakly-compressible speed gate "
                    f"at root_step={current_step}: {peak_speed:.6g} > "
                    f"{args.maximum_health_speed:.6g}",
                )
            minimum_population = min(
                float(record["minimum_population"]) for record in level_health
            )
            if minimum_population < args.minimum_health_population:
                raise FloatingPointError(
                    "hierarchy crossed the population-health floor "
                    f"at root_step={current_step}: {minimum_population:.6g} < "
                    f"{args.minimum_health_population:.6g}",
                )
            if (
                maximum_limiter_fraction
                > args.maximum_positivity_limited_fraction
            ):
                raise FloatingPointError(
                    "hierarchy exceeded the positivity-limiter gate "
                    f"at root_step={current_step}: "
                    f"{maximum_limiter_fraction:.6g} > "
                    f"{args.maximum_positivity_limited_fraction:.6g}",
                )
            maximum_interface_correction = max(
                ledger.maximum_applied_correction_fraction
                for ledger in ledgers
            )
            if (
                maximum_interface_correction
                > args.maximum_reflux_applied_correction_fraction
            ):
                raise FloatingPointError(
                    "hierarchy exceeded the reflux-correction gate "
                    f"at root_step={current_step}: "
                    f"{maximum_interface_correction:.6g} > "
                    f"{args.maximum_reflux_applied_correction_fraction:.6g}",
                )
        # The common averager owns both the recursive sample-count invariant
        # and its denominator.  This must evolve with refinement depth.
        cv_mean = force_averager.mean(
            (item["cv"] for item in force_samples), observable="CV force",
        )
        for index, ledger in enumerate(ledgers):
            maximum_reflux_residual[index] = max(
                maximum_reflux_residual[index],
                float(ledger.residual.abs().max()),
            )
            maximum_reflux_limited_directions[index] = max(
                maximum_reflux_limited_directions[index],
                ledger.limited_directions,
            )
            maximum_reflux_applied_correction_fraction[index] = max(
                maximum_reflux_applied_correction_fraction[index],
                ledger.maximum_applied_correction_fraction,
            )
            maximum_transfer_limited_fraction[index] = max(
                maximum_transfer_limited_fraction[index],
                ledger.restriction_limited_fraction,
                ledger.prolongation_limited_fraction,
            )
            minimum_transfer_alpha[index] = min(
                minimum_transfer_alpha[index],
                ledger.restriction_minimum_alpha,
                ledger.prolongation_minimum_alpha,
            )
            maximum_raw_mass_mismatch[index] = max(
                maximum_raw_mass_mismatch[index],
                raw_mismatch_moments[index][0],
            )
            maximum_raw_momentum_mismatch[index] = max(
                maximum_raw_momentum_mismatch[index],
                raw_mismatch_moments[index][1],
            )
        bfl_mean = force_averager.mean(
            (item["bfl"] for item in force_samples), observable="BFL force",
        )
        pressure_mean = force_averager.mean(
            (item["pressure"] for item in force_samples),
            observable="BFL pressure force",
        )
        friction_mean = force_averager.mean(
            (item["friction"] for item in force_samples),
            observable="wall shear force",
        )
        source_mean = force_averager.mean(
            (item["source"] for item in force_samples),
            observable="numerical momentum source",
        )
        auxiliary_means = (
            {
                margin: force_averager.mean(
                    (
                        item["auxiliary"][margin]
                        for item in force_samples
                    ),
                    observable=f"auxiliary CV force margin {margin}",
                )
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
        y_plus_summaries = [
            item["y_plus_summary"] for item in force_samples
            if item["y_plus_summary"] is not None
        ]
        y_plus_aggregate = (
            aggregate_wall_exchange_yplus_summaries(
                y_plus_summaries,
            ).to_dict()
            if y_plus_summaries else None
        )
        pressure_gradient_samples = [
            item for item in force_samples
            if item["pressure_gradient_parameter_mean"] is not None
        ]
        pressure_gradient_summaries = [
            item["pressure_gradient_summary"] for item in force_samples
            if item["pressure_gradient_summary"] is not None
        ]
        pressure_gradient_aggregate = (
            aggregate_wall_pressure_gradient_summaries(
                pressure_gradient_summaries,
            ).to_dict()
            if pressure_gradient_summaries else None
        )
        wall_shear_profiles = [
            item["wall_shear_axial_profile"] for item in force_samples
            if item["wall_shear_axial_profile"] is not None
        ]
        link_force_samples = [
            item["link_force_decomposition"] for item in force_samples
            if item["link_force_decomposition"] is not None
        ]
        link_force_aggregate = None
        if link_force_samples:
            link_force_aggregate = {
                "scope": "diagnostic_population_impulse_not_pressure_shear",
                "force_frame": link_force_samples[0]["force_frame"],
                "samples": len(link_force_samples),
                "minimum_active_links": min(
                    item["active_links"] for item in link_force_samples
                ),
                "maximum_active_links": max(
                    item["active_links"] for item in link_force_samples
                ),
                "minimum_decomposed_links": min(
                    item["decomposed_links"] for item in link_force_samples
                ),
                "maximum_undecomposed_links": max(
                    item["undecomposed_links"] for item in link_force_samples
                ),
                "minimum_coverage_fraction": min(
                    item["coverage_fraction"] for item in link_force_samples
                ),
                "normal_completion": {
                    "scheme": link_force_samples[0]["normal_completion"][
                        "scheme"
                    ],
                    "maximum_fallback_nodes": max(
                        item["normal_completion"]["fallback_nodes"]
                        for item in link_force_samples
                    ),
                    "maximum_fallback_links": max(
                        item["normal_completion"]["fallback_links"]
                        for item in link_force_samples
                    ),
                    "maximum_unresolved_nodes": max(
                        item["normal_completion"]["unresolved_nodes"]
                        for item in link_force_samples
                    ),
                },
                "mean_total_force_n": [
                    sum(item["total_force"][axis] for item in link_force_samples)
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "mean_geometry_normal_force_n": [
                    sum(item["normal_force"][axis] for item in link_force_samples)
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "mean_geometry_tangential_force_n": [
                    sum(
                        item["tangential_force"][axis]
                        for item in link_force_samples
                    )
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "mean_unresolved_force_n": [
                    sum(
                        item["unresolved_force"][axis]
                        for item in link_force_samples
                    )
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "mean_stationary_interpolation_force_n": [
                    sum(
                        item["stationary_interpolation_force"][axis]
                        for item in link_force_samples
                    )
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "mean_moving_wall_population_correction_force_n": [
                    sum(
                        item["moving_wall_population_correction_force"][axis]
                        for item in link_force_samples
                    )
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "mean_frame_correction_force_n": [
                    sum(
                        item["frame_correction_force"][axis]
                        for item in link_force_samples
                    )
                    / len(link_force_samples)
                    * scale
                    for axis in range(3)
                ],
                "maximum_closure_error_n": max(
                    item["maximum_closure_error"] for item in link_force_samples
                ) * scale,
                "maximum_relative_closure_error": max(
                    item["maximum_relative_closure_error"]
                    for item in link_force_samples
                ),
                "maximum_relative_component_closure_error": max(
                    item["maximum_relative_component_closure_error"]
                    for item in link_force_samples
                ),
            }
        record = {
            "step": current_step,
            "collision_resolved_reynolds": instantaneous_reynolds,
            "cv_resistance_n": cv_mean * scale,
            "bfl_plus_wall_stress_n": bfl_mean * scale,
            "bfl_pressure_n": pressure_mean * scale,
            "conservative_bfl_link_impulse_n": pressure_mean * scale,
            "bfl_pressure_field_status": (
                "deprecated_alias_for_conservative_link_impulse_not_pressure"
            ),
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
            "wall_y_plus_distribution": y_plus_aggregate,
            "wall_pressure_gradient_parameter_mean": (
                sum(
                    item["pressure_gradient_parameter_mean"]
                    for item in pressure_gradient_samples
                ) / len(pressure_gradient_samples)
                if pressure_gradient_samples else None
            ),
            "wall_pressure_gradient_parameter_p95": (
                max(
                    item["pressure_gradient_parameter_p95"]
                    for item in pressure_gradient_samples
                ) if pressure_gradient_samples else None
            ),
            "wall_pressure_gradient_parameter_max": (
                max(
                    item["pressure_gradient_parameter_max"]
                    for item in pressure_gradient_samples
                ) if pressure_gradient_samples else None
            ),
            "wall_pressure_gradient_distribution": pressure_gradient_aggregate,
            "wall_shear_axial_profile": (
                wall_shear_profiles[-1] if wall_shear_profiles else None
            ),
            "bfl_link_force_decomposition": link_force_aggregate,
            "wall_fully_activated": current_step >= max(
                wall_normal_ramp_steps, wall_shear_ramp_steps,
            ),
            "surface_pressure_plus_wall_stress_n": None,
            "surface_pressure_observer_status": "rejected_diagnostic_only",
            "projected_bfl_pressure_n": None,
            "projected_bfl_pressure_plus_wall_stress_n": None,
            "projected_bfl_pressure_diagnostics": None,
            "projected_bfl_pressure_observer_status": (
                "candidate_diagnostic_only_not_an_acceptance_gate"
            ),
        }
        if (
            args.enable_projected_bfl_pressure_diagnostic
            and current_step % args.surface_force_interval == 0
        ):
            # The isothermal D3Q19 pressure is c_s^2 (rho-rho_infinity).
            # Removing the unit far-field density also avoids subtracting two
            # large directional sums; a closed projected body is invariant to
            # this constant by construction.
            projected_pressure_field = (
                hierarchy.finest_f.sum(dim=0) - 1.0
            ) / 3.0
            projected_force, projected_diagnostics = (
                integrate_bfl_projected_pressure(
                    projected_pressure_field,
                    bfl_mask,
                    bfl_q,
                    solid=finest_solid,
                    reconstruction=args.projected_bfl_pressure_reconstruction,
                )
            )
            projected_pressure_x = projected_force[0]
            record["projected_bfl_pressure_n"] = (
                projected_pressure_x * scale
            )
            record["projected_bfl_pressure_plus_wall_stress_n"] = (
                (projected_pressure_x + friction_mean) * scale
            )
            record["projected_bfl_pressure_diagnostics"] = {
                **asdict(projected_diagnostics),
                "reconstruction": args.projected_bfl_pressure_reconstruction,
                "pressure_definition": "cs2_times_rho_minus_unit_far_field",
                "force_surface": "axial_bfl_projected_faces",
            }
        if (
            args.enable_rejected_surface_pressure_diagnostic
            and current_step % args.surface_force_interval == 0
        ):
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
                f"Rlink={step_records[-1]['bfl_pressure_n']:.3f} N "
                f"Rf={step_records[-1]['wall_shear_n']:.3f} N "
                f"closure={step_records[-1]['source_corrected_observer_difference_pct']:.5f}%",
                flush=True,
            )
        if (
            args.checkpoint is not None
            and args.checkpoint_interval > 0
            and current_step % args.checkpoint_interval == 0
        ):
            save_checkpoint(current_step)

    wall_activated_records = [
        record for record in step_records if record["wall_fully_activated"]
    ]
    all_target_reynolds_records = [
        record for record in step_records
        if math.isclose(
            record["collision_resolved_reynolds"],
            args.resolved_reynolds,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ]
    target_reynolds_records = [
        record for record in wall_activated_records
        if math.isclose(
            record["collision_resolved_reynolds"],
            args.resolved_reynolds,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ]
    maximum_corrected_difference = (
        max(
            record["source_corrected_observer_difference_pct"]
            for record in target_reynolds_records
        )
        if target_reynolds_records else None
    )
    conservative_force_observer_acceptable = (
        maximum_corrected_difference is not None
        and maximum_corrected_difference <= 0.1
    )
    finite = all(
        bool(torch.isfinite(level).all()) for level in hierarchy.level_populations
    )
    open_boundary_history = [
        record["open_boundary_population_delta"]
        for record in health_records
        if record.get("open_boundary_population_delta") is not None
    ]
    open_boundary_audit = audit_open_boundary_history(
        open_boundary_history,
        reference_mass=float(math.prod(shape)),
        reference_momentum=float(math.prod(shape)) * args.lattice_speed,
    )
    geometry_resolution = assess_suboff_geometry_resolution(
        finest_solid,
        hull_type=args.hull_type,
        fine_hull_length_cells=finest_length,
        center_yz=(finest_center[1], finest_center[2]),
        bare_hull=bare_solid,
        with_sail=with_sail_solid,
        appendage_halfway_links=appendage_boundary_links,
    )
    geometry_resolution_output = geometry_resolution.to_dict()
    if args.hull_type == "full":
        geometry_resolution_output["appendage_boundary_links"] = (
            appendage_boundary_links
        )
        geometry_resolution_output["appendage_halfway_links"] = 0
        geometry_resolution_output["appendage_link_scheme"] = (
            SUBOFF_APPENDAGE_LINK_SCHEME
        )
    maximum_observed_speed = (
        max(
            float(level["maximum_speed"])
            for record in health_records
            for level in record["levels"]
            if level["maximum_speed"] is not None
        )
        if health_records else None
    )
    minimum_observed_population = (
        min(
            float(level["minimum_population"])
            for record in health_records
            for level in record["levels"]
        )
        if health_records else None
    )
    minimum_observed_density = (
        min(
            float(level["minimum_density"])
            for record in health_records
            for level in record["levels"]
            if level["minimum_density"] is not None
        )
        if health_records else None
    )
    maximum_observed_density = (
        max(
            float(level["maximum_density"])
            for record in health_records
            for level in record["levels"]
            if level["maximum_density"] is not None
        )
        if health_records else None
    )
    population_health_acceptable = (
        maximum_observed_speed is not None
        and maximum_observed_speed <= args.maximum_health_speed
        and minimum_observed_population is not None
        and minimum_observed_population >= args.minimum_health_population
        and minimum_observed_density is not None
        and minimum_observed_density > 0.0
        and maximum_observed_density is not None
        and math.isfinite(maximum_observed_density)
    )
    admitted = (
        finite
        and bool(target_reynolds_records)
        and conservative_force_observer_acceptable
        and max(maximum_reflux_residual) <= 1.0e-6
        and max(maximum_reflux_limited_directions) == 0
        and max(maximum_reflux_applied_correction_fraction)
        <= args.maximum_reflux_applied_correction_fraction
        and max(maximum_transfer_limited_fraction) <= 1.0e-3
        and maximum_limiter_fraction <= args.maximum_positivity_limited_fraction
        and maximum_rejected_fraction <= 0.01
        and (args.health_interval == 0 or population_health_acceptable)
    )
    post_warmup_records = [
        record for record in target_reynolds_records
        if record["step"] > args.warmup_steps
    ]
    statistics_window_steps = (
        args.statistics_window_steps or len(post_warmup_records)
    )
    selected_records = post_warmup_records[-statistics_window_steps:]
    total_convective_times = (
        args.steps * args.lattice_speed / args.hull_length
    )
    target_reynolds_convective_times = (
        len(all_target_reynolds_records)
        * args.lattice_speed
        / args.hull_length
    )
    fully_physical_convective_times = (
        len(target_reynolds_records)
        * args.lattice_speed
        / args.hull_length
    )
    sampling_convective_times = (
        len(selected_records) * args.lattice_speed / args.hull_length
    )
    duration_acceptable = (
        total_convective_times >= args.minimum_convective_times
        and target_reynolds_convective_times
        >= args.minimum_target_reynolds_convective_times
        and sampling_convective_times
        >= args.minimum_statistics_convective_times
    )
    force_stationarity = None
    mean_resistance = None
    mean_bfl = None
    mean_bfl_pressure = None
    mean_wall_shear = None
    pressure_fraction = None
    wall_shear_fraction = None
    resistance_component_audit = None
    mean_source = None
    reference_error_pct = None
    wall_records = [
        record for record in selected_records
        if record["mean_y_plus"] is not None
    ]
    wall_y_plus_distributions = [
        record["wall_y_plus_distribution"] for record in wall_records
        if record.get("wall_y_plus_distribution") is not None
    ]
    wall_y_plus_distribution = (
        aggregate_wall_exchange_yplus_summaries(
            wall_y_plus_distributions,
        ).to_dict()
        if wall_y_plus_distributions else None
    )
    wall_y_plus_applicability_acceptable = (
        wall_y_plus_distribution is not None
        and bool(wall_y_plus_distribution["admitted"])
    )
    wall_pressure_gradient_records = [
        record for record in wall_records
        if record.get("wall_pressure_gradient_parameter_mean") is not None
    ]
    wall_pressure_gradient_distributions = [
        record["wall_pressure_gradient_distribution"]
        for record in wall_pressure_gradient_records
        if record.get("wall_pressure_gradient_distribution") is not None
    ]
    wall_pressure_gradient_distribution = (
        aggregate_wall_pressure_gradient_summaries(
            wall_pressure_gradient_distributions,
        ).to_dict()
        if wall_pressure_gradient_distributions else None
    )
    if selected_records:
        cv_values = [record["cv_resistance_n"] for record in selected_records]
        mean_resistance = sum(cv_values) / len(cv_values)
        mean_bfl = sum(
            record["bfl_plus_wall_stress_n"] for record in selected_records
        ) / len(selected_records)
        mean_bfl_pressure = sum(
            record["bfl_pressure_n"] for record in selected_records
        ) / len(selected_records)
        mean_wall_shear = sum(
            record["wall_shear_n"] for record in selected_records
        ) / len(selected_records)
        pressure_fraction = mean_bfl_pressure / max(abs(mean_bfl), 1.0e-30)
        wall_shear_fraction = mean_wall_shear / max(abs(mean_bfl), 1.0e-30)
        friction_reference = None
        if args.hull_type == "bare_hull":
            wetted_area_m2 = (
                float(finest_geometry["wetted_area_lu2"])
                * (MODEL_LENGTH_M / finest_length) ** 2
            )
            friction_reference = (
                0.5
                * args.rho_water
                * point.speed_mps**2
                * wetted_area_m2
                * ittc57_friction_coefficient(physical_re)
            )
        resistance_component_audit = audit_resistance_components(
            total_resistance=mean_resistance,
            pressure_resistance=mean_bfl_pressure,
            wall_shear_resistance=mean_wall_shear,
            experimental_total=point.resistance_n,
            friction_reference=friction_reference,
        ).to_dict()
        resistance_component_audit["pressure_input_status"] = (
            "deprecated_link_impulse_alias_not_physical_pressure"
        )
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
    collision_viscosity_acceptable = (
        args.collision_model in {
            "cumulant_smagorinsky", "cumulant_wale", "cumulant_vreman",
        }
    )
    wall_exchange_ratio = args.stress_exchange_distance / finest_length
    wall_exchange_scaling_acceptable = math.isclose(
        wall_exchange_ratio,
        args.wall_exchange_distance_over_length_target,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    auxiliary_cv_difference_pct = None
    nested_cv_acceptable = False
    surface_observer_difference_pct = None
    surface_observer_acceptable = False
    projected_bfl_observer_difference_pct = None
    projected_bfl_mean_pressure_n = None
    projected_bfl_mean_total_n = None
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
        projected_records = [
            record for record in selected_records
            if record["projected_bfl_pressure_plus_wall_stress_n"] is not None
        ]
        if projected_records:
            projected_bfl_mean_pressure_n = sum(
                record["projected_bfl_pressure_n"]
                for record in projected_records
            ) / len(projected_records)
            projected_bfl_mean_total_n = sum(
                record["projected_bfl_pressure_plus_wall_stress_n"]
                for record in projected_records
            ) / len(projected_records)
            paired_corrected_cv_mean = sum(
                record["source_corrected_cv_n"] for record in projected_records
            ) / len(projected_records)
            projected_bfl_observer_difference_pct = (
                abs(projected_bfl_mean_total_n - paired_corrected_cv_mean)
                / max(abs(paired_corrected_cv_mean), 1.0e-30)
                * 100.0
            )
    single_grid_candidate = (
        admitted
        and duration_acceptable
        and stationarity_acceptable
        and nested_cv_acceptable
        and reference_error_pct is not None
        and reference_error_pct <= 5.0
        and geometry_resolution.absolute_reference_resolved
        and not args.disable_wall_stress
        and population_health_acceptable
        and collision_viscosity_acceptable
        and wall_exchange_scaling_acceptable
        and wall_y_plus_applicability_acceptable
    )
    peak_gib_by_device = {
        str(level_device): torch.cuda.max_memory_allocated(level_device) / 2**30
        for level_device in dict.fromkeys(level_devices)
        if level_device.type == "cuda"
    }
    peak_gib = (
        next(iter(peak_gib_by_device.values()))
        if len(peak_gib_by_device) == 1 else None
    )
    invocation_elapsed_seconds = time.perf_counter() - invocation_started
    root_steps_advanced = args.steps - start_step
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
                *continuation.tau_by_level(
                    args.steps,
                    lattice_speed=args.lattice_speed,
                    root_hull_length=args.hull_length,
                    levels=level_count,
                ),
            ],
            "population_storage_dtype": args.population_storage_dtype,
            "d3q19_weight_precision_scheme": WEIGHT_PRECISION_SCHEME,
            "initial_tau_by_level": list(initial_tau_by_level),
            "resolved_wall_normal_ramp_steps": wall_normal_ramp_steps,
            "resolved_wall_shear_ramp_steps": wall_shear_ramp_steps,
            "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_interval": args.checkpoint_interval,
            "gradient_sgs_solid_velocity": [0.0, 0.0, 0.0],
            "force_samples_per_root_step": force_averager.expected_samples,
            "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
            "link_force_decomposition_scheme": (
                "actual_population_impulse_geometry_projection_v1"
            ),
            "conservative_force_observer_scheme": (
                "fixed_control_volume_plus_laboratory_bfl_link_impulse_v1"
            ),
            "appendage_link_scheme": (
                SUBOFF_APPENDAGE_LINK_SCHEME
                if args.hull_type == "full"
                else "analytic_axisymmetric_bisection_v1"
            ),
            "gradient_sgs_uses_finest_solid_mask": (
                args.collision_model in {"cumulant_wale", "cumulant_vreman"}
            ),
            "stress_exchange_distance_over_finest_length": (
                wall_exchange_ratio
            ),
            "resumed_from_step": resumed_from_step,
            "resumed_legacy_v2_checkpoint": resumed_legacy_v2_checkpoint,
            "resumed_legacy_v3_checkpoint": resumed_legacy_v3_checkpoint,
            "resumed_pre_gradient_sgs_checkpoint": (
                resumed_pre_gradient_sgs_checkpoint
            ),
            "resumed_pre_inlet_sponge_checkpoint": (
                resumed_pre_inlet_sponge_checkpoint
            ),
            "resumed_pre_collision_chunk_checkpoint": (
                resumed_pre_collision_chunk_checkpoint
            ),
            "resumed_pre_y_plus_distribution_checkpoint": (
                resumed_pre_y_plus_distribution_checkpoint
            ),
        },
        "planning": planning | {
            "measured_peak_allocated_gib": peak_gib,
            "measured_peak_allocated_gib_by_device": peak_gib_by_device,
        },
        "runtime": {
            "invocation_elapsed_seconds": invocation_elapsed_seconds,
            "root_steps_advanced": root_steps_advanced,
            "seconds_per_root_step": (
                invocation_elapsed_seconds / root_steps_advanced
            ),
        },
        "geometry": finest_geometry | {
            "resolution": geometry_resolution_output,
            "area_weighting": vars(area_diagnostics),
            "appendage_boundary_links": appendage_boundary_links,
            "appendage_halfway_links": 0,
            "appendage_link_intersection": (
                appendage_link_diagnostics.to_dict()
                if appendage_link_diagnostics is not None else None
            ),
            "geometry_owner_level": refinement_depth,
            "force_owner_level": refinement_depth,
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
            "maximum_reflux_applied_correction_fraction_by_interface": (
                maximum_reflux_applied_correction_fraction
            ),
            "maximum_transfer_limited_fraction_by_interface": (
                maximum_transfer_limited_fraction
            ),
            "minimum_transfer_alpha_by_interface": minimum_transfer_alpha,
            "maximum_raw_mass_mismatch_by_interface": (
                maximum_raw_mass_mismatch
            ),
            "maximum_raw_momentum_mismatch_by_interface": (
                maximum_raw_momentum_mismatch
            ),
            "maximum_positivity_limited_fraction": maximum_limiter_fraction,
            "maximum_wall_sample_rejected_fraction": maximum_rejected_fraction,
            "force_sample_aggregation": force_averager.provenance(
                force_averager.expected_samples,
            ),
            "collision_execution": natural_kbc_executor.diagnostics(),
            "finite": finite,
            "population_health": health_records,
            "open_boundary_population_delta_audit": (
                open_boundary_audit.to_dict()
            ),
            "maximum_observed_speed": maximum_observed_speed,
            "minimum_observed_population": minimum_observed_population,
            "minimum_observed_density": minimum_observed_density,
            "maximum_observed_density": maximum_observed_density,
            "statistics": {
                "warmup_steps": args.warmup_steps,
                "wall_activated_steps_available": len(wall_activated_records),
                "target_reynolds_steps_available": len(target_reynolds_records),
                "statistics_window_steps_requested": args.statistics_window_steps,
                "statistics_window_steps_resolved": len(selected_records),
                "total_convective_times": total_convective_times,
                "target_reynolds_convective_times": (
                    target_reynolds_convective_times
                ),
                "fully_physical_convective_times": (
                    fully_physical_convective_times
                ),
                "sampling_convective_times": sampling_convective_times,
                "mean_resistance_n": mean_resistance,
                "mean_bfl_plus_wall_stress_n": mean_bfl,
                "mean_bfl_pressure_n": mean_bfl_pressure,
                "mean_conservative_bfl_link_impulse_n": mean_bfl_pressure,
                "mean_bfl_pressure_field_status": (
                    "deprecated_alias_for_conservative_link_impulse_not_pressure"
                ),
                "mean_wall_shear_n": mean_wall_shear,
                "bfl_pressure_fraction": pressure_fraction,
                "conservative_bfl_link_impulse_fraction": pressure_fraction,
                "wall_shear_fraction": wall_shear_fraction,
                "resistance_component_audit": resistance_component_audit,
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
                "surface_pressure_observer_scope": (
                    "enabled_rejected_diagnostic_only_not_an_acceptance_gate"
                    if args.enable_rejected_surface_pressure_diagnostic
                    else "disabled_rejected_diagnostic_not_an_acceptance_gate"
                ),
                "projected_bfl_pressure_observer": {
                    "scope": (
                        "candidate_diagnostic_only_not_an_acceptance_gate"
                    ),
                    "enabled": args.enable_projected_bfl_pressure_diagnostic,
                    "reconstruction": (
                        args.projected_bfl_pressure_reconstruction
                    ),
                    "mean_pressure_n": projected_bfl_mean_pressure_n,
                    "mean_pressure_plus_wall_stress_n": (
                        projected_bfl_mean_total_n
                    ),
                    "source_corrected_cv_difference_pct": (
                        projected_bfl_observer_difference_pct
                    ),
                },
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
                    "y_plus_distribution": wall_y_plus_distribution,
                    "pressure_gradient_parameter": {
                        "samples": len(wall_pressure_gradient_records),
                        "mean": (
                            sum(
                                record["wall_pressure_gradient_parameter_mean"]
                                for record in wall_pressure_gradient_records
                            ) / len(wall_pressure_gradient_records)
                            if wall_pressure_gradient_records else None
                        ),
                        "maximum_p95": (
                            max(
                                record["wall_pressure_gradient_parameter_p95"]
                                for record in wall_pressure_gradient_records
                            ) if wall_pressure_gradient_records else None
                        ),
                        "maximum": (
                            max(
                                record["wall_pressure_gradient_parameter_max"]
                                for record in wall_pressure_gradient_records
                            ) if wall_pressure_gradient_records else None
                        ),
                        "distribution": wall_pressure_gradient_distribution,
                        "scope": "diagnostic_only_not_a_force_correction",
                    },
                },
            },
        },
        "acceptance": {
            "integration_smoke_admitted": admitted,
            "fully_activated_steps_assessed": len(wall_activated_records),
            "target_reynolds_steps_assessed": len(target_reynolds_records),
            "target_reynolds_reached": bool(target_reynolds_records),
            "duration_target_met": duration_acceptable,
            "target_reynolds_duration_target_met": (
                target_reynolds_convective_times
                >= args.minimum_target_reynolds_convective_times
            ),
            "stationarity_target_met": stationarity_acceptable,
            "nested_control_volume_target_met": nested_cv_acceptable,
            "surface_observer_target_met": surface_observer_acceptable,
            "surface_observer_used_for_acceptance": False,
            "conservative_force_observer_target_met": (
                conservative_force_observer_acceptable
            ),
            "reference_error_target_met": (
                reference_error_pct is not None and reference_error_pct <= 5.0
            ),
            "population_health_target_met": population_health_acceptable,
            "collision_viscosity_target_met": collision_viscosity_acceptable,
            "wall_exchange_scaling_target_met": (
                wall_exchange_scaling_acceptable
            ),
            "wall_exchange_y_plus_applicability_target_met": (
                wall_y_plus_applicability_acceptable
            ),
            "wall_exchange_distance_over_finest_length_target": (
                args.wall_exchange_distance_over_length_target
            ),
            "minimum_population_target": args.minimum_health_population,
            "positivity_limited_fraction_target": (
                args.maximum_positivity_limited_fraction
            ),
            "reflux_applied_correction_fraction_target": (
                args.maximum_reflux_applied_correction_fraction
            ),
            "reflux_applied_correction_target_met": (
                max(maximum_reflux_applied_correction_fraction)
                <= args.maximum_reflux_applied_correction_fraction
            ),
            "force_sample_aggregation_target_met": True,
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
