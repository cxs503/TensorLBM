#!/usr/bin/env python3
"""Run a DARPA SUBOFF resistance candidate with conservative static AMR.

The coarse grid carries the external domain and far-field boundaries.  One
strictly interior 2:1 block owns the complete CAD hull plus a downstream wake
region, advances twice per coarse step, and is conservatively restricted and
refluxed.  The fine block regenerates the analytical SUBOFF geometry rather
than repeating coarse voxels.

This is an engineering/validation runner, not a claim of validated drag.  A
result is admitted only after force stability plus grid/time convergence.
"""
from __future__ import annotations

import argparse
import json
import math
import time
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
    assess_nested_control_volume_invariance,
    box_control_volume,
    fluid_momentum_change,
    observe_control_volume_force,
)
from tensorlbm.cuda_memory_budget import require_cuda_memory_budget
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_3d,
)
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.population_positivity import limit_nonequilibrium_for_positivity
from tensorlbm.solver3d import stream3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_static_amr import (
    apply_suboff_appendage_halfway_links,
    assess_suboff_geometry_resolution,
    build_fine_suboff_mask,
    plan_suboff_static_amr,
)
from tensorlbm.surface_area_weights import bfl_surface_area_weights
from tensorlbm.turbulence import (
    collide_smagorinsky_mrt3d,
    collide_wale_mrt3d,
)
from tensorlbm.wall_model import (
    WALL_TRACTION_SOURCE_SCHEME,
    WallStressDiagnostics,
    bfl_wall_function_3d,
    physical_wall_lattice_viscosity,
)


def _sponge(
    shape: tuple[int, int, int], width: int, strength: float,
    device: torch.device,
) -> torch.Tensor:
    nz, ny, nx = shape
    if width <= 0 or strength <= 0.0:
        return torch.zeros((1, nz, ny, nx), device=device)
    z = torch.arange(nz, device=device, dtype=torch.float32)
    y = torch.arange(ny, device=device, dtype=torch.float32)
    x = torch.arange(nx, device=device, dtype=torch.float32)
    wz, wy, wx = min(width, nz // 4), min(width, ny // 4), min(width, nx // 4)

    def edge_weight(axis: torch.Tensor, size: int, span: int) -> torch.Tensor:
        edge = torch.minimum(axis, (size - 1) - axis)
        return torch.clamp((span - edge) / max(span, 1), 0.0, 1.0).square()

    field = torch.maximum(
        edge_weight(z, nz, wz).view(nz, 1, 1),
        edge_weight(y, ny, wy).view(1, ny, 1),
    )
    field = torch.maximum(field, edge_weight(x, nx, wx).view(1, 1, nx))
    return (strength * field).unsqueeze(0)


def run(args: argparse.Namespace) -> dict:
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup-steps must lie in [0, steps)")
    if args.wall_diagnostic_interval < 1:
        raise ValueError("wall-diagnostic-interval must be positive")
    if args.stress_exchange_distance < 0.0:
        raise ValueError("stress-exchange-distance must be non-negative")
    if min(args.report_interval, args.average_window) < 1:
        raise ValueError("report-interval and average-window must be positive")
    if not 0 <= args.statistics_window_steps <= args.steps - args.warmup_steps:
        raise ValueError(
            "statistics-window-steps must be zero or fit after warmup",
        )
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.surface_force_interval < 1:
        raise ValueError("surface-force-interval must be positive")
    if not 0.0 < args.maximum_reflux_correction_fraction <= 1.0:
        raise ValueError(
            "maximum-reflux-correction-fraction must lie in (0,1]",
        )
    if args.resume and not args.checkpoint:
        raise ValueError("resume requires --checkpoint")
    if min(
        args.error_target,
        args.drift_target,
        args.force_observer_target,
        args.nested_cv_target,
        args.surface_observer_target,
        args.numerical_source_target,
        args.minimum_convective_times,
        args.minimum_sampling_convective_times,
    ) < 0.0:
        raise ValueError("acceptance targets must be non-negative")
    aux_cv_margins = tuple(sorted({
        int(value) for value in args.aux_cv_margins.split(",") if value.strip()
    }))
    if any(margin < 1 for margin in aux_cv_margins):
        raise ValueError("aux-cv-margins must contain positive integers")
    if len({args.cv_margin, *aux_cv_margins}) < 3:
        raise ValueError(
            "primary plus auxiliary control volumes must provide three distinct margins",
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    point = experimental_point(args.hull_type, args.speed_knots)
    shape = (args.nz, args.ny, args.nx)
    center = (args.nx * args.center_x_fraction, args.ny / 2.0, args.nz / 2.0)
    config = SuboffConfig()
    coarse_solid, _ = build_suboff_mask(
        args.hull_type, args.nx, args.ny, args.nz,
        cx=center[0], cy=center[1], cz=center[2], length=args.hull_length,
        config=config, device=device,
    )
    plan = plan_suboff_static_amr(
        coarse_solid,
        coarse_hull_length=args.hull_length,
        wall_margin=args.wall_margin,
        wake_cells=args.wake_cells,
    )
    memory_budget = require_cuda_memory_budget(
        device, estimated_peak_gib=plan.estimated_peak_gib(),
        reserve_gib=1.0, label="SUBOFF static-AMR run",
    )
    fine_solid, fine_geometry = build_fine_suboff_mask(
        plan, hull_type=args.hull_type, coarse_center=center,
        config=config, device=device,
    )

    physical_re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    collision_re = args.resolved_reynolds or physical_re
    nu_coarse = args.lattice_speed * args.hull_length / collision_re
    wall_nu_fine = physical_wall_lattice_viscosity(
        args.lattice_speed, args.hull_length * 2.0, physical_re,
    )
    tau_coarse = 0.5 + 3.0 * nu_coarse
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)
    amr = StaticBlockAMR3D(
        coarse_f,
        StaticBlockAMRConfig(
            plan.box, tau_coarse=tau_coarse,
            reflux=not args.disable_reflux,
            maximum_reflux_correction_fraction=(
                args.maximum_reflux_correction_fraction
            ),
        ),
        fine_solid=fine_solid,
    )
    fine_solid_g = amr.fine_solid_with_ghost
    assert fine_solid_g is not None

    g = amr.config.ghost
    nz_f, ny_f, nx_f = fine_solid_g.shape
    fine_center = (
        center[0] * 2.0 - plan.box.x0 * 2.0 + g,
        center[1] * 2.0 - plan.box.y0 * 2.0 + g,
        center[2] * 2.0 - plan.box.z0 * 2.0 + g,
    )
    print("building fine-grid BFL link distances", flush=True)
    bfl_mask, bfl_q = compute_q_suboff(
        nx_f, ny_f, nz_f, *fine_center, args.hull_length * 2.0,
        hull_type=args.hull_type, config=config, device=device,
        solid_mask=fine_solid_g,
    )
    appendage_links = 0
    if args.hull_type == "full":
        appendage_links = apply_suboff_appendage_halfway_links(
            fine_solid_g,
            bfl_mask,
            bfl_q,
            center=fine_center,
            length=args.hull_length * 2.0,
            config=config,
        )
    fine_near = get_near_wall_3d(fine_solid_g)
    bare_solid = None
    with_sail_solid = None
    if args.hull_type == "bare_hull":
        fine_surface = SurfaceMesh.from_suboff(
            fine_solid_g, fine_near, *fine_center,
            args.hull_length * 2.0, args.hull_length * 2.0 / (2.0 * 8.57),
            config=config,
        )
        fine_area_weight, surface_area_diagnostics = bfl_surface_area_weights(
            bfl_mask,
            (fine_surface.nx_n, fine_surface.ny_n, fine_surface.nz_n),
            reference_area=float(fine_geometry["wetted_area_lu2"]),
            boundary_mask=fine_near,
        )
    else:
        fine_surface = SurfaceMesh.from_gradient(fine_solid_g, fine_near)
        bare_solid, _ = build_suboff_mask(
            "bare_hull", nx_f, ny_f, nz_f,
            cx=fine_center[0], cy=fine_center[1], cz=fine_center[2],
            length=args.hull_length * 2.0, config=config, device=device,
        )
        with_sail_solid, _ = build_suboff_mask(
            "with_sail", nx_f, ny_f, nz_f,
            cx=fine_center[0], cy=fine_center[1], cz=fine_center[2],
            length=args.hull_length * 2.0, config=config, device=device,
        )
        bare_near = get_near_wall_3d(bare_solid)
        bare_surface = SurfaceMesh.from_gradient(bare_solid, bare_near)
        bare_bfl_mask, _ = compute_q_suboff(
            nx_f, ny_f, nz_f, *fine_center, args.hull_length * 2.0,
            hull_type="bare_hull", config=config, device=device,
        )
        _, bare_area_diagnostics = bfl_surface_area_weights(
            bare_bfl_mask,
            (bare_surface.nx_n, bare_surface.ny_n, bare_surface.nz_n),
            reference_area=float(fine_geometry["wetted_area_lu2"]),
            boundary_mask=bare_near,
        )
        fine_area_weight, surface_area_diagnostics = bfl_surface_area_weights(
            bfl_mask,
            (fine_surface.nx_n, fine_surface.ny_n, fine_surface.nz_n),
            calibration_factor=bare_area_diagnostics.calibration_factor,
            boundary_mask=fine_near,
        )
    geometry_resolution = assess_suboff_geometry_resolution(
        fine_solid_g,
        hull_type=args.hull_type,
        fine_hull_length_cells=args.hull_length * 2.0,
        center_yz=(fine_center[1], fine_center[2]),
        bare_hull=bare_solid,
        with_sail=with_sail_solid,
        appendage_halfway_links=appendage_links,
    )
    fine_surface.dA = fine_area_weight
    fine_solid_q = fine_solid_g.unsqueeze(0).expand_as(amr.fine_f)
    fine_indices = fine_solid_g.nonzero(as_tuple=False)
    z_min, y_min, x_min = (
        int(fine_indices[:, axis].min().item()) for axis in range(3)
    )
    z_max, y_max, x_max = (
        int(fine_indices[:, axis].max().item()) + 1 for axis in range(3)
    )

    def build_owned_control_volume(margin: int) -> torch.Tensor:
        bounds = (
            x_min - margin, x_max + margin, nx_f,
            y_min - margin, y_max + margin, ny_f,
            z_min - margin, z_max + margin, nz_f,
        )
        for lower, upper, size in zip(
            bounds[0::3], bounds[1::3], bounds[2::3], strict=True,
        ):
            if lower <= g or upper >= size - g:
                raise ValueError(
                    f"control-volume margin {margin} reaches the AMR ghost/interface layer",
                )
        return box_control_volume(
            fine_solid_g.shape,
            x0=x_min - margin, x1=x_max + margin,
            y0=y_min - margin, y1=y_max + margin,
            z0=z_min - margin, z1=z_max + margin,
            device=device,
        )

    fine_cv = build_owned_control_volume(args.cv_margin)
    auxiliary_cvs: dict[int, torch.Tensor] = {}
    for margin in aux_cv_margins:
        if margin == args.cv_margin:
            continue
        auxiliary_cvs[margin] = build_owned_control_volume(margin)

    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    if args.sponge_inlet:
        sponge_faces = ("x-",) + sponge_faces
    sponge = build_sponge_sigma_3d(
        shape, width=args.sponge_width, max_strength=args.sponge_strength,
        device=device, faces=sponge_faces,
    )
    force_samples: list[tuple[float, float, float]] = []
    paired_primary_cv_samples: list[tuple[int, float]] = []
    auxiliary_cv_samples: dict[int, list[tuple[int, float]]] = {
        margin: [] for margin in auxiliary_cvs
    }
    surface_pressure_samples: list[tuple[int, float]] = []
    surface_total_samples: list[tuple[int, float]] = []
    paired_bfl_total_samples: list[tuple[int, float]] = []
    numerical_momentum_source_samples: list[tuple[int, float]] = []
    corrected_cv_samples: list[tuple[int, float]] = []
    wall_diagnostic_samples: list[WallStressDiagnostics] = []
    positivity_fractions: list[float] = []
    current_step = 0
    history: list[dict] = []
    recent_forces: list[float] = []
    recent_bfl_pressure: list[float] = []
    recent_wall_shear: list[float] = []
    force_history: list[float] = []
    bfl_total_history: list[float] = []
    pressure_history: list[float] = []
    wall_shear_history: list[float] = []
    wall_y_plus_min_history: list[float] = []
    wall_y_plus_mean_history: list[float] = []
    wall_y_plus_max_history: list[float] = []
    wall_rejected_fraction_history: list[float] = []
    maximum_positivity_limited_fraction = 0.0
    maximum_reflux_population_residual = 0.0
    maximum_reflux_requested_correction = 0.0
    maximum_reflux_applied_correction = 0.0
    maximum_reflux_limited_directions = 0
    maximum_raw_kinetic_mismatch = 0.0

    def advance(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        def collide(state: torch.Tensor) -> torch.Tensor:
            if args.collision_model == "cumulant_smagorinsky":
                result = collide_cumulant_d3q19(
                    state, tau=tau, C_s=args.cs_smag,
                )
            elif args.les_model == "wale":
                result = collide_wale_mrt3d(state, tau, C_w=args.cw_wale)
            else:
                result = collide_smagorinsky_mrt3d(state, tau, C_s=args.cs_smag)
            if not args.disable_positivity_limiter:
                result, diagnostic = limit_nonequilibrium_for_positivity(result)
                positivity_fractions.append(diagnostic.limited_fraction)
            return result

        if level == 0:
            post_collision = collide(f)
            out = stream3d(post_collision)
            if args.far_field_mode == "non_equilibrium_extrapolation":
                out = non_equilibrium_far_field_bc_3d(
                    out, u_in=args.lattice_speed,
                )
            else:
                out = far_field_bc_3d(out, u_in=args.lattice_speed)
            if args.sponge_width > 0 and args.sponge_strength > 0.0:
                out = apply_equilibrium_difference_sponge(
                    out, sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
            if args.far_field_mode == "non_equilibrium_extrapolation":
                out = non_equilibrium_far_field_bc_3d(
                    out, u_in=args.lattice_speed,
                )
            else:
                out = far_field_bc_3d(out, u_in=args.lattice_speed)
            return AMRAdvanceResult(out, post_collision)

        before = f
        collided = collide(f)
        post_collision = torch.where(fine_solid_q, before, collided)
        out = stream3d(post_collision)
        activation = smooth_ramp_factor(current_step, args.ramp_steps)
        collect_wall_diagnostics = (
            current_step > args.warmup_steps
            and current_step % args.wall_diagnostic_interval == 0
        )
        wall_result = bfl_wall_function_3d(
            out, post_collision, fine_solid_g, wall_nu_fine,
            bfl_mask, bfl_q, y_val=args.wall_distance,
            wall_law=args.wall_law, near_mask=fine_near,
            bfl_wall_mode="wall_model_slip", wall_activation=activation,
            stress_exchange_distance=(
                args.stress_exchange_distance
                if args.stress_exchange_distance > 0.0 else None
            ),
            wall_normals=(
                fine_surface.nx_n, fine_surface.ny_n, fine_surface.nz_n,
            ),
            area_weight=fine_area_weight,
            apply_wall_stress=not args.diagnostic_uncoupled_wall_stress,
            return_wall_diagnostics=collect_wall_diagnostics,
        )
        if collect_wall_diagnostics:
            out, friction, pressure, wall_diagnostics = wall_result
            wall_diagnostic_samples.append(wall_diagnostics)
        else:
            out, friction, pressure = wall_result
        before_positivity = out
        if not args.disable_positivity_limiter:
            out, diagnostic = limit_nonequilibrium_for_positivity(out)
            positivity_fractions.append(diagnostic.limited_fraction)
        cv_force = float(observe_control_volume_force(
            before, out, post_collision, fine_cv, solid=fine_solid_g,
        ).force_on_body[0].item())
        force_samples.append((pressure, friction, cv_force))
        if (
            current_step > args.warmup_steps
            and current_step % args.surface_force_interval == 0
        ):
            collision_source = float(fluid_momentum_change(
                before, post_collision, fine_cv, solid=fine_solid_g,
            )[0].item())
            positivity_source = float(fluid_momentum_change(
                before_positivity, out, fine_cv, solid=fine_solid_g,
            )[0].item())
            numerical_source = collision_source + positivity_source
            numerical_momentum_source_samples.append(
                (current_step, numerical_source),
            )
            corrected_cv_samples.append(
                (current_step, cv_force + numerical_source),
            )
            paired_primary_cv_samples.append((current_step, cv_force))
            for margin, auxiliary_cv in auxiliary_cvs.items():
                auxiliary_force = float(observe_control_volume_force(
                    before, out, post_collision, auxiliary_cv,
                    solid=fine_solid_g,
                ).force_on_body[0].item())
                auxiliary_cv_samples[margin].append(
                    (current_step, auxiliary_force),
                )
        return AMRAdvanceResult(out, post_collision)

    dx_fine_m = MODEL_LENGTH_M / (2.0 * args.hull_length)
    scale = force_scale_newton(
        rho_water=args.rho_water, dx_m=dx_fine_m,
        speed_mps=point.speed_mps, lattice_speed=args.lattice_speed,
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_signature = {
        "schema_version": 8,
        "coarse_shape_zyx": list(shape),
        "hull_type": args.hull_type,
        "speed_knots": args.speed_knots,
        "hull_length": args.hull_length,
        "center_x_fraction": args.center_x_fraction,
        "refinement_box": vars(plan.box),
        "wall_margin": args.wall_margin,
        "wake_cells": args.wake_cells,
        "cv_margin": args.cv_margin,
        "aux_cv_margins": list(auxiliary_cvs),
        "surface_force_interval": args.surface_force_interval,
        "pressure_reference": args.pressure_reference,
        "surface_pressure_extrapolation": args.surface_pressure_extrapolation,
        "reflux_enabled": not args.disable_reflux,
        "reflux_method": "face_local_conserved_moment_flux",
        "maximum_reflux_correction_fraction": (
            args.maximum_reflux_correction_fraction
        ),
        "wall_stress_coupled": not args.diagnostic_uncoupled_wall_stress,
        "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
        "positivity_limiter_enabled": not args.disable_positivity_limiter,
        "warmup_steps": args.warmup_steps,
        "report_interval": args.report_interval,
        "average_window": args.average_window,
        "ramp_steps": args.ramp_steps,
        "lattice_speed": args.lattice_speed,
        "resolved_reynolds": args.resolved_reynolds,
        "nu_water": args.nu_water,
        "rho_water": args.rho_water,
        "cs_smag": args.cs_smag,
        "cw_wale": args.cw_wale,
        "les_model": args.les_model,
        "collision_model": args.collision_model,
        "wall_law": args.wall_law,
        "wall_distance": args.wall_distance,
        "stress_exchange_distance": args.stress_exchange_distance,
        "wall_diagnostic_interval": args.wall_diagnostic_interval,
        "sponge_width": args.sponge_width,
        "sponge_strength": args.sponge_strength,
        "sponge_inlet": args.sponge_inlet,
        "far_field_mode": args.far_field_mode,
        "boundary_treatment": "bfl_wall_model",
        "link_force_frame": "laboratory_after_wall_activation",
        "refinement_ratio": plan.ratio,
        "wall_viscosity_basis": "physical_reynolds",
    }

    if args.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        stored_configuration = state.get("configuration")
        legacy_none_signature = dict(checkpoint_signature)
        legacy_none_signature.pop("surface_pressure_extrapolation")
        resumed_legacy_none_observer = (
            args.surface_pressure_extrapolation == "none"
            and stored_configuration == legacy_none_signature
        )
        if (
            stored_configuration != checkpoint_signature
            and not resumed_legacy_none_observer
        ):
            raise ValueError("checkpoint configuration does not match static-AMR run")
        current_step = int(state["step"])
        if current_step >= args.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")
        amr.coarse_f = state["coarse_populations"].to(device=device)
        amr.fine_f = state["fine_populations"].to(device=device)
        force_history = state["force_history"].tolist()
        bfl_total_history = state["bfl_total_history"].tolist()
        pressure_history = state["pressure_history"].tolist()
        wall_shear_history = state["wall_shear_history"].tolist()
        paired_primary_cv_samples = [
            tuple(item) for item in state["paired_primary_cv_samples"].tolist()
        ]
        auxiliary_cv_samples = {
            int(margin): [tuple(item) for item in samples.tolist()]
            for margin, samples in state["auxiliary_cv_samples"].items()
        }
        surface_pressure_samples = [
            tuple(item) for item in state["surface_pressure_samples"].tolist()
        ]
        surface_total_samples = [
            tuple(item) for item in state["surface_total_samples"].tolist()
        ]
        paired_bfl_total_samples = [
            tuple(item) for item in state["paired_bfl_total_samples"].tolist()
        ]
        numerical_momentum_source_samples = [
            tuple(item)
            for item in state["numerical_momentum_source_samples"].tolist()
        ]
        corrected_cv_samples = [
            tuple(item) for item in state["corrected_cv_samples"].tolist()
        ]
        recent_forces = state["recent_forces"].tolist()
        recent_bfl_pressure = state["recent_bfl_pressure"].tolist()
        recent_wall_shear = state["recent_wall_shear"].tolist()
        wall_y_plus_min_history = state["wall_y_plus_min_history"].tolist()
        wall_y_plus_mean_history = state["wall_y_plus_mean_history"].tolist()
        wall_y_plus_max_history = state["wall_y_plus_max_history"].tolist()
        wall_rejected_fraction_history = state[
            "wall_rejected_fraction_history"
        ].tolist()
        maximum_positivity_limited_fraction = float(
            state["maximum_positivity_limited_fraction"],
        )
        maximum_reflux_population_residual = float(
            state["maximum_reflux_population_residual"],
        )
        maximum_reflux_requested_correction = float(
            state["maximum_reflux_requested_correction"],
        )
        maximum_reflux_applied_correction = float(
            state["maximum_reflux_applied_correction"],
        )
        maximum_reflux_limited_directions = int(
            state["maximum_reflux_limited_directions"],
        )
        maximum_raw_kinetic_mismatch = float(
            state["maximum_raw_kinetic_mismatch"],
        )
        history = list(state["history"])

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({
            "schema": "tensorlbm-suboff-static-amr-checkpoint-v8",
            "configuration": checkpoint_signature,
            "step": step,
            "coarse_populations": amr.coarse_f.detach().cpu(),
            "fine_populations": amr.fine_f.detach().cpu(),
            "force_history": torch.tensor(force_history, dtype=torch.float64),
            "bfl_total_history": torch.tensor(
                bfl_total_history, dtype=torch.float64,
            ),
            "pressure_history": torch.tensor(pressure_history, dtype=torch.float64),
            "wall_shear_history": torch.tensor(
                wall_shear_history, dtype=torch.float64,
            ),
            "paired_primary_cv_samples": torch.tensor(
                paired_primary_cv_samples, dtype=torch.float64,
            ).reshape(-1, 2),
            "auxiliary_cv_samples": {
                margin: torch.tensor(samples, dtype=torch.float64).reshape(-1, 2)
                for margin, samples in auxiliary_cv_samples.items()
            },
            "surface_pressure_samples": torch.tensor(
                surface_pressure_samples, dtype=torch.float64,
            ).reshape(-1, 2),
            "surface_total_samples": torch.tensor(
                surface_total_samples, dtype=torch.float64,
            ).reshape(-1, 2),
            "paired_bfl_total_samples": torch.tensor(
                paired_bfl_total_samples, dtype=torch.float64,
            ).reshape(-1, 2),
            "numerical_momentum_source_samples": torch.tensor(
                numerical_momentum_source_samples, dtype=torch.float64,
            ).reshape(-1, 2),
            "corrected_cv_samples": torch.tensor(
                corrected_cv_samples, dtype=torch.float64,
            ).reshape(-1, 2),
            "recent_forces": torch.tensor(recent_forces, dtype=torch.float64),
            "recent_bfl_pressure": torch.tensor(
                recent_bfl_pressure, dtype=torch.float64,
            ),
            "recent_wall_shear": torch.tensor(
                recent_wall_shear, dtype=torch.float64,
            ),
            "wall_y_plus_min_history": torch.tensor(
                wall_y_plus_min_history, dtype=torch.float64,
            ),
            "wall_y_plus_mean_history": torch.tensor(
                wall_y_plus_mean_history, dtype=torch.float64,
            ),
            "wall_y_plus_max_history": torch.tensor(
                wall_y_plus_max_history, dtype=torch.float64,
            ),
            "wall_rejected_fraction_history": torch.tensor(
                wall_rejected_fraction_history, dtype=torch.float64,
            ),
            "maximum_positivity_limited_fraction": (
                maximum_positivity_limited_fraction
            ),
            "maximum_reflux_population_residual": (
                maximum_reflux_population_residual
            ),
            "maximum_reflux_requested_correction": (
                maximum_reflux_requested_correction
            ),
            "maximum_reflux_applied_correction": (
                maximum_reflux_applied_correction
            ),
            "maximum_reflux_limited_directions": (
                maximum_reflux_limited_directions
            ),
            "maximum_raw_kinetic_mismatch": maximum_raw_kinetic_mismatch,
            "history": history,
        }, checkpoint)

    started = time.time()
    start_step = current_step
    for current_step in range(start_step + 1, args.steps + 1):
        force_samples.clear()
        wall_diagnostic_samples.clear()
        positivity_fractions.clear()
        ledger = amr.step(advance)
        pressure = sum(item[0] for item in force_samples) / len(force_samples)
        friction = sum(item[1] for item in force_samples) / len(force_samples)
        cv_force = sum(item[2] for item in force_samples) / len(force_samples)
        resistance = cv_force * scale
        bfl_resistance = (pressure + friction) * scale
        if (
            current_step > args.warmup_steps
            and current_step % args.surface_force_interval == 0
        ):
            surface_pressure = drag_pressure_integration(
                amr.fine_f, fine_surface, 1.0,
                extrap=args.surface_pressure_extrapolation,
                p0_method=args.pressure_reference, solid=fine_solid_g,
                fluid_boundary_mask=bfl_mask, q_field=bfl_q,
            )[0]
            surface_pressure_samples.append((current_step, surface_pressure))
            surface_total_samples.append(
                (current_step, surface_pressure + friction),
            )
            paired_bfl_total_samples.append(
                (current_step, pressure + friction),
            )
        maximum_positivity_limited_fraction = max(
            maximum_positivity_limited_fraction,
            max(positivity_fractions, default=0.0),
        )
        maximum_reflux_population_residual = max(
            maximum_reflux_population_residual,
            float(ledger.residual.abs().max().item()),
        )
        maximum_reflux_requested_correction = max(
            maximum_reflux_requested_correction,
            float(ledger.replacement_mismatch.abs().max().item()),
        )
        maximum_reflux_applied_correction = max(
            maximum_reflux_applied_correction,
            float(ledger.applied_shell_correction.abs().max().item()),
        )
        maximum_reflux_limited_directions = max(
            maximum_reflux_limited_directions, ledger.limited_directions,
        )
        if ledger.raw_kinetic_mismatch is None:
            raise RuntimeError("AMR ledger omitted raw kinetic mismatch")
        maximum_raw_kinetic_mismatch = max(
            maximum_raw_kinetic_mismatch,
            float(ledger.raw_kinetic_mismatch.abs().max().item()),
        )
        if current_step > args.warmup_steps:
            force_history.append(resistance)
            bfl_total_history.append(bfl_resistance)
            pressure_history.append(pressure * scale)
            wall_shear_history.append(friction * scale)
        if wall_diagnostic_samples:
            finite_y_min = [
                diagnostic.y_plus_min for diagnostic in wall_diagnostic_samples
                if diagnostic.y_plus_min is not None
            ]
            finite_y_mean = [
                diagnostic.y_plus_mean for diagnostic in wall_diagnostic_samples
                if diagnostic.y_plus_mean is not None
            ]
            finite_y_max = [
                diagnostic.y_plus_max for diagnostic in wall_diagnostic_samples
                if diagnostic.y_plus_max is not None
            ]
            if finite_y_mean:
                wall_y_plus_min_history.append(min(finite_y_min))
                wall_y_plus_mean_history.append(
                    sum(finite_y_mean) / len(finite_y_mean),
                )
                wall_y_plus_max_history.append(max(finite_y_max))
            wall_rejected_fraction_history.append(max(
                diagnostic.rejected_fraction
                for diagnostic in wall_diagnostic_samples
            ))
        recent_forces.append(resistance)
        recent_bfl_pressure.append(pressure * scale)
        recent_wall_shear.append(friction * scale)
        if len(recent_forces) > args.average_window:
            recent_forces.pop(0)
            recent_bfl_pressure.pop(0)
            recent_wall_shear.pop(0)
        if (
            not bool(torch.isfinite(amr.coarse_f).all())
            or not bool(torch.isfinite(amr.fine_f).all())
        ):
            raise FloatingPointError(f"non-finite population at step {current_step}")
        if current_step % args.report_interval == 0 or current_step == args.steps:
            mean_force = sum(recent_forces) / len(recent_forces)
            row = {
                "step": current_step,
                "instantaneous_resistance_n": resistance,
                "window_resistance_n": mean_force,
                "instantaneous_bfl_pressure_n": pressure * scale,
                "instantaneous_wall_shear_n": friction * scale,
                "instantaneous_bfl_link_plus_wall_stress_n": bfl_resistance,
                "instantaneous_force_observer_difference_n": (
                    bfl_resistance - resistance
                ),
                "error_pct": abs(mean_force - point.resistance_n) / point.resistance_n * 100.0,
                "reflux_mass_residual": ledger.mass_residual,
                "reflux_max_population_residual": float(
                    ledger.residual.abs().max().item()
                ),
                "reflux_max_requested_correction": float(
                    ledger.replacement_mismatch.abs().max().item()
                ),
                "reflux_max_applied_correction": float(
                    ledger.applied_shell_correction.abs().max().item()
                ),
                "reflux_limited_directions": ledger.limited_directions,
                "raw_kinetic_mismatch_max": float(
                    ledger.raw_kinetic_mismatch.abs().max().item()
                ),
                "maximum_positivity_limited_fraction": max(
                    positivity_fractions, default=0.0,
                ),
            }
            history.append(row)
            print(
                f"step={current_step}/{args.steps} Rt_win={mean_force:.3f} N "
                f"Rt_inst={resistance:.3f} N "
                f"P_bfl={pressure * scale:.3f} N "
                f"T_wall={friction * scale:.3f} N "
                f"Rt_bfl={bfl_resistance:.3f} N "
                f"closure={bfl_resistance - resistance:.3e} N "
                f"exp={point.resistance_n:.3f} N err={row['error_pct']:.2f}% "
                f"reflux={ledger.mass_residual:.3e}", flush=True,
            )
        if (
            checkpoint is not None and args.checkpoint_interval
            and current_step % args.checkpoint_interval == 0
        ):
            save_checkpoint(current_step)

    statistics_window_steps = args.statistics_window_steps or len(force_history)
    selected_force_history = force_history[-statistics_window_steps:]
    selected_bfl_total_history = bfl_total_history[-statistics_window_steps:]
    selected_pressure_history = pressure_history[-statistics_window_steps:]
    selected_wall_shear_history = wall_shear_history[-statistics_window_steps:]
    mean_force = sum(selected_force_history) / len(selected_force_history)
    mean_bfl_total = (
        sum(selected_bfl_total_history) / len(selected_bfl_total_history)
    )
    mean_pressure = sum(selected_pressure_history) / len(selected_pressure_history)
    mean_wall_shear = (
        sum(selected_wall_shear_history) / len(selected_wall_shear_history)
    )
    final_window_start = args.steps - statistics_window_steps

    def sampled_mean(samples: list[tuple[int, float]]) -> float:
        selected = [value for step, value in samples if step > final_window_start]
        return sum(selected) / len(selected) if selected else math.nan

    paired_primary_cv_mean = sampled_mean(paired_primary_cv_samples)
    auxiliary_cv_means = {
        margin: sampled_mean(samples)
        for margin, samples in auxiliary_cv_samples.items()
    }
    surface_pressure_mean = sampled_mean(surface_pressure_samples)
    surface_total_mean = sampled_mean(surface_total_samples)
    paired_bfl_total_mean = sampled_mean(paired_bfl_total_samples)
    numerical_momentum_source_mean = sampled_mean(
        numerical_momentum_source_samples,
    )
    corrected_cv_mean = sampled_mean(corrected_cv_samples)
    auxiliary_items = list(auxiliary_cv_means.items())
    nested_cv_assessment = assess_nested_control_volume_invariance(
        paired_primary_cv_mean,
        [value for _, value in auxiliary_items],
    )
    auxiliary_cv_difference_pct = {
        str(margin): difference
        for (margin, _), difference in zip(
            auxiliary_items, nested_cv_assessment.differences_pct,
            strict=True,
        )
    }
    force_stationarity = assess_force_stationarity(
        selected_force_history,
        block_size=max(1, len(selected_force_history) // 8),
    )
    reference_error_pct = (
        abs(mean_force - point.resistance_n) / point.resistance_n * 100.0
    )
    force_observer_difference_pct = (
        abs(mean_bfl_total - mean_force) / max(abs(mean_force), 1e-30) * 100.0
    )
    surface_observer_difference_pct = (
        abs(surface_total_mean - paired_primary_cv_mean)
        / max(abs(paired_primary_cv_mean), 1e-30) * 100.0
    )
    corrected_cv_observer_difference_pct = (
        abs(corrected_cv_mean - paired_bfl_total_mean)
        / max(abs(paired_bfl_total_mean), 1e-30) * 100.0
    )
    numerical_momentum_source_fraction_pct = (
        abs(numerical_momentum_source_mean)
        / max(abs(paired_bfl_total_mean), 1e-30) * 100.0
    )
    maximum_rejected_fraction = max(
        wall_rejected_fraction_history, default=0.0,
    )
    wall_sampling_acceptable = (
        bool(wall_rejected_fraction_history)
        and maximum_rejected_fraction <= 0.01
    )
    limiter_acceptable = maximum_positivity_limited_fraction <= 1e-3
    reflux_acceptable = maximum_reflux_population_residual <= 1e-6
    surface_area_acceptable = (
        surface_area_diagnostics.unweighted_nodes == 0
        and surface_area_diagnostics.calibrated_area > 0.0
        and (
            args.hull_type == "bare_hull"
            or surface_area_diagnostics.calibrated_area
            > float(fine_geometry["bare_hull_wetted_area_lu2"])
        )
    )
    geometry_convergence_member_acceptable = (
        geometry_resolution.convergence_member_resolved
    )
    absolute_reference_geometry_acceptable = (
        geometry_resolution.absolute_reference_resolved
    )
    finite = (
        bool(torch.isfinite(amr.coarse_f).all())
        and bool(torch.isfinite(amr.fine_f).all())
        and math.isfinite(mean_force)
    )
    total_convective_times = (
        args.steps * args.lattice_speed / args.hull_length
    )
    post_warmup_convective_times = (
        (args.steps - args.warmup_steps)
        * args.lattice_speed / args.hull_length
    )
    sampling_convective_times = (
        statistics_window_steps * args.lattice_speed / args.hull_length
    )
    duration_acceptable = (
        total_convective_times >= args.minimum_convective_times
        and sampling_convective_times
        >= args.minimum_sampling_convective_times
    )
    numerical_quality_admitted = (
        finite
        and duration_acceptable
        and not args.diagnostic_uncoupled_wall_stress
        and force_stationarity.meets(args.drift_target)
        and nested_cv_assessment.meets(
            args.nested_cv_target, minimum_auxiliary_count=2,
        )
        and surface_observer_difference_pct <= args.surface_observer_target
        and numerical_momentum_source_fraction_pct
        <= args.numerical_source_target
        and limiter_acceptable
        and reflux_acceptable
        and wall_sampling_acceptable
        and surface_area_acceptable
        and geometry_convergence_member_acceptable
    )
    single_grid_admitted = (
        numerical_quality_admitted
        and reference_error_pct <= args.error_target
        and absolute_reference_geometry_acceptable
    )
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    )
    rho_c, ux_c, uy_c, uz_c = macroscopic3d(amr.coarse_f)
    result = {
        "schema": "tensorlbm-suboff-static-amr-v8",
        "status": (
            "single_grid_candidate" if single_grid_admitted
            else (
                "numerically_converged_grid"
                if numerical_quality_admitted else "single_grid_rejected"
            )
        ),
        "configuration": checkpoint_signature | {
            "device": str(device),
            "coarse_hull_length_cells": args.hull_length,
            "fine_hull_length_cells": args.hull_length * 2.0,
            "fine_diameter_cells": plan.effective_diameter_cells,
            "tau_coarse": tau_coarse,
            "tau_fine": amr.config.tau_fine, "physical_reynolds": physical_re,
            "collision_reynolds": collision_re, "steps": args.steps,
            "wall_model_reynolds": physical_re,
            "wall_nu_fine": wall_nu_fine,
            "report_average_window": args.average_window,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "reflux_method": "face_local_conserved_moment_flux",
            "les_constant": (
                args.cw_wale if args.les_model == "wale" else args.cs_smag
            ),
            "stress_exchange_distance": (
                args.stress_exchange_distance
                if args.stress_exchange_distance > 0.0 else None
            ),
            "sponge_inlet_enabled": args.sponge_inlet,
            "wall_activation_ramp_steps": args.ramp_steps,
            "total_convective_times": total_convective_times,
            "post_warmup_convective_times": post_warmup_convective_times,
            "sampling_convective_times": sampling_convective_times,
            "statistics_window_steps_requested": args.statistics_window_steps,
            "statistics_window_steps_resolved": statistics_window_steps,
        },
        "mesh": {
            "coarse_cells": plan.coarse_cells,
            "fine_allocated_cells": plan.fine_allocated_cells,
            "total_allocated_cells": plan.total_allocated_cells,
            "uniform_fine_cells": plan.uniform_fine_cells,
            "saving_fraction": plan.cell_saving_fraction,
            "estimated_peak_gib": plan.estimated_peak_gib(),
            "measured_peak_allocated_gib": peak_gib,
            "cuda_memory_preflight": (
                memory_budget.to_dict() if memory_budget is not None else None
            ),
        },
        "geometry": fine_geometry | {
            "appendage_halfway_links": appendage_links,
            "geometry_resolution": geometry_resolution.to_dict(),
            "force_integration_area_scope": args.hull_type,
            "force_integration_calibrated_area_lu2": (
                surface_area_diagnostics.calibrated_area
            ),
            "surface_area_weighting": vars(surface_area_diagnostics),
        },
        "result": {
            "mean_resistance_n": mean_force,
            "mean_bfl_pressure_n_diagnostic": (
                mean_pressure
            ),
            "mean_wall_shear_n_diagnostic": (
                mean_wall_shear
            ),
            "mean_bfl_link_plus_wall_stress_n_diagnostic": mean_bfl_total,
            "force_observer_difference_pct": force_observer_difference_pct,
            "paired_primary_control_volume_resistance_n": (
                paired_primary_cv_mean * scale
            ),
            "paired_control_volume_samples_in_window": sum(
                step > final_window_start
                for step, _ in paired_primary_cv_samples
            ),
            "auxiliary_control_volume_resistance_n": {
                str(margin): value * scale
                for margin, value in auxiliary_cv_means.items()
            },
            "auxiliary_control_volume_difference_pct": (
                auxiliary_cv_difference_pct
            ),
            "nested_control_volume_invariance": {
                "auxiliary_count": nested_cv_assessment.auxiliary_count,
                "maximum_difference_pct": (
                    nested_cv_assessment.maximum_difference_pct
                ),
                "finite": nested_cv_assessment.finite,
            },
            "mean_surface_pressure_n_diagnostic": surface_pressure_mean * scale,
            "mean_surface_pressure_plus_wall_stress_n_diagnostic": (
                surface_total_mean * scale
            ),
            "paired_bfl_link_plus_wall_stress_n": paired_bfl_total_mean * scale,
            "mean_numerical_momentum_source_n_diagnostic": (
                numerical_momentum_source_mean * scale
            ),
            "numerical_momentum_source_fraction_pct": (
                numerical_momentum_source_fraction_pct
            ),
            "source_corrected_control_volume_resistance_n_diagnostic": (
                corrected_cv_mean * scale
            ),
            "source_corrected_cv_vs_bfl_difference_pct": (
                corrected_cv_observer_difference_pct
            ),
            "surface_pressure_samples_in_window": sum(
                step > final_window_start for step, _ in surface_pressure_samples
            ),
            "surface_vs_control_volume_observer_difference_pct": (
                surface_observer_difference_pct
            ),
            "experimental_resistance_n": point.resistance_n,
            "error_pct": reference_error_pct,
            "force_stationarity": force_stationarity.to_dict(),
            "maximum_positivity_limited_fraction": (
                maximum_positivity_limited_fraction
            ),
            "maximum_reflux_population_residual": (
                maximum_reflux_population_residual
            ),
            "maximum_reflux_requested_correction": (
                maximum_reflux_requested_correction
            ),
            "maximum_reflux_applied_correction": (
                maximum_reflux_applied_correction
            ),
            "maximum_reflux_limited_directions": (
                maximum_reflux_limited_directions
            ),
            "maximum_raw_kinetic_mismatch": maximum_raw_kinetic_mismatch,
            "wall_stress_applicability": {
                "samples": len(wall_y_plus_mean_history),
                "y_plus_min": min(wall_y_plus_min_history, default=None),
                "y_plus_mean": (
                    sum(wall_y_plus_mean_history) / len(wall_y_plus_mean_history)
                    if wall_y_plus_mean_history else None
                ),
                "y_plus_max": max(wall_y_plus_max_history, default=None),
                "maximum_rejected_fraction": maximum_rejected_fraction,
            },
            "coarse_density_min_max": [float(rho_c.min()), float(rho_c.max())],
            "coarse_speed_max": float(torch.sqrt(ux_c**2 + uy_c**2 + uz_c**2).max()),
            "finite": finite,
        },
        "history": history,
        "elapsed_s": time.time() - started,
        "claim_boundary": (
            "Single-grid static-AMR candidate only. Three-grid/domain "
            "convergence and paired AFF-1/AFF-8 validation remain mandatory."
        ),
        "acceptance": {
            "force_error_target_pct": args.error_target,
            "stationarity_target_pct": args.drift_target,
            "force_observer_target_pct": args.force_observer_target,
            "nested_control_volume_target_pct": args.nested_cv_target,
            "surface_observer_target_pct": args.surface_observer_target,
            "numerical_momentum_source_target_pct": (
                args.numerical_source_target
            ),
            "minimum_convective_times": args.minimum_convective_times,
            "minimum_sampling_convective_times": (
                args.minimum_sampling_convective_times
            ),
            "maximum_limiter_fraction": 1e-3,
            "maximum_reflux_population_residual": 1e-6,
            "maximum_exchange_rejected_fraction": 0.01,
            "force_target_met": reference_error_pct <= args.error_target,
            "stationarity_target_met": force_stationarity.meets(
                args.drift_target,
            ),
            "bfl_vs_control_volume_diagnostic_target_met": (
                force_observer_difference_pct <= args.force_observer_target
            ),
            "nested_control_volume_target_met": (
                nested_cv_assessment.meets(
                    args.nested_cv_target, minimum_auxiliary_count=2,
                )
            ),
            "surface_observer_target_met": (
                surface_observer_difference_pct <= args.surface_observer_target
            ),
            "numerical_momentum_source_target_met": (
                numerical_momentum_source_fraction_pct
                <= args.numerical_source_target
            ),
            "source_corrected_cv_vs_bfl_diagnostic_target_met": (
                corrected_cv_observer_difference_pct
                <= args.force_observer_target
            ),
            "duration_target_met": duration_acceptable,
            "limiter_target_met": limiter_acceptable,
            "reflux_target_met": reflux_acceptable,
            "wall_sampling_target_met": wall_sampling_acceptable,
            "surface_area_target_met": surface_area_acceptable,
            "geometry_convergence_member_target_met": (
                geometry_convergence_member_acceptable
            ),
            "absolute_reference_geometry_target_met": (
                absolute_reference_geometry_acceptable
            ),
            "wall_stress_coupled": not args.diagnostic_uncoupled_wall_stress,
            "single_grid_admitted": single_grid_admitted,
            "numerical_quality_admitted": numerical_quality_admitted,
            "physical_validation": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if checkpoint is not None:
        save_checkpoint(args.steps)
    print(f"wrote {output}; peak={peak_gib} GiB", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hull-type", choices=("bare_hull", "full"), default="bare_hull")
    p.add_argument("--speed-knots", type=float, default=5.92)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=300)
    p.add_argument("--ny", type=int, default=120)
    p.add_argument("--nz", type=int, default=120)
    p.add_argument("--hull-length", type=float, default=120.0)
    p.add_argument("--center-x-fraction", type=float, default=0.35)
    p.add_argument("--wall-margin", type=int, default=8)
    p.add_argument("--wake-cells", type=int, default=50)
    p.add_argument("--cv-margin", type=int, default=8)
    p.add_argument(
        "--aux-cv-margins", default="4,10",
        help="Comma-separated independent nested control-volume margins.",
    )
    p.add_argument("--surface-force-interval", type=int, default=25)
    p.add_argument(
        "--pressure-reference",
        choices=("near_wall", "far_field", "domain_avg", "inlet"),
        default="near_wall",
    )
    p.add_argument(
        "--surface-pressure-extrapolation",
        choices=("none", "linear", "quadratic", "bfl_quadratic"),
        default="none",
        help="wall-normal extrapolation used only by the surface-pressure observer",
    )
    p.add_argument("--disable-reflux", action="store_true")
    p.add_argument("--maximum-reflux-correction-fraction", type=float, default=0.2)
    p.add_argument("--diagnostic-uncoupled-wall-stress", action="store_true")
    p.add_argument("--disable-positivity-limiter", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=2500)
    p.add_argument("--report-interval", type=int, default=100)
    p.add_argument("--average-window", type=int, default=500)
    p.add_argument(
        "--statistics-window-steps", type=int, default=0,
        help=(
            "Explicit final force-statistics tail; zero uses all post-warmup "
            "samples."
        ),
    )
    p.add_argument("--ramp-steps", type=int, default=1000)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--resolved-reynolds", type=float, default=2.0e6)
    p.add_argument("--nu-water", type=float, default=1.004e-6)
    p.add_argument("--rho-water", type=float, default=998.2)
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument("--les-model", choices=("wale", "smagorinsky"), default="wale")
    p.add_argument(
        "--collision-model",
        choices=("mrt_les", "cumulant_smagorinsky"),
        default="mrt_les",
    )
    p.add_argument("--wall-law", choices=("log", "reichardt", "musker"), default="reichardt")
    p.add_argument("--wall-distance", type=float, default=0.5)
    p.add_argument("--stress-exchange-distance", type=float, default=0.0)
    p.add_argument("--wall-diagnostic-interval", type=int, default=50)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sponge-width", type=int, default=12)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--sponge-inlet", action="store_true")
    p.add_argument("--error-target", type=float, default=5.0)
    p.add_argument("--drift-target", type=float, default=1.0)
    p.add_argument("--force-observer-target", type=float, default=1.0)
    p.add_argument("--nested-cv-target", type=float, default=1.0)
    p.add_argument("--surface-observer-target", type=float, default=5.0)
    p.add_argument("--numerical-source-target", type=float, default=1.0)
    p.add_argument("--minimum-convective-times", type=float, default=8.0)
    p.add_argument(
        "--minimum-sampling-convective-times", type=float, default=5.0,
    )
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
