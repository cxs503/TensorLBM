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
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d
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
    build_fine_suboff_mask,
    plan_suboff_static_amr,
)
from tensorlbm.turbulence import (
    collide_smagorinsky_mrt3d,
    collide_wale_mrt3d,
)
from tensorlbm.wall_model import WallStressDiagnostics, bfl_wall_function_3d


def _appendage_halfway_links(
    solid: torch.Tensor,
    mask: torch.Tensor,
    q: torch.Tensor,
    *,
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    device: torch.device,
) -> int:
    """Use analytical q on the body and halfway links on voxel appendages."""
    bare, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz, cx=cx, cy=cy, cz=cz, length=length,
        config=SuboffConfig(), device=device,
    )
    count = 0
    for direction in range(1, 19):
        dcx, dcy, dcz = (int(value) for value in C19[direction].tolist())
        full_neighbor = torch.roll(
            solid, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2),
        )
        bare_neighbor = torch.roll(
            bare, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2),
        )
        halfway = mask[direction] & full_neighbor & ~bare_neighbor
        count += int(halfway.sum().item())
        q[direction][halfway] = 0.5
    return count


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
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.resume and not args.checkpoint:
        raise ValueError("resume requires --checkpoint")
    if min(args.error_target, args.drift_target, args.force_observer_target) < 0.0:
        raise ValueError("acceptance targets must be non-negative")
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
    fine_solid, fine_geometry = build_fine_suboff_mask(
        plan, hull_type=args.hull_type, coarse_center=center,
        config=config, device=device,
    )

    physical_re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    collision_re = args.resolved_reynolds or physical_re
    nu_coarse = args.lattice_speed * args.hull_length / collision_re
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
    )
    # The independent geometry constructions must agree before a force run.
    q_solid, _ = build_suboff_mask(
        args.hull_type, nx_f, ny_f, nz_f, cx=fine_center[0],
        cy=fine_center[1], cz=fine_center[2], length=args.hull_length * 2.0,
        config=config, device=device,
    )
    if not torch.equal(q_solid, fine_solid_g):
        raise RuntimeError("fine CAD mask and BFL geometry disagree")
    appendage_links = 0
    if args.hull_type == "full":
        appendage_links = _appendage_halfway_links(
            fine_solid_g, bfl_mask, bfl_q, nx=nx_f, ny=ny_f, nz=nz_f,
            cx=fine_center[0], cy=fine_center[1], cz=fine_center[2],
            length=args.hull_length * 2.0, device=device,
        )
    fine_near = get_near_wall_3d(fine_solid_g)
    if args.hull_type == "bare_hull":
        fine_surface = SurfaceMesh.from_suboff(
            fine_solid_g, fine_near, *fine_center,
            args.hull_length * 2.0, args.hull_length * 2.0 / (2.0 * 8.57),
            config=config,
        )
        fine_area_weight = torch.full_like(
            fine_near,
            float(fine_geometry["wetted_area_lu2"])
            / max(float(fine_near.sum().item()), 1.0),
            dtype=amr.fine_f.dtype,
        )
    else:
        fine_surface = SurfaceMesh.from_gradient(fine_solid_g, fine_near)
        fine_area_weight = None
    fine_solid_q = fine_solid_g.unsqueeze(0).expand_as(amr.fine_f)
    fine_indices = fine_solid_g.nonzero(as_tuple=False)
    z_min, y_min, x_min = (
        int(fine_indices[:, axis].min().item()) for axis in range(3)
    )
    z_max, y_max, x_max = (
        int(fine_indices[:, axis].max().item()) + 1 for axis in range(3)
    )
    fine_cv = box_control_volume(
        fine_solid_g.shape,
        x0=x_min - args.cv_margin, x1=x_max + args.cv_margin,
        y0=y_min - args.cv_margin, y1=y_max + args.cv_margin,
        z0=z_min - args.cv_margin, z1=z_max + args.cv_margin,
        device=device,
    )

    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    if args.sponge_inlet:
        sponge_faces = ("x-",) + sponge_faces
    sponge = build_sponge_sigma_3d(
        shape, width=args.sponge_width, max_strength=args.sponge_strength,
        device=device, faces=sponge_faces,
    )
    force_samples: list[tuple[float, float, float]] = []
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
            out, post_collision, fine_solid_g, 2.0 * nu_coarse,
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
        if not args.disable_positivity_limiter:
            out, diagnostic = limit_nonequilibrium_for_positivity(out)
            positivity_fractions.append(diagnostic.limited_fraction)
        cv_force = float(observe_control_volume_force(
            before, out, post_collision, fine_cv, solid=fine_solid_g,
        ).force_on_body[0].item())
        force_samples.append((pressure, friction, cv_force))
        return AMRAdvanceResult(out, post_collision)

    dx_fine_m = MODEL_LENGTH_M / (2.0 * args.hull_length)
    scale = force_scale_newton(
        rho_water=args.rho_water, dx_m=dx_fine_m,
        speed_mps=point.speed_mps, lattice_speed=args.lattice_speed,
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_signature = {
        "coarse_shape_zyx": list(shape),
        "hull_type": args.hull_type,
        "speed_knots": args.speed_knots,
        "hull_length": args.hull_length,
        "center_x_fraction": args.center_x_fraction,
        "refinement_box": vars(plan.box),
        "wall_margin": args.wall_margin,
        "wake_cells": args.wake_cells,
        "cv_margin": args.cv_margin,
        "reflux_enabled": not args.disable_reflux,
        "wall_stress_coupled": not args.diagnostic_uncoupled_wall_stress,
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
    }

    if args.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("configuration") != checkpoint_signature:
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
        history = list(state["history"])

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({
            "schema": "tensorlbm-suboff-static-amr-checkpoint-v2",
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
        maximum_positivity_limited_fraction = max(
            maximum_positivity_limited_fraction,
            max(positivity_fractions, default=0.0),
        )
        maximum_reflux_population_residual = max(
            maximum_reflux_population_residual,
            float(ledger.residual.abs().max().item()),
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
                "maximum_positivity_limited_fraction": max(
                    positivity_fractions, default=0.0,
                ),
            }
            history.append(row)
            print(
                f"step={current_step}/{args.steps} Rt={mean_force:.3f} N "
                f"exp={point.resistance_n:.3f} N err={row['error_pct']:.2f}% "
                f"reflux={ledger.mass_residual:.3e}", flush=True,
            )
        if (
            checkpoint is not None and args.checkpoint_interval
            and current_step % args.checkpoint_interval == 0
        ):
            save_checkpoint(current_step)

    mean_force = sum(force_history) / len(force_history)
    mean_bfl_total = sum(bfl_total_history) / len(bfl_total_history)
    mean_pressure = sum(pressure_history) / len(pressure_history)
    mean_wall_shear = sum(wall_shear_history) / len(wall_shear_history)
    force_stationarity = assess_force_stationarity(
        force_history,
        block_size=max(1, len(force_history) // 8),
    )
    reference_error_pct = (
        abs(mean_force - point.resistance_n) / point.resistance_n * 100.0
    )
    force_observer_difference_pct = (
        abs(mean_bfl_total - mean_force) / max(abs(mean_force), 1e-30) * 100.0
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
    finite = (
        bool(torch.isfinite(amr.coarse_f).all())
        and bool(torch.isfinite(amr.fine_f).all())
        and math.isfinite(mean_force)
    )
    single_grid_admitted = (
        finite
        and not args.diagnostic_uncoupled_wall_stress
        and reference_error_pct <= args.error_target
        and force_stationarity.meets(args.drift_target)
        and force_observer_difference_pct <= args.force_observer_target
        and limiter_acceptable
        and reflux_acceptable
        and wall_sampling_acceptable
    )
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    )
    rho_c, ux_c, uy_c, uz_c = macroscopic3d(amr.coarse_f)
    result = {
        "schema": "tensorlbm-suboff-static-amr-v2",
        "status": (
            "single_grid_candidate" if single_grid_admitted
            else "single_grid_rejected"
        ),
        "configuration": {
            "device": str(device), "coarse_shape_zyx": list(shape),
            "coarse_hull_length_cells": args.hull_length,
            "fine_hull_length_cells": args.hull_length * 2.0,
            "fine_diameter_cells": plan.effective_diameter_cells,
            "refinement_box": vars(plan.box), "tau_coarse": tau_coarse,
            "tau_fine": amr.config.tau_fine, "physical_reynolds": physical_re,
            "collision_reynolds": collision_re, "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "report_interval": args.report_interval,
            "report_average_window": args.average_window,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "reflux_enabled": amr.config.reflux,
            "reflux_method": "face_local_post_collision_kinetic_flux",
            "les_model": args.les_model,
            "collision_model": args.collision_model,
            "les_constant": (
                args.cw_wale if args.les_model == "wale" else args.cs_smag
            ),
            "wall_stress_coupled": not args.diagnostic_uncoupled_wall_stress,
            "wall_law": args.wall_law,
            "wall_distance": args.wall_distance,
            "stress_exchange_distance": (
                args.stress_exchange_distance
                if args.stress_exchange_distance > 0.0 else None
            ),
            "wall_diagnostic_interval": args.wall_diagnostic_interval,
            "positivity_limiter_enabled": not args.disable_positivity_limiter,
            "far_field_mode": args.far_field_mode,
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "sponge_inlet_enabled": args.sponge_inlet,
            "wall_activation_ramp_steps": args.ramp_steps,
        },
        "mesh": {
            "coarse_cells": plan.coarse_cells,
            "fine_allocated_cells": plan.fine_allocated_cells,
            "total_allocated_cells": plan.total_allocated_cells,
            "uniform_fine_cells": plan.uniform_fine_cells,
            "saving_fraction": plan.cell_saving_fraction,
            "estimated_peak_gib": plan.estimated_peak_gib(),
            "measured_peak_allocated_gib": peak_gib,
        },
        "geometry": fine_geometry | {"appendage_halfway_links": appendage_links},
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
            "experimental_resistance_n": point.resistance_n,
            "error_pct": reference_error_pct,
            "force_stationarity": force_stationarity.to_dict(),
            "maximum_positivity_limited_fraction": (
                maximum_positivity_limited_fraction
            ),
            "maximum_reflux_population_residual": (
                maximum_reflux_population_residual
            ),
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
            "maximum_limiter_fraction": 1e-3,
            "maximum_reflux_population_residual": 1e-6,
            "maximum_exchange_rejected_fraction": 0.01,
            "force_target_met": reference_error_pct <= args.error_target,
            "stationarity_target_met": force_stationarity.meets(
                args.drift_target,
            ),
            "force_observer_target_met": (
                force_observer_difference_pct <= args.force_observer_target
            ),
            "limiter_target_met": limiter_acceptable,
            "reflux_target_met": reflux_acceptable,
            "wall_sampling_target_met": wall_sampling_acceptable,
            "wall_stress_coupled": not args.diagnostic_uncoupled_wall_stress,
            "single_grid_admitted": single_grid_admitted,
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
    p.add_argument("--disable-reflux", action="store_true")
    p.add_argument("--diagnostic-uncoupled-wall-stress", action="store_true")
    p.add_argument("--disable-positivity-limiter", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=2500)
    p.add_argument("--report-interval", type=int, default=100)
    p.add_argument("--average-window", type=int, default=500)
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
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
