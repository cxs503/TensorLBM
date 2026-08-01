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
from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.solver3d import stream3d
from tensorlbm.static_block_amr import StaticBlockAMR3D, StaticBlockAMRConfig
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_static_amr import build_fine_suboff_mask, plan_suboff_static_amr
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import bfl_wall_function_3d


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
        StaticBlockAMRConfig(plan.box, tau_coarse=tau_coarse),
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
    fine_solid_q = fine_solid_g.unsqueeze(0).expand_as(amr.fine_f)

    coarse_free = equilibrium3d(rho, ux, zero, zero, device=device)
    sponge = _sponge(shape, args.sponge_width, args.sponge_strength, device)
    force_samples: list[tuple[float, float]] = []
    current_step = 0

    def advance(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        if level == 0:
            out = stream3d(collide_smagorinsky_mrt3d(f, tau, C_s=args.cs_smag))
            out = far_field_bc_3d(out, u_in=args.lattice_speed)
            if args.sponge_width > 0 and args.sponge_strength > 0.0:
                out = (1.0 - sponge) * out + sponge * coarse_free
            return far_field_bc_3d(out, u_in=args.lattice_speed)

        before = f
        collided = collide_smagorinsky_mrt3d(f, tau, C_s=args.cs_smag)
        post_collision = torch.where(fine_solid_q, before, collided)
        out = stream3d(post_collision)
        activation = smooth_ramp_factor(current_step, args.ramp_steps)
        out, friction, pressure = bfl_wall_function_3d(
            out, post_collision, fine_solid_g, 2.0 * nu_coarse,
            bfl_mask, bfl_q, y_val=args.wall_distance,
            wall_law=args.wall_law, near_mask=fine_near,
            bfl_wall_mode="wall_model_slip", wall_activation=activation,
        )
        force_samples.append((pressure, friction))
        return out

    dx_fine_m = MODEL_LENGTH_M / (2.0 * args.hull_length)
    scale = force_scale_newton(
        rho_water=args.rho_water, dx_m=dx_fine_m,
        speed_mps=point.speed_mps, lattice_speed=args.lattice_speed,
    )
    started = time.time()
    history: list[dict] = []
    recent_forces: list[float] = []
    for current_step in range(1, args.steps + 1):
        force_samples.clear()
        ledger = amr.step(advance)
        pressure = sum(item[0] for item in force_samples) / len(force_samples)
        friction = sum(item[1] for item in force_samples) / len(force_samples)
        resistance = (pressure + friction) * scale
        recent_forces.append(resistance)
        if len(recent_forces) > args.average_window:
            recent_forces.pop(0)
        if not bool(torch.isfinite(amr.coarse_f).all()) or not bool(torch.isfinite(amr.fine_f).all()):
            raise FloatingPointError(f"non-finite population at step {current_step}")
        if current_step % args.report_interval == 0 or current_step == args.steps:
            mean_force = sum(recent_forces) / len(recent_forces)
            row = {
                "step": current_step,
                "instantaneous_resistance_n": resistance,
                "window_resistance_n": mean_force,
                "error_pct": abs(mean_force - point.resistance_n) / point.resistance_n * 100.0,
                "reflux_mass_residual": ledger.mass_residual,
            }
            history.append(row)
            print(
                f"step={current_step}/{args.steps} Rt={mean_force:.3f} N "
                f"exp={point.resistance_n:.3f} N err={row['error_pct']:.2f}% "
                f"reflux={ledger.mass_residual:.3e}", flush=True,
            )

    mean_force = sum(recent_forces) / len(recent_forces)
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    )
    rho_c, ux_c, uy_c, uz_c = macroscopic3d(amr.coarse_f)
    result = {
        "schema": "tensorlbm-suboff-static-amr-v1",
        "status": "grid_candidate_not_yet_validated",
        "configuration": {
            "device": str(device), "coarse_shape_zyx": list(shape),
            "coarse_hull_length_cells": args.hull_length,
            "fine_hull_length_cells": args.hull_length * 2.0,
            "fine_diameter_cells": plan.effective_diameter_cells,
            "refinement_box": vars(plan.box), "tau_coarse": tau_coarse,
            "tau_fine": amr.config.tau_fine, "physical_reynolds": physical_re,
            "collision_reynolds": collision_re, "steps": args.steps,
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
            "experimental_resistance_n": point.resistance_n,
            "error_pct": abs(mean_force - point.resistance_n) / point.resistance_n * 100.0,
            "coarse_density_min_max": [float(rho_c.min()), float(rho_c.max())],
            "coarse_speed_max": float(torch.sqrt(ux_c**2 + uy_c**2 + uz_c**2).max()),
        },
        "history": history,
        "elapsed_s": time.time() - started,
        "claim_boundary": (
            "Static-AMR execution candidate only. Force-observer verification, "
            "three-grid convergence and time-window convergence remain mandatory."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--report-interval", type=int, default=100)
    p.add_argument("--average-window", type=int, default=500)
    p.add_argument("--ramp-steps", type=int, default=1000)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--resolved-reynolds", type=float, default=2.0e6)
    p.add_argument("--nu-water", type=float, default=1.004e-6)
    p.add_argument("--rho-water", type=float, default=998.2)
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--wall-law", choices=("log", "reichardt", "musker"), default="reichardt")
    p.add_argument("--wall-distance", type=float, default=0.5)
    p.add_argument("--sponge-width", type=int, default=12)
    p.add_argument("--sponge-strength", type=float, default=1.0)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
