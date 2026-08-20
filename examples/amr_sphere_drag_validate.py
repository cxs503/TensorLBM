#!/usr/bin/env python3
"""AMR sphere drag validation: StaticBlockAMR3D + BFL sphere + control-volume force.

Coarse grid carries the external domain and far-field boundaries. One strictly
interior 2:1 block owns the sphere plus a downstream wake region, advances
twice per coarse step, and is conservatively restricted and refluxed.

This is a validation runner: the result is compared against the uniform-grid
sphere reference (Schiller-Naumann Cd) to test whether AMR preserves accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import far_field_bc_3d, sphere_mask
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_bfl_control_volume import schiller_naumann_cd
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import AMRAdvanceResult, StaticBlockAMR3D, StaticBlockAMRConfig


def _ramp(step: int, steps: int) -> float:
    if steps <= 0:
        return 1.0
    return min(1.0, step / steps)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=112)
    p.add_argument("--nz", type=int, default=112)
    p.add_argument("--radius", type=float, default=8.0)  # coarse-grid sphere radius
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--wall-margin", type=int, default=6)
    p.add_argument("--wake-cells", type=int, default=30)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--ramp-steps", type=int, default=500)
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--cv-margin", type=int, default=5)
    p.add_argument("--report-interval", type=int, default=500)
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument("--max-reflux-fraction", type=float, default=0.2)
    p.add_argument("--no-reflux", action="store_true")
    p.add_argument(
        "--ghost-interpolation",
        choices=("injection", "trilinear"),
        default="injection",
    )
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    shape = (args.nz, args.ny, args.nx)
    cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0

    # ---- coarse solid (used only to plan the box; fine mask is authoritative)
    solid_coarse = sphere_mask(args.nx, args.ny, args.nz, cx, cy, cz, args.radius, device=device)
    solid_coarse_q = solid_coarse.unsqueeze(0).expand(19, *shape).contiguous()
    indices = solid_coarse.nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError("no sphere cells on coarse grid")
    z_min, y_min, x_min = (int(indices[:, a].min().item()) for a in range(3))
    z_max, y_max, x_max = (int(indices[:, a].max().item()) + 1 for a in range(3))
    m = args.wall_margin
    x0 = max(1, x_min - m)
    x1 = min(args.nx - 1, x_max + m + args.wake_cells)
    y0 = max(1, y_min - m)
    y1 = min(args.ny - 1, y_max + m)
    z0 = max(1, z_min - m)
    z1 = min(args.nz - 1, z_max + m)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError("refinement box too small")
    box = BoxRegion(x0, x1, y0, y1, z0, z1)
    ratio = 2
    g = 1

    nu_coarse = args.lattice_speed * (2.0 * args.radius) / args.reynolds
    tau_coarse = 0.5 + 3.0 * nu_coarse

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ---- fine sphere geometry (authoritative)
    fine_shape = ((z1 - z0) * ratio, (y1 - y0) * ratio, (x1 - x0) * ratio)
    fine_center = (
        cx * ratio - x0 * ratio + g,
        cy * ratio - y0 * ratio + g,
        cz * ratio - z0 * ratio + g,
    )
    fine_solid = sphere_mask(
        fine_shape[2],
        fine_shape[1],
        fine_shape[0],
        fine_center[0],
        fine_center[1],
        fine_center[2],
        args.radius * ratio,
        device=device,
    )
    fine_solid_g = torch.zeros(
        (fine_shape[0] + 2 * g, fine_shape[1] + 2 * g, fine_shape[2] + 2 * g),
        dtype=torch.bool,
        device=device,
    )
    fine_solid_g[g:-g, g:-g, g:-g] = fine_solid
    bfl_mask, bfl_q = compute_q_sphere(
        fine_solid_g.shape[2],
        fine_solid_g.shape[1],
        fine_solid_g.shape[0],
        fine_center[0],
        fine_center[1],
        fine_center[2],
        args.radius * ratio,
        device=device,
    )
    solid_q = (
        fine_solid_g.unsqueeze(0)
        .expand(
            19,
            *fine_solid_g.shape,
        )
        .contiguous()
    )

    print(
        f"coarse={list(shape)} fine_box={[x0, x1, y0, y1, z0, z1]} "
        f"fine_shape={list(fine_shape)} fine_center={tuple(fine_center)} "
        f"tau_coarse={tau_coarse:.6f} Re={args.reynolds}",
        flush=True,
    )

    amr = StaticBlockAMR3D(
        coarse_f,
        StaticBlockAMRConfig(
            box,
            tau_coarse=tau_coarse,
            reflux=not args.no_reflux,
            maximum_reflux_correction_fraction=args.max_reflux_fraction,
            ghost_interpolation=args.ghost_interpolation,
        ),
        fine_solid=fine_solid,
    )
    fine_solid_g = amr.fine_solid_with_ghost
    assert fine_solid_g is not None

    # ---- control volume on the fine grid (with ghost offset)
    fg = fine_solid_g.shape
    cv = box_control_volume(
        tuple(fg),
        x0=int(math.floor(fine_center[0] - args.radius * ratio)) - args.cv_margin,
        x1=int(math.ceil(fine_center[0] + args.radius * ratio)) + args.cv_margin + 1,
        y0=int(math.floor(fine_center[1] - args.radius * ratio)) - args.cv_margin,
        y1=int(math.ceil(fine_center[1] + args.radius * ratio)) + args.cv_margin + 1,
        z0=int(math.floor(fine_center[2] - args.radius * ratio)) - args.cv_margin,
        z1=int(math.ceil(fine_center[2] + args.radius * ratio)) + args.cv_margin + 1,
        device=device,
    )
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=sponge_faces,
    )
    dynamic_area = 0.5 * args.lattice_speed**2 * math.pi * (args.radius * ratio) ** 2

    force_samples: list[float] = []
    reflux_residuals: list[float] = []
    max_reflux_residual = 0.0
    max_reflux_correction = 0.0
    started = time.time()

    def advance(
        f: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        nonlocal max_reflux_residual, max_reflux_correction
        if level == 0:
            before = f
            collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
            post_collision = torch.where(solid_coarse_q, before, collided)
            out = stream3d(post_collision)
            if args.far_field_mode == "non_equilibrium_extrapolation":
                out = non_equilibrium_far_field_bc_3d(out, u_in=args.lattice_speed)
            else:
                out = far_field_bc_3d(out, u_in=args.lattice_speed)
            out = apply_equilibrium_difference_sponge(
                out,
                sigma,
                velocity_target=(args.lattice_speed, 0.0, 0.0),
            )
            if args.far_field_mode == "non_equilibrium_extrapolation":
                out = non_equilibrium_far_field_bc_3d(out, u_in=args.lattice_speed)
            else:
                out = far_field_bc_3d(out, u_in=args.lattice_speed)
            return AMRAdvanceResult(out, collided)

        before = f
        collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
        post_collision = torch.where(solid_q, before, collided)
        out = stream3d(post_collision)
        rho_post, ux_post, uy_post, uz_post = macroscopic3d(post_collision)
        # NOTE: bouzidi_bounce_back_d3q19 was refactored to the q-field API and
        # now assumes a stationary wall (no wall_velocity kwarg, single return
        # value).  Static obstacle => wall velocity is zero, so the new call is
        # correct here.
        out = bouzidi_bounce_back_d3q19(out, post_collision, bfl_mask, bfl_q)
        if substep == 1:  # one sample per coarse step
            cv_force = float(
                observe_control_volume_force(
                    before,
                    out,
                    post_collision,
                    cv,
                    solid=fine_solid_g,
                )
                .force_on_body[0]
                .item()
            )
            force_samples.append(cv_force)
        return AMRAdvanceResult(out, post_collision)

    current_step = 0
    for current_step in range(1, args.steps + 1):
        ledger = amr.step(advance)
        r = float(ledger.residual.abs().max().item())
        max_reflux_residual = max(max_reflux_residual, r)
        max_reflux_correction = max(
            max_reflux_correction,
            float(ledger.replacement_mismatch.abs().max().item()),
        )
        if current_step % args.report_interval == 0:
            recent = force_samples[-min(len(force_samples), args.report_interval) :]
            recent_cd = sum(recent) / len(recent) / dynamic_area if recent else math.nan
            elapsed = time.time() - started
            print(
                f"step={current_step}/{args.steps} recent_Cd={recent_cd:.6f} "
                f"steps/s={current_step / elapsed:.2f} max_ref_res={max_reflux_residual:.2e}",
                flush=True,
            )

    statistics_window = args.statistics_window_steps or len(force_samples)
    selected = force_samples[-statistics_window:]
    mean_force = sum(selected) / len(selected)
    cd = mean_force / dynamic_area
    reference = schiller_naumann_cd(args.reynolds)
    cd_history = [f_ / dynamic_area for f_ in selected]
    stationarity = assess_force_stationarity(
        cd_history,
        block_size=max(1, len(cd_history) // 8),
    )
    stationarity_dict = (
        asdict(stationarity) if hasattr(stationarity, "__dataclass_fields__") else stationarity
    )
    reference_error = abs(cd - reference) / reference * 100.0
    result = {
        "schema": "tensorlbm-sphere-amr-cv-v1",
        "status": "measured_candidate",
        "physical_validation": False,
        "case": (
            f"AMR sphere Re={args.reynolds}: coarse {list(shape)} + 2:1 fine block "
            f"{[x0, x1, y0, y1, z0, z1]} effective radius {args.radius * ratio}"
        ),
        "configuration": {
            "coarse_shape_zyx": list(shape),
            "fine_shape_zyx": list(fine_shape),
            "fine_box": [x0, x1, y0, y1, z0, z1],
            "radius_coarse": args.radius,
            "radius_fine": args.radius * ratio,
            "reynolds": args.reynolds,
            "lattice_speed": args.lattice_speed,
            "tau_coarse": tau_coarse,
            "tau_fine": 0.5 + ratio * (tau_coarse - 0.5),
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "far_field_mode": args.far_field_mode,
            "reflux": not args.no_reflux,
            "ghost_interpolation": args.ghost_interpolation,
        },
        "result": {
            "cd_control_volume": cd,
            "reference_cd": reference,
            "reference_error_pct": reference_error,
            "mean_force_lu": mean_force,
            "dynamic_area_lu2": dynamic_area,
            "stationarity": stationarity_dict,
            "max_reflux_residual": max_reflux_residual,
            "max_reflux_correction": max_reflux_correction,
            "wall_time_s": time.time() - started,
        },
        "artifacts": {
            "output": args.output,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
