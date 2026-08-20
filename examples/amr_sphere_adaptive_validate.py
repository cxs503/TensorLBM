#!/usr/bin/env python3
"""Adaptive AMR sphere drag validation: AdaptiveSolver3D + error-indicator patches.

The coarse grid carries the external domain. Fine patches are added/removed
dynamically by the nonequilibrium error indicator (Lagrava et al. 2012),
tracking the sphere wake. Solid handling follows the uniform/static-AMR
reference (examples/amr_sphere_drag_validate.py): solid cells are frozen at
their pre-collision state during collision — with stream-after-collide this
acts as a simple bounce-back without needing the pre-collision distribution
inside the boundary stage — and per-patch solid masks are block-upsampled
from the coarse mask in patch coordinates.

This is a validation runner: compares adaptive-AMR Cd against Schiller-Naumann
and the uniform-grid controls. status=measured_candidate, physical_validation
is only set when the run passes all gates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tensorlbm.adaptive_refinement import (
    AdaptationSchedule,
    AdaptiveSolver3D,
    nonequilibrium_indicator_3d,
)
from tensorlbm.boundaries3d import bounce_back_cells_3d, sphere_mask
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_bfl_control_volume import schiller_naumann_cd
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)


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
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--ramp-steps", type=int, default=500)
    p.add_argument("--adapt-interval", type=int, default=50)
    p.add_argument("--adapt-warmup", type=int, default=200)
    p.add_argument("--max-patches", type=int, default=8)
    p.add_argument("--max-levels", type=int, default=2)
    p.add_argument("--refine-threshold", type=float, default=1e-3)
    p.add_argument("--coarsen-threshold", type=float, default=1e-5)
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
    p.add_argument("--indicator", choices=("nonequilibrium", "vorticity"), default="nonequilibrium")
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
    solid = sphere_mask(args.nx, args.ny, args.nz, cx, cy, cz, args.radius, device=device)
    solid_q = solid.unsqueeze(0).expand(19, *shape).contiguous()
    shape_q = (19, *shape)

    nu = args.lattice_speed * (2.0 * args.radius) / args.reynolds
    tau = 0.5 + 3.0 * nu

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, zero, zero, device=device)

    schedule = AdaptationSchedule(
        interval=args.adapt_interval,
        warmup=args.adapt_warmup,
        max_patches=args.max_patches,
        max_levels=args.max_levels,
        refine_threshold=args.refine_threshold,
        coarsen_threshold=args.coarsen_threshold,
        tau=tau,
    )
    solver = AdaptiveSolver3D(f, schedule=schedule, mask=solid)

    cv = box_control_volume(
        shape,
        x0=int(math.floor(cx - args.radius)) - args.cv_margin,
        x1=int(math.ceil(cx + args.radius)) + args.cv_margin + 1,
        y0=int(math.floor(cy - args.radius)) - args.cv_margin,
        y1=int(math.ceil(cy + args.radius)) + args.cv_margin + 1,
        z0=int(math.floor(cz - args.radius)) - args.cv_margin,
        z1=int(math.ceil(cz + args.radius)) + args.cv_margin + 1,
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
    dynamic_area = 0.5 * args.lattice_speed**2 * math.pi * args.radius**2

    forces: list[float] = []
    patch_counts: list[int] = []
    started = time.time()

    def _patch_solid_mask(patch) -> torch.Tensor:
        """Block-upsample the coarse solid mask into a patch's fine grid.

        The fine tensor produced by the solver's FH interpolation has shape
        ``(19, (z1-z0)*r, (y1-y0)*r, (x1-x0)*r)``; block-upsampling the
        coarse mask with the same ratio yields a per-patch solid mask of
        exactly that spatial shape, geometrically consistent with the
        coarse sphere (no gaps or overlaps at the coarse/fine interface).
        """
        b = patch.box
        r = patch.ratio
        coarse_slice = solid[b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1]
        return (
            coarse_slice.repeat_interleave(r, dim=0)
            .repeat_interleave(r, dim=1)
            .repeat_interleave(r, dim=2)
        )

    def collide_fn(state: torch.Tensor) -> torch.Tensor:
        # Freeze solid cells at their pre-collision state (same scheme as
        # examples/amr_sphere_drag_validate.py).  With the stream-after-
        # collide ordering used here, a frozen solid cell reflects the
        # populations streamed into it on the following step, which is a
        # simple no-slip bounce-back.  This avoids the half-way BB pitfall
        # of bounce_back_cells_3d, whose f_pre (pre-collision) argument is
        # not accessible inside the solver's boundary stage.
        collided = collide_cumulant_d3q19(state, tau, C_s=0.0)
        if state.shape == shape_q:  # coarse background grid
            return torch.where(solid_q, state, collided)
        for p in solver.patches:
            if p.f is state:
                local = _patch_solid_mask(p)
                return torch.where(
                    local.unsqueeze(0).expand_as(state),
                    state,
                    collided,
                )
        return collided

    def boundary_fn(state: torch.Tensor) -> torch.Tensor:
        # Bounce-back is essential here: the freeze-only scheme (reference
        # amr_sphere_drag_validate.py relies on a fine Bouzidi BFL to seed
        # the perturbation) leaves the uniform free stream an exact fixed
        # point, so the sphere would be invisible (Cd=0, no patches).
        # Plain post-stream BB is used (not f_pre): with solid cells frozen
        # during collision, the post-collision state at solid cells equals
        # the pre-collision state, so an f_pre-based half-way BB feeds back
        # the previous step's values and drives a period-2 oscillation.
        if state.shape == shape_q:
            state = bounce_back_cells_3d(state, solid)
            if args.far_field_mode == "non_equilibrium_extrapolation":
                state = non_equilibrium_far_field_bc_3d(state, u_in=args.lattice_speed)
            else:
                from tensorlbm.boundaries3d import far_field_bc_3d

                state = far_field_bc_3d(state, u_in=args.lattice_speed)
            state = apply_equilibrium_difference_sponge(
                state,
                sigma,
                velocity_target=(args.lattice_speed, 0.0, 0.0),
            )
            if args.far_field_mode == "non_equilibrium_extrapolation":
                state = non_equilibrium_far_field_bc_3d(state, u_in=args.lattice_speed)
            else:
                from tensorlbm.boundaries3d import far_field_bc_3d

                state = far_field_bc_3d(state, u_in=args.lattice_speed)
            return state
        # Patch tensors are interior; the border is re-injected from the
        # parent every patch substep.  Apply plain bounce-back with the
        # per-patch solid mask (no pre-collision capture available here;
        # patches act as correction layers on top of the coarse grid).
        for p in solver.patches:
            if p.f is state:
                local = _patch_solid_mask(p)
                return bounce_back_cells_3d(state, local)
        return state

    current_step = 0
    for current_step in range(1, args.steps + 1):
        before = solver.coarse_f.clone()
        solver.step(collide_fn, stream3d, boundary_fn)
        # Control-volume force on the coarse grid (adaptive patches are
        # correction layers; the conservative CV force is measured on the
        # base grid to remain well-defined under dynamic patch changes).
        if current_step > args.warmup_steps:
            # observe_control_volume_force's third argument must be the
            # post-collision, pre-stream state (streaming_momentum_import
            # samples populations from their pre-stream source cells).
            # collide is deterministic and side-effect free, so recompute
            # it from the pre-step state.
            post_collide = collide_fn(before)
            cv_force = float(
                observe_control_volume_force(
                    before,
                    solver.coarse_f,
                    post_collide,
                    cv,
                    solid=solid,
                )
                .force_on_body[0]
                .item()
            )
            forces.append(cv_force)
        patch_counts.append(len(solver.patches))
        if solver.should_adapt(current_step):
            rho, ux, uy, uz = macroscopic3d(solver.coarse_f)
            if args.indicator == "vorticity":
                from tensorlbm.adaptive_refinement import vorticity_indicator_3d

                indicator = vorticity_indicator_3d(ux, uy, uz)
            else:
                indicator = nonequilibrium_indicator_3d(
                    solver.coarse_f,
                    rho,
                    ux,
                    uy,
                    uz,
                )
            print(
                f"  adapt@{current_step}: indicator max={float(indicator.max()):.3e} "
                f"mean={float(indicator.mean()):.3e} "
                f"refine_cells={int((indicator > args.refine_threshold).sum())} "
                f"patches={len(solver.patches)}",
                flush=True,
            )
            solver.adapt(indicator)
        if args.report_interval and current_step % args.report_interval == 0:
            recent = forces[-min(len(forces), args.report_interval) :]
            recent_cd = sum(recent) / len(recent) / dynamic_area if recent else math.nan
            print(
                f"step={current_step}/{args.steps} recent_Cd={recent_cd:.6f} "
                f"patches={len(solver.patches)} steps/s={current_step / (time.time() - started):.2f}",
                flush=True,
            )
        if not bool(torch.isfinite(solver.coarse_f).all()):
            raise FloatingPointError(f"adaptive sphere diverged at step {current_step}")

    statistics_window = args.statistics_window_steps or len(forces)
    selected = forces[-statistics_window:]
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
        "schema": "tensorlbm-sphere-adaptive-amr-cv-v1",
        "status": "measured_candidate",
        "physical_validation": False,
        "case": (
            f"adaptive AMR sphere Re={args.reynolds}: coarse {list(shape)} "
            f"indicator={args.indicator} max_levels={args.max_levels} "
            f"max_patches={args.max_patches}"
        ),
        "configuration": {
            "shape_zyx": list(shape),
            "radius": args.radius,
            "reynolds": args.reynolds,
            "lattice_speed": args.lattice_speed,
            "tau": tau,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "adapt_interval": args.adapt_interval,
            "adapt_warmup": args.adapt_warmup,
            "max_patches": args.max_patches,
            "max_levels": args.max_levels,
            "refine_threshold": args.refine_threshold,
            "coarsen_threshold": args.coarsen_threshold,
            "indicator": args.indicator,
            "far_field_mode": args.far_field_mode,
        },
        "result": {
            "cd_control_volume": cd,
            "reference_cd": reference,
            "reference_error_pct": reference_error,
            "mean_force_lu": mean_force,
            "dynamic_area_lu2": dynamic_area,
            "stationarity": stationarity_dict,
            "mean_patch_count": (sum(patch_counts) / len(patch_counts) if patch_counts else 0.0),
            "max_patch_count": max(patch_counts, default=0),
            "wall_time_s": time.time() - started,
        },
        "artifacts": {"output": args.output},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
