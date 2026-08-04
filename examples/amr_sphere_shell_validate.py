#!/usr/bin/env python3
"""Body-fitted shell AMR sphere validation: hull-proximity shell + wake block.

Route-1 ("贴体壳层加密") variant of the static-block sphere validation: instead of
refining a fat box around the whole sphere (see amr_sphere_drag_validate.py),
the fine 2:1 block is built from a body-fitted surface shell — the coarse cells
within ``shell_margin`` of the sphere surface (HullProximityRegion) plus a
downstream wake slab (WakeRegion) — and its bounding box.

The fine block therefore only covers a thin shell hugging the sphere surface,
not the entire coarse domain and (for a partial shell) possibly not even the
sphere centre.  Two independent solid freeze layers keep the geometry exact:

* coarse level 0: ``solid_coarse_q`` freezes every coarse sphere cell that the
  fine block does *not* own (the fine block is restricted back over its box
  every root step, so coarse cells inside the box are fine-owned);
* fine level 1: ``fine_solid`` (the fine sphere mask restricted to the fine
  block — computed in block-local coordinates so it only ever marks sphere
  cells the block actually covers) freezes fine solid cells at collision, and
  the Bouzidi BFL interpolated bounce-back enforces the curved surface after
  streaming, with q-values evaluated on the fine grid including the ghost layer.

The drag is measured with an independent control volume on the fine grid
(including ghost), non-dimensionalised by the fine dynamic area
``0.5 u^2 pi R_fine^2`` and compared against Schiller-Naumann.

Run (CPU smoke)::

    PYTHONPATH=src .venv/bin/python examples/amr_sphere_shell_validate.py \
        --device cpu --nx 96 --ny 64 --nz 64 --radius 6 --reynolds 100 \
        --steps 200 --warmup-steps 60 --ramp-steps 30 --sponge-width 8 \
        --shell-margin 8 --wake-cells 25 --report-interval 50 \
        --output /tmp/shell_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.evidence_io import (
    common_schema_fields,
    json_safe,
    write_evidence,
)
from tensorlbm.sphere_amr_common import (
    build_control_volume,
    build_fine_block_geometry,
    build_sphere_geometry,
    fine_sphere_advance,
    root_advance,
    summarize_force_history,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=112)
    p.add_argument("--nz", type=int, default=112)
    p.add_argument("--radius", type=float, default=8.0)      # coarse-grid sphere radius
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--ramp-steps", type=int, default=500)
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--cv-margin", type=int, default=5)
    p.add_argument("--report-interval", type=int, default=500)
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument(
        "--ghost-interpolation",
        choices=("injection", "trilinear"),
        default="injection",
    )
    p.add_argument(
        "--shell-margin",
        type=int,
        default=8,
        help="body-fitted shell thickness in coarse cells around the sphere "
        "surface (HullProximityRegion margin)",
    )
    p.add_argument(
        "--wake-cells",
        type=int,
        default=30,
        help="downstream wake extension in coarse cells behind the sphere",
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

    # ---- coarse sphere solid (level-0 freeze mask for coarse cells the fine
    # ---- block does not own; the whole sphere is inside the closed-shell box
    # ---- here, so the level-0 freeze is dormant but keeps partial-shell runs
    # ---- geometrically consistent).
    solid_coarse, solid_coarse_q = build_sphere_geometry(
        args.nx, args.ny, args.nz, cx, cy, cz, args.radius, device,
    )

    # ---- body-fitted shell refinement region: hull-proximity surface shell
    # ---- (cells within shell_margin of the sphere surface) + downstream wake.
    plan = plan_body_shell_box(
        solid_coarse, args.shell_margin, args.wake_cells, pad=2,
    )
    box = plan.box
    x0, x1, y0, y1, z0, z1 = box.x0, box.x1, box.y0, box.y1, box.z0, box.z1
    ratio, g = 2, 1

    nu_coarse = args.lattice_speed * (2.0 * args.radius) / args.reynolds
    tau_coarse = 0.5 + 3.0 * nu_coarse

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ---- fine sphere geometry (authoritative).  The mask is evaluated on the
    # ---- fine block's own coordinates, so fine_solid only ever marks sphere
    # ---- cells the fine block covers — robust whether or not the shell box
    # ---- contains the sphere centre (partial-shell case).
    fine_shape, fine_center, r_fine, fs = build_fine_block_geometry(
        box, (cx, cy, cz), args.radius, ratio, g, device,
    )
    fine_solid, fine_solid_g, solid_q, bfl_mask, bfl_q = (
        fs.solid, fs.solid_g, fs.solid_q, fs.bfl_mask, fs.bfl_q,
    )

    print(
        f"coarse={list(shape)} fine_box={[x0, x1, y0, y1, z0, z1]} "
        f"fine_shape={list(fine_shape)} fine_center={tuple(fine_center)} "
        f"shell_margin={args.shell_margin} wake_cells={args.wake_cells} "
        f"refine_cells={plan.refine_cells} "
        f"tau_coarse={tau_coarse:.6f} Re={args.reynolds}",
        flush=True,
    )

    amr = StaticBlockAMR3D(
        coarse_f,
        StaticBlockAMRConfig(
            box, tau_coarse=tau_coarse,
            reflux=True,
            maximum_reflux_correction_fraction=0.2,
            ghost_interpolation=args.ghost_interpolation,
        ),
        fine_solid=fine_solid,
    )
    print(
        f"allocated_cells={amr.total_allocated_cells} "
        f"uniform_fine_equivalent={amr.uniform_fine_equivalent_cells} "
        f"cell_saving_fraction={amr.cell_saving_fraction:.4f}",
        flush=True,
    )
    fine_solid_g = amr.fine_solid_with_ghost
    assert fine_solid_g is not None

    # ---- control volume on the fine grid (with ghost offset), clamped to the
    # ---- strictly-interior range required by box_control_volume.
    cv = build_control_volume(
        fine_solid_g.shape, fine_center, r_fine, args.cv_margin, device,
    )

    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(
        shape, width=args.sponge_width,
        max_strength=args.sponge_strength, device=device,
        faces=sponge_faces,
    )
    dynamic_area = 0.5 * args.lattice_speed**2 * math.pi * r_fine**2

    force_samples: list[float] = []
    max_reflux_residual = 0.0
    max_reflux_correction = 0.0
    started = time.time()

    def advance(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        nonlocal max_reflux_residual, max_reflux_correction
        if level == 0:
            out, _post_collision, collided = root_advance(
                f, tau, solid_coarse_q, sigma, args.lattice_speed,
            )
            return AMRAdvanceResult(out, collided)

        out, post_collision, cv_force = fine_sphere_advance(
            f, tau,
            solid_q=solid_q, bfl_mask=bfl_mask, bfl_q=bfl_q,
            step=current_step, ramp_steps=args.ramp_steps,
            sample_cv=(substep == 1), cv=cv, solid_g=fine_solid_g,
        )
        if substep == 1:  # one sample per coarse step
            assert cv_force is not None
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
            coarse_finite = bool(torch.isfinite(amr.coarse_f).all())
            fine_finite = bool(torch.isfinite(amr.fine_f).all())
            if not (coarse_finite and fine_finite):
                raise FloatingPointError(
                    f"AMR shell run diverged (non-finite populations) at step "
                    f"{current_step}",
                )
            recent = force_samples[-min(len(force_samples), args.report_interval):]
            recent_cd = sum(recent) / len(recent) / dynamic_area if recent else math.nan
            elapsed = time.time() - started
            print(
                f"step={current_step}/{args.steps} recent_Cd={recent_cd:.6f} "
                f"steps/s={current_step / elapsed:.2f} "
                f"max_ref_res={max_reflux_residual:.2e}",
                flush=True,
            )

    summary = summarize_force_history(
        force_samples, dynamic_area, args.reynolds, args.statistics_window_steps,
    )
    cd = summary["cd"]
    reference = summary["reference_cd"]
    reference_error = summary["reference_error_pct"]
    mean_force = summary["mean_force_lu"]
    stationarity_dict = summary["stationarity"]
    result = {
        **common_schema_fields("sphere-shell-amr-cv-v1"),
        "case": (
            f"AMR body-fitted shell sphere Re={args.reynolds}: coarse "
            f"{list(shape)} + 2:1 shell block {[x0, x1, y0, y1, z0, z1]} "
            f"shell_margin={args.shell_margin} wake_cells={args.wake_cells} "
            f"effective radius {r_fine}"
        ),
        "configuration": {
            "coarse_shape_zyx": list(shape),
            "fine_shape_zyx": list(fine_shape),
            "fine_box": [x0, x1, y0, y1, z0, z1],
            "shell_margin": args.shell_margin,
            "wake_cells": args.wake_cells,
            "refine_cells_coarse": plan.refine_cells,
            "cell_saving_fraction": amr.cell_saving_fraction,
            "radius_coarse": args.radius,
            "radius_fine": r_fine,
            "reynolds": args.reynolds,
            "lattice_speed": args.lattice_speed,
            "tau_coarse": tau_coarse,
            "tau_fine": 0.5 + ratio * (tau_coarse - 0.5),
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "cv_margin": args.cv_margin,
            "reflux": True,
            "ghost_interpolation": args.ghost_interpolation,
            "far_field_mode": "non_equilibrium_extrapolation",
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
    write_evidence(result, args.output)
    print(json.dumps(json_safe(result["result"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
