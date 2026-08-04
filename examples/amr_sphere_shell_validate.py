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
from dataclasses import asdict
from pathlib import Path

import torch

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.refinement import BoxRegion, HullProximityRegion, WakeRegion
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_bfl_control_volume import schiller_naumann_cd
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
)


def _ramp(step: int, steps: int) -> float:
    if steps <= 0:
        return 1.0
    return min(1.0, step / steps)


def _json_safe(value: object) -> object:
    """Recursively map non-finite floats to None for strict JSON validity.

    ForceStationarityReport may hold ``math.inf`` (autocorrelation fields) and
    tuples; ``json.dumps`` would emit non-standard ``Infinity`` tokens.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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
    solid_coarse = sphere_mask(
        args.nx, args.ny, args.nz, cx, cy, cz, args.radius, device=device,
    )
    if not bool(solid_coarse.any()):
        raise ValueError("no sphere cells on the coarse grid")
    solid_coarse_q = solid_coarse.unsqueeze(0).expand(19, *shape).contiguous()

    # ---- body-fitted shell refinement region: hull-proximity surface shell
    # ---- (cells within shell_margin of the sphere surface) + downstream wake.
    shell_mask = HullProximityRegion(
        solid_coarse, margin=args.shell_margin,
    ).expand_mask()
    wake_mask = WakeRegion(
        solid_coarse, extend_x=args.wake_cells,
    ).expand_mask()
    # WakeRegion fills the full downstream cross-section plane; clip it
    # laterally to the body's shell extent so the fine block stays body-fitted
    # instead of spanning the entire coarse domain height/depth.
    shell_idx = shell_mask.nonzero(as_tuple=False)
    if shell_idx.numel() == 0:
        raise ValueError(
            "empty shell refinement region; adjust --shell-margin",
        )
    sz0, sy0 = int(shell_idx[:, 0].min().item()), int(shell_idx[:, 1].min().item())
    sz1, sy1 = int(shell_idx[:, 0].max().item()), int(shell_idx[:, 1].max().item())
    wake_mask[:sz0, :, :] = False
    wake_mask[sz1 + 1:, :, :] = False
    wake_mask[:, :sy0, :] = False
    wake_mask[:, sy1 + 1:, :] = False
    refine_mask = shell_mask | wake_mask
    indices = refine_mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError(
            "empty shell+wake refinement region; adjust --shell-margin/--wake-cells",
        )
    z_min, y_min, x_min = (int(indices[:, a].min().item()) for a in range(3))
    z_max, y_max, x_max = (int(indices[:, a].max().item()) + 1 for a in range(3))
    pad = 2  # keep the coarse-fine interface off the solid shell surface
    x0 = max(1, x_min - pad)
    x1 = min(args.nx - 1, x_max + pad)
    y0 = max(1, y_min - pad)
    y1 = min(args.ny - 1, y_max + pad)
    z0 = max(1, z_min - pad)
    z1 = min(args.nz - 1, z_max + pad)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError(
            "refinement box too small: try a larger --wake-cells or a smaller "
            "--shell-margin",
        )
    box = BoxRegion(x0, x1, y0, y1, z0, z1)
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
    fine_shape = ((z1 - z0) * ratio, (y1 - y0) * ratio, (x1 - x0) * ratio)
    fine_center = (
        cx * ratio - x0 * ratio + g,
        cy * ratio - y0 * ratio + g,
        cz * ratio - z0 * ratio + g,
    )
    fine_solid = sphere_mask(
        fine_shape[2], fine_shape[1], fine_shape[0],
        fine_center[0], fine_center[1], fine_center[2],
        args.radius * ratio, device=device,
    )
    fine_solid_g = torch.zeros(
        (fine_shape[0] + 2 * g, fine_shape[1] + 2 * g, fine_shape[2] + 2 * g),
        dtype=torch.bool, device=device,
    )
    fine_solid_g[g:-g, g:-g, g:-g] = fine_solid
    bfl_mask, bfl_q = compute_q_sphere(
        fine_solid_g.shape[2], fine_solid_g.shape[1], fine_solid_g.shape[0],
        fine_center[0], fine_center[1], fine_center[2],
        args.radius * ratio, device=device,
    )
    solid_q = fine_solid_g.unsqueeze(0).expand(
        19, *fine_solid_g.shape,
    ).contiguous()

    print(
        f"coarse={list(shape)} fine_box={[x0, x1, y0, y1, z0, z1]} "
        f"fine_shape={list(fine_shape)} fine_center={tuple(fine_center)} "
        f"shell_margin={args.shell_margin} wake_cells={args.wake_cells} "
        f"refine_cells={int(refine_mask.sum().item())} "
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
    r_fine = args.radius * ratio
    fg = fine_solid_g.shape

    def _cv_lo(centre: float) -> int:
        return int(max(1, math.floor(centre - r_fine) - args.cv_margin))

    def _cv_hi(centre: float, extent: int) -> int:
        return int(min(extent - 2, math.ceil(centre + r_fine) + args.cv_margin + 1))

    cv = box_control_volume(
        tuple(fg),
        x0=_cv_lo(fine_center[0]), x1=_cv_hi(fine_center[0], fg[2]),
        y0=_cv_lo(fine_center[1]), y1=_cv_hi(fine_center[1], fg[1]),
        z0=_cv_lo(fine_center[2]), z1=_cv_hi(fine_center[2], fg[0]),
        device=device,
    )
    # Enclosure guard: the fine control volume must fully contain the fine sphere.
    cv_bounds = {
        "x": (int(_cv_lo(fine_center[0])), int(_cv_hi(fine_center[0], fg[2]))),
        "y": (int(_cv_lo(fine_center[1])), int(_cv_hi(fine_center[1], fg[1]))),
        "z": (int(_cv_lo(fine_center[2])), int(_cv_hi(fine_center[2], fg[0]))),
    }
    for name, (cv0, cv1) in cv_bounds.items():
        axis = 0 if name == "x" else 1 if name == "y" else 2
        centre = fine_center[axis]
        if cv0 > math.floor(centre - r_fine) or \
           cv1 < math.ceil(centre + r_fine) + 1:
            raise ValueError(
                f"fine control volume does not enclose the sphere along {name} "
                f"(cv=[{cv0},{cv1})); enlarge the fine block via --wake-cells / "
                f"--shell-margin or shrink --cv-margin",
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
            before = f
            collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
            post_collision = torch.where(solid_coarse_q, before, collided)
            out = stream3d(post_collision)
            out = non_equilibrium_far_field_bc_3d(out, u_in=args.lattice_speed)
            out = apply_equilibrium_difference_sponge(
                out, sigma, velocity_target=(args.lattice_speed, 0.0, 0.0),
            )
            out = non_equilibrium_far_field_bc_3d(out, u_in=args.lattice_speed)
            return AMRAdvanceResult(out, collided)

        before = f
        collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
        post_collision = torch.where(solid_q, before, collided)
        out = stream3d(post_collision)
        rho_post, ux_post, uy_post, uz_post = macroscopic3d(post_collision)
        activation = _ramp(current_step, args.ramp_steps)
        wall_velocity = (
            (1.0 - activation) * ux_post,
            (1.0 - activation) * uy_post,
            (1.0 - activation) * uz_post,
        )
        out, _bfl_force = bouzidi_bounce_back_d3q19(
            out, post_collision, bfl_mask, bfl_q,
            wall_velocity=wall_velocity, wall_density=rho_post,
            return_force=True,
        )
        if substep == 1:  # one sample per coarse step
            cv_force = float(observe_control_volume_force(
                before, out, post_collision, cv, solid=fine_solid_g,
            ).force_on_body[0].item())
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

    statistics_window = args.statistics_window_steps or len(force_samples)
    selected = force_samples[-statistics_window:]
    mean_force = sum(selected) / len(selected)
    cd = mean_force / dynamic_area
    reference = schiller_naumann_cd(args.reynolds)
    cd_history = [f_ / dynamic_area for f_ in selected]
    stationarity = assess_force_stationarity(
        cd_history, block_size=max(1, len(cd_history) // 8),
    )
    stationarity_dict = (
        asdict(stationarity)
        if hasattr(stationarity, "__dataclass_fields__")
        else stationarity
    )
    reference_error = abs(cd - reference) / reference * 100.0
    result = {
        "schema": "tensorlbm-sphere-shell-amr-cv-v1",
        "status": "measured_candidate",
        "physical_validation": False,
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
            "refine_cells_coarse": int(refine_mask.sum().item()),
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(result["result"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
