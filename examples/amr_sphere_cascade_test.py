#!/usr/bin/env python3
"""Two-level nested body-fitted AMR sphere drag validation.

Hierarchy (NestedStaticBlockAMR3D, both interfaces forced reflux):

  level 0  root coarse grid            — external domain, far-field BC, sponge
  level 1  L1 block (2:1)              — encloses sphere + wake, BFL sphere at
                                         analytically resolved radius R*2
  level 2  L2 surface-hugging shell    — thin fluid shell hugging the sphere
      (2:1)                              surface, BFL sphere at radius R*4

The L2 box is defined in the L1 fine (with-ghost) coordinate system and only
covers the sphere surface neighbourhood.  Control-volume drag is observed on
the finest level (amr.finest_f, the L2 tensor including its ghost layer).
The result is compared against Schiller-Naumann to test whether the nested
shell recovers the uniform-grid R32 accuracy (reference uniform R16: ~4.45%).

Coordinate conventions (mirroring examples/amr_sphere_drag_validate.py):

  * interface ``i``'s box lives in its parent tensor's allocated coordinates;
    the parent of interface 1 is L1's fine tensor *with* its one-cell ghost
    layer, so the L2 box is expressed in L1 with-ghost indices.
  * sphere centre on the L1 fine grid: fc1 = (cx*2 - box.x0*2 + g, ...).
  * sphere centre in L1 with-ghost coordinates: c1_w = fc1 + g (this is the
    centre of the frozen-solid sphere in the L1 fine tensor).
  * sphere centre on the L2 fine grid (same conversion, one more level):
    fc2 = (c1_w*2 - box2.x0*2 + g, ...), radius R*4.
  * BFL q-fields are computed on each level's with-ghost tensor using that
    level's fine centre (same convention as the single-level reference).
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
from tensorlbm.cascaded_collision import collide_cascaded_d3q19
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
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)

RATIO = 2
GHOST = 1


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
    p.add_argument("--wall-margin", type=int, default=16)  # L1 block margin
    p.add_argument("--wake-cells", type=int, default=45)  # L1 wake extension
    p.add_argument("--l2-margin", type=int, default=8)  # L2 shell thickness
    p.add_argument("--shell-margin", type=int, default=8)  # L1 body-fitted shell thickness
    p.add_argument(
        "--collision",
        choices=("cumulant", "cascaded"),
        default="cumulant",
    )
    p.add_argument("--output", required=True)
    return p


def _clamp_axis(lo: int, hi: int, limit: int, axis: str) -> tuple[int, int]:
    """Keep the L2 box strictly interior to its parent (L1 with-ghost) tensor."""
    lo = max(1, lo)
    hi = min(limit - 2, hi)
    if hi - lo < 3:
        raise ValueError(
            f"L2 box degenerate on {axis} axis ({lo}, {hi}) in parent limit {limit} "
            "-- reduce --l2-margin or enlarge the L1 block",
        )
    return lo, hi


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    shape = (args.nz, args.ny, args.nx)
    cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0

    # ------------------------------------------------------------------
    # level 0 (root) geometry
    # ------------------------------------------------------------------
    solid_coarse = sphere_mask(
        args.nx,
        args.ny,
        args.nz,
        cx,
        cy,
        cz,
        args.radius,
        device=device,
    )
    solid_coarse_q = solid_coarse.unsqueeze(0).expand(19, *shape).contiguous()

    # ---- L1 block: body-fitted shell (surface-proximity) + downstream wake,
    # ---- instead of the full sphere + uniform margin. Saves most of the
    # ---- fine cells far from the body while keeping the sphere surface and
    # ---- wake refined.
    shell_mask = HullProximityRegion(
        solid_coarse,
        margin=args.shell_margin,
    ).expand_mask()
    wake_mask = WakeRegion(
        solid_coarse,
        extend_x=args.wake_cells,
    ).expand_mask()
    shell_idx = shell_mask.nonzero(as_tuple=False)
    if shell_idx.numel() == 0:
        raise ValueError(
            "empty shell refinement region; adjust --shell-margin",
        )
    sz0, sy0 = int(shell_idx[:, 0].min().item()), int(shell_idx[:, 1].min().item())
    sz1, sy1 = int(shell_idx[:, 0].max().item()), int(shell_idx[:, 1].max().item())
    wake_mask[:sz0, :, :] = False
    wake_mask[sz1 + 1 :, :, :] = False
    wake_mask[:, :sy0, :] = False
    wake_mask[:, sy1 + 1 :, :] = False
    refine_mask = shell_mask | wake_mask
    r_indices = refine_mask.nonzero(as_tuple=False)
    if r_indices.numel() == 0:
        raise ValueError(
            "empty shell+wake refinement region; adjust --shell-margin/--wake-cells",
        )
    z_min, y_min, x_min = (int(r_indices[:, a].min().item()) for a in range(3))
    z_max, y_max, x_max = (int(r_indices[:, a].max().item()) + 1 for a in range(3))
    pad = args.wall_margin
    x0 = max(1, x_min - pad)
    x1 = min(args.nx - 1, x_max + pad)
    y0 = max(1, y_min - pad)
    y1 = min(args.ny - 1, y_max + pad)
    z0 = max(1, z_min - pad)
    z1 = min(args.nz - 1, z_max + pad)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError("L1 shell refinement box too small")
    box1 = BoxRegion(x0, x1, y0, y1, z0, z1)

    nu_coarse = args.lattice_speed * (2.0 * args.radius) / args.reynolds
    tau_coarse = 0.5 + 3.0 * nu_coarse

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ------------------------------------------------------------------
    # level 1 (interface 0): L1 block, sphere at R*2
    # ------------------------------------------------------------------
    s1 = (
        (z1 - z0) * RATIO,
        (y1 - y0) * RATIO,
        (x1 - x0) * RATIO,
    )  # physical (nz, ny, nx), no ghost
    fc1 = (
        cx * RATIO - x0 * RATIO + GHOST,
        cy * RATIO - y0 * RATIO + GHOST,
        cz * RATIO - z0 * RATIO + GHOST,
    )
    radius1 = args.radius * RATIO
    l1_solid = sphere_mask(
        s1[2],
        s1[1],
        s1[0],
        fc1[0],
        fc1[1],
        fc1[2],
        radius1,
        device=device,
    )
    l1_solid_g = torch.zeros(
        tuple(size + 2 * GHOST for size in s1),
        dtype=torch.bool,
        device=device,
    )
    l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l1_solid
    solid_q1 = l1_solid_g.unsqueeze(0).expand(19, *l1_solid_g.shape).contiguous()
    bfl_mask1, bfl_q1 = compute_q_sphere(
        l1_solid_g.shape[2],
        l1_solid_g.shape[1],
        l1_solid_g.shape[0],
        fc1[0],
        fc1[1],
        fc1[2],
        radius1,
        device=device,
    )

    # ------------------------------------------------------------------
    # level 2 (interface 1): surface-hugging shell in L1 fine coordinates
    # ------------------------------------------------------------------
    # Sphere centre in the L1 fine *with-ghost* tensor: the frozen-solid
    # sphere of L1 sits at fc1 in physical local coordinates, i.e. fc1 + g in
    # the with-ghost tensor used as this interface's parent.
    c1_w = tuple(value + GHOST for value in fc1)
    s1g = l1_solid_g.shape  # (nz, ny, nx) of the L1 with-ghost tensor
    half2 = int(math.floor(radius1 + args.l2_margin))
    x0_2, x1_2 = _clamp_axis(
        int(math.floor(c1_w[0] - half2)),
        int(math.ceil(c1_w[0] + half2)),
        s1g[2],
        "x",
    )
    y0_2, y1_2 = _clamp_axis(
        int(math.floor(c1_w[1] - half2)),
        int(math.ceil(c1_w[1] + half2)),
        s1g[1],
        "y",
    )
    z0_2, z1_2 = _clamp_axis(
        int(math.floor(c1_w[2] - half2)),
        int(math.ceil(c1_w[2] + half2)),
        s1g[0],
        "z",
    )
    box2 = BoxRegion(x0_2, x1_2, y0_2, y1_2, z0_2, z1_2)
    s2 = (
        (z1_2 - z0_2) * RATIO,
        (y1_2 - y0_2) * RATIO,
        (x1_2 - x0_2) * RATIO,
    )  # physical (nz, ny, nx), no ghost
    fc2 = (
        2.0 * c1_w[0] - 2.0 * x0_2 + GHOST,
        2.0 * c1_w[1] - 2.0 * y0_2 + GHOST,
        2.0 * c1_w[2] - 2.0 * z0_2 + GHOST,
    )
    radius2 = args.radius * RATIO * RATIO
    l2_solid = sphere_mask(
        s2[2],
        s2[1],
        s2[0],
        fc2[0],
        fc2[1],
        fc2[2],
        radius2,
        device=device,
    )
    l2_solid_g = torch.zeros(
        tuple(size + 2 * GHOST for size in s2),
        dtype=torch.bool,
        device=device,
    )
    l2_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l2_solid
    solid_q2 = l2_solid_g.unsqueeze(0).expand(19, *l2_solid_g.shape).contiguous()
    bfl_mask2, bfl_q2 = compute_q_sphere(
        l2_solid_g.shape[2],
        l2_solid_g.shape[1],
        l2_solid_g.shape[0],
        fc2[0],
        fc2[1],
        fc2[2],
        radius2,
        device=device,
    )

    # ------------------------------------------------------------------
    # tau chain: interface i+1's tau_coarse must equal interface i's tau_fine
    # ------------------------------------------------------------------
    config1 = StaticBlockAMRConfig(
        box1,
        tau_coarse=tau_coarse,
        reflux=True,
        ghost_interpolation=args.ghost_interpolation,
    )
    config2 = StaticBlockAMRConfig(
        box2,
        tau_coarse=config1.tau_fine,
        reflux=True,
        ghost_interpolation=args.ghost_interpolation,
    )
    tau_fine2 = config2.tau_fine

    amr = NestedStaticBlockAMR3D(
        coarse_f,
        (config1, config2),
        fine_solids=(l1_solid, l2_solid),
    )
    level_shapes = tuple(level.shape for level in amr.level_populations)
    if len(set(level_shapes)) != 3:
        raise ValueError(
            "level population shapes must be distinct for shape-based advance "
            f"dispatch, got {level_shapes}",
        )
    level0_shape, level1_shape, level2_shape = level_shapes

    # ------------------------------------------------------------------
    # control volume on the finest level (L2 with-ghost tensor)
    # ------------------------------------------------------------------
    fg = l2_solid_g.shape
    cv = box_control_volume(
        (fg[0], fg[1], fg[2]),
        x0=int(math.floor(fc2[0] - radius2)) - args.cv_margin,
        x1=int(math.ceil(fc2[0] + radius2)) + args.cv_margin + 1,
        y0=int(math.floor(fc2[1] - radius2)) - args.cv_margin,
        y1=int(math.ceil(fc2[1] + radius2)) + args.cv_margin + 1,
        z0=int(math.floor(fc2[2] - radius2)) - args.cv_margin,
        z1=int(math.ceil(fc2[2] + radius2)) + args.cv_margin + 1,
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
    dynamic_area = 0.5 * args.lattice_speed**2 * math.pi * radius2**2

    print(
        f"coarse={list(shape)} L1_box={[x0, x1, y0, y1, z0, z1]} "
        f"L1_shape={list(s1)} fc1={tuple(fc1)} R1={radius1} "
        f"L2_box_l1fine={[x0_2, x1_2, y0_2, y1_2, z0_2, z1_2]} "
        f"L2_shape={list(s2)} fc2={tuple(fc2)} R2={radius2} "
        f"tau=[{tau_coarse:.6f},{config1.tau_fine:.6f},{tau_fine2:.6f}] "
        f"Re={args.reynolds} cell_saving={amr.cell_saving_fraction:.3f}",
        flush=True,
    )

    force_samples: list[float] = []
    max_reflux_residual = 0.0
    reflux_residual_by_level = [0.0, 0.0]
    started = time.time()

    def _level_index(f: torch.Tensor) -> int:
        if f.shape == level0_shape:
            return 0
        if f.shape == level1_shape:
            return 1
        if f.shape == level2_shape:
            return 2
        raise ValueError(f"advance received unexpected population shape {tuple(f.shape)}")

    def advance(
        f: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        nonlocal max_reflux_residual
        level_index = _level_index(f)
        if level != level_index:
            raise ValueError(
                f"runtime level {level} disagrees with shape-derived level {level_index}",
            )
        if level_index == 0:
            # root: coarse sphere frozen, far-field + sponge (patch-free region)
            before = f
            if args.collision == "cascaded":
                collided = collide_cascaded_d3q19(f, tau)
            else:
                collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
            post_collision = torch.where(solid_coarse_q, before, collided)
            out = stream3d(post_collision)
            out = non_equilibrium_far_field_bc_3d(out, u_in=args.lattice_speed)
            out = apply_equilibrium_difference_sponge(
                out,
                sigma,
                velocity_target=(args.lattice_speed, 0.0, 0.0),
            )
            out = non_equilibrium_far_field_bc_3d(out, u_in=args.lattice_speed)
            return AMRAdvanceResult(out, post_collision)

        before = f
        if args.collision == "cascaded":
            collided = collide_cascaded_d3q19(f, tau)
        else:
            collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
        if level_index == 1:
            post_collision = torch.where(solid_q1, before, collided)
            bfl_mask, bfl_q, solid_g = bfl_mask1, bfl_q1, l1_solid_g
        else:
            post_collision = torch.where(solid_q2, before, collided)
            bfl_mask, bfl_q, solid_g = bfl_mask2, bfl_q2, l2_solid_g
        out = stream3d(post_collision)
        rho_post, ux_post, uy_post, uz_post = macroscopic3d(post_collision)
        activation = _ramp(current_step, args.ramp_steps)
        wall_velocity = (
            (1.0 - activation) * ux_post,
            (1.0 - activation) * uy_post,
            (1.0 - activation) * uz_post,
        )
        out, _bfl_force = bouzidi_bounce_back_d3q19(
            out,
            post_collision,
            bfl_mask,
            bfl_q,
            wall_velocity=wall_velocity,
            wall_density=rho_post,
            return_force=True,
        )
        if level_index == 2 and substep == 0:
            # one control-volume sample per root step, on the finest level
            cv_force = float(
                observe_control_volume_force(
                    before,
                    out,
                    post_collision,
                    cv,
                    solid=solid_g,
                )
                .force_on_body[0]
                .item()
            )
            if not math.isfinite(cv_force):
                raise FloatingPointError(f"non-finite control-volume force at step {current_step}")
            force_samples.append(cv_force)
        return AMRAdvanceResult(out, post_collision)

    current_step = 0
    for current_step in range(1, args.steps + 1):
        ledgers = amr.step(advance)
        for index, ledger in enumerate(ledgers):
            residual = float(ledger.residual.abs().max().item())
            reflux_residual_by_level[index] = max(reflux_residual_by_level[index], residual)
        max_reflux_residual = max(max_reflux_residual, *reflux_residual_by_level)
        if current_step % args.report_interval == 0:
            if not all(bool(torch.isfinite(level).all()) for level in amr.level_populations):
                raise FloatingPointError(f"non-finite populations at step {current_step}")
            recent = force_samples[-min(len(force_samples), args.report_interval) :]
            recent_cd = sum(recent) / len(recent) / dynamic_area if recent else math.nan
            elapsed = time.time() - started
            print(
                f"step={current_step}/{args.steps} recent_Cd={recent_cd:.6f} "
                f"steps/s={current_step / elapsed:.2f} "
                f"max_ref_res={max_reflux_residual:.2e} "
                f"ref_res_by_level={[f'{r:.2e}' for r in reflux_residual_by_level]}",
                flush=True,
            )

    if not all(bool(torch.isfinite(level).all()) for level in amr.level_populations):
        raise FloatingPointError("non-finite populations at end of run")

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
        "schema": "tensorlbm-sphere-shell-l2-amr-cv-v1",
        "status": "measured_candidate",
        "physical_validation": False,
        "case": (
            f"nested AMR sphere Re={args.reynolds}: coarse {list(shape)} + "
            f"L1 block {[x0, x1, y0, y1, z0, z1]} (R{args.radius * RATIO}) + "
            f"L2 surface shell {[x0_2, x1_2, y0_2, y1_2, z0_2, z1_2]} in L1 "
            f"fine coords (R{radius2}), BFL sphere + control-volume drag"
        ),
        "configuration": {
            "coarse_shape_zyx": list(shape),
            "reynolds": args.reynolds,
            "lattice_speed": args.lattice_speed,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "cv_margin": args.cv_margin,
            "ghost_interpolation": args.ghost_interpolation,
            "wall_margin": args.wall_margin,
            "wake_cells": args.wake_cells,
            "l2_margin": args.l2_margin,
            "ratio": RATIO,
            "levels": [
                {
                    "level": 0,
                    "box": None,
                    "radius": args.radius,
                    "tau": tau_coarse,
                },
                {
                    "level": 1,
                    "box_l0_coarse_coords": [x0, x1, y0, y1, z0, z1],
                    "fine_shape_zyx": list(s1),
                    "fine_center_physical": list(fc1),
                    "radius": radius1,
                    "tau": config1.tau_fine,
                },
                {
                    "level": 2,
                    "box_l1_fine_coords": [x0_2, x1_2, y0_2, y1_2, z0_2, z1_2],
                    "fine_shape_zyx": list(s2),
                    "fine_center_physical": list(fc2),
                    "radius": radius2,
                    "tau": tau_fine2,
                },
            ],
            "cell_saving_fraction": amr.cell_saving_fraction,
        },
        "result": {
            "cd_control_volume": cd,
            "reference_cd": reference,
            "reference_error_pct": reference_error,
            "mean_force_lu": mean_force,
            "dynamic_area_lu2": dynamic_area,
            "stationarity": stationarity_dict,
            "max_reflux_residual": max_reflux_residual,
            "max_reflux_residual_by_level": reflux_residual_by_level,
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
