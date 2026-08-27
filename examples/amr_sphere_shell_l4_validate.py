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
from pathlib import Path

import torch

from tensorlbm.amr_checkpoint import (
    build_checkpoint_signature,
    resume_amr_checkpoint,
    save_amr_checkpoint,
)
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.evidence_io import common_schema_fields, write_evidence
from tensorlbm.sphere_amr_common import (
    build_control_volume,
    build_fine_block_geometry,
    build_l2_shell_geometry,
    build_sphere_geometry,
    distinct_level_shapes,
    fine_sphere_advance,
    level_index_of,
    root_advance,
    summarize_force_history,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)

RATIO = 2
GHOST = 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--devices",
        default=None,
        help="comma-separated per-interface devices for L1,L2,L3 "
        "(e.g. 'cuda:0,cuda:1,cuda:1'). Root runs on --device. "
        "Omit to keep everything on --device.",
    )
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=112)
    p.add_argument("--nz", type=int, default=112)
    p.add_argument("--radius", type=float, default=8.0)  # coarse-grid sphere radius
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument(
        "--resolved-reynolds",
        type=float,
        default=None,
        help="Reynolds used for the lattice collision (stability). When "
        "omitted, equals --reynolds. Set lower than --reynolds to keep the "
        "collision in the stable regime while the wall function still uses "
        "the physical --reynolds (standard high-Re LBM practice, same as "
        "SUBOFF runner).",
    )
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
        default=None,
        help="Explicit collision operator (no LES). When omitted, falls "
        "through to --les-model (WALE/Smagorinsky) for high-Re runs.",
    )
    p.add_argument(
        "--les-model",
        choices=("wale", "smagorinsky"),
        default="wale",
        help="LES subgrid model used when --collision is omitted",
    )
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument(
        "--lattice",
        choices=("D3Q19", "D3Q27"),
        default="D3Q19",
        help="lattice stencil (D3Q19 keeps the legacy path; D3Q27 uses the "
        "D3Q27 equilibrium/collision/stream/BFL kernels)",
    )
    p.add_argument("--checkpoint", default=None, help="checkpoint .ckpt path")
    p.add_argument(
        "--checkpoint-interval", type=int, default=0, help="save every N root steps (0=off)"
    )
    p.add_argument("--resume", action="store_true", help="resume from --checkpoint")
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

    # ------------------------------------------------------------------
    # level 0 (root) geometry
    # ------------------------------------------------------------------
    solid_coarse, solid_coarse_q = build_sphere_geometry(
        args.nx,
        args.ny,
        args.nz,
        cx,
        cy,
        cz,
        args.radius,
        device,
        lattice=args.lattice,
    )

    # ---- L1 block: body-fitted shell (surface-proximity) + downstream wake,
    # ---- instead of the full sphere + uniform margin. Saves most of the
    # ---- fine cells far from the body while keeping the sphere surface and
    # ---- wake refined.
    plan = plan_body_shell_box(
        solid_coarse,
        args.shell_margin,
        args.wake_cells,
        pad=args.wall_margin,
    )
    box1 = plan.box
    x0, x1, y0, y1, z0, z1 = box1.x0, box1.x1, box1.y0, box1.y1, box1.z0, box1.z1

    collision_re = args.resolved_reynolds or args.reynolds
    nu_coarse = args.lattice_speed * (2.0 * args.radius) / collision_re
    tau_coarse = 0.5 + 3.0 * nu_coarse

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    if args.lattice == "D3Q27":
        coarse_f = equilibrium27(rho, ux, zero, zero, device=device)
    else:
        coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ------------------------------------------------------------------
    # device plan: root on --device; L1/L2/L3 on --devices when given
    # ------------------------------------------------------------------
    fine_devices = None
    if args.devices:
        device_list = [value.strip() for value in args.devices.split(",")]
        if len(device_list) != 3:
            raise ValueError("--devices must name exactly 3 devices (L1,L2,L3)")
        fine_devices = [torch.device(value) for value in device_list]

    device1 = fine_devices[0] if fine_devices else device
    device2 = fine_devices[1] if fine_devices else device
    device3 = fine_devices[2] if fine_devices else device

    # ------------------------------------------------------------------
    # level 1 (interface 0): L1 block, sphere at R*2
    # ------------------------------------------------------------------
    s1, fc1, radius1, l1 = build_fine_block_geometry(
        box1,
        (cx, cy, cz),
        args.radius,
        RATIO,
        GHOST,
        device1,
        lattice=args.lattice,
    )
    l1_solid, l1_solid_g, solid_q1, bfl_mask1, bfl_q1 = (
        l1.solid,
        l1.solid_g,
        l1.solid_q,
        l1.bfl_mask,
        l1.bfl_q,
    )

    # ------------------------------------------------------------------
    # level 2 (interface 1): surface-hugging shell in L1 fine coordinates
    # ------------------------------------------------------------------
    # Sphere centre in the L1 fine *with-ghost* tensor: the frozen-solid
    # sphere of L1 sits at fc1 in physical local coordinates, i.e. fc1 + g in
    # the with-ghost tensor used as this interface's parent.
    c1_w = tuple(value + GHOST for value in fc1)
    s1g = l1_solid_g.shape  # (nz, ny, nx) of the L1 with-ghost tensor
    box2, s2, fc2, l2 = build_l2_shell_geometry(
        c1_w,
        s1g,
        radius1,
        args.l2_margin,
        RATIO,
        GHOST,
        device2,
        lattice=args.lattice,
    )
    x0_2, x1_2, y0_2, y1_2, z0_2, z1_2 = (
        box2.x0,
        box2.x1,
        box2.y0,
        box2.y1,
        box2.z0,
        box2.z1,
    )
    radius2 = args.radius * RATIO * RATIO
    l2_solid, l2_solid_g, solid_q2, bfl_mask2, bfl_q2 = (
        l2.solid,
        l2.solid_g,
        l2.solid_q,
        l2.bfl_mask,
        l2.bfl_q,
    )

    # ------------------------------------------------------------------
    # level 3 (interface 2): one more surface-hugging shell in L2 fine
    # coordinates (sphere surface resolution R*8)
    # ------------------------------------------------------------------
    c2_w = tuple(value + GHOST for value in fc2)
    s2g = l2_solid_g.shape  # (nz, ny, nx) of the L2 with-ghost tensor
    box3, s3, fc3, l3 = build_l2_shell_geometry(
        c2_w,
        s2g,
        radius2,
        args.l2_margin,
        RATIO,
        GHOST,
        device3,
        lattice=args.lattice,
    )
    x0_3, x1_3, y0_3, y1_3, z0_3, z1_3 = (
        box3.x0,
        box3.x1,
        box3.y0,
        box3.y1,
        box3.z0,
        box3.z1,
    )
    radius3 = args.radius * RATIO * RATIO * RATIO
    l3_solid, l3_solid_g, solid_q3, bfl_mask3, bfl_q3 = (
        l3.solid,
        l3.solid_g,
        l3.solid_q,
        l3.bfl_mask,
        l3.bfl_q,
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
    config3 = StaticBlockAMRConfig(
        box3,
        tau_coarse=config2.tau_fine,
        reflux=True,
        ghost_interpolation=args.ghost_interpolation,
    )
    tau_fine3 = config3.tau_fine

    amr = NestedStaticBlockAMR3D(
        coarse_f,
        (config1, config2, config3),
        fine_solids=(l1_solid, l2_solid, l3_solid),
        fine_devices=fine_devices,
    )
    level_shapes = distinct_level_shapes(amr.level_populations, 4)

    # ------------------------------------------------------------------
    # control volume on the finest level (L3 with-ghost tensor)
    # ------------------------------------------------------------------
    cv = build_control_volume(
        l3_solid_g.shape,
        fc3,
        radius3,
        args.cv_margin,
        device3,
    )
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=sponge_faces,
    )
    dynamic_area = 0.5 * args.lattice_speed**2 * math.pi * radius3**2

    print(
        f"coarse={list(shape)} L1_box={[x0, x1, y0, y1, z0, z1]} "
        f"L1_shape={list(s1)} fc1={tuple(fc1)} R1={radius1} "
        f"L2_box_l1fine={[x0_2, x1_2, y0_2, y1_2, z0_2, z1_2]} "
        f"L2_shape={list(s2)} fc2={tuple(fc2)} R2={radius2} "
        f"L3_box_l2fine={[x0_3, x1_3, y0_3, y1_3, z0_3, z1_3]} "
        f"L3_shape={list(s3)} fc3={tuple(fc3)} R3={radius3} "
        f"tau=[{tau_coarse:.6f},{config1.tau_fine:.6f},{config2.tau_fine:.6f},{tau_fine3:.6f}] "
        f"Re={args.reynolds} cell_saving={amr.cell_saving_fraction:.3f}",
        flush=True,
    )

    force_samples: list[float] = []
    max_reflux_residual = 0.0
    reflux_residual_by_level = [0.0, 0.0, 0.0]
    started = time.time()

    def advance(
        f: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        nonlocal max_reflux_residual
        level_index = level_index_of(f, level_shapes)
        if level != level_index:
            raise ValueError(
                f"runtime level {level} disagrees with shape-derived level {level_index}",
            )
        if level_index == 0:
            # root: coarse sphere frozen, far-field + sponge (patch-free region)
            out, post_collision, _collided = root_advance(
                f,
                tau,
                solid_coarse_q,
                sigma,
                args.lattice_speed,
                collision=args.collision,
                lattice=args.lattice,
                les_model=args.les_model,
                cs_smag=args.cs_smag,
                cw_wale=args.cw_wale,
            )
            return AMRAdvanceResult(out, post_collision)

        if level_index == 1:
            solid_q, bfl_mask, bfl_q, solid_g = (
                solid_q1,
                bfl_mask1,
                bfl_q1,
                l1_solid_g,
            )
        elif level_index == 2:
            solid_q, bfl_mask, bfl_q, solid_g = (
                solid_q2,
                bfl_mask2,
                bfl_q2,
                l2_solid_g,
            )
        else:
            solid_q, bfl_mask, bfl_q, solid_g = (
                solid_q3,
                bfl_mask3,
                bfl_q3,
                l3_solid_g,
            )
        out, post_collision, cv_force = fine_sphere_advance(
            f,
            tau,
            solid_q=solid_q,
            bfl_mask=bfl_mask,
            bfl_q=bfl_q,
            step=current_step,
            ramp_steps=args.ramp_steps,
            sample_cv=(level_index == 3 and substep == 0),
            cv=cv,
            solid_g=solid_g,
            collision=args.collision,
            lattice=args.lattice,
            les_model=args.les_model,
            cs_smag=args.cs_smag,
            cw_wale=args.cw_wale,
        )
        if level_index == 3 and substep == 0:
            # one control-volume sample per root step, on the finest level
            assert cv_force is not None
            if not math.isfinite(cv_force):
                raise FloatingPointError(f"non-finite control-volume force at step {current_step}")
            force_samples.append(cv_force)
        return AMRAdvanceResult(out, post_collision)

    checkpoint_signature = build_checkpoint_signature(
        shape=shape,
        radius=args.radius,
        reynolds=args.reynolds,
        lattice_speed=args.lattice_speed,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        ramp_steps=args.ramp_steps,
        shell_margin=args.shell_margin,
        wake_cells=args.wake_cells,
        l2_margin=args.l2_margin,
        ghost_interpolation=args.ghost_interpolation,
        collision=args.collision,
        lattice=args.lattice,
        les_model=args.les_model,
        cs_smag=args.cs_smag,
        cw_wale=args.cw_wale,
        tau_chain=[tau_coarse, config1.tau_fine, config2.tau_fine, tau_fine3],
        ratio=RATIO,
        ghost=GHOST,
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    start_step = 0
    if args.resume:
        assert checkpoint_path is not None, "--resume requires --checkpoint"
        start_step, force_samples = resume_amr_checkpoint(
            amr,
            configuration=checkpoint_signature,
            path=checkpoint_path,
        )
        print(f"resumed from checkpoint at step {start_step}", flush=True)

    current_step = 0
    for current_step in range(start_step + 1, args.steps + 1):
        ledgers = amr.step(advance)
        for index, ledger in enumerate(ledgers):
            residual = float(ledger.residual.abs().max().item())
            reflux_residual_by_level[index] = max(reflux_residual_by_level[index], residual)
        max_reflux_residual = max(max_reflux_residual, *reflux_residual_by_level)
        if (
            checkpoint_path is not None
            and args.checkpoint_interval > 0
            and current_step % args.checkpoint_interval == 0
        ):
            save_amr_checkpoint(
                amr,
                step=current_step,
                force_samples=force_samples,
                configuration=checkpoint_signature,
                path=checkpoint_path,
            )
            print(f"checkpoint saved at step {current_step}", flush=True)
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

    summary = summarize_force_history(
        force_samples,
        dynamic_area,
        args.reynolds,
        args.statistics_window_steps,
    )
    cd = summary["cd"]
    reference = summary["reference_cd"]
    reference_error = summary["reference_error_pct"]
    mean_force = summary["mean_force_lu"]
    stationarity_dict = summary["stationarity"]
    schema_prefix = (
        "sphere-shell-l3-amr-cv-v1"
        if args.lattice == "D3Q19"
        else "sphere-shell-l3-amr-cv-v1-d3q27"
    )
    result = {
        **common_schema_fields(schema_prefix),
        "case": (
            f"nested AMR sphere Re={args.reynolds}: coarse {list(shape)} + "
            f"L1 block {[x0, x1, y0, y1, z0, z1]} (R{args.radius * RATIO}) + "
            f"L2 surface shell {[x0_2, x1_2, y0_2, y1_2, z0_2, z1_2]} in L1 "
            f"fine coords (R{radius2}) + L3 surface shell "
            f"{[x0_3, x1_3, y0_3, y1_3, z0_3, z1_3]} in L2 fine coords "
            f"(R{radius3}), BFL sphere + control-volume drag"
        ),
        "configuration": {
            "coarse_shape_zyx": list(shape),
            "lattice": args.lattice,
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
                    "tau": config2.tau_fine,
                },
                {
                    "level": 3,
                    "box_l2_fine_coords": [x0_3, x1_3, y0_3, y1_3, z0_3, z1_3],
                    "fine_shape_zyx": list(s3),
                    "fine_center_physical": list(fc3),
                    "radius": radius3,
                    "tau": tau_fine3,
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
    write_evidence(result, args.output)
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
