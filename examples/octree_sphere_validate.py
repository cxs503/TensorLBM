#!/usr/bin/env python3
"""Hybrid octree-boundary sphere drag validation (P3 acceptance).

Hierarchy (design doc ``docs/octree-boundary-design.md`` §3.0):

  level 0  L0 coarse global grid          — NestedStaticBlockAMR3D root,
                                           far-field BC + sponge, frozen
                                           coarse sphere (no BFL)
  level 1  L1 rectangular fine block      — 2:1 nested block (BFL disabled:
                                           the octree shell owns the wall),
                                           frozen sphere at radius 2R
  level 2  octree boundary shell          — body-fitted leaves embedded in
                                           the L1 block, gather-based Bouzidi
                                           BFL at leaf resolution (R*4)

Per root step the L1 block advances ``ratio`` substeps (NestedStaticBlock
AMR3D), then the shell advances ``2^d_max`` lockstep substeps with
time-lerped ghost fill, gather streaming, gather BFL and a per-leaf-weighted
momentum-exchange force; the shell is restricted back into the covered L1
cells and the kinetic-flux reflux ledger is applied on the L1 side
(``step_octree_shell``).  The control-volume drag is observed once per root
step on the L1 block with a fail-closed clearance gate (the CV surface must
not intersect the shell interface, the AMR interface filter shell or the
body).

P3 CV closure (``scripts/diag_cv_components.py`` / ``diag_momentum_audit.py``):
the CV momentum change is sampled on a snapshot of the live L1 tensor taken
right after the block's own advance and *before* ``step_octree_shell``'s
in-place restriction/reflux rewrites, and the per-root-step leaf wall force
(MEM, converted to L1 units) is re-added explicitly:

``F_cv = import - dP_L1advance + wall_mom_l1``

Counting the restriction + reflux deltas as fluid momentum change
double-counts the wall transfer and overestimates Cd_cv (historically
Cd_cv ~ 5-9 vs Cd_mem ~ 1.3); dropping them without the wall term reverses
the sign.  The two closures then agree to the acceptance tolerance.

Acceptance (design doc §4, P3):
  * Cd vs Schiller-Naumann (Re=100 -> 1.0917) within 2%;
  * MEM (leaf-lattice, substep-averaged) vs CV (L1 lattice) within 5%;
  * interface-position invariance: shifting the shell/wall by one leaf cell
    changes Cd by < 1% (``--check-invariance``).

Example (CPU):
  PYTHONPATH=src .venv/bin/python examples/octree_sphere_validate.py \
      --device cpu --nx 96 --ny 64 --nz 64 --radius 6 --reynolds 100 \
      --steps 1000 --warmup-steps 300 --ramp-steps 100 --output out.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cascaded_collision import collide_cascaded_d3q19
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather,
    bfl_ramp_wall_velocity,
    leaf_force_weights,
)
from tensorlbm.octree_boundary.force import (
    ShellForceLedger,
    build_shell_control_volume,
)
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    step_octree_shell,
)
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry,
    build_sphere_geometry,
    root_advance,
    summarize_force_history,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)

Q = 19
GHOST = 1
RATIO = 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--ramp-steps", type=int, default=100)
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--cv-margin", type=int, default=6)
    p.add_argument("--wall-margin", type=int, default=8)      # L1 box padding
    p.add_argument("--shell-margin", type=int, default=6)     # L1 shell margin
    p.add_argument("--wake-cells", type=int, default=32)      # L1 wake extent
    p.add_argument("--bl-thickness", type=float, default=3.0)  # shell band
    p.add_argument("--d-max", type=int, default=1)
    p.add_argument("--collision", choices=("cumulant", "cascaded"),
                   default=None,
                   help="Explicit collision (no LES). Omit to use LES dispatch "
                        "(--les-model) for high-Re runs.")
    p.add_argument("--les-model", choices=("wale", "smagorinsky"), default="wale")
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument("--ghost-interpolation", choices=("injection", "trilinear"),
                   default="injection")
    p.add_argument("--report-interval", type=int, default=100)
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument("--interface-shift", type=float, default=0.0,
                   help="sphere-centre shift in COARSE cells (0.25 = one "
                        "leaf cell / half an L1 cell)")
    p.add_argument("--check-invariance", action="store_true",
                   help="also run with a one-leaf-cell shift and report the "
                        "Cd change (acceptance: < 1%)")
    p.add_argument("--output", required=True)
    return p


def _collide_dispatch(f: torch.Tensor, tau: float, collision: str,
                      les_model: str = "wale", cs_smag: float = 0.05,
                      cw_wale: float = 0.5) -> torch.Tensor:
    if collision == "cascaded":
        return collide_cascaded_d3q19(f, tau)
    if collision == "cumulant":
        return collide_cumulant_d3q19(f, tau, C_s=0.0)
    # LES dispatch (collision is None)
    if les_model == "wale":
        from tensorlbm.turbulence import collide_wale_mrt3d
        return collide_wale_mrt3d(f, tau, C_w=cw_wale)
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    return collide_smagorinsky_mrt3d(f, tau, C_s=cs_smag)


def _schiller_naumann(reynolds: float) -> float:
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds ** 0.687)


def run_case(
    args: argparse.Namespace,
    center_offset: tuple[float, float, float],
    label: str,
) -> dict:
    """One full hybrid run (L0 + L1 block + octree shell) for a sphere."""
    device = torch.device(args.device)
    shape = (args.nz, args.ny, args.nx)
    cx, cy, cz = (
        args.nx * 0.5 + center_offset[0],
        args.ny / 2.0 + center_offset[1],
        args.nz / 2.0 + center_offset[2],
    )
    radius_coarse = args.radius
    nu = args.lattice_speed * (2.0 * radius_coarse) / args.reynolds
    tau_coarse = 0.5 + 3.0 * nu

    # ---- level 0 (root) ----------------------------------------------------
    solid_coarse, solid_coarse_q = build_sphere_geometry(
        args.nx, args.ny, args.nz, cx, cy, cz, radius_coarse, device,
    )
    plan = plan_body_shell_box(
        solid_coarse, args.shell_margin, args.wake_cells, pad=args.wall_margin,
    )
    box1 = plan.box
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ---- level 1 (L1 rectangular block) ------------------------------------
    s1, fc1, radius1, _l1 = build_fine_block_geometry(
        box1, (cx, cy, cz), radius_coarse, RATIO, GHOST, device,
    )
    nz1, ny1, nx1 = s1
    config1 = StaticBlockAMRConfig(
        box1, tau_coarse=tau_coarse, reflux=True,
        ghost_interpolation=args.ghost_interpolation,
    )
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    l1_fine = amr.interfaces[0].fine_f                 # with-ghost tensor

    # ---- octree shell on the L1 physical grid ------------------------------
    # The sphere centre in L1 *physical* coordinates is the with-ghost centre
    # shifted by the ghost width; the shell's own `_solid` (computed at this
    # physical centre) is the consistent L1 freeze mask (the FineSphere solid
    # is evaluated at the with-ghost centre and sits one cell off).
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    radius_l1 = radius1
    octree = build_octree_shell(
        s1, phys_center, radius_l1,
        bl_thickness_cells=args.bl_thickness, d_max=args.d_max,
        transition=1, device=device,
    )
    shell_band = octree.meta["delta_mask"]
    host = octree.leaf_host_cell
    octree.f_leaf = l1_fine[
        :, host[:, 0] + GHOST, host[:, 1] + GHOST, host[:, 2] + GHOST
    ].clone()
    leaf_weights = leaf_force_weights(octree)
    ghost_plan = build_ghost_plan(octree, s1)
    dx_leaf = 2.0 ** (-octree.d_max)
    dt_leaf = dx_leaf  # convective scaling

    # ---- L1 freeze masks (from the octree solid at the physical centre) ----
    assert octree._solid is not None, "shell builder must provide the solid mask"
    l1_solid_phys = octree._solid
    l1_solid_g = torch.zeros(
        (nz1 + 2 * GHOST, ny1 + 2 * GHOST, nx1 + 2 * GHOST),
        dtype=torch.bool, device=device,
    )
    l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l1_solid_phys
    l1_solid_q = l1_solid_g.unsqueeze(0).expand(Q, *l1_solid_g.shape).contiguous()

    # ---- control volume (fail-closed clearance gate) -----------------------
    filter_shell = amr.interfaces[0]._interface_filter_blend
    assert octree._shell_mask is not None, "shell builder must provide the mask"
    cv_w = build_shell_control_volume(
        (int(l1_solid_g.shape[0]), int(l1_solid_g.shape[1]), int(l1_solid_g.shape[2])),
        fc1, radius_l1, shell_band, args.cv_margin,
        covered=octree._shell_mask, filter_shell=filter_shell,
        solid=l1_solid_g, device=device,
    )
    cv = cv_w[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]   # physical slice

    # ---- sponge + dynamic area ---------------------------------------------
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(
        shape, width=args.sponge_width, max_strength=args.sponge_strength,
        device=device, faces=sponge_faces,
    )
    dynamic_area_cv = 0.5 * args.lattice_speed ** 2 * math.pi * radius_l1 ** 2
    radius_leaf = radius_l1 / dx_leaf
    dynamic_area_mem = (
        0.5 * args.lattice_speed ** 2 * math.pi * radius_leaf ** 2
    )

    # ---- per-root-step state -------------------------------------------------
    l1_posts: list[torch.Tensor] = []
    mem_samples: list[float] = []
    cv_samples: list[float] = []
    ledger = ShellForceLedger(octree)
    max_reflux_residual = 0.0
    joint_mass0: float | None = None
    joint_mass_end: float | None = None
    current_step = 0
    started = time.time()

    def advance(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        nonlocal current_step
        if level == 0:
            out, post_collision, _ = root_advance(
                f, tau, solid_coarse_q, sigma, args.lattice_speed,
                collision=args.collision,
            )
            return AMRAdvanceResult(out, post_collision)
        if level == 1:
            before = f
            collided = _collide_dispatch(
            f, tau, args.collision,
            les_model=args.les_model, cs_smag=args.cs_smag,
            cw_wale=args.cw_wale,
        )
            post = torch.where(l1_solid_q, before, collided)
            out = stream3d(post)
            l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
            return AMRAdvanceResult(out, post)
        raise ValueError(f"unexpected hierarchy level {level}")

    def shell_advance(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        collided = _collide_dispatch(
            f.view(Q, 1, 1, -1), tau, args.collision,
            les_model=args.les_model, cs_smag=args.cs_smag,
            cw_wale=args.cw_wale,
        )
        post = collided.view_as(f)
        return AMRAdvanceResult(post.clone(), post)

    def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
        rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
            octree_, post, current_step, args.ramp_steps,
        )
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
            wall_velocity=(uwx, uwy, uwz), wall_density=rho_w,
            force_weights=leaf_weights, return_force=True,
        )

    def joint_mass() -> float:
        assert octree._shell_mask is not None
        covered = octree._shell_mask
        l1_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
        exterior = float(l1_phys[:, ~covered].sum().item())
        shell = float(
            (octree.f_leaf.sum(dim=0) * octree.leaf_volume().to(device)).sum().item()
        )
        return exterior + shell

    joint_mass0 = joint_mass()

    for current_step in range(1, args.steps + 1):
        # StaticBlockAMR3D.step() rebinds `fine_f` to a fresh tensor after
        # each advance (streaming/collision return new tensors), so the
        # cached ``l1_fine`` reference goes stale after the first root step.
        # Re-fetch the live tensor before and after the step — otherwise the
        # shell stepping, restriction/reflux and the CV observation all
        # operate on a frozen pre-advance state and the wall never couples
        # into the L1 flow field (CV force ~ 0, flow through the sphere).
        l1_fine = amr.interfaces[0].fine_f
        l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
        l1_posts.clear()
        ledgers = amr.step(advance)
        l1_fine = amr.interfaces[0].fine_f
        l1_f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
        # P3 CV fix (scripts/diag_cv_components.py): snapshot the live L1
        # tensor right after the block's own advance and BEFORE
        # step_octree_shell rewrites it in place (restriction + reflux).
        # The CV momentum change must exclude those in-place projection
        # deltas (they double-count the wall transfer); the wall momentum is
        # re-added explicitly from the force ledger (wall_momentum_l1), so
        #   F_cv = import - dP_L1advance + wall_mom_l1.
        l1_pre_shell = l1_f_phys.clone()
        shell_ledger = step_octree_shell(
            octree, shell_advance, l1_old_phys, l1_f_phys,
            tau_coarse=config1.tau_fine,
            l1_post=l1_posts if config1.reflux else None,
            shell_level=1, ghost_plan=ghost_plan,
            bfl_fn=bfl_callback, force_ledger=ledger,
        )
        max_reflux_residual = max(
            max_reflux_residual, abs(shell_ledger.mass_residual),
        )
        ledger.observe_cv_force(
            l1_old_phys, l1_pre_shell, l1_posts, cv, solid=l1_solid_phys,
            wall_mom_l1=ledger.wall_momentum_l1(dx_leaf, dt_leaf),
        )
        if current_step > args.warmup_steps:
            mem_samples.append(float(ledger.mem_force[0].item()))
            assert ledger.cv_force is not None
            cv_samples.append(float(ledger.cv_force[0].item()))
        ledger.reset()
        if not bool(torch.isfinite(l1_fine).all()):
            raise FloatingPointError(
                f"non-finite L1 populations at step {current_step}",
            )
        if not bool(torch.isfinite(octree.f_leaf).all()):
            raise FloatingPointError(
                f"non-finite shell populations at step {current_step}",
            )
        if current_step % args.report_interval == 0 and mem_samples:
            recent_mem = mem_samples[-args.report_interval:]
            recent_cv = cv_samples[-args.report_interval:]
            print(
                f"[{label}] step={current_step}/{args.steps} "
                f"Cd_mem={sum(recent_mem)/len(recent_mem)/dynamic_area_mem:.6f} "
                f"Cd_cv={sum(recent_cv)/len(recent_cv)/dynamic_area_cv:.6f} "
                f"max_ref_res={max_reflux_residual:.2e} "
                f"steps/s={current_step/(time.time()-started):.2f}",
                flush=True,
            )

    joint_mass_end = joint_mass()
    stats_window = args.statistics_window_steps or len(mem_samples)
    mem_mean = sum(mem_samples[-stats_window:]) / stats_window
    cv_mean = sum(cv_samples[-stats_window:]) / stats_window
    cd_mem = mem_mean / dynamic_area_mem
    cd_cv = cv_mean / dynamic_area_cv
    reference = _schiller_naumann(args.reynolds)
    cd_cv_error_pct = abs(cd_cv - reference) / reference * 100.0
    mem_cv_deviation_pct = (
        abs(cd_mem - cd_cv) / max(abs(cd_cv), 1e-30) * 100.0
    )
    mass_drift = (
        abs(joint_mass_end - joint_mass0) / joint_mass0 if joint_mass0 else 0.0
    )
    print(
        f"[{label}] final: Cd_mem={cd_mem:.6f} Cd_cv={cd_cv:.6f} "
        f"ref={reference:.6f} ref_err={cd_cv_error_pct:.3f}% "
        f"mem/cv={mem_cv_deviation_pct:.3f}% mass_drift={mass_drift:.2e} "
        f"max_ref_res={max_reflux_residual:.2e}",
        flush=True,
    )

    return {
        "label": label,
        "cd_mem": cd_mem,
        "cd_cv": cd_cv,
        "reference_cd": reference,
        "reference_error_pct_cv": cd_cv_error_pct,
        "reference_error_pct_mem": (
            abs(cd_mem - reference) / reference * 100.0
        ),
        "mem_cv_deviation_pct": mem_cv_deviation_pct,
        "mean_mem_force_leaf_lu": mem_mean,
        "mean_cv_force_l1_lu": cv_mean,
        "dynamic_area_cv": dynamic_area_cv,
        "dynamic_area_mem": dynamic_area_mem,
        "max_reflux_residual": max_reflux_residual,
        "joint_mass_drift": mass_drift,
        "samples": stats_window,
        "n_leaf": int(octree.n_leaf),
        "n_bfl_links": int(octree.bfl_mask.sum().item()),
        "shell_cell_saving": octree.stats.get("saving_fraction", 0.0),
        "l1_shape": list(s1),
        "l1_box_coarse": [box1.x0, box1.x1, box1.y0, box1.y1, box1.z0, box1.z1],
        "phys_center_l1": list(phys_center),
        "radius_l1": radius_l1,
        "radius_leaf": radius_leaf,
        "center_offset_coarse": list(center_offset),
    }


def main() -> None:
    args = parser().parse_args()
    if args.d_max not in (1, 2):
        raise ValueError("--d-max must be 1 or 2")
    if not 0.0 <= args.interface_shift:
        raise ValueError("--interface-shift must be non-negative")
    if args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be < --steps")

    base = run_case(args, (0.0, 0.0, 0.0), "base")
    runs = [base]
    invariance = None
    if args.check_invariance:
        shift = args.interface_shift if args.interface_shift > 0 else 0.25
        shifted = run_case(args, (shift, 0.0, 0.0), "shifted")
        runs.append(shifted)
        cd_change = (
            abs(shifted["cd_cv"] - base["cd_cv"])
            / max(abs(base["cd_cv"]), 1e-30) * 100.0
        )
        invariance = {
            "shift_coarse_cells": shift,
            "cd_base": base["cd_cv"],
            "cd_shifted": shifted["cd_cv"],
            "cd_change_pct": cd_change,
            "accepts_1pct": cd_change < 1.0,
        }
        print(f"invariance: |dCd| = {cd_change:.4f}% (accept < 1%)", flush=True)

    result = {
        "schema": "octree-hybrid-sphere-p3-v1",
        "configuration": {
            "device": args.device,
            "coarse_shape_zyx": [args.nz, args.ny, args.nx],
            "radius_coarse": args.radius,
            "reynolds": args.reynolds,
            "lattice_speed": args.lattice_speed,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "sponge_width": args.sponge_width,
            "cv_margin": args.cv_margin,
            "wall_margin": args.wall_margin,
            "shell_margin": args.shell_margin,
            "wake_cells": args.wake_cells,
            "bl_thickness": args.bl_thickness,
            "d_max": args.d_max,
            "collision": args.collision,
            "ghost_interpolation": args.ghost_interpolation,
            "statistics_window_steps": args.statistics_window_steps,
            "tau_coarse": 0.5 + 3.0 * args.lattice_speed * 2.0 * args.radius / args.reynolds,
        },
        "runs": runs,
        "invariance": invariance,
        "acceptance": {
            "cd_within_2pct": runs[0]["reference_error_pct_cv"] <= 2.0,
            "mem_cv_within_5pct": runs[0]["mem_cv_deviation_pct"] <= 5.0,
            "invariance_within_1pct": (
                invariance is not None and invariance["accepts_1pct"]
            ),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)
    print(
        f"ACCEPTANCE: Cd<=2%: {result['acceptance']['cd_within_2pct']} "
        f"| MEM/CV<=5%: {result['acceptance']['mem_cv_within_5pct']} "
        f"| invariance<1%: {result['acceptance']['invariance_within_1pct']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
