#!/usr/bin/env python3
"""Hybrid octree-boundary sphere drag validation — multi-GPU sharded shell.

Same physics and output schema as ``octree_sphere_validate.py`` (L0 root +
L1 ``NestedStaticBlockAMR3D`` block + octree boundary shell, P3 CV closure),
with the shell leaves **sharded across ``--fine-devices``**:

  level 0  L0 coarse global grid          — root device (``--device``)
  level 1  L1 rectangular fine block      — root device
  level 2  octree boundary shell          — Morton-split across fine devices

The shell is advanced with ``step_octree_shell_sharded`` (per-substep
collision runs on each shard's own device; ghost fill, restriction, reflux
interface transfers and the MEM force are reassembled on the root device in
the global order).  See ``src/tensorlbm/octree_boundary/sharding.py`` for the
numerical-equivalence contract: state-affecting reductions are assembled in
global order and cross-device exchanges are exact copies; collision kernels
may differ by a few floating-point ulps when shard sizes differ.

One shard (``--fine-devices cuda:0``) agrees with the unsharded
``step_octree_shell`` run to roundoff on the same device; multiple shards are
checked with the same tolerance.  Leaf LES (D3Q19 or D3Q27) runs the sparse-leaf MRT closure;
the sharded stepper exchanges remote macroscopic velocities before collision,
so cross-shard and coarse/fine gradients are retained across Morton cuts.

Example (dual GPU on the supercomputer, ``CUDA_VISIBLE_DEVICES=1,2``):

  PYTHONPATH=src python -u examples/octree_multigpu_validate.py \\
      --device cuda:0 --fine-devices cuda:0,cuda:1 \\
      --nx 96 --ny 64 --nz 64 --radius 6 --reynolds 100 \\
      --steps 1000 --warmup-steps 300 --ramp-steps 100 --output out_2gpu.json

Equivalence check (single vs dual GPU, 50 steps, roundoff-level Cd):

  ... --fine-devices cuda:0    --steps 50 --warmup-steps 10 --output s1.json
  ... --fine-devices cuda:0,cuda:1 --steps 50 --warmup-steps 10 --output s2.json

High-Re dual-GPU stability (Re=1e4):

  ... --fine-devices cuda:0,cuda:1 --reynolds 10000 --lattice D3Q27 \\
      --les-model wale --q-min 0.2 --no-moving-wall \\
      --steps 400 --warmup-steps 50 --ramp-steps 100 --output hi_re_2gpu.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from tensorlbm.octree_boundary.sharding import (
    shard_octree_shell,
    shards_all_finite,
)
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    step_octree_shell_sharded,
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

Q = 19  # placeholder, set in main
GHOST = 1
RATIO = 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--fine-devices",
        default=None,
        help="comma-separated device list for the octree shell leaves "
        "(e.g. 'cuda:0,cuda:1'); one shard per device, leaves split in "
        "Morton order.  Default = the root --device (single shard, "
        "roundoff-equivalent to the unsharded shell stepper).",
    )
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument(
        "--lattice",
        choices=("D3Q19", "D3Q27"),
        default="D3Q19",
        help="lattice stencil (D3Q19 default; D3Q27 for high-Re "
        "stability, requires cascaded/cumulant d3q27 kernels)",
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--ramp-steps", type=int, default=100)
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--cv-margin", type=int, default=6)
    p.add_argument("--wall-margin", type=int, default=8)  # L1 box padding
    p.add_argument("--shell-margin", type=int, default=6)  # L1 shell margin
    p.add_argument("--wake-cells", type=int, default=32)  # L1 wake extent
    p.add_argument(
        "--bl-thickness",
        type=float,
        default=None,
        help="Shell band thickness in L1 cells. Default None = "
        "R/2 (scaled with sphere radius; validated R6->3, "
        "R8->4 giving best accuracy).",
    )
    p.add_argument("--d-max", type=int, default=1)
    p.add_argument(
        "--collision",
        choices=("cumulant", "cascaded"),
        default=None,
        help="Explicit collision (no LES). Omit to use LES dispatch "
        "(--les-model) for high-Re runs.",
    )
    p.add_argument("--les-model", choices=("wale", "smagorinsky"), default="wale")
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument(
        "--q-min",
        type=float,
        default=None,
        help="Clamp BFL q to this minimum (high-Re safeguard vs 1/(2q) "
        "divergence). None = disabled (validated low-Re default).",
    )
    p.add_argument(
        "--no-moving-wall",
        action="store_true",
        help="Disable BFL moving-wall ramp correction (fixed no-slip wall). "
        "Stable at very high Re where the (3/q)*moving_base term "
        "diverges as tau->0.5; correct for a stationary body.",
    )
    p.add_argument(
        "--no-bfl",
        action="store_true",
        help="Disable the BFL wall reconstruction entirely: the shell wall "
        "falls back to the stream_gather SOLID bounce-back (staircase "
        "no-slip at the leaf centers) and no MEM force is produced. "
        "Diagnostic isolation of the BFL contribution to the drag.",
    )
    p.add_argument("--ghost-interpolation", choices=("injection", "trilinear"), default="injection")
    p.add_argument(
        "--ghost-fallback",
        choices=("on", "off"),
        default="on",
        help="solid-host fallback in build_ghost_plan (off = "
        "diagnostic: trilinear sample over frozen-solid cells)",
    )
    p.add_argument("--report-interval", type=int, default=100)
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument(
        "--minimum-statistics-convective-times",
        type=float,
        default=5.0,
        help="Minimum trailing statistics duration in body-diameter convective "
        "times required for physical acceptance (default: 5).",
    )
    p.add_argument(
        "--interface-shift",
        type=float,
        default=0.0,
        help="sphere-centre shift in COARSE cells (0.25 = one leaf cell / half an L1 cell)",
    )
    p.add_argument(
        "--check-invariance",
        action="store_true",
        help="also run with a one-leaf-cell shift and report the Cd change (acceptance: < 1%)",
    )
    p.add_argument(
        "--force-trace",
        action="store_true",
        help="record the per-root-step streamwise MEM/CV force into the JSON "
        "(full-precision float64 trace for single-vs-dual-GPU "
        "roundoff-equivalence checks).",
    )
    p.add_argument("--output", required=True)
    return p


def _collide_dispatch(
    f: torch.Tensor,
    tau: float,
    collision: str,
    les_model: str = "wale",
    cs_smag: float = 0.05,
    cw_wale: float = 0.5,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    if lattice == "D3Q27":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        from tensorlbm.cumulant import collide_cumulant_d3q27

        if collision == "cascaded":
            return collide_cascaded_d3q27(f, tau)
        if collision == "cumulant":
            return collide_cumulant_d3q27(f, tau, C_s=0.0)
        # LES on D3Q27
        if les_model == "wale":
            from tensorlbm.turbulence import collide_wale_bgk27

            return collide_wale_bgk27(f, tau, C_w=cw_wale)
        from tensorlbm.turbulence import collide_smagorinsky_bgk27

        return collide_smagorinsky_bgk27(f, tau, C_s=cs_smag)
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
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687)


def run_case(
    args: argparse.Namespace,
    center_offset: tuple[float, float, float],
    label: str,
) -> dict:
    """One full hybrid run (L0 + L1 block + sharded octree shell)."""
    global Q
    Q = 27 if args.lattice == "D3Q27" else 19
    device = torch.device(args.device)
    fine_devices = [torch.device(d) for d in args.fine_devices]
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
        args.nx,
        args.ny,
        args.nz,
        cx,
        cy,
        cz,
        radius_coarse,
        device,
        lattice=args.lattice,
    )
    plan = plan_body_shell_box(
        solid_coarse,
        args.shell_margin,
        args.wake_cells,
        pad=args.wall_margin,
    )
    box1 = plan.box
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    if args.lattice == "D3Q27":
        from tensorlbm.d3q27 import equilibrium27

        coarse_f = equilibrium27(rho, ux, zero, zero, device=device)
    else:
        coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ---- level 1 (L1 rectangular block) ------------------------------------
    s1, fc1, radius1, _l1 = build_fine_block_geometry(
        box1,
        (cx, cy, cz),
        radius_coarse,
        RATIO,
        GHOST,
        device,
    )
    nz1, ny1, nx1 = s1
    config1 = StaticBlockAMRConfig(
        box1,
        tau_coarse=tau_coarse,
        reflux=True,
        ghost_interpolation=args.ghost_interpolation,
    )
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    l1_fine = amr.interfaces[0].fine_f  # with-ghost tensor

    # ---- octree shell on the L1 physical grid ------------------------------
    # The sphere centre in L1 *physical* coordinates is the with-ghost centre
    # shifted by the ghost width; the shell's own `_solid` (computed at this
    # physical centre) is the consistent L1 freeze mask (the FineSphere solid
    # is evaluated at the with-ghost centre and sits one cell off).
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    radius_l1 = radius1
    bl_cells = (
        args.bl_thickness if args.bl_thickness is not None else (max(2.0, round(radius_l1 / 2.0)))
    )
    octree = build_octree_shell(
        s1,
        phys_center,
        radius_l1,
        bl_thickness_cells=bl_cells,
        d_max=args.d_max,
        transition=1,
        device=device,
        lattice=args.lattice,
    )
    shell_band = octree.meta["delta_mask"]
    host = octree.leaf_host_cell
    octree.f_leaf = l1_fine[:, host[:, 0] + GHOST, host[:, 1] + GHOST, host[:, 2] + GHOST].clone()
    if len(fine_devices) > octree.n_leaf:
        raise ValueError(
            f"more fine devices ({len(fine_devices)}) than shell leaves "
            f"({octree.n_leaf}) — reduce --fine-devices",
        )
    leaf_weights = leaf_force_weights(octree)
    ghost_plan = build_ghost_plan(
        octree,
        s1,
        solid_fallback=(args.ghost_fallback == "on"),
    )
    # ---- shard the shell across the fine devices ---------------------------
    # One contiguous Morton slice per device; every leaf-local tensor moves to
    # the shard device, the ghost plan's donor arrays stay on the root device.
    shards = shard_octree_shell(
        octree,
        fine_devices,
        ghost_plan=ghost_plan,
        solid_fallback=(args.ghost_fallback == "on"),
    )
    shard_by_device = {shard.device: shard for shard in shards}
    dx_leaf = 2.0 ** (-octree.d_max)
    dt_leaf = dx_leaf  # convective scaling

    # ---- L1 freeze masks (from the octree solid at the physical centre) ----
    assert octree._solid is not None, "shell builder must provide the solid mask"
    l1_solid_phys = octree._solid
    l1_solid_g = torch.zeros(
        (nz1 + 2 * GHOST, ny1 + 2 * GHOST, nx1 + 2 * GHOST),
        dtype=torch.bool,
        device=device,
    )
    l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l1_solid_phys
    l1_solid_q = l1_solid_g.unsqueeze(0).expand(Q, *l1_solid_g.shape).contiguous()

    # ---- control volume (fail-closed clearance gate) -----------------------
    filter_shell = amr.interfaces[0]._interface_filter_blend
    assert octree._shell_mask is not None, "shell builder must provide the mask"
    cv_w = build_shell_control_volume(
        (int(l1_solid_g.shape[0]), int(l1_solid_g.shape[1]), int(l1_solid_g.shape[2])),
        fc1,
        radius_l1,
        shell_band,
        args.cv_margin,
        covered=octree._shell_mask,
        filter_shell=filter_shell,
        solid=l1_solid_g,
        device=device,
    )
    cv = cv_w[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]  # physical slice

    # ---- sponge + dynamic area ---------------------------------------------
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=sponge_faces,
    )
    dynamic_area_cv = 0.5 * args.lattice_speed**2 * math.pi * radius_l1**2
    radius_leaf = radius_l1 / dx_leaf
    dynamic_area_mem = 0.5 * args.lattice_speed**2 * math.pi * radius_leaf**2

    # ---- per-root-step state -------------------------------------------------
    l1_posts: list[torch.Tensor] = []
    mem_samples: list[float] = []
    cv_samples: list[float] = []
    ledger = ShellForceLedger(octree)
    mem_trace: list[float] = []
    cv_trace: list[float] = []
    max_reflux_residual = 0.0
    min_density_seen = float("inf")
    min_population_seen = float("inf")
    joint_mass0: float | None = None
    joint_mass_end: float | None = None
    current_step = 0
    started = time.time()

    def advance(
        f: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        nonlocal current_step
        if level == 0:
            out, post_collision, _ = root_advance(
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
        if level == 1:
            before = f
            collided = _collide_dispatch(
                f,
                tau,
                args.collision,
                les_model=args.les_model,
                cs_smag=args.cs_smag,
                cw_wale=args.cw_wale,
                lattice=args.lattice,
            )
            post = torch.where(l1_solid_q, before, collided)
            if args.lattice == "D3Q27":
                from tensorlbm.d3q27 import stream27_roll

                out = stream27_roll(post)
            else:
                out = stream3d(post)
            l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
            return AMRAdvanceResult(out, post)
        raise ValueError(f"unexpected hierarchy level {level}")

    def shell_advance(
        f: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
        *,
        shard=None,
    ) -> AMRAdvanceResult:
        # The sharded stepper calls this once per shard with the shard's own
        # f_leaf tensor; resolve the shard by device so the LES neighbour
        # gather uses the shard-local (REMOTE-rewritten) table.
        if os.environ.get("OCTREE_DEBUG_NAN"):
            bad = ~torch.isfinite(f)
            if bool(bad.any()):
                lev = shard_by_device.get(f.device)
                lev_a = lev.leaf_level if lev is not None else octree.leaf_level
                cen_a = octree.leaf_center  # single-shard: local == global
                print(
                    f"[dbg] step={current_step} substep={substep} "
                    f"shell_advance ENTRY f non-finite: "
                    f"{int(bad.sum().item())}/{f.numel()} "
                    f"levels={lev_a[bad.any(dim=0)][:5].tolist()} "
                    f"centers={cen_a[bad.any(dim=0)][:3].tolist()} "
                    f"min={float(f.min().item()):.4e} max={float(f.max().item()):.4e}",
                    flush=True,
                )
        # The sharded stepper passes the active shard explicitly when the
        # callback accepts ``shard``.  Fall back to the device map only for
        # legacy callers; a device map cannot distinguish two same-device
        # shards.
        shard = shard or shard_by_device.get(f.device)
        if shard is not None:
            nt = shard.neighbor_table
            n_leaf = shard.n_leaf
            leaf_level = shard.leaf_level
        else:
            nt = octree.neighbor_table
            n_leaf = octree.n_leaf
            leaf_level = octree.leaf_level
        if args.collision is None:
            # LES on octree leaves via neighbour-table gathers (spatially
            # correct; the regular-grid WALE roll semantics are wrong on SoA).
            from tensorlbm.octree_boundary.les import leaf_les_collide

            f4 = f.view(Q, 1, 1, -1)
            if f4.numel() != n_leaf * Q:
                raise RuntimeError(
                    f"shell_advance f numel {f4.numel()} != Q*n_leaf "
                    f"{Q}*{n_leaf}; f.shape={tuple(f.shape)} "
                    f"shard={shard.device if shard is not None else device}",
                )
            collided = leaf_les_collide(
                f4,
                tau,
                nt,
                model="wale" if args.les_model == "wale" else "smagorinsky",
                C_w=args.cw_wale,
                C_s=args.cs_smag,
                dx=2.0 ** (-octree.d_max) * 0.5,
                leaf_level=leaf_level,
                leaf_center=(
                    octree.leaf_center[shard.lo : shard.hi]
                    if shard is not None
                    else octree.leaf_center
                ),
                neighbor_velocity=(shard.les_neighbor_velocity if shard is not None else None),
                neighbor_distance=(shard.les_neighbor_distance if shard is not None else None),
            ).view(f.shape)
            if not bool(torch.isfinite(collided).all()):
                raise FloatingPointError(
                    f"leaf LES produced non-finite populations at step "
                    f"{current_step} substep {substep}",
                )
        else:
            collided = _collide_dispatch(
                f.view(Q, 1, 1, -1),
                tau,
                args.collision,
                les_model=args.les_model,
                cs_smag=args.cs_smag,
                cw_wale=args.cw_wale,
                lattice=args.lattice,
            )
            collided = collided.view_as(f)
        post = collided.view_as(f)
        return AMRAdvanceResult(post.clone(), post)

    def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
        if args.no_moving_wall:
            wall_velocity = None
            wall_density = None
        else:
            rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
                octree_,
                post,
                current_step,
                args.ramp_steps,
            )
            wall_velocity = (uwx, uwy, uwz)
            wall_density = rho_w
        # shard facade exposes shard-local weights on the shard device; the
        # plain octree falls back to the root-side global weights
        fw = getattr(octree_, "force_weights", None)
        if fw is None:
            fw = leaf_weights
        return bfl_apply_gather(
            octree_,
            out,
            post,
            ghost_plan=ghost_plan_,
            ghost_vals=ghost_vals,
            wall_velocity=wall_velocity,
            wall_density=wall_density,
            force_weights=fw,
            return_force=True,
            q_min=args.q_min,
        )

    # Tell the generic sharded stepper whether it must exchange LES neighbour
    # velocities.  Explicit cumulant/cascaded runs avoid this extra exchange.
    shell_advance.uses_sparse_les = args.collision is None

    def joint_mass() -> float:
        assert octree._shell_mask is not None
        covered = octree._shell_mask
        l1_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
        exterior = float(l1_phys[:, ~covered].sum().item())
        shell = float((octree.f_leaf.sum(dim=0) * octree.leaf_volume().to(device)).sum().item())
        return exterior + shell

    joint_mass0 = joint_mass()

    for current_step in range(1, args.steps + 1):
        # StaticBlockAMR3D.step() rebinds `fine_f` to a fresh tensor after
        # each advance, so the cached ``l1_fine`` reference goes stale after
        # the first root step. Re-fetch the live tensor before and after the
        # step — otherwise the shell stepping, restriction/reflux and the CV
        # observation all operate on a frozen pre-advance state.
        l1_fine = amr.interfaces[0].fine_f
        l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
        l1_posts.clear()
        ledgers = amr.step(advance)
        l1_fine = amr.interfaces[0].fine_f
        l1_f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
        # P3 CV fix: snapshot the live L1 tensor right after the block's own
        # advance and BEFORE the shell stepper rewrites it in place
        # (restriction + reflux). F_cv = import - dP_L1advance + wall_mom_l1.
        l1_pre_shell = l1_f_phys.clone()
        shell_ledger = step_octree_shell_sharded(
            octree,
            shards,
            shell_advance,
            l1_old_phys,
            l1_f_phys,
            tau_coarse=config1.tau_fine,
            l1_post=l1_posts if config1.reflux else None,
            shell_level=1,
            ghost_plan=ghost_plan,
            bfl_fn=(None if args.no_bfl else bfl_callback),
            force_ledger=ledger,
        )
        max_reflux_residual = max(
            max_reflux_residual,
            abs(shell_ledger.mass_residual),
        )
        ledger.observe_cv_force(
            l1_old_phys,
            l1_pre_shell,
            l1_posts,
            cv,
            solid=l1_solid_phys,
            wall_mom_l1=(None if args.no_bfl else ledger.wall_momentum_l1(dx_leaf, dt_leaf)),
        )
        if args.force_trace:
            mem_trace.append(
                float(ledger.mem_force[0].item()) if not args.no_bfl else 0.0,
            )
            assert ledger.cv_force is not None
            cv_trace.append(float(ledger.cv_force[0].item()))
        if current_step > args.warmup_steps:
            if not args.no_bfl:
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
        if not shards_all_finite(shards):
            raise FloatingPointError(
                f"non-finite shard populations at step {current_step}",
            )
        if current_step % args.report_interval == 0 or current_step == args.steps:
            min_density_seen = min(
                min_density_seen,
                float(l1_fine.sum(dim=0).min().item()),
            )
            min_population_seen = min(
                min_population_seen,
                float(l1_fine.min().item()),
                *(float(s.f_leaf.min().item()) for s in shards),
            )
        if current_step % args.report_interval == 0 and mem_samples:
            recent_mem = mem_samples[-args.report_interval :]
            recent_cv = cv_samples[-args.report_interval :]
            cd_mem_s = (
                f"Cd_mem={sum(recent_mem) / len(recent_mem) / dynamic_area_mem:.6f} "
                if recent_mem
                else "Cd_mem=      n/a "
            )
            print(
                f"[{label}] step={current_step}/{args.steps} "
                f"{cd_mem_s}"
                f"Cd_cv={sum(recent_cv) / len(recent_cv) / dynamic_area_cv:.6f} "
                f"max_ref_res={max_reflux_residual:.2e} "
                f"steps/s={current_step / (time.time() - started):.2f}",
                flush=True,
            )

    wall_seconds = time.time() - started
    joint_mass_end = joint_mass()
    stats_window = args.statistics_window_steps or (len(mem_samples) or len(cv_samples) or 1)
    mem_mean = sum(mem_samples[-stats_window:]) / stats_window if mem_samples else float("nan")
    cv_mean = sum(cv_samples[-stats_window:]) / stats_window if cv_samples else float("nan")
    cd_mem = mem_mean / dynamic_area_mem if mem_samples else float("nan")
    cd_cv = cv_mean / dynamic_area_cv if cv_samples else float("nan")
    reference = _schiller_naumann(args.reynolds)
    cd_cv_error_pct = abs(cd_cv - reference) / reference * 100.0
    mem_cv_deviation_pct = (
        abs(cd_mem - cd_cv) / max(abs(cd_cv), 1e-30) * 100.0 if mem_samples else float("nan")
    )
    mass_drift = abs(joint_mass_end - joint_mass0) / joint_mass0 if joint_mass0 else 0.0
    cv_summary = (
        summarize_force_history(
            cv_samples,
            dynamic_area_cv,
            args.reynolds,
            stats_window,
        )
        if cv_samples
        else None
    )
    statistics_convective_times = (
        stats_window * args.lattice_speed / max(2.0 * radius_leaf, 1.0e-30)
    )
    print(
        f"[{label}] final: Cd_mem={cd_mem:.6f} Cd_cv={cd_cv:.6f} "
        f"ref={reference:.6f} ref_err={cd_cv_error_pct:.3f}% "
        f"mem/cv={mem_cv_deviation_pct:.3f}% mass_drift={mass_drift:.2e} "
        f"max_ref_res={max_reflux_residual:.2e} "
        f"min_rho={min_density_seen:.3e} min_f={min_population_seen:.3e} "
        f"steps/s={args.steps / wall_seconds:.2f}",
        flush=True,
    )
    if args.force_trace:
        print(
            f"[{label}] trace[0]={mem_trace[0]!r} trace[-1]={mem_trace[-1]!r} "
            f"cd_mem={cd_mem!r} cd_cv={cd_cv!r}",
            flush=True,
        )

    return {
        "label": label,
        "cd_mem": cd_mem,
        "cd_cv": cd_cv,
        "reference_cd": reference,
        "reference_error_pct_cv": cd_cv_error_pct,
        "reference_error_pct_mem": (abs(cd_mem - reference) / reference * 100.0),
        "mem_cv_deviation_pct": mem_cv_deviation_pct,
        "mean_mem_force_leaf_lu": mem_mean,
        "mean_cv_force_l1_lu": cv_mean,
        "dynamic_area_cv": dynamic_area_cv,
        "dynamic_area_mem": dynamic_area_mem,
        "max_reflux_residual": max_reflux_residual,
        "min_density_seen": min_density_seen,
        "min_population_seen": min_population_seen,
        "joint_mass_drift": mass_drift,
        "samples": stats_window,
        "statistics_convective_times": statistics_convective_times,
        "stationarity": (cv_summary or {}).get("stationarity"),
        "n_leaf": int(octree.n_leaf),
        "n_bfl_links": int(octree.bfl_mask.sum().item()),
        "shell_cell_saving": octree.stats.get("saving_fraction", 0.0),
        "l1_shape": list(s1),
        "l1_box_coarse": [box1.x0, box1.x1, box1.y0, box1.y1, box1.z0, box1.z1],
        "phys_center_l1": list(phys_center),
        "radius_l1": radius_l1,
        "radius_leaf": radius_leaf,
        "center_offset_coarse": list(center_offset),
        "fine_devices": [str(d) for d in fine_devices],
        "n_shards": len(shards),
        "perf": {
            "steps_per_second": args.steps / wall_seconds,
            "wall_seconds": wall_seconds,
        },
        "force_trace": ({"mem_x": mem_trace, "cv_x": cv_trace} if args.force_trace else None),
    }


def main() -> None:
    args = parser().parse_args()
    if args.d_max not in (1, 2):
        raise ValueError("--d-max must be 1 or 2")
    if not 0.0 <= args.interface_shift:
        raise ValueError("--interface-shift must be non-negative")
    if args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be < --steps")
    # ---- --fine-devices parsing --------------------------------------------
    if args.fine_devices is None:
        args.fine_devices = [args.device]
    else:
        devices = [d.strip() for d in args.fine_devices.split(",") if d.strip()]
        if not devices:
            raise ValueError(
                "--fine-devices must be a non-empty comma-separated device "
                "list (e.g. 'cuda:0,cuda:1')",
            )
        args.fine_devices = devices
    for d in args.fine_devices:
        torch.device(d)  # raise early on malformed device strings

    base = run_case(args, (0.0, 0.0, 0.0), "base")
    runs = [base]
    invariance = None
    if args.check_invariance:
        shift = args.interface_shift if args.interface_shift > 0 else 0.25
        shifted = run_case(args, (shift, 0.0, 0.0), "shifted")
        runs.append(shifted)
        cd_change = abs(shifted["cd_cv"] - base["cd_cv"]) / max(abs(base["cd_cv"]), 1e-30) * 100.0
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
            "fine_devices": [str(d) for d in args.fine_devices],
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
            "lattice": args.lattice,
            "collision": args.collision,
            "les_model": args.les_model,
            "cw_wale": args.cw_wale,
            "cs_smag": args.cs_smag,
            "q_min": args.q_min,
            "no_moving_wall": args.no_moving_wall,
            "ghost_interpolation": args.ghost_interpolation,
            "statistics_window_steps": args.statistics_window_steps,
            "minimum_statistics_convective_times": (args.minimum_statistics_convective_times),
            "tau_coarse": 0.5 + 3.0 * args.lattice_speed * 2.0 * args.radius / args.reynolds,
            "acceptance_requires_samples": 200,
        },
        "runs": runs,
        "invariance": invariance,
        "acceptance": {
            "cd_within_2pct": runs[0]["reference_error_pct_cv"] <= 2.0,
            "mem_cv_within_5pct": runs[0]["mem_cv_deviation_pct"] <= 5.0,
            "finite_positive_state": (
                runs[0]["min_density_seen"] > 0.0 and runs[0]["min_population_seen"] >= -1.0e-10
            ),
            "invariance_within_1pct": (invariance is not None and invariance["accepts_1pct"]),
            "physical_accuracy_admitted": (
                runs[0]["reference_error_pct_cv"] <= 2.0
                and runs[0]["mem_cv_deviation_pct"] <= 5.0
                and runs[0]["joint_mass_drift"] <= 1.0e-4
                and runs[0]["samples"] >= 200
                and runs[0]["statistics_convective_times"]
                >= (args.minimum_statistics_convective_times)
                and runs[0]["stationarity"] is not None
                and runs[0]["stationarity"]["relative_range_pct"] <= 1.0
                and runs[0]["stationarity"]["half_mean_drift_pct"] <= 0.5
                and runs[0]["min_density_seen"] > 0.0
                and runs[0]["min_population_seen"] >= -1.0e-10
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
        f"| state-positive: {result['acceptance']['finite_positive_state']} "
        f"| physical-admitted: {result['acceptance']['physical_accuracy_admitted']} "
        f"| invariance<1%: {result['acceptance']['invariance_within_1pct']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
