#!/usr/bin/env python3
"""Full integration: 16-card coarse D3Q27 cumulant + octree shell.

Coarse:  x-domain-decomposed D3Q27 cumulant with all_gather halo (the
         validated ``dg_suboff_cumulant_d3q27_multicard`` pattern).
Shell:   all_gather-distributed octree shell (geometry_adapters + BFL MEM).

Every root step:
  1. coarse collide (local) + halo all_gather + stream27_roll + BB + far-field
  2. shell ghost sampled from the *evolved* coarse field (all-gathered shell
     region, small), shell collide -> all_gather -> stream -> BFL -> MEM force
  3. restrict shell back into coarse (rank 0 patch + broadcast)
  4. drag: friction/pressure all_reduce + MEM all_reduce

Usage (torchrun, one device per rank):
  torchrun --nproc_per_node=16 examples/octree_integrated_validate.py \
      --geo sphere --nx 96 --ny 64 --nz 64 --radius 6 --steps 200 \
      --warmup-steps 50 --report-interval 50 --output /tmp/integrated.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geo", choices=("sphere", "suboff"), default="sphere")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--hull", type=float, default=40.0)
    p.add_argument("--hull-type", default="bare_hull")
    p.add_argument("--cx", type=float, default=30.0)
    p.add_argument("--bl", type=float, default=None)
    p.add_argument("--d-max", type=int, default=2)
    p.add_argument("--l1-block", action="store_true", default=True,
                   help="enable the L1 middle block (coarse -> L1 2x -> shell "
                        "leaf three-level hierarchy, design doc "
                        "L1_MIDDLE_BLOCK_INTEGRATION_DESIGN.md). "
                        "Pass --no-l1-block for the legacy two-level path.")
    p.add_argument("--no-l1-block", dest="l1_block", action="store_false")
    p.add_argument("--shell-margin", type=int, default=6,
                   help="L1 box: hull-proximity shell thickness in coarse "
                        "cells (plan_body_shell_box)")
    p.add_argument("--wake-cells", type=int, default=32,
                   help="L1 box: downstream wake extension in coarse cells")
    p.add_argument("--wall-margin", type=int, default=8,
                   help="L1 box: coarse-cell padding around the shell+wake "
                        "mask (also raises GHOST_PAD to max(6, pad+2))")
    p.add_argument("--interleave", action="store_true")
    p.add_argument("--u-in", type=float, default=0.06)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--q-min", type=float, default=None)
    p.add_argument("--no-coarse-bb", action="store_true",
                   help="disable coarse halfway BB (wall handled only by shell BFL)")
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--report-interval", type=int, default=50)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--blockage-correction",
        choices=("simple", "glauert", "off"), default="simple",
        help=("Blockage (wind-tunnel) correction for the confined domain. "
              "The infinite-domain Cd_ref is unfair in a small domain: the "
              "lateral walls accelerate the flow around the body, so a "
              "correct solver should compute a HIGHER Cd. 'simple' (default, "
              "Maskell/Bartz): f=1/(1-beta)^2; 'glauert': "
              "f=1/(1-1.5*beta); 'off': no correction. beta=D/Ly (D=body "
              "diameter, Ly=domain width). R6 sphere: D=12, Ly=64 -> "
              "beta=0.1875 -> simple f=1.5148 -> Cd_ref_blocked=1.654."),
    )
    p.add_argument(
        "--domain-scale",
        type=float, default=None,
        help=("Target Ly/D ratio for a recommended larger domain (e.g. 8.0 "
              "means Ly=8D, beta=12.5%%). If set, prints the recommended ny "
              "and resulting beta. If unset and beta>12.5%%, prints a "
              "recommendation to scale to Ly>=8D."),
    )
    args = p.parse_args()

    dist.init_process_group("tccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = torch.device(f"sdaa:{rank % 32}")
    torch.sdaa.set_device(dev)

    nx, ny, nz = args.nx, args.ny, args.nz
    q = 27
    u_in = args.u_in

    # ---------------- geometry ----------------
    from tensorlbm.octree_boundary.geometry import (
        build_octree_shell,
        sphere_distance_field,
    )
    from tensorlbm.octree_boundary.geometry_adapters import (
        solid_mask_inside_fn,
        sphere_inside_fn,
    )
    from tensorlbm.amr_shell_planning import plan_body_shell_box

    solid_coarse = None  # coarse-frame solid mask (L1 box planning / coarse BB)

    if args.geo == "sphere":
        center = (nx * 0.5, ny * 0.5, nz * 0.5)
        radius = args.radius
        D_body = 2.0 * radius          # body diameter for blockage ratio
        bl = args.bl if args.bl is not None else max(2.0, round(radius / 2.0))
        # Sphere uses the ANALYTIC path (no inside_fn): the analytic q-field
        # (0.02-0.97) and symmetric BFL mask are required for a correct Cd.
        # inside_fn (sphere_inside_fn) has an asymmetric mask + missing
        # self-fluid exclusion (audit P1) — analytic avoids it entirely.
        solid_coarse = None
        if args.l1_block:
            # Coarse solid mask for the L1 box planning (same cell-centre
            # convention as the octree's analytic solid).
            solid_coarse = sphere_distance_field(
                (nz, ny, nx), center, radius, dev,
            ) <= 0.0
    else:
        from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
        L = args.hull
        config = SuboffConfig()
        srad = config.r_over_l * L
        D_body = 2.0 * srad            # hull diameter for blockage ratio
        solid_cpu, _ = build_suboff_mask(
            hull_type=args.hull_type, nx=nx, ny=ny, nz=nz,
            cx=args.cx, cy=ny * 0.5, cz=nz * 0.5,
            length=L, radius=srad, config=config, device="cpu",
        )
        center = (args.cx, ny * 0.5, nz * 0.5)
        bl = args.bl if args.bl is not None else max(2.0, round(srad / 2.0))
        if args.l1_block:
            solid_coarse = solid_cpu.bool().to(dev)

    if args.l1_block:
        # ---- L1 middle-block geometry (design §3a) ----
        # Body-fitted box: shell band + downstream wake + wall padding,
        # planned on the COARSE solid mask, in coarse global coordinates.
        assert solid_coarse is not None, "L1 path requires a coarse solid mask"
        plan = plan_body_shell_box(
            solid_coarse, shell_margin=args.shell_margin,
            wake_cells=args.wake_cells, pad=args.wall_margin,
        )
        box = plan.box
        l1_shape = (
            (box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2,
            (box.x1 - box.x0) * 2,
        )
        # Body centre in L1 physical coordinates and radius in L1 units.
        center_l1 = (
            center[0] * 2.0 - box.x0 * 2,
            center[1] * 2.0 - box.y0 * 2,
            center[2] * 2.0 - box.z0 * 2,
        )
        if args.geo == "sphere":
            radius_l1 = radius * 2.0
            octree = build_octree_shell(
                l1_shape, center=center_l1, radius=radius_l1,
                bl_thickness_cells=bl, d_max=args.d_max,
                lattice="D3Q27", device=dev,
            )
        else:
            radius_l1 = max(srad, bl * 2) * 2.0

            def _suboff_inside_l1(centers):
                # Leaf centres are in L1-local world units; map them back to
                # the coarse mask frame (mask index = box origin + local/2).
                coarse = 0.5 * centers + torch.tensor(
                    (box.x0, box.y0, box.z0), dtype=torch.float64,
                    device=centers.device,
                )
                return solid_mask_inside_fn(
                    solid_cpu.bool(), device=dev,
                )(coarse)

            octree = build_octree_shell(
                l1_shape, center=center_l1, radius=radius_l1,
                bl_thickness_cells=bl, d_max=args.d_max,
                lattice="D3Q27", device=dev,
                # device=dev keeps inside_fn evaluation on the GPU.
                inside_fn=_suboff_inside_l1,
            )
        solid = octree._solid              # L1-frame solid mask
    else:
        # ---- legacy two-level path: octree hosted on the coarse grid ----
        l1_shape = (nz, ny, nx)
        box = None
        if args.geo == "sphere":
            octree = build_octree_shell(
                l1_shape, center=center, radius=radius,
                bl_thickness_cells=bl, d_max=args.d_max,
                lattice="D3Q27", device=dev,
            )
            solid = octree._solid
        else:
            octree = build_octree_shell(
                l1_shape, center=center, radius=max(srad, bl * 2),
                bl_thickness_cells=bl, d_max=args.d_max,
                lattice="D3Q27", device=dev,
                # device=dev keeps inside_fn evaluation on the GPU (the
                # default mask device would be CPU and break
                # _level1_leaves arithmetic).
                inside_fn=solid_mask_inside_fn(solid_cpu.bool(), device=dev),
            )
            solid = octree._solid
    n_leaf = octree.n_leaf
    q_oct = octree.Q
    if rank == 0:
        box_desc = "None" if box is None else (
            f"z:[{box.z0},{box.z1}) y:[{box.y0},{box.y1}) "
            f"x:[{box.x0},{box.x1})"
        )
        print(
            f"[geo] l1_block={args.l1_block} box={box_desc} "
            f"l1_shape={l1_shape} "
            f"n_leaf_l1={octree.stats.get('n_leaf_l1')} "
            f"n_leaf_l2={octree.stats.get('n_leaf_l2')} "
            f"n_interface_links={octree.stats.get('n_interface_links')}",
            flush=True,
        )

    # ---------------- coarse domain decomposition (x-slab) ----------------
    # Each rank owns nx_local = nx // world_size columns of the coarse field.
    assert nx % world_size == 0, "nx must divide evenly across ranks"
    nx_local = nx // world_size
    lo = rank * nx_local
    hi = lo + nx_local
    # Coarse f layout (27, nz, ny, nx_local+2) with 1 halo on each x side.
    coarse_f = torch.zeros(q, nz, ny, nx_local + 2, device=dev)
    rho = torch.ones(nz, ny, nx_local, device=dev)
    ux = torch.full((nz, ny, nx_local), u_in, device=dev)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    from tensorlbm.d3q27 import equilibrium27
    eq = equilibrium27(rho, ux, uy, uz)
    coarse_f[:, :, :, 1:-1] = eq
    coarse_f[:, :, :, 0:1] = eq[:, :, :, 0:1]
    coarse_f[:, :, :, -1:] = eq[:, :, :, -1:]

    # Shell host cells for ghost sampling from coarse.
    host = octree.leaf_host_cell  # (n_leaf, 3) (z, y, x) global
    from tensorlbm.octree_boundary.distributed_stepping import (
        interleaved_leaf_indices,
    )
    if args.interleave:
        lidx = interleaved_leaf_indices(n_leaf, world_size, rank)
    else:
        # MUST match distributed_stepping.split_leaf_bounds (base+extra), so
        # all ranks agree on shard sizes when n_leaf % world_size != 0.
        from tensorlbm.octree_boundary.distributed_stepping import split_leaf_bounds
        lo_l, hi_l = split_leaf_bounds(n_leaf, world_size)[rank]
        lidx = torch.arange(lo_l, hi_l, dtype=torch.int64)
    n_local = lidx.shape[0]
    # Initialise the shell leaves from the uniform inflow equilibrium using
    # GLOBAL host coordinates (host[lidx] are (z, y, x) in the octree's host
    # grid — coarse for the legacy path, L1 for the L1-block path); build a
    # full-host equilibrium tensor for that (eq above is the local slab).
    # The field is spatially uniform, so any in-bounds indexing is exact.
    oshape = tuple(octree.meta["shape"])
    eq_global = equilibrium27(
        torch.ones(oshape, device=dev),
        torch.full(oshape, u_in, device=dev),
        torch.zeros(oshape, device=dev),
        torch.zeros(oshape, device=dev),
    )
    octree.f_leaf = eq_global[:, host[lidx, 0], host[lidx, 1], host[lidx, 2]].clone()
    print(f"[r{rank}] n_leaf={n_leaf} lidx_len={lidx.shape[0]} "
          f"f_leaf_cols={octree.f_leaf.shape[1]}", flush=True)

    # coarse tau + L1/shell tau chain
    tau_coarse = 0.5 + 3.0 * (u_in * args.radius / args.reynolds) \
        if args.geo == "sphere" else 0.5 + 3.0 * (u_in * args.hull / args.reynolds)
    from tensorlbm.octree_boundary.stepping import _tau_chain
    taus = _tau_chain(tau_coarse, octree.d_max)
    # Legacy two-level (host=coarse, d_max=1): taus = [tau_c, tau_shell].
    # L1 block (host=L1, d_max=2): taus = [tau_c, tau_l1, tau_shell]; the
    # shell stepper is handed tau_coarse=tau_l1 and derives its own chain.
    tau_l1 = taus[1] if args.l1_block else tau_coarse
    tau_shell = taus[octree.d_max]

    # ---------------- coarse operators (domain-decomposed) ----------------
    from tensorlbm.cumulant import collide_cumulant_d3q27
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q27 import OPPOSITE as OPP27
    S27 = [(int(C27[d, 0]), int(C27[d, 1]), int(C27[d, 2])) for d in range(27)]

    # Coarse solid mask for the coarse halfway BB: the L1 path keeps the
    # COARSE solid (octree._solid is L1-shaped there), the legacy path uses
    # octree._solid directly (coarse-shaped).
    if args.l1_block:
        assert solid_coarse is not None, "L1 path requires a coarse solid mask"
        coarse_solid = solid_coarse
    else:
        coarse_solid = solid
    solid_local = coarse_solid[:, :, lo:hi]   # (nz, ny, nx_local) no halo
    solid_full = torch.zeros(nz, ny, nx_local + 2, dtype=torch.bool, device=dev)
    solid_full[:, :, 1:-1] = solid_local
    # Sponge toward uniform inflow on y/z/x+ faces (inlet x- is driven by the
    # far-field reset).  Shape matches the local slab (halo included).
    from tensorlbm.sponge_layer import (
        apply_equilibrium_difference_sponge,
        build_sponge_sigma_3d,
    )
    sponge_sigma = build_sponge_sigma_3d(
        (nz, ny, nx_local + 2), width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=dev,
        # y/z faces are global per-rank (not decomposed); x faces are
        # handled by the far-field reset (rank 0 inlet / last rank outlet),
        # so no x sponge is applied to avoid mid-rank artefacts.
        faces=("y-", "y+", "z-", "z+"),
    )
    # Fill the solid halo columns from the neighbour ranks (periodic), so
    # bounce-back at the domain boundary sees the correct wall geometry.
    if world_size > 1:
        left_s = solid_local[:, :, 0:1].contiguous()
        right_s = solid_local[:, :, -1:].contiguous()
        ls_g = [torch.empty_like(left_s) for _ in range(world_size)]
        dist.all_gather(ls_g, left_s)
        rs_g = [torch.empty_like(right_s) for _ in range(world_size)]
        dist.all_gather(rs_g, right_s)
        solid_full[:, :, 0:1] = rs_g[(rank - 1) % world_size]
        solid_full[:, :, -1:] = ls_g[(rank + 1) % world_size]
    else:
        solid_full[:, :, 0:1] = solid_local[:, :, -1:]
        solid_full[:, :, -1:] = solid_local[:, :, 0:1]

    def collide_coarse(f, tau):
        f4 = f.view(q, 1, 1, -1)
        c4 = collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)
        # Sponge (equilibrium-difference damping) toward uniform inflow on
        # the non-inlet faces; suppresses far-field reflection / blocking.
        if sponge_sigma is not None:
            c4 = apply_equilibrium_difference_sponge(
                c4, sponge_sigma,
                rho_target=1.0,
                velocity_target=(u_in, 0.0, 0.0),
            )
        return c4

    def halo_exchange(f_local):
        left_interior = f_local[:, :, :, 1:2].contiguous()
        right_interior = f_local[:, :, :, -2:-1].contiguous()
        right_gather = [torch.empty_like(right_interior)
                        for _ in range(world_size)]
        dist.all_gather(right_gather, right_interior)
        left_halo = right_gather[(rank - 1) % world_size]
        left_gather = [torch.empty_like(left_interior)
                       for _ in range(world_size)]
        dist.all_gather(left_gather, left_interior)
        right_halo = left_gather[(rank + 1) % world_size]
        f_local[:, :, :, 0:1] = left_halo
        f_local[:, :, :, -1:] = right_halo

    def stream27_roll(f):
        out = torch.empty_like(f)
        for d in range(27):
            sx, sy, sz = S27[d]
            out[d] = torch.roll(f[d], shifts=(sz, sy, sx), dims=(0, 1, 2))
        return out

    def apply_halfway_bounce_back(streamed, postcollision):
        fluid = ~solid_full
        opposite = OPP27.to(dev)
        out = streamed.clone()
        for d in range(27):
            sx, sy, sz = S27[d]
            solid_source = torch.roll(solid_full, shifts=(sz, sy, sx),
                                      dims=(0, 1, 2))
            wall_link = fluid & solid_source
            out[d] = torch.where(wall_link, postcollision[opposite[d]], out[d])
        return out

    def far_field(f, u_in=0.06):
        # Physical boundaries (matching the validated domain-decomposed
        # reference): rank 0 resets the inlet interior plane (global x=0) to
        # uniform inflow; the last rank extrapolates the outlet plane
        # (global x=nx-1); every rank resets the y/z boundary planes (y/z are
        # not decomposed, so each rank's y/z edges ARE the global boundary).
        # The halo columns are NOT reset here — they are overwritten by
        # halo_exchange before the next stream anyway.
        rho_ff = torch.ones(nz, ny, 1, device=dev)
        ux_ff = torch.full((nz, ny, 1), u_in, device=dev)
        uy_ff = torch.zeros(nz, ny, 1, device=dev)
        uz_ff = torch.zeros(nz, ny, 1, device=dev)
        eq_ff = equilibrium27(rho_ff, ux_ff, uy_ff, uz_ff)
        if rank == 0:
            f[:, :, :, 1:2] = eq_ff
        if rank == world_size - 1:
            f[:, :, :, -2:-1] = f[:, :, :, -3:-2]  # outlet extrapolation
        # y/z boundary planes (each rank owns the full y/z extent).
        eq_ff_plane = equilibrium27(
            torch.ones(1, ny, nx_local + 2, device=dev),
            torch.full((1, ny, nx_local + 2), u_in, device=dev),
            torch.zeros(1, ny, nx_local + 2, device=dev),
            torch.zeros(1, ny, nx_local + 2, device=dev),
        )
        f[:, 0, :, :] = eq_ff_plane[:, 0]
        f[:, -1, :, :] = eq_ff_plane[:, 0]
        eq_ff_y = equilibrium27(
            torch.ones(nz, 1, nx_local + 2, device=dev),
            torch.full((nz, 1, nx_local + 2), u_in, device=dev),
            torch.zeros(nz, 1, nx_local + 2, device=dev),
            torch.zeros(nz, 1, nx_local + 2, device=dev),
        )
        f[:, :, 0, :] = eq_ff_y[:, :, 0]
        f[:, :, -1, :] = eq_ff_y[:, :, 0]
        return f

    # ---------------- shell advance + BFL (all_gather distributed) ----------------
    from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
    leaf_weights = leaf_force_weights(octree).to(dev)[lidx]

    # ---- TEMP DIAGNOSTICS (SUBOFF L1 force-deficit audit) ----
    # dbg_fx: accumulated x-force (weighted); dbg_nw: weighted link count;
    # dbg_nraw: raw (unweighted) wall-link count.
    dbg_fx = torch.zeros(1, dtype=torch.float64, device=dev)
    dbg_nw = torch.zeros(1, dtype=torch.float64, device=dev)
    dbg_nraw = torch.zeros(1, dtype=torch.float64, device=dev)
    dbg_fx_last = torch.zeros(1, dtype=torch.float64, device=dev)
    dbg_nw_last = torch.zeros(1, dtype=torch.float64, device=dev)
    dbg_nraw_last = torch.zeros(1, dtype=torch.float64, device=dev)
    dbg_step_last = 0

    def bfl_fn(octree_, out, post, gplan, ghost_vals, *, substep):
        fout, force = bfl_apply_gather(
            octree_, out, post,
            ghost_plan=gplan, ghost_vals=ghost_vals,
            force_weights=leaf_weights, return_force=True,
            q_min=args.q_min,
        )
        # local BFL mask on the facade is (Q, n_local)
        mask_loc = octree_.bfl_mask
        w_leaf = 2.0 ** (-(octree_.d_max
                           - octree_.leaf_level.to(torch.float64)))
        dbg_fx[0] += float(force[0].item())
        dbg_nw[0] += float((mask_loc * w_leaf.unsqueeze(0)).sum().item())
        dbg_nraw[0] += float(mask_loc.sum().item())
        return fout, force

    def advance_shell(f, tau, level, substep):
        f4 = f.view(q_oct, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    from tensorlbm.octree_boundary.distributed_stepping import (
        step_octree_shell_distributed,
    )

    # ---------------- L1 middle block (stage 1, design §3b/3d) ----------------
    from tensorlbm.octree_boundary.l1_block import (
        L1BlockDistributed,
        gather_window_chunked,
        restrict_l1_block_to_coarse,
        step_l1_block_distributed,
        write_window_back,
    )

    def advance_l1(f, tau):
        f4 = f.view(q, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    win = None
    if args.l1_block:
        assert box is not None, "L1 path requires a planned box"
        l1_block = L1BlockDistributed(
            box, (nz, ny, nx), tau_coarse, q=q, ratio=2, ghost=3,
            device=dev, solid_l1=solid,
            collide_fn=advance_l1, stream_fn=stream27_roll,
        )
        l1_block.initialize_uniform(u_in)
        win = l1_block.win
        print(f"[r{rank}] L1 block shape={l1_block.l1_shape} "
              f"ghost={l1_block.ghost} window_ring={l1_block.window_ring} "
              f"window_cells={win.cells.shape[0]} "
              f"window_shape={win.shape} "
              f"tau_l1={l1_block.tau_l1:.6f}", flush=True)
    else:
        l1_block = None

    # Leaf resolution relative to the COARSE grid, from the ACTUAL
    # leaf-level distribution (audit 2026-08-16).  d_max=2 shells mix
    # level-1 and level-2 leaves (e.g. SUBOFF L1: ~87k level-1 + ~91k
    # level-2), and the old formulas (2^-(1+d_max) L1 / 2^-d_max legacy)
    # assumed every leaf sits at d_max, overestimating the resolved
    # resolution by 2x on the level-1 fraction.  Correct per-level dx:
    #   legacy: host = coarse, level-l leaf dx = 2^-l      coarse cells
    #   L1:     host = 2x coarse, level-l leaf dx = 2^-(1+l) coarse cells
    # ``dx_leaf_coarse`` is the leaf-count-weighted mean of the per-leaf
    # dx (the effective shell resolution); ``dx_leaf_levelmean`` is the
    # alternative 2^-(1+level_mean) form.  Both are printed for audit.
    lev = octree.leaf_level.to(torch.float64)
    n_leaf_lv = int(lev.numel())
    n_l1 = int((lev == 1).sum().item())
    n_l2 = int((lev == 2).sum().item())
    level_mean = float(lev.mean().item()) if n_leaf_lv else 0.0
    if n_leaf_lv:
        dx_per_leaf = 2.0 ** (-(1.0 + lev)) if args.l1_block \
            else 2.0 ** (-lev)
        dx_leaf_coarse = float(dx_per_leaf.mean().item())
        dx_leaf_levelmean = 2.0 ** (-(1.0 + level_mean)) if args.l1_block \
            else 2.0 ** (-level_mean)
    else:
        dx_leaf_coarse = 2.0 ** (-(1 + octree.d_max)) if args.l1_block \
            else 2.0 ** (-octree.d_max)
        dx_leaf_levelmean = dx_leaf_coarse
    dx_leaf_old = 2.0 ** (-(1 + octree.d_max)) if args.l1_block \
        else 2.0 ** (-octree.d_max)
    if args.geo == "sphere":
        radius_leaf = args.radius / dx_leaf_coarse
        dynamic_area = 0.5 * u_in ** 2 * math.pi * radius_leaf ** 2
    else:
        L_leaf = args.hull / dx_leaf_coarse
        radius_leaf = L_leaf
        dynamic_area = 0.5 * u_in ** 2 * L_leaf ** 2
    if rank == 0:
        print(f"[area] leaf_level counts: L1={n_l1} L2={n_l2} "
              f"total={n_leaf_lv} level_mean={level_mean:.4f}", flush=True)
        print(f"[area] dx_leaf_coarse(count-weighted mean)={dx_leaf_coarse:.6f} "
              f"dx_leaf(2^-(1+lev_mean))={dx_leaf_levelmean:.6f} "
              f"dx_leaf(old 2^-(1+d_max))={dx_leaf_old:.6f}", flush=True)
        print(f"[area] radius_leaf={radius_leaf:.3f} "
              f"dynamic_area={dynamic_area:.6f}", flush=True)

    # Legacy two-level path: precompute the dilated shell-region coarse cells
    # once (the sparse coarse field fed to the shell ghost fill).  GHOST_PAD
    # is raised to max(6, wall_margin+2) so the L1 box+ring window stays
    # inside the dilated band (design §3a).  The L1 path replaces this whole
    # machinery with the box+ring window gather (ring = L1 ghost depth).
    GHOST_PAD = max(6, args.wall_margin + 2)
    n_shell = 0
    shell_cells = torch.zeros((0, 3), dtype=torch.int64, device=dev)
    sc_x = torch.zeros(0, dtype=torch.int64, device=dev)
    sc_in = torch.zeros(0, dtype=torch.bool, device=dev)
    sc_z = torch.zeros(0, dtype=torch.int64, device=dev)
    sc_y = torch.zeros(0, dtype=torch.int64, device=dev)
    sc_xx = torch.zeros(0, dtype=torch.int64, device=dev)
    if not args.l1_block:
        shell_mask_full = octree._shell_mask
        import torch.nn.functional as Fnn
        dilated = shell_mask_full.float().unsqueeze(0).unsqueeze(0)
        for _ in range(GHOST_PAD):
            dilated = Fnn.max_pool3d(
                dilated, kernel_size=3, stride=1, padding=1,
            )
        shell_mask_full = (dilated.squeeze(0).squeeze(0) > 0.5)
        shell_cells = torch.nonzero(
            shell_mask_full, as_tuple=False,
        )  # (n_shell, 3) (z, y, x) global
        n_shell = shell_cells.shape[0]
        sc_x = shell_cells[:, 2]
        sc_in = (sc_x >= lo) & (sc_x < hi)   # cells in this rank's x-slab
        sc_z = shell_cells[sc_in, 0]
        sc_y = shell_cells[sc_in, 1]
        sc_xx = sc_x[sc_in] - lo + 1          # local column (with halo)
        print(f"[r{rank}] n_shell={n_shell} in_rank={int(sc_in.sum())}",
              flush=True)

    # ---------------- main loop ----------------
    from tensorlbm.d3q27 import equilibrium27 as eq27b
    t0 = time.time()
    mem_accum = torch.zeros(3, dtype=torch.float64, device=dev)
    for step in range(1, args.steps + 1):
        # 1. coarse evolve (domain-decomposed) — unchanged.
        coarse_old = coarse_f.clone()
        post = collide_coarse(coarse_f, tau_coarse)
        halo_exchange(post)
        streamed = stream27_roll(post)
        if not args.no_coarse_bb:
            streamed = apply_halfway_bounce_back(streamed, post)
        coarse_f = far_field(streamed, u_in)

        if args.l1_block:
            # 2. window gather: box + ring-cell ring (ring = L1 ghost
            #    depth, 3 in the default L1 config — the deepened coarse
            #    supply band of audit 2026-08) at the three time points
            #    old / new / post (chunked all_gather, <3MB/msg).  ``win``
            #    and ``l1_block`` are guaranteed set in this branch.
            assert win is not None and l1_block is not None
            cw_old, _in_slab = gather_window_chunked(
                coarse_old, win, lo, hi,
                rank=rank, world_size=world_size)
            cw_new, in_slab = gather_window_chunked(
                coarse_f, win, lo, hi,
                rank=rank, world_size=world_size)
            cw_post, _in_slab2 = gather_window_chunked(
                post, win, lo, hi,
                rank=rank, world_size=world_size)
            # 3. L1 block stage: 2 time-interpolated substeps (ghost <-
            #    lerp(coarse_old, coarse_new, s/2); cumulant collide +
            #    stream27_roll + frozen solid) -> l1_posts.
            l1_phys_pre, l1_posts_phys, _posts_ghost = \
                step_l1_block_distributed(l1_block, cw_old, cw_new)
            # 4. shell stage hosted on the real L1 field: the two lerp
            #    anchors are the genuine root-step-start / root-step-end L1
            #    physical slices (design §3c — fixes the old P3 defect where
            #    l1_old == l1_f).  tau_coarse = tau_l1; l1_post is the list
            #    of the two L1 post-collision slices.
            l1_f_phys = l1_block.physical_copy()
            _ledger_shell, local_mem, _restricted, _cells = \
                step_octree_shell_distributed(
                    octree, advance_shell, l1_phys_pre, l1_f_phys,
                    tau_coarse=l1_block.tau_l1, l1_post=l1_posts_phys,
                    ghost_plan=None, bfl_fn=bfl_fn, rank=rank,
                    world_size=world_size, reflux=True,
                    interleave=args.interleave,
                )
            l1_block.set_physical(l1_f_phys)
            # 5. L1 -> coarse restriction (box interior) + face-local kinetic
            #    reflux on the box interface (design §3d).
            ledger_l1c = restrict_l1_block_to_coarse(
                l1_block, cw_new, cw_post,
            )
            if step % args.report_interval == 0 and rank == 0:
                print(
                    f"[r{rank}] step={step} l1c_reflux_residual="
                    f"{ledger_l1c.mass_residual:.3e} "
                    f"corrected_links={ledger_l1c.shell_cells}",
                    flush=True,
                )
            # 6. window patch write-back to this rank's slab (box restriction
            #    + ring reflux corrections); halo columns are refreshed by
            #    the next halo_exchange.
            write_window_back(coarse_f, cw_new, win, in_slab, lo)
        else:
            # ---- legacy two-level path (unchanged) ----
            # 2. shell ghost from evolved coarse field: build the sparse
            #    coarse field (only shell-region cells, tiny all-gather) and
            #    feed it to the shell stepper's ghost fill.
            #
            # 3. shell step: ghost fill needs the *coarse field* (4D, global
            #    coordinates).  Build a sparse coarse field holding the
            #    shell-region cells PLUS a ~GHOST_PAD-cell dilation buffer so
            #    ghost donors just outside the band see the real
            #    wake/defect, not the uniform-inflow fill (audit P1-3:
            #    clamping the band exterior to uniform inflow corrupts
            #    near-wake pressure/shear).
            sc_local = torch.zeros(q_oct, n_shell, device=dev)
            if bool(sc_in.any()):
                sc_local[:, sc_in] = coarse_f[:, sc_z, sc_y, sc_xx]
            # TCCL deadlock guard: chunk the shell gather (<3MB/msg).
            sc_chunk = max(1, int(3 * 1024 * 1024 // (q_oct * 4)))
            full_sc = torch.zeros(q_oct, n_shell, device=dev)
            for c0 in range(0, n_shell, sc_chunk):
                c1 = min(c0 + sc_chunk, n_shell)
                piece = sc_local[:, c0:c1].contiguous()
                g_piece = [torch.empty_like(piece) for _ in range(world_size)]
                dist.all_gather(g_piece, piece)
                for r in range(world_size):
                    full_sc[:, c0:c1] = full_sc[:, c0:c1] + g_piece[r]
            # Build the coarse post-collision sparse field for the reflux
            # observation.  ``l1_post`` is the post-collision (pre-stream)
            # coarse state, observed on the shell boundary links to pair
            # with the fine-side transfer accumulated over the shell
            # substeps.  Same sparse-field pattern as ``coarse_sparse``:
            # real post-collision values at the dilated shell cells, uniform
            # inflow equilibrium elsewhere (the observation_links masks are
            # nonzero only at the shell boundary + 1-cell border, which lies
            # inside the dilation, so every observed cell has a real value).
            sc_local_post = torch.zeros(q_oct, n_shell, device=dev)
            if bool(sc_in.any()):
                sc_local_post[:, sc_in] = post[:, sc_z, sc_y, sc_xx]
            full_sc_post = torch.zeros(q_oct, n_shell, device=dev)
            for c0 in range(0, n_shell, sc_chunk):
                c1 = min(c0 + sc_chunk, n_shell)
                piece = sc_local_post[:, c0:c1].contiguous()
                g_piece = [torch.empty_like(piece) for _ in range(world_size)]
                dist.all_gather(g_piece, piece)
                for r in range(world_size):
                    full_sc_post[:, c0:c1] = \
                        full_sc_post[:, c0:c1] + g_piece[r]
            l1_post = eq27b(
                torch.ones(nz, ny, nx, device=dev),
                torch.full((nz, ny, nx), u_in, device=dev),
                torch.zeros(nz, ny, nx, device=dev),
                torch.zeros(nz, ny, nx, device=dev),
            )
            l1_post[:, shell_cells[:, 0], shell_cells[:, 1],
                    shell_cells[:, 2]] = full_sc_post
            # Sparse coarse field (4D) with shell-region values; fill the
            # rest with uniform inflow equilibrium (ghost donors must never
            # see 0).
            coarse_sparse = eq27b(
                torch.ones(nz, ny, nx, device=dev),
                torch.full((nz, ny, nx), u_in, device=dev),
                torch.zeros(nz, ny, nx, device=dev),
                torch.zeros(nz, ny, nx, device=dev),
            )
            coarse_sparse[:, shell_cells[:, 0], shell_cells[:, 1],
                          shell_cells[:, 2]] = full_sc
            l1_old = coarse_sparse
            l1_f = coarse_sparse
            _ledger, local_mem, restricted, cells = \
                step_octree_shell_distributed(
                    octree, advance_shell, l1_old, l1_f,
                    tau_coarse=tau_coarse, l1_post=l1_post,
                    ghost_plan=None, bfl_fn=bfl_fn, rank=rank,
                    world_size=world_size, reflux=True,
                    interleave=args.interleave,
                )
            # ---- bidirectional coupling: shell restriction + reflux ->
            # coarse.  ``l1_f`` (= ``coarse_sparse``) now carries the fine
            # restriction at covered cells AND the face-local reflux
            # correction at the exterior interface cells.  Writing the
            # dilated shell region back to the per-rank ``coarse_f``
            # propagates BOTH to the real coarse field.  ``coarse_f`` is
            # THIS rank's x-slab (Q, nz, ny, nx_local+2) with one halo
            # column on each x side, so the local x index of a global column
            # is ``global_x - lo + 1``.  Each rank writes only the cells
            # inside its own slab [lo, hi); neighbour halo columns are
            # refreshed by halo_exchange at the start of the next root step.
            if bool(sc_in.any()):
                coarse_f[:, sc_z, sc_y, sc_xx] = \
                    l1_f[:, sc_z, sc_y, sc_x[sc_in]]
        if step > args.warmup_steps:
            # Only accumulate force after the startup transient (warmup);
            # the initial impact force is unphysical and must not pollute Cd.
            mem_accum += local_mem

        if step > args.warmup_steps and step % args.report_interval == 0:
            # Cd uses the GLOBAL (all-reduced) force — rank0's local mem_accum
            # is only half the force on 2 ranks and misleads the trend.
            glob = mem_accum.clone()
            dist.all_reduce(glob, op=dist.ReduceOp.SUM)
            if rank == 0:
                cd_cur = float(glob[0].item()) / max(step - args.warmup_steps, 1) / dynamic_area
                print(f"[r{rank}] step={step} Cd_mem={cd_cur:.4f}", flush=True)

        # ---- TEMP DIAGNOSTICS: per-report force/link/per-link ----
        if step % args.report_interval == 0 or step <= 5:
            gfx = dbg_fx.clone()
            gnw = dbg_nw.clone()
            gnr = dbg_nraw.clone()
            dist.all_reduce(gfx, op=dist.ReduceOp.SUM)
            dist.all_reduce(gnw, op=dist.ReduceOp.SUM)
            dist.all_reduce(gnr, op=dist.ReduceOp.SUM)
            if rank == 0:
                d_fx = gfx[0].item() - dbg_fx_last[0].item()
                d_nw = gnw[0].item() - dbg_nw_last[0].item()
                d_nr = gnr[0].item() - dbg_nraw_last[0].item()
                n_steps = step - dbg_step_last
                print(
                    f"[dbg] step={step} fx_sum={gfx[0].item():.6e} "
                    f"links_w={gnw[0].item():.0f} links_raw={gnr[0].item():.0f} "
                    f"per-link_w={gfx[0].item()/max(gnw[0].item(),1.0):.6e} "
                    f"per-link_raw={gfx[0].item()/max(gnr[0].item(),1.0):.6e} | "
                    f"Δfx={d_fx:.6e} Δlinks_w={d_nw:.0f} "
                    f"Δper-link_w={d_fx/max(d_nw,1.0):.6e} (n_steps={n_steps})",
                    flush=True,
                )
                dbg_fx_last[0] = gfx[0].item()
                dbg_nw_last[0] = gnw[0].item()
                dbg_nraw_last[0] = gnr[0].item()
                dbg_step_last = step
        if step % args.report_interval == 0:
            print(f"[r{rank}] step={step}/{args.steps} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

    dist.all_reduce(mem_accum, op=dist.ReduceOp.SUM)
    n_samples = max(args.steps - args.warmup_steps, 1)
    if rank == 0:
        # MEM force x-component sign: +F_x is the drag (matches the validated
        # single-card convention cd = +mem_mean/dynamic_area).
        cd_mem = float(mem_accum[0].item()) / n_samples / dynamic_area

        # ---- Blockage (wind-tunnel) correction for the confined domain ----
        # The infinite-domain reference Cd_ref is unfair in a small domain:
        # the lateral walls accelerate the flow around the body, so a correct
        # solver in a confined domain should compute a HIGHER Cd than the
        # infinite-domain value.  We correct the reference upward by the
        # standard wind-tunnel factor so the reported error reflects the true
        # grid/discretisation error rather than a domain-size artefact.
        #   beta = D / Ly   (D = body diameter, Ly = domain width = ny)
        #   simple  (Maskell/Bartz):  f = 1 / (1 - beta)^2
        #   glauert (more conservative): f = 1 / (1 - 1.5*beta)
        # R6 sphere: D=12, Ly=ny=64 -> beta=0.1875
        #   simple:  f = 1/(0.8125)^2 = 1.5148 -> Cd_ref_blocked = 1.654
        Ly = float(ny)
        beta = D_body / Ly if Ly > 0 else 0.0
        ref_inf = 1.0917 if args.geo == "sphere" else 0.004
        bc = args.blockage_correction
        if bc == "off" or beta <= 0.0:
            corr_factor = 1.0
            bc_note = "off"
        elif bc == "glauert":
            corr_factor = 1.0 / (1.0 - 1.5 * beta)
            bc_note = "glauert 1/(1-1.5*beta)"
        else:  # "simple" (default)
            corr_factor = 1.0 / (1.0 - beta) ** 2
            bc_note = "simple 1/(1-beta)^2"
        ref = ref_inf * corr_factor
        err_pct = 100.0 * (cd_mem - ref) / ref
        err_pct_inf = 100.0 * (cd_mem - ref_inf) / ref_inf

        # Domain-scale recommendation: a larger domain lowers beta and thus
        # the blockage bias.  Ly >= 8D keeps beta <= 12.5% (a common
        # wind-tunnel rule of thumb).
        scale_note = ""
        if args.domain_scale is not None:
            target_ratio = args.domain_scale
            ny_rec = int(math.ceil(target_ratio * D_body))
            beta_rec = D_body / ny_rec if ny_rec > 0 else 0.0
            corr_rec = (1.0 / (1.0 - beta_rec) ** 2) if beta_rec > 0 else 1.0
            scale_note = (
                f"domain_scale={target_ratio:.1f}D -> recommended ny={ny_rec} "
                f"(beta={beta_rec:.4f}={beta_rec*100:.2f}%, "
                f"corr_factor={corr_rec:.4f}, "
                f"Cd_ref_blocked={ref_inf*corr_rec:.4f})"
            )
        elif beta > 0.125:
            ny_rec = int(math.ceil(8.0 * D_body))
            beta_rec = D_body / ny_rec if ny_rec > 0 else 0.0
            scale_note = (
                f"WARNING: beta={beta*100:.2f}% > 12.5% (blockage is large); "
                f"recommend enlarging the domain to Ly>=8D (ny>={ny_rec}, "
                f"beta<={beta_rec*100:.2f}%) or pass --domain-scale 8.0"
            )

        result = {
            "geo": args.geo, "n_leaf": n_leaf, "world_size": world_size,
            "steps": args.steps, "warmup": args.warmup_steps,
            "cd_mem": cd_mem,
            "ref_Cd_inf": ref_inf,
            "ref_Cd": ref,
            "blockage_ratio": beta,
            "blockage_correction": bc_note,
            "blockage_factor": corr_factor,
            "err_pct": err_pct,
            "err_pct_inf_domain": err_pct_inf,
            "per_step_s": (time.time() - t0) / args.steps,
        }
        if scale_note:
            result["domain_scale_note"] = scale_note
        print(json.dumps(result, indent=2), flush=True)
        # Human-readable blockage summary.
        print(
            f"[blockage] beta=D/Ly={D_body:.3f}/{Ly:.0f}="
            f"{beta*100:.2f}%  correction={bc_note}  "
            f"factor={corr_factor:.4f}",
            flush=True,
        )
        print(
            f"[blockage] Cd_ref: inf-domain={ref_inf:.4f} -> "
            f"blocked={ref:.4f}",
            flush=True,
        )
        print(
            f"[blockage] Cd_mem={cd_mem:.4f}  "
            f"err vs blocked={err_pct:+.2f}%  "
            f"err vs inf-domain={err_pct_inf:+.2f}%",
            flush=True,
        )
        if scale_note:
            print(f"[blockage] {scale_note}", flush=True)
        if args.output:
            with open(args.output, "w") as fh:
                json.dump(result, fh, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
