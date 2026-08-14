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
    p.add_argument("--d-max", type=int, default=1)
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
    from tensorlbm.octree_boundary.geometry import build_octree_shell
    from tensorlbm.octree_boundary.geometry_adapters import (
        solid_mask_inside_fn,
        sphere_inside_fn,
    )
    l1_shape = (nz, ny, nx)

    if args.geo == "sphere":
        center = (nx * 0.5, ny * 0.5, nz * 0.5)
        radius = args.radius
        bl = args.bl if args.bl is not None else max(2.0, round(radius / 2.0))
        # Sphere uses the ANALYTIC path (no inside_fn): the analytic q-field
        # (0.02-0.97) and symmetric BFL mask are required for a correct Cd.
        # inside_fn (sphere_inside_fn) has an asymmetric mask + missing
        # self-fluid exclusion (audit P1) — analytic avoids it entirely.
        octree = build_octree_shell(
            l1_shape, center=center, radius=radius,
            bl_thickness_cells=bl, d_max=args.d_max,
            lattice="D3Q27", device=dev,
        )
        solid = octree._solid
    else:
        from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
        L = args.hull
        config = SuboffConfig()
        srad = config.r_over_l * L
        solid_cpu, _ = build_suboff_mask(
            hull_type=args.hull_type, nx=nx, ny=ny, nz=nz,
            cx=args.cx, cy=ny * 0.5, cz=nz * 0.5,
            length=L, radius=srad, config=config, device="cpu",
        )
        solid = solid_cpu.bool().to(dev)
        center = (args.cx, ny * 0.5, nz * 0.5)
        bl = args.bl if args.bl is not None else max(2.0, round(srad / 2.0))
        octree = build_octree_shell(
            l1_shape, center=center, radius=max(srad, bl * 2),
            bl_thickness_cells=bl, d_max=args.d_max,
            lattice="D3Q27", device=dev,
            # device=dev keeps inside_fn evaluation on the GPU (the default
            # mask device would be CPU and break _level1_leaves arithmetic).
            inside_fn=solid_mask_inside_fn(solid_cpu.bool(), device=dev),
        )
    n_leaf = octree.n_leaf
    q_oct = octree.Q

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
    # GLOBAL host coordinates (host[lidx] are global (z, y, x)); build a
    # full-domain equilibrium tensor for that (eq above is the local slab).
    eq_global = equilibrium27(
        torch.ones(nz, ny, nx, device=dev),
        torch.full((nz, ny, nx), u_in, device=dev),
        torch.zeros(nz, ny, nx, device=dev),
        torch.zeros(nz, ny, nx, device=dev),
    )
    octree.f_leaf = eq_global[:, host[lidx, 0], host[lidx, 1], host[lidx, 2]].clone()
    print(f"[r{rank}] n_leaf={n_leaf} lidx_len={lidx.shape[0]} "
          f"f_leaf_cols={octree.f_leaf.shape[1]}", flush=True)

    # coarse tau + shell tau chain
    tau_coarse = 0.5 + 3.0 * (u_in * args.radius / args.reynolds) \
        if args.geo == "sphere" else 0.5 + 3.0 * (u_in * args.hull / args.reynolds)
    from tensorlbm.octree_boundary.stepping import _tau_chain
    taus = _tau_chain(tau_coarse, octree.d_max)
    tau_shell = taus[1]

    # ---------------- coarse operators (domain-decomposed) ----------------
    from tensorlbm.cumulant import collide_cumulant_d3q27
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q27 import OPPOSITE as OPP27
    S27 = [(int(C27[d, 0]), int(C27[d, 1]), int(C27[d, 2])) for d in range(27)]

    solid_local = solid[:, :, lo:hi]          # (nz, ny, nx_local) no halo
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

    def bfl_fn(octree_, out, post, gplan, ghost_vals, *, substep):
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=gplan, ghost_vals=ghost_vals,
            force_weights=leaf_weights, return_force=True,
            q_min=args.q_min,
        )

    def advance_shell(f, tau, level, substep):
        f4 = f.view(q_oct, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    from tensorlbm.octree_boundary.distributed_stepping import (
        step_octree_shell_distributed,
    )

    dx_leaf = 2.0 ** (-octree.d_max)
    if args.geo == "sphere":
        radius_leaf = args.radius / dx_leaf
        dynamic_area = 0.5 * u_in ** 2 * math.pi * radius_leaf ** 2
    else:
        L_leaf = args.hull / dx_leaf
        dynamic_area = 0.5 * u_in ** 2 * L_leaf ** 2

    # Precompute the shell-region coarse cells once (used every step for the
    # sparse coarse field fed to ghost fill).  Dilate the shell band by
    # GHOST_PAD cells so ghost donors just outside the band see the real
    # wake/defect from the coarse field instead of the uniform-inflow fill.
    GHOST_PAD = 6
    shell_mask_full = octree._shell_mask
    import torch.nn.functional as Fnn
    dilated = shell_mask_full.float().unsqueeze(0).unsqueeze(0)
    for _ in range(GHOST_PAD):
        dilated = Fnn.max_pool3d(dilated, kernel_size=3, stride=1, padding=1)
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
    print(f"[r{rank}] n_shell={n_shell} in_rank={int(sc_in.sum())}", flush=True)

    # ---------------- main loop ----------------
    from tensorlbm.d3q27 import equilibrium27 as eq27b
    t0 = time.time()
    mem_accum = torch.zeros(3, dtype=torch.float64, device=dev)
    for step in range(1, args.steps + 1):
        # 1. coarse evolve (domain-decomposed).
        post = collide_coarse(coarse_f, tau_coarse)
        halo_exchange(post)
        streamed = stream27_roll(post)
        if not args.no_coarse_bb:
            streamed = apply_halfway_bounce_back(streamed, post)
        coarse_f = far_field(streamed, u_in)

        # 2. shell ghost from evolved coarse field: build the sparse coarse
        #    field (only shell-region cells, tiny all-gather) and feed it to
        #    the shell stepper's ghost fill.

        # 3. shell step: ghost fill needs the *coarse field* (4D, global
        #    coordinates).  Build a sparse coarse field holding the
        #    shell-region cells PLUS a ~6-cell dilation buffer so ghost
        #    donors just outside the band see the real wake/defect, not the
        #    uniform-inflow fill (audit P1-3: clamping the band exterior to
        #    uniform inflow corrupts near-wake pressure/shear).
        sc_local = torch.zeros(q_oct, n_shell, device=dev)
        if bool(sc_in.any()):
            sc_local[:, sc_in] = coarse_f[:, sc_z, sc_y, sc_xx]
        gathered_sc = [torch.empty_like(sc_local) for _ in range(world_size)]
        # TCCL deadlock guard: chunk the shell gather (<3MB/msg) for big R10.
        sc_chunk = max(1, int(3 * 1024 * 1024 // (q_oct * 4)))
        full_sc = torch.zeros(q_oct, n_shell, device=dev)
        for c0 in range(0, n_shell, sc_chunk):
            c1 = min(c0 + sc_chunk, n_shell)
            piece = sc_local[:, c0:c1].contiguous()
            g_piece = [torch.empty_like(piece) for _ in range(world_size)]
            dist.all_gather(g_piece, piece)
            for r in range(world_size):
                full_sc[:, c0:c1] = full_sc[:, c0:c1] + g_piece[r]
        # Sparse coarse field (4D) with shell-region values; fill the rest
        # with uniform inflow equilibrium (ghost donors must never see 0).
        coarse_sparse = eq27b(
            torch.ones(nz, ny, nx, device=dev),
            torch.full((nz, ny, nx), u_in, device=dev),
            torch.zeros(nz, ny, nx, device=dev),
            torch.zeros(nz, ny, nx, device=dev),
        )
        coarse_sparse[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]] = full_sc
        l1_old = coarse_sparse
        l1_f = coarse_sparse
        _ledger, local_mem, restricted, cells = step_octree_shell_distributed(
            octree, advance_shell, l1_old, l1_f,
            tau_coarse=tau_coarse, l1_post=None,
            ghost_plan=None, bfl_fn=bfl_fn, rank=rank, world_size=world_size,
            reflux=False, interleave=args.interleave,
        )
        # ---- bidirectional coupling: shell restriction -> coarse field ----
        # ``cells`` are the GLOBAL (z, y, x) coordinates of the shell-covered
        # L1 cells and ``restricted`` the (Q, n_cells) fine->coarse mean that
        # rank 0 computed and broadcast (both identical on every rank).
        # ``coarse_f`` is THIS rank's x-slab (Q, nz, ny, nx_local+2) with one
        # halo column on each x side, so the local x index of a global column
        # is ``global_x - lo + 1``.  Each rank writes only the cells inside
        # its own slab [lo, hi); neighbour halo columns are refreshed by
        # halo_exchange at the start of the next root step, and the y/z
        # edges of the shell region are handled by the far-field planes
        # (overwritten by the restriction only if a shell cell actually lies
        # on a boundary plane, which the finer shell solution is entitled to).
        cell_z = cells[:, 0]
        cell_y = cells[:, 1]
        cell_x = cells[:, 2]
        mine = (cell_x >= lo) & (cell_x < hi)
        if bool(mine.any()):
            coarse_f[:, cell_z[mine], cell_y[mine], cell_x[mine] - lo + 1] = \
                restricted[:, mine]
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
        if step % args.report_interval == 0:
            print(f"[r{rank}] step={step}/{args.steps} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

    dist.all_reduce(mem_accum, op=dist.ReduceOp.SUM)
    n_samples = max(args.steps - args.warmup_steps, 1)
    if rank == 0:
        # MEM force x-component sign: +F_x is the drag (matches the validated
        # single-card convention cd = +mem_mean/dynamic_area).
        cd_mem = float(mem_accum[0].item()) / n_samples / dynamic_area
        ref = (1.0917 if args.geo == "sphere" else 0.004)
        result = {
            "geo": args.geo, "n_leaf": n_leaf, "world_size": world_size,
            "steps": args.steps, "warmup": args.warmup_steps,
            "cd_mem": cd_mem, "ref_Cd": ref,
            "err_pct": 100.0 * (cd_mem - ref) / ref,
            "per_step_s": (time.time() - t0) / args.steps,
        }
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            with open(args.output, "w") as fh:
                json.dump(result, fh, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
