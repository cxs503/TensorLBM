#!/usr/bin/env python3
"""Distributed octree-shell sphere/SUBOFF drag validation (torchrun, all_gather).

Common driver for the distributed octree shell on SDAA:
- geometry is selected by ``--geo sphere|suboff`` and built through the
  common geometry adapters (``geometry_adapters``);
- the shell is advanced with ``step_octree_shell_distributed`` (all_gather
  comms, one process per device);
- drag is reported via the shell MEM force (rank 0) with the
  Schiller-Naumann (sphere) / AFF-8 (SUBOFF) reference.

Usage:
    PYTHONPATH=src torchrun --nproc_per_node=2 examples/octree_distributed_validate.py \
        --geo sphere --nx 96 --ny 64 --nz 64 --radius 6 --steps 50
    PYTHONPATH=src torchrun --nproc_per_node=2 examples/octree_distributed_validate.py \
        --geo suboff --nx 96 --ny 64 --nz 64 --steps 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "src")

import torch
import torch.distributed as dist

from tensorlbm.octree_boundary.distributed_stepping import (
    split_leaf_bounds,
    step_octree_shell_distributed,
)
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import (
    solid_mask_inside_fn,
    sphere_inside_fn,
)


def _schiller_naumann(re: float) -> float:
    if re < 1.0:
        return 24.0 / re
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def build_geometry(args, dev):
    """Return (octree, l1_shape, solid_cpu) for sphere or SUBOFF."""
    nx, ny, nz = args.nx, args.ny, args.nz
    l1_shape = (nz, ny, nx)
    if args.geo == "sphere":
        center = (nx * 0.5, ny * 0.5, nz * 0.5)
        radius = args.radius
        bl = args.bl if args.bl is not None else max(2.0, round(radius / 2.0))
        octree = build_octree_shell(
            l1_shape, center=center, radius=radius,
            bl_thickness_cells=bl, d_max=args.d_max,
            lattice="D3Q27", device=dev,
            inside_fn=sphere_inside_fn(center, radius),
        )
        solid = None
    else:
        from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
        L = args.hull
        config = SuboffConfig()
        radius = config.r_over_l * L
        solid, _ = build_suboff_mask(
            hull_type=args.hull_type, nx=nx, ny=ny, nz=nz,
            cx=args.cx, cy=ny * 0.5, cz=nz * 0.5,
            length=L, radius=radius, config=config, device="cpu",
        )
        solid = solid.bool()
        center = (args.cx, ny * 0.5, nz * 0.5)
        bl = args.bl if args.bl is not None else max(2.0, round(radius / 2.0))
        octree = build_octree_shell(
            l1_shape, center=center, radius=max(radius, bl * 2),
            bl_thickness_cells=bl, d_max=args.d_max,
            lattice="D3Q27", device=dev,
            inside_fn=solid_mask_inside_fn(solid),
        )
    return octree, l1_shape, solid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geo", choices=("sphere", "suboff"), default="sphere")
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--hull", type=float, default=40.0)
    p.add_argument("--hull-type", default="bare_hull")
    p.add_argument("--cx", type=float, default=30.0)
    p.add_argument("--bl", type=float, default=None,
                   help="shell band thickness (default max(2, R/2))")
    p.add_argument("--d-max", type=int, default=1)
    p.add_argument("--interleave", action="store_true",
                   help="round-robin leaf sharding (load balance for d_max=2)")
    p.add_argument("--u-in", type=float, default=0.06)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--q-min", type=float, default=None,
                   help="clamp tiny BFL q values (high-Re safeguard)")
    p.add_argument("--report-interval", type=int, default=10)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    dist.init_process_group("tccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # One device per process: rank -> sdaa:{rank} (32 devices available).
    dev = torch.device(f"sdaa:{rank % torch.sdaa.device_count()}")
    torch.sdaa.set_device(dev)

    octree, l1_shape, solid = build_geometry(args, dev)
    q = octree.Q
    n_leaf = octree.n_leaf
    lo, hi = split_leaf_bounds(n_leaf, world_size)[rank]

    # L1 fluid state (uniform inflow) — identical on every rank.
    nz, ny, nx = l1_shape
    u_in = args.u_in
    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), u_in, device=dev)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    from tensorlbm.d3q27 import equilibrium27
    l1_f = equilibrium27(rho, ux, uy, uz)
    l1_old = l1_f.clone()
    l1_post = l1_f.clone()

    # Initialise the shell leaves by sampling the L1 field (not zeros).
    host = octree.leaf_host_cell  # (n_leaf, 3) (z, y, x) in L1 block
    f_leaf_full = l1_f[:, host[:, 0], host[:, 1], host[:, 2]].clone()
    # Each rank owns its leaf slice (contiguous [lo:hi) or interleaved).
    from tensorlbm.octree_boundary.distributed_stepping import (
        interleaved_leaf_indices,
    )
    if args.interleave:
        lidx = interleaved_leaf_indices(n_leaf, world_size, rank)
    else:
        lidx = torch.arange(lo, hi, dtype=torch.int64)
    octree.f_leaf = f_leaf_full[:, lidx].contiguous()
    print(f"[r{rank}] geo={args.geo} n_leaf={n_leaf} range=[{lo},{hi}) "
          f"Q={q} d_max={octree.d_max}", flush=True)

    # L1 solid mask (for coarse bounce-back): solid cells of the body.
    l1_solid = octree._solid
    if l1_solid is None:
        l1_solid = torch.zeros(l1_shape, dtype=torch.bool, device=dev)
    else:
        l1_solid = l1_solid.to(dev)
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q27 import OPPOSITE as OPP27
    from tensorlbm.cumulant import collide_cumulant_d3q27 as _collide_cumulant27

    def advance_l1(f, tau):
        """One coarse L1 collide (cumulant) — returns collided populations."""
        f4 = f.view(q, 1, 1, -1)
        c4 = _collide_cumulant27(f4, tau, C_s=0.0)
        return c4.view_as(f)

    def stream_l1(f):
        """D3Q27 pull-stream on the L1 block with solid bounce-back."""
        out = torch.empty_like(f)
        for d in range(27):
            cd = C27[d]
            src = torch.roll(f[d], shifts=(-int(cd[0]), -int(cd[1]), -int(cd[2])), dims=(0, 1, 2))
            # Solid cells: bounce-back (opposite direction, same cell).
            out[d] = torch.where(l1_solid, f[OPP27[d]], src)
        return out

    # Re→tau via the shared module (diameter convention L_ref = 2R — the
    # old inline `u_in * args.radius / args.reynolds` used the bare radius
    # and silently doubled the Reynolds number).
    from tensorlbm.lbm_re_tau import tau_from_re
    tau_coarse = tau_from_re(u_in, 2.0 * args.radius, args.reynolds)
    # shell tau follows the convective chain.
    from tensorlbm.octree_boundary.stepping import _tau_chain
    taus = _tau_chain(tau_coarse, octree.d_max)
    tau_shell = taus[1]

    # ---- real MEM force (Bouzidi BFL + ShellForceLedger) ----
    from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
    from tensorlbm.octree_boundary.force import ShellForceLedger
    # leaf_weights is global (n_leaf,); the BFL facade sees only this rank's
    # columns, so slice by the rank's leaf indices.
    if args.interleave:
        lidx_w = interleaved_leaf_indices(n_leaf, world_size, rank)
    else:
        lidx_w = torch.arange(lo, hi, dtype=torch.int64)
    leaf_weights = leaf_force_weights(octree).to(dev)[lidx_w]
    dx_leaf = 2.0 ** (-octree.d_max)
    dt_leaf = dx_leaf  # convective scaling

    def bfl_fn(octree_, out, post, gplan, ghost_vals, *, substep):
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=gplan, ghost_vals=ghost_vals,
            force_weights=leaf_weights, return_force=True,
            q_min=args.q_min,
        )

    def advance(f, tau, level, substep):
        from tensorlbm.cumulant import collide_cumulant_d3q27
        # SoA (Q, n_leaf) -> 4D (Q, 1, 1, n_leaf) for the regular-grid collide.
        f4 = f.view(q, 1, 1, -1)
        collided = collide_cumulant_d3q27(f4, tau, C_s=0.0)
        return collided.view_as(f)

    # Dynamic area: sphere uses projected area; SUBOFF uses L^2 scale.
    if args.geo == "sphere":
        radius_leaf = args.radius / dx_leaf
        dynamic_area = 0.5 * u_in ** 2 * math.pi * radius_leaf ** 2
    else:
        L_leaf = args.hull / dx_leaf
        dynamic_area = 0.5 * u_in ** 2 * L_leaf ** 2
    mem_samples: list[float] = []

    # Placeholder force: sum of |f| on the rank's leaves (drag later via MEM).
    t0 = time.time()
    forces = []
    mem_accum = torch.zeros(3, dtype=torch.float64, device=dev)
    for step in range(1, args.steps + 1):
        # 0. Evolve the coarse L1 field: collide + stream + solid BB.
        l1_f = stream_l1(advance_l1(l1_f, tau_coarse))
        # far-field: reset boundary cells to uniform inflow.
        l1_f[:, :, :, 0] = equilibrium27(rho, ux, uy, uz)[:, :, :, 0]
        l1_f[:, :, :, -1] = equilibrium27(rho, ux, uy, uz)[:, :, :, -1]
        l1_f = l1_f.clone()
        _ledger, local_mem, _restricted, _cells = step_octree_shell_distributed(
            octree, advance, l1_old, l1_f,
            tau_coarse=tau_coarse, l1_post=l1_post,
            ghost_plan=None, bfl_fn=bfl_fn, rank=rank, world_size=world_size,
            reflux=False,  # 先测 all_gather 通信; reflux 后续实现
            interleave=args.interleave,
        )
        # local_mem: (3,) MEM force in leaf-lattice units (this rank's leaves),
        # already time-averaged over the root step's substeps.
        mem_accum += local_mem
        l1_old = l1_f.clone()
        # rank-local drag proxy (all-reduced below); f_leaf is ALREADY the
        # local [lo:hi) slice, so sum it directly (do NOT slice by lo:hi again).
        forces.append(float(octree.f_leaf.abs().sum().item()))
        if step > args.warmup_steps and step % args.report_interval == 0 and rank == 0:
            cd_cur = float(mem_accum[0].item()) / max(step - args.warmup_steps, 1) / dynamic_area
            print(f"[r{rank}] step={step} Cd_mem={cd_cur:.4f}", flush=True)
        if step % args.report_interval == 0:
            print(f"[r{rank}] step={step}/{args.steps} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

    # Global MEM force (all-reduce the per-rank sums).
    dist.all_reduce(mem_accum, op=dist.ReduceOp.SUM)
    n_samples = max(args.steps - args.warmup_steps, 1)
    if rank == 0:
        # Drag = -F_x (MEM force is the fluid-on-body impulse, opposing the
        # +x inflow; Cd uses the drag magnitude).
        cd_mem = -float(mem_accum[0].item()) / n_samples / dynamic_area
        ref = _schiller_naumann(args.reynolds) if args.geo == "sphere" else 0.004
        result = {
            "geo": args.geo, "n_leaf": n_leaf, "world_size": world_size,
            "steps": args.steps, "warmup": args.warmup_steps,
            "cd_mem": cd_mem, "ref_Cd": ref,
            "err_pct": 100.0 * (cd_mem - ref) / ref,
            "force_proxy": float(force_sum := torch.tensor([sum(forces)], device=dev, dtype=torch.float32).item()),
            "device": str(dev), "per_step_s": (time.time() - t0) / args.steps,
        }
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            with open(args.output, "w") as fh:
                json.dump(result, fh, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
