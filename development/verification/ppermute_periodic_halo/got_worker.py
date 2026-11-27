"""Gloo worker for the GOT / CONTROL-LANE arms (CPU only, real P2P).

Every rank:
  * builds the REAL ``DistributedTritonFusedSolver3D`` (only the CUDA gate and
    the GPU lattice-table factory are stubbed — see ``arms.make_cpu_solver``);
  * runs ``n_steps`` of collide+stream with the REAL
    ``_start_halo_exchange`` / ``_finalize_halo``;
  * all-gathers the owned planes and compares the assembled global field with
    the single-domain ground truth (computed redundantly on every rank).

Print ``XVALS_METRICS=<json>`` on rank 0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import arms  # noqa: E402


def _gt(f0, n_steps, tau, nz, ny, nx):
    """Ground truth: whole-domain collide + periodic pull-stream."""
    C_t, W_t = arms._tables_t()
    f = f0.clone()
    for _ in range(n_steps):
        f = arms.stream_global3d(arms._collide_t(f, tau, C_t, W_t), C_t)
    return f


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master = os.environ.get("XVALS_MASTER", "tcp://127.0.0.1:29517")
    nz, ny, nx = 16, 8, 8
    n_steps, tau = int(os.environ.get("XVALS_STEPS", "20")), 0.6
    kind = os.environ.get("XVALS_FIELD", "rand")
    seed = int(os.environ.get("XVALS_SEED", "0"))
    wire = os.environ.get("XVALS_WIRE", "fp64")
    arm = os.environ.get("XVALS_ARM", "GOT")

    dist.init_process_group("gloo", init_method=master, rank=rank, world_size=world)
    try:
        halo_dtype = {"fp64": torch.float64, "fp16": torch.float16}[wire]
        s = arms.make_cpu_solver(
            nz, ny, nx, tau=tau, halo_dtype=halo_dtype, lane_swap=(arm == "CONTROL-LANE")
        )
        f0_np = arms.make_field_3d(nz, ny, nx, seed, kind=kind)
        f0 = torch.as_tensor(f0_np, dtype=torch.float64)

        nzl = s.nz_local
        f = arms.run_got_steps(
            s, f0_np, n_steps,
            stream_fn=lambda buf: arms.stream_local3d(buf, arms._tables_t()[0], nzl),
            w1_periodic_fill=bool(int(os.environ.get("XVALS_W1FILL", "0"))),
        )
        owned = f[:, 1:nzl + 1].contiguous()
        parts = [torch.empty_like(owned) for _ in range(world)]
        dist.all_gather(parts, owned)
        f_glob = torch.cat(parts, dim=1)

        gt = _gt(f0, n_steps, tau, nz, ny, nx)
        num = float(torch.linalg.vector_norm(f_glob - gt))
        den = float(torch.linalg.vector_norm(gt))
        metrics = {
            "arm": arm + ("-W1FILL" if os.environ.get("XVALS_W1FILL", "0") == "1" else ""),
            "ranks": world,
            "nz_global": nz, "ny": ny, "nx": nx,
            "n_steps": n_steps,
            "field": kind,
            "seed": seed,
            "wire": wire,
            "n_cross": s.n_cross,
            "cross_up": list(s._cross_up),
            "cross_dn": list(s._cross_dn),
            "staging_bytes": s.staging_bytes(),
            "halo_bytes_per_step": s.halo_bytes_per_step(),
            "halo_lane_planes_per_step": 2 * s.n_cross,
            "max_abs": float((f_glob - gt).abs().max()),
            "rel_l2": num / den,
            "bitwise_equal": bool(torch.equal(f_glob, gt)),
            "ke_got": arms.kinetic_energy(f_glob, arms._tables_t()[0]),
            "ke_gt": arms.kinetic_energy(gt, arms._tables_t()[0]),
        }
        if rank == 0:
            print("XVALS_METRICS=" + json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()