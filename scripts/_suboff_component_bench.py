#!/usr/bin/env python3
"""Ground-truth wall time of each step component in isolation (one sync/loop).

For a pipelined op the reported time is max(CPU_enqueue, device_time).
"""
from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401

from tensorlbm import boundaries3d as b3  # noqa: E402
from tensorlbm import solver3d as s3  # noqa: E402
from tensorlbm.d3q19 import C, OPPOSITE, macroscopic3d, equilibrium3d  # noqa: E402
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402
from tensorlbm.solver3d import _get_d3q19_mrt_matrices, _mrt3d_s_vec  # noqa: E402


def bench(fn, n=20, label=""):
    for _ in range(3):
        fn()
    torch.sdaa.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.sdaa.synchronize()
    dt = (time.time() - t) / n * 1000
    print(f"{label:44s} {dt:9.2f} ms", flush=True)
    return dt


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    dev = "sdaa:0"
    from _suboff_fast_step import build, stream3d_cat, FusedMRT
    eng = build(L, dev)
    torch.sdaa.synchronize()
    f = eng.f
    solid = eng.solid
    tau = eng.uc.tau
    opp = OPPOSITE.to(f.device)
    cf, ck = eng._get_collide_fn()
    ff_fn = functools.partial(b3.far_field_bc_3d, bc_config=eng._build_bc_config())
    print(f"L={L} shape={tuple(f.shape)} cells={f[0].numel()/1e6:.2f}M "
          f"tau={tau:.4f} solid={int(solid.sum())}", flush=True)

    bench(lambda: s3.stream3d(f), 10, "stream: int64 gather (current)")
    bench(lambda: stream3d_cat(f), 10, "stream: cat shifts")
    bench(lambda: torch.roll(f, (0, 0, 1), (0, 1, 2)), 10, "single full-tensor roll dim2")
    bench(lambda: cf(f, tau=tau, **ck), 10, "collide: MRT low_memory (3 gemm)")
    bench(lambda: FusedMRT(tau)(f), 10, "collide: FusedMRT (1 gemm)")
    bench(lambda: ff_fn(f, eng.uc.u_lb), 10, "far_field_bc_3d")

    fpo = None

    def nodyn_bb():
        nonlocal fpo
        fpo = f[opp].clone()
        return torch.where(solid.unsqueeze(0), fpo, f)

    bench(nodyn_bb, 10, "nodyn+bb: f[opp].clone()+where (fused)")
    bench(lambda: f.clone(), 10, "clone")
    bench(lambda: torch.isfinite(f).all(), 10, "isfinite(f).all()")
    bench(lambda: bool(torch.isfinite(f).all()), 10, "bool(isfinite(f).all()) sync")
    bench(lambda: solid.unsqueeze(0).expand_as(f), 10, "solid.unsqueeze.expand view")
    bench(lambda: lbm_step_correct(f, cf, tau, solid, eng.uc.u_lb, ff_fn,
                                   step=1, wall_treatment="bb", **ck),
          5, "FULL lbm_step_correct (baseline)")

    # full step, baseline components
    def full_orig():
        return s3.lbm_step_correct(f, cf, tau, solid, eng.uc.u_lb, ff_fn,
                                   step=1, wall_treatment="bb", **ck)

    print("done", flush=True)


if __name__ == "__main__":
    main()