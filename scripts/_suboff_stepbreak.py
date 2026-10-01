#!/usr/bin/env python3
"""Pin down the full optimized-step wall vs sum-of-components on SDAA."""
from __future__ import annotations

import time
import sys
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401

from _suboff_fast_step import build, make_step, stream3d_cat, FusedMRT
from tensorlbm import solver3d as s3  # noqa: E402


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    eng = build(L, "sdaa:0")
    torch.sdaa.synchronize()
    f0 = eng.f.clone()
    print(f"L={L} shape={tuple(eng.f.shape)}", flush=True)

    cf, ck = eng._get_collide_fn()
    fmrt = FusedMRT(eng.uc.tau)
    st = make_step(fmrt, True, {})
    s3.stream3d = stream3d_cat
    import functools
    from tensorlbm import boundaries3d as b3
    ffn = functools.partial(b3.far_field_bc_3d, bc_config=eng._build_bc_config())
    solid = eng.solid
    tau = eng.uc.tau
    u_in = eng.uc.u_lb

    # prime R
    st(eng.f, tau, solid, u_in, ffn, None, None, 1, 200)
    torch.sdaa.synchronize()

    def run(n, sync_each):
        f = f0.clone()
        torch.sdaa.synchronize()
        t = time.time()
        for i in range(n):
            f = st(f, tau, solid, u_in, ffn, None, None, i + 1, 200)
            if sync_each:
                torch.sdaa.synchronize()
        torch.sdaa.synchronize()
        return (time.time() - t) / n * 1000

    print(f"full opt step, sync every step : {run(20, True):8.2f} ms/step", flush=True)
    print(f"full opt step, sync once       : {run(20, False):8.2f} ms/step", flush=True)
    print(f"full opt step, sync once (40)  : {run(40, False):8.2f} ms/step", flush=True)

    # device-only time via events
    s = torch.sdaa.Event(enable_timing=True)
    e = torch.sdaa.Event(enable_timing=True)
    n = 20
    f = f0.clone()
    torch.sdaa.synchronize()
    s.record()
    for i in range(n):
        f = st(f, tau, solid, u_in, ffn, None, None, i + 1, 200)
    e.record()
    torch.sdaa.synchronize()
    print(f"device event time              : {s.elapsed_time(e)/n:8.2f} ms/step", flush=True)

    # cpu enqueue time (device busy) — time python issuing calls without sync
    torch.sdaa.synchronize()
    # saturate device first
    for i in range(40):
        f = st(f, tau, solid, u_in, ffn, None, None, i + 1, 200)
    t = time.time()
    for i in range(20):
        f = st(f, tau, solid, u_in, ffn, None, None, i + 1, 200)
    cpu_enq = (time.time() - t) / 20 * 1000
    torch.sdaa.synchronize()
    print(f"cpu enqueue (device saturated) : {cpu_enq:8.2f} ms/step", flush=True)

    # breakdown: collide / stream / farfield / nodynbb individually, pipelined
    from tensorlbm.d3q19 import OPPOSITE
    opp = OPPOSITE.to(eng.f.device)

    def b(label, fn, n2=20):
        for _ in range(5):
            fn()
        torch.sdaa.synchronize()
        t = time.time()
        for _ in range(n2):
            fn()
        torch.sdaa.synchronize()
        print(f"  {label:34s} {(time.time()-t)/n2*1000:8.2f} ms", flush=True)

    print("isolated components:", flush=True)
    b("fusedMRT", lambda: fmrt(f0))
    b("stream3d_cat", lambda: stream3d_cat(f0))
    b("far_field", lambda: ffn(f0, u_in))
    b("nodyn+bb fused", lambda: torch.where(solid.unsqueeze(0), f0[opp].clone(), f0))
    b("collide orig 3gemm", lambda: cf(f0, tau=tau, **ck))


if __name__ == "__main__":
    main()