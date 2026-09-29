#!/usr/bin/env python3
"""Benchmark MRT collision building blocks + gemm shapes on SDAA."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401

from tensorlbm.d3q19 import (  # noqa: E402
    equilibrium3d, equilibrium3d_low_memory,
    macroscopic3d, macroscopic3d_low_memory,
)
from tensorlbm.solver3d import (  # noqa: E402
    _get_d3q19_mrt_matrices, _mrt3d_s_vec, collide_mrt3d, collide_mrt3d_low_memory,
)


def bench(fn, n=20, label=""):
    for _ in range(5):
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
    ny = int(round(L * 102 / 48)); nx, nz = L * 6, ny
    N = nz * ny * nx
    dev = torch.device("sdaa:0")
    torch.sdaa.set_device(0)
    Q = 19
    print(f"L={L} N={N/1e6:.2f}M", flush=True)
    f = torch.rand(Q, nz, ny, nx, device=dev)
    tau = 0.5072
    M, Minv = _get_d3q19_mrt_matrices(dev, torch.float32)
    s_vec = _mrt3d_s_vec(1.19, 1.4, 1.2, 1.19, 1.0 / tau, dtype=torch.float32, device=dev)
    R = Minv @ (s_vec.unsqueeze(1) * M)
    ff = f.reshape(Q, -1).contiguous()
    ft = ff.t().contiguous()  # (N,19)

    bench(lambda: macroscopic3d(f), 20, "macroscopic3d (vectorized)")
    bench(lambda: macroscopic3d_low_memory(f), 20, "macroscopic3d_low_memory")
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    bench(lambda: equilibrium3d(rho, ux, uy, uz), 20, "equilibrium3d (vectorized)")
    bench(lambda: equilibrium3d_low_memory(rho, ux, uy, uz), 20, "equilibrium3d_low_memory")

    print("--- gemm shapes (19x19 @ 19xN) ---", flush=True)
    bench(lambda: R @ ff, 20, "R(19,19) @ X(19,N)")
    bench(lambda: torch.einsum("ij,jn->in", R, ff), 20, "einsum ij,jn->in")
    bench(lambda: torch.mm(R, ff), 20, "torch.mm")
    bench(lambda: torch.tensordot(R, ff, dims=1), 20, "tensordot")
    bench(lambda: ft @ R.t(), 20, "(N,19) @ R.T(19,19)")
    bench(lambda: ff.t().matmul(R.t()), 20, "X.T @ R.T")
    if hasattr(torch, "addmm"):
        b = torch.zeros_like(ff)
        bench(lambda: torch.addmm(b, R, ff), 20, "addmm(b,R,X)")

    print("--- full collision ---", flush=True)
    bench(lambda: collide_mrt3d_low_memory(f, tau), 20, "collide_mrt3d_low_memory (3gemm)")
    bench(lambda: collide_mrt3d(f, tau), 20, "collide_mrt3d (vec, 3gemm)")

    def fused1(f, tau=tau):
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        return (f.reshape(Q, -1) - R @ (f - feq).reshape(Q, -1)).reshape(f.shape)

    bench(lambda: fused1(f), 20, "FusedMRT (vec macro+eq, 1gemm)")
    print("done", flush=True)


if __name__ == "__main__":
    main()