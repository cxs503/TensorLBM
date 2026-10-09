#!/usr/bin/env python3
"""Try to beat tecoblas 49ms for the dense (19,19)@(19,N) MRT contraction."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401
from tensorlbm.solver3d import _get_d3q19_mrt_matrices, _mrt3d_s_vec  # noqa: E402


def bench(fn, n=20, label=""):
    try:
        for _ in range(5):
            fn()
        torch.sdaa.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.sdaa.synchronize()
        dt = (time.time() - t) / n * 1000
        print(f"{label:46s} {dt:9.2f} ms", flush=True)
        return dt
    except Exception as e:  # noqa: BLE001
        print(f"{label:46s} FAILED: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    ny = int(round(L * 102 / 48)); nx, nz = L * 6, ny
    N = nz * ny * nx
    dev = torch.device("sdaa:0")
    torch.sdaa.set_device(0)
    Q = 19
    print(f"L={L} N={N/1e6:.2f}M", flush=True)
    tau = 0.5072
    M, Minv = _get_d3q19_mrt_matrices(dev, torch.float32)
    s_vec = _mrt3d_s_vec(1.19, 1.4, 1.2, 1.19, 1.0 / tau, dtype=torch.float32, device=dev)
    R = (Minv @ (s_vec.unsqueeze(1) * M)).contiguous()
    X = torch.rand(Q, N, device=dev)
    x3 = X.view(1, Q, nz, ny, nx)

    bench(lambda: R @ X, 20, "baseline R@X (19,19)@(19,N)")

    # conv1d kernel-1
    w1 = R.view(Q, Q, 1).contiguous()
    xin = X.view(1, Q, N).contiguous()
    bench(lambda: torch.nn.functional.conv1d(xin, w1), 20, "conv1d k=1 (1,19,N)*(19,19,1)")

    # padded gemm 32x32
    Rp = torch.zeros(32, 32, device=dev); Rp[:Q, :Q] = R
    Xp = torch.zeros(32, N, device=dev); Xp[:Q] = X
    bench(lambda: (Rp @ Xp)[:Q], 20, "padded (32,32)@(32,N)")

    # padded 64
    R64 = torch.zeros(64, 64, device=dev); R64[:Q, :Q] = R
    X64 = torch.zeros(64, N, device=dev); X64[:Q] = X
    bench(lambda: (R64 @ X64)[:Q], 20, "padded (64,64)@(64,N)")

    # addcmul accumulate loop over input directions
    def acc_loop():
        out = R[:, 0].view(Q, 1) * X[0].view(1, -1)
        for p in range(1, Q):
            out.addcmul_(R[:, p].view(Q, 1), X[p].view(1, -1))
        return out

    bench(acc_loop, 10, "addcmul accumulate (19 terms)")

    # einsum with output index first (transposed contraction)
    bench(lambda: torch.einsum("qp,pn->qn", R, X), 20, "einsum qp,pn->qn")

    # bmm batched: reshape N -> (B, N/B)
    for B in (4, 16):
        Nb = N // B
        Xb = X[:, : Nb * B].view(Q, B, Nb).permute(1, 0, 2).contiguous()  # (B,Q,Nb)
        Rb = R.unsqueeze(0).expand(B, Q, Q).contiguous()
        bench(lambda: torch.bmm(Rb, Xb), 10, f"bmm B={B}")

    # half-split gemms
    h = N // 2
    bench(lambda: torch.cat([R @ X[:, :h], R @ X[:, h:]], dim=1), 10, "2x half-size gemm")

    # gemm on transposed layout (row-major N-first)
    Xt = X.t().contiguous()
    bench(lambda: (Xt @ R.t()), 20, "Xt(N,19)@R.T")

    print("done", flush=True)


if __name__ == "__main__":
    main()