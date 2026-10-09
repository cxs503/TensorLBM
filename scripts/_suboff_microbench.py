#!/usr/bin/env python3
"""Microbenchmark SDAA primitive ops relevant to the LBM step."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401
from tensorlbm.d3q19 import C  # noqa: E402

DEV = torch.device("sdaa:0")
torch.sdaa.set_device(0)


def bench(fn, n=10, warm=3, label=""):
    for _ in range(warm):
        fn()
    torch.sdaa.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.sdaa.synchronize()
    dt = (time.time() - t) / n * 1000
    print(f"{label:42s} {dt:9.3f} ms", flush=True)
    return dt


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    # domain_lu for padding (1,4,1,1,1,1): x=6L, y/z=2L+2R? approximate; use measured L48 ratio
    # L=48 -> (288,102,102); scale linearly
    ny = int(round(L * 102 / 48))
    nx = L * 6
    nz = ny
    N = nz * ny * nx
    Q = 19
    print(f"L={L} grid {nz}x{ny}x{nx} cells={N/1e6:.2f}M f_bytes={Q*N*4/1e6:.1f}MB", flush=True)

    f = torch.rand(Q, nz, ny, nx, device=DEV, dtype=torch.float32)

    # 0. achievable bandwidth: contiguous copy
    def copy():
        return f.clone()

    dt = bench(copy, label="clone f (contig copy)")
    print(f"   -> BW ~ {Q*N*4*2/dt/1e6:.1f} GB/s", flush=True)

    def mul_scalar():
        return f * 1.0001

    dt = bench(mul_scalar, label="f * scalar")
    print(f"   -> BW ~ {Q*N*4*2/dt/1e6:.1f} GB/s", flush=True)

    # 1. current stream3d gather
    c = C.to(DEV)
    z_src = (torch.arange(nz, device=DEV).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
    y_src = (torch.arange(ny, device=DEV).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
    x_src = (torch.arange(nx, device=DEV).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
    q_idx64 = torch.arange(Q, device=DEV).view(Q, 1, 1, 1).expand(Q, nz, ny, nx)
    z_idx64 = z_src.view(Q, nz, 1, 1).expand(Q, nz, ny, nx)
    y_idx64 = y_src.view(Q, 1, ny, 1).expand(Q, nz, ny, nx)
    x_idx64 = x_src.view(Q, 1, 1, nx).expand(Q, nz, ny, nx)

    def stream_gather64():
        return f[q_idx64, z_idx64, y_idx64, x_idx64]

    bench(stream_gather64, label="stream: 4D int64 gather (current)")

    # int32 variant
    q32 = q_idx64.to(torch.int32)
    z32 = z_idx64.to(torch.int32)
    y32 = y_idx64.to(torch.int32)
    x32 = x_idx64.to(torch.int32)

    def stream_gather32():
        return f[q32, z32, y32, x32]

    bench(stream_gather32, label="stream: 4D int32 gather")

    # flat single-index gather int64
    flat64 = (
        q_idx64 * (nz * ny * nx) + z_idx64 * (ny * nx) + y_idx64 * nx + x_idx64
    ).reshape(Q, N)

    def stream_flat64():
        return f.reshape(Q, N).gather(1, flat64).reshape(Q, nz, ny, nx)

    bench(stream_flat64, label="stream: flat int64 gather (1 idx)")

    # 2. slice-based streaming, no wraparound indices (interior); wrap via narrow+copies
    def stream_slices():
        out = torch.empty_like(f)
        # rest
        out[0].copy_(f[0])
        for q in range(1, Q):
            sx, sy, sz = int(c[q, 0]), int(c[q, 1]), int(c[q, 2])
            # destination region (interior, no wrap) and source
            zi0, zi1 = max(0, -sz), nz - max(0, sz)
            yi0, yi1 = max(0, -sy), ny - max(0, sy)
            xi0, xi1 = max(0, -sx), nx - max(0, sx)
            zs0, zs1 = max(0, sz), nz - max(0, -sz)
            ys0, ys1 = max(0, sy), ny - max(0, -sy)
            xs0, xs1 = max(0, sx), nx - max(0, -sx)
            tgt = out[q, zi0:zi1, yi0:yi1, xi0:xi1]
            src = f[q, zs0:zs1, ys0:ys1, xs0:xs1]
            tgt.copy_(src)
            # wrap pieces (periodic) — only boundary slabs
            if sz:
                if sz > 0:
                    out[q, 0:sz, yi0:yi1, xi0:xi1].copy_(f[q, nz - sz:nz, ys0:ys1, xs0:xs1])
                else:
                    out[q, zi1:nz, yi0:yi1, xi0:xi1].copy_(f[q, 0:-sz, ys0:ys1, xs0:xs1])
            if sy:
                if sy > 0:
                    out[q, :, 0:sy, xi0:xi1].copy_(f[q, :, ny - sy:ny, xs0:xs1])
                else:
                    out[q, :, yi1:ny, xi0:xi1].copy_(f[q, :, 0:-sy, xs0:xs1])
            if sx:
                if sx > 0:
                    out[q, :, :, 0:sx].copy_(f[q, :, :, nx - sx:nx])
                else:
                    out[q, :, :, xi1:nx].copy_(f[q, :, :, 0:-sx])
        return out

    try:
        bench(stream_slices, label="stream: slice copies (19 dirs)")
    except Exception as e:  # noqa: BLE001
        print("   slice stream failed:", e, flush=True)

    # 3. MRT matmul
    from tensorlbm.solver3d import _get_d3q19_mrt_matrices

    M, Minv = _get_d3q19_mrt_matrices(DEV, torch.float32)
    ff = f.reshape(Q, -1)

    def mm():
        return M @ ff

    dt = bench(mm, label="MRT: (19,19)@(19,N) fp32 gemm")
    print(f"   -> BW ~ {2*Q*N*4/dt/1e6:.1f} GB/s", flush=True)

    if Minv is not None:
        def mm2():
            return Minv @ ff

        bench(mm2, label="MRT: inv matmul")

    # 4. isfinite + all
    def finite_all():
        return torch.isfinite(f).all()

    bench(finite_all, label="isfinite(f).all()  [no sync]")
    bench(lambda: bool(torch.isfinite(f).all()), label="bool(isfinite(f).all())  [sync]")

    # 5. where masked restore full-domain
    mask = torch.rand(nz, ny, nx, device=DEV) < 0.001
    fpre = f.clone()

    def where_restore():
        return torch.where(mask.unsqueeze(0), fpre, f)

    bench(where_restore, label="NoDynamics: full-domain where")

    # masked index restore (current style)
    def mask_restore():
        fcopy = f.clone()
        fcopy[:, mask] = fpre[:, mask]
        return fcopy

    try:
        bench(mask_restore, label="NoDynamics: bool-mask index assign (current)")
    except Exception as e:  # noqa: BLE001
        print("   mask restore failed:", e, flush=True)

    # precomputed flat index restore
    sidx = mask.reshape(-1).nonzero().squeeze(1)

    def idx_restore():
        fcopy = f.clone()
        ffl = fcopy.reshape(Q, -1)
        ffl[:, sidx] = fpre.reshape(Q, -1)[:, sidx]
        return fcopy

    bench(idx_restore, label="NoDynamics: precomputed long-index assign")

    # 6. reduce sum for correct_mass
    def summ():
        return f.sum()

    bench(summ, label="f.sum()  [full reduce]")


if __name__ == "__main__":
    main()