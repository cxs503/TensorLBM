#!/usr/bin/env python3
"""Benchmark alternative streaming implementations on SDAA."""
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
    try:
        for _ in range(warm):
            fn()
        torch.sdaa.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.sdaa.synchronize()
        dt = (time.time() - t) / n * 1000
        print(f"{label:46s} {dt:9.3f} ms", flush=True)
        return dt
    except Exception as e:  # noqa: BLE001
        print(f"{label:46s} FAILED: {e}", flush=True)
        return None


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    ny = int(round(L * 102 / 48))
    nx, nz = L * 6, ny
    N = nz * ny * nx
    Q = 19
    print(f"L={L} grid {nz}x{ny}x{nx} cells={N/1e6:.2f}M", flush=True)
    f = torch.rand(Q, nz, ny, nx, device=DEV, dtype=torch.float32)
    c = C.to(DEV)
    shifts = [(int(c[q, 0]), int(c[q, 1]), int(c[q, 2])) for q in range(Q)]

    # A. whole-tensor roll along each axis (all q same shift) — measures roll cost
    bench(lambda: torch.roll(f, (1,), (3,)), label="roll all-q dx=+1 (dim3)")
    bench(lambda: torch.roll(f, (1,), (2,)), label="roll all-q dy=+1 (dim2)")
    bench(lambda: torch.roll(f, (1,), (1,)), label="roll all-q dz=+1 (dim1)")

    # B. cat-based single-axis shift (periodic), whole tensor
    def cat_last():
        return torch.cat([f[..., -1:], f[..., :-1]], dim=3)

    bench(cat_last, label="cat dx=+1 (dim3)")

    def cat_mid():
        return torch.cat([f[:, :, -1:, :], f[:, :, :-1, :]], dim=2)

    bench(cat_mid, label="cat dy=+1 (dim2)")

    def cat_first():
        return torch.cat([f[:, -1:, :, :], f[:, :-1, :, :]], dim=1)

    bench(cat_first, label="cat dz=+1 (dim1)")

    # C. per-direction roll (19 rolls on 1-slab each) — proper periodic streaming
    def stream_roll():
        out = torch.empty_like(f)
        for q in range(Q):
            sx, sy, sz = shifts[q]
            if sx == 0 and sy == 0 and sz == 0:
                out[q].copy_(f[q])
            else:
                out[q] = torch.roll(f[q], (sz, sy, sx), (0, 1, 2))
        return out

    bench(stream_roll, label="stream: per-q roll (19 dirs)")

    # D. per-direction cat-based (shift each axis by cat) on 3D slab
    def shift3(d, sx, sy, sz):
        if sz > 0:
            d = torch.cat([d[-sz:], d[:-sz]], dim=0)
        elif sz < 0:
            d = torch.cat([d[-sz:], d[:-sz]], dim=0)
        if sy > 0:
            d = torch.cat([d[:, -sy:], d[:, :-sy]], dim=1)
        elif sy < 0:
            d = torch.cat([d[:, -sy:], d[:, :-sy]], dim=1)
        if sx > 0:
            d = torch.cat([d[:, :, -sx:], d[:, :, :-sx]], dim=2)
        elif sx < 0:
            d = torch.cat([d[:, :, -sx:], d[:, :, :-sx]], dim=2)
        return d

    def stream_cat():
        out = torch.empty_like(f)
        for q in range(Q):
            sx, sy, sz = shifts[q]
            if sx == 0 and sy == 0 and sz == 0:
                out[q].copy_(f[q])
            else:
                out[q] = shift3(f[q], sx, sy, sz)
        return out

    bench(stream_cat, label="stream: per-q cat-shift (19 dirs)")

    # E. vectorised per-axis shifts across all q at once (shift groups)
    #    group directions by sign of each axis; do 3 whole-tensor cats per sign set
    #    instead: apply shift along an axis to ALL q, where each q needs its own sign.
    #    Build sign-specific index vectors and use index_select-free cat per group.
    def stream_grouped():
        out = torch.empty_like(f)
        # handle rest
        out[0].copy_(f[0])
        for axis, dim in ((3, 3), (2, 2), (1, 1)):
            pass
        return out

    # F. try narrow+copy_ contiguous (whole-tensor shift of last dim)
    def narrow_last():
        out = torch.empty_like(f)
        out[..., 1:].copy_(f[..., :-1])
        out[..., 0].copy_(f[..., -1])
        return out

    bench(narrow_last, label="narrow+copy dx=+1 (dim3)")

    # G. flatten-shift: view (Q*nz*ny, nx) and cat rows
    def flat_cat():
        v = f.reshape(Q * nz * ny, nx)
        return torch.cat([v[:, -1:], v[:, :-1]], dim=1).reshape(Q, nz, ny, nx)

    bench(flat_cat, label="flat row-cat dx=+1")

    # H. index_select on last dim with cached int index (contiguous-ish gather)
    xidx = torch.cat([torch.tensor([nx - 1], device=DEV),
                      torch.arange(0, nx - 1, device=DEV)])

    def idxsel_last():
        v = f.reshape(Q * nz * ny, nx)
        return v.index_select(1, xidx).reshape(Q, nz, ny, nx)

    bench(idxsel_last, label="index_select dx=+1 (dim3)")

    print("done", flush=True)


if __name__ == "__main__":
    main()