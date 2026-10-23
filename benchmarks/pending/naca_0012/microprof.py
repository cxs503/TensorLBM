"""Micro-profile of the fast step components + alternative O(1)-launch bulk MRT.

Usage: python microprof.py [C] [--device sdaa:0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fast_kernel import _mats  # noqa: E402
from speed_probe import build  # noqa: E402


def timeit(fn, f, n, dev):
    for _ in range(3):
        f = fn(f)
    torch.sdaa.synchronize()
    t0 = time.time()
    for _ in range(n):
        f = fn(f)
    torch.sdaa.synchronize()
    return (time.time() - t0) / n * 1e3


def bulk_broadcast(f, tau, chunk=128):
    """new[i] = f[i] - sum_j A0[i,j] d[j] with 2 launches per chunk."""
    A0, K, m7, m8, mi7, mi8 = _mats(f.device, f.dtype, tau)
    ny, nx = f.shape[1], f.shape[2]
    from tensorlbm.d2q9 import C, equilibrium
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    c = C.to(f.device).to(f.dtype)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    d = f - equilibrium(rho, ux, uy)
    out = torch.empty_like(f)
    for y0 in range(0, ny, chunk):
        y1 = min(ny, y0 + chunk)
        dd = d[:, y0:y1]  # 9, h, nx
        acc = (A0[:, :, None, None] * dd[None]).sum(dim=1)  # 9, h, nx
        out[:, y0:y1] = f[:, y0:y1] - acc
    return out


def bulk_broadcast_full(f, tau):
    """single broadcast (9,9,ny,nx) then sum -> 2 launches (memory heavy)."""
    A0, K, m7, m8, mi7, mi8 = _mats(f.device, f.dtype, tau)
    from tensorlbm.d2q9 import C, equilibrium
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    c = C.to(f.device).to(f.dtype)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    d = f - equilibrium(rho, ux, uy)
    acc = (A0[:, :, None, None] * d[None]).sum(dim=1)
    return f - acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("C", type=int, nargs="?", default=64)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    dev, f, tau, tau_field, solid, nx, ny = build(a.C, a.device)

    from fast_kernel import fast_collide_mrt, fast_stream
    r = {}
    r["collide_fast"] = timeit(lambda ff: fast_collide_mrt(ff, tau, tau_field=tau_field), f, a.steps, dev)
    r["stream_fast"] = timeit(lambda ff: fast_stream(ff), f, a.steps, dev)
    from tensorlbm.boundaries import far_field_bc_2d
    r["bc"] = timeit(lambda ff: far_field_bc_2d(ff, 0.1, solid), f, a.steps, dev)

    def bcast(ff):
        return fast_stream(bulk_broadcast(ff, tau))
    r["collide_bcast128"] = timeit(lambda ff: bulk_broadcast(ff, tau, 128), f, a.steps, dev)
    r["collide_bcast256"] = timeit(lambda ff: bulk_broadcast(ff, tau, 256), f, a.steps, dev)
    try:
        r["collide_bcast_full"] = timeit(lambda ff: bulk_broadcast_full(ff, tau), f, a.steps, dev)
    except Exception as e:  # noqa: BLE001
        r["collide_bcast_full_err"] = str(e)[:100]

    # correctness vs fast loop
    ref = fast_collide_mrt(f, tau, tau_field=tau_field)
    for k, fn in [("bcast128", lambda: bulk_broadcast(f, tau, 128)),
                  ("bcast_full", lambda: bulk_broadcast_full(f, tau))]:
        try:
            r[f"diff_{k}"] = (fn() - ref).abs().max().item()
        except Exception as e:  # noqa: BLE001
            r[f"diff_{k}_err"] = str(e)[:80]

    import json
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, indent=2))


if __name__ == "__main__":
    main()