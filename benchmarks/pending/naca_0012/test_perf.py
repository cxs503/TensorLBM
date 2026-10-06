"""Controlled interleaved A/B timing of the naca0012 step components."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from tensorlbm.boundaries import far_field_bc_2d  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402

from fast_kernel import _mats, fast_collide_mrt, fast_stream  # noqa: E402
from speed_probe import build  # noqa: E402


def loop_collide(f, tau, tau_field=None):
    A0, K, m7, m8, mi7, mi8 = _mats(f.device, f.dtype, tau)
    ny, nx = f.shape[1], f.shape[2]
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    from tensorlbm.d2q9 import C
    c = C.to(f.device).to(f.dtype)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    d = f - equilibrium(rho, ux, uy)
    if tau_field is None:
        new = torch.empty_like(f)
        for i in range(9):
            acc = A0[i, 0] * d[0]
            for j in range(1, 9):
                acc = acc + A0[i, j] * d[j]
            new[i] = f[i] - acc
        return new
    p7 = (m7.view(9, 1, 1) * d).sum(dim=0)
    p8 = (m8.view(9, 1, 1) * d).sum(dim=0)
    delta = (1.0 / tau_field) - (1.0 / tau)
    new = torch.empty_like(f)
    for i in range(9):
        acc = A0[i, 0] * d[0]
        for j in range(1, 9):
            acc = acc + A0[i, j] * d[j]
        new[i] = f[i] - acc - (mi7[i] * p7 + mi8[i] * p8) * delta
    return new


def bench(fn, f, n, dev, reps=5):
    best = 1e9
    for _ in range(reps):
        for _ in range(2):
            f = fn(f)
        torch.sdaa.synchronize()
        t0 = time.time()
        for _ in range(n):
            f = fn(f)
        torch.sdaa.synchronize()
        best = min(best, (time.time() - t0) / n * 1e3)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("C", type=int, nargs="?", default=64)
    ap.add_argument("--device", default="sdaa:7")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    dev, f, tau, tau_field, solid, nx, ny = build(a.C, a.device)
    r = {"C": a.C, "device": a.device}
    r["bcast_collide"] = bench(lambda ff: fast_collide_mrt(ff, tau, tau_field=tau_field), f, a.steps, dev)
    r["loop_collide"] = bench(lambda ff: loop_collide(ff, tau, tau_field=tau_field), f, a.steps, dev)
    try:
        from tensorlbm.solver import collide_mrt, stream
        r["lib_collide"] = bench(lambda ff: collide_mrt(ff, tau, tau_field=tau_field), f, min(a.steps, 6), dev, reps=2)
        r["lib_stream"] = bench(lambda ff: stream(ff), f, min(a.steps, 6), dev, reps=2)
    except Exception as e:  # noqa: BLE001
        r["lib_err"] = str(e)[:100]
    r["fast_stream"] = bench(lambda ff: fast_stream(ff), f, a.steps, dev)
    r["bc"] = bench(lambda ff: far_field_bc_2d(ff, 0.1, solid), f, a.steps, dev)
    step_b = lambda ff: far_field_bc_2d(fast_stream(fast_collide_mrt(ff, tau, tau_field=tau_field)), 0.1, solid)
    step_l = lambda ff: far_field_bc_2d(fast_stream(loop_collide(ff, tau, tau_field=tau_field)), 0.1, solid)
    r["step_bcast_total"] = bench(step_b, f, a.steps, dev)
    r["step_loop_total"] = bench(step_l, f, a.steps, dev)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, indent=2))


if __name__ == "__main__":
    main()