"""ms/step speed probe: fast_kernel path vs (small) library path.

Usage: python speed_probe.py [C ...] [--device sdaa:0] [--steps 40]
Writes speed_probe.json next to this file.
"""
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

from fast_kernel import fast_collide_mrt, fast_stream  # noqa: E402
from run import SPONGE_ALPHA, SPONGE_C, UPSTREAM_C, DOWNSTREAM_C, HALFHEIGHT_C, airfoil_mask  # noqa: E402
from tensorlbm.boundaries import far_field_bc_2d, make_sponge_strength  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402


def build(C, device):
    dev = torch.device(device)
    re, u_in = 1000.0, 0.1
    nx = int(round((UPSTREAM_C + 1.0 + DOWNSTREAM_C) * C))
    ny = int(round(2.0 * HALFHEIGHT_C * C))
    x_le = UPSTREAM_C * C
    y_c = ny / 2.0
    nu = u_in * C / re
    tau = 0.5 + 3.0 * nu
    solid = airfoil_mask(nx, ny, float(C), 0.0, x_le, y_c, dev)
    sigma = make_sponge_strength(ny, nx, int(nx - SPONGE_C * C), int(SPONGE_C * C), power=2.0, device=dev)
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)
    rho0 = torch.ones((ny, nx), device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))
    return dev, f, tau, tau_field, solid, nx, ny


def timeit(fn, f, n, dev):
    # warmup
    for _ in range(3):
        f = fn(f)
    torch.sdaa.synchronize()
    t0 = time.time()
    for _ in range(n):
        f = fn(f)
    torch.sdaa.synchronize()
    return (time.time() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    out = {}
    for C in a.grids:
        dev, f, tau, tau_field, solid, nx, ny = build(C, a.device)
        ms_fast = timeit(lambda ff: far_field_bc_2d(fast_stream(fast_collide_mrt(ff, tau, tau_field=tau_field)), 0.1, solid), f, a.steps, dev)
        rec = {"C": C, "nx": nx, "ny": ny, "ms_per_step_fast": ms_fast}
        try:
            from tensorlbm.solver import collide_mrt, stream
            ms_lib = timeit(lambda ff: far_field_bc_2d(stream(collide_mrt(ff, tau, tau_field=tau_field)), 0.1, solid), f, min(a.steps, 8), dev)
            rec["ms_per_step_lib"] = ms_lib
            rec["speedup"] = ms_lib / ms_fast
        except Exception as e:  # noqa: BLE001
            rec["lib_err"] = str(e)[:120]
        print(json.dumps(rec), flush=True)
        out[f"C{C}"] = rec
    (_HERE.parent / "speed_probe.json").write_text(json.dumps(out, indent=2))
    print("wrote", _HERE.parent / "speed_probe.json")


if __name__ == "__main__":
    main()