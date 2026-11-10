"""Wall-clock benchmark of the torch mirror of the compressible Euler kernel.

The four-grid resolvability ladder tops out at 1200x400 cells run to ``t = 4``
(``~3.2e4`` SSP-RK2 steps).  Before committing cluster hours to it, this script
measures the throughput of the reference numpy kernel and of the torch mirror
(same discretisation, see ``tensorlbm/compressible_shock/fast2d.py``) on the
production grids and prints a projected wall time for each ``t_end``.

Usage::

    python scripts/bench_torch_backend.py --grids 600x200 1200x400 --t-end 0.02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.compressible_shock.cases import forward_facing_step_case  # noqa: E402


def bench(nx, ny, t_end, backend, threads=None):
    solver, _ = forward_facing_step_case(nx=nx, ny=ny, mach=3.0, t_end=0.0)
    if backend == "torch":
        from tensorlbm.compressible_shock.fast2d import to_torch
        solver = to_torch(solver, threads=threads)
    t0 = time.time()
    solver.advance(t_end)
    wall = time.time() - t0
    return {
        "nx": nx, "ny": ny, "backend": backend, "threads": threads,
        "t_end": t_end, "wall_s": wall,
        "cell_steps_per_s": nx * ny * max(1, int(round(t_end / 1e-9))) / wall,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=["600x200", "1200x400"])
    ap.add_argument("--t-end", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out = []
    for g in args.grids:
        nx, ny = (int(v) for v in g.lower().split("x"))
        for backend in ("numpy", "torch"):
            rec = bench(nx, ny, args.t_end, backend,
                        threads=(args.threads or None) if backend == "torch" else None)
            out.append(rec)
            print(json.dumps(rec))

    # projected wall time to a list of production end times
    print("\nprojected wall time (s) for full runs:")
    for rec in out:
        if rec["backend"] != "torch":
            continue
        per_t = rec["wall_s"] / rec["t_end"]
        print(json.dumps({"grid": f'{rec["nx"]}x{rec["ny"]}',
                          "projected": {t: round(per_t * t, 1) for t in (2.0, 4.0, 6.0)}}))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())