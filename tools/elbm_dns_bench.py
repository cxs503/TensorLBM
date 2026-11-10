"""Throughput benchmark of the ELBM channel kernel (one grid, ``--steps`` steps).

Used to size the wall-resolved DNS runs: the entropy solve is ``(1+n_iter)*27``
logarithms per cell, so the cost per cell-step is ~10x a BGK collision and the
grid/time-step budget has to be measured, not guessed.

Example::

    OMP_NUM_THREADS=64 python tools/elbm_dns_bench.py --nx 384 --ny 194 --nz 192
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tensorlbm.elbm_channel_dns import DnsChannelConfig, initial_field
from tensorlbm.elbm_channel_native import native_channel_step_inplace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=194)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.010)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args(argv)

    cfg = DnsChannelConfig(nx=args.nx, ny=args.ny, nz=args.nz, re_tau=args.re_tau, u_tau=args.u_tau)
    cfg.validate()
    f, alpha = initial_field(cfg)
    scratch = np.empty_like(f)
    stats = np.zeros(5, dtype=np.float64)
    kw = dict(nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, tau=cfg.tau, force=cfg.body_force)
    native_channel_step_inplace(f, scratch, alpha, **kw, stats=stats)  # warm-up / page-in
    t0 = time.time()
    for _ in range(args.steps):
        native_channel_step_inplace(f, scratch, alpha, **kw, stats=stats)
    dt = (time.time() - t0) / args.steps
    print(
        json.dumps(
            {
                "grid": [cfg.nx, cfg.ny, cfg.nz],
                "cells": cfg.cells,
                "delta_plus": cfg.delta_plus,
                "tau": cfg.tau,
                "s_per_step": dt,
                "Mcell_per_s": cfg.cells / dt / 1e6,
                "alpha_mean": float(stats[0]),
                "finite": bool(np.isfinite(f).all()),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())