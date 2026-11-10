#!/usr/bin/env python3
"""IN99-N03: run one native LES channel case (TensorLBM paper-grade campaign).

Example
-------
    python3 benchmarks/in99_n03_les_channel/run_case.py \
        --name R180_h32 --re-tau 178.12 --ny 66 --nx 200 --nz 100 \
        --steps 90000 --warmup 15000 --cs 0.1 --threads 32 \
        --out-dir runs/in99_n03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.turbulence.les_channel import (  # noqa: E402
    LesChannelConfig,
    run_native,
    wall_units,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=66)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--steps", type=int, default=90000)
    ap.add_argument("--warmup", type=int, default=15000)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--re-tau", type=float, default=178.12)
    ap.add_argument("--u-tau", type=float, default=0.006)
    ap.add_argument("--cs", type=float, default=0.1)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.1875)
    ap.add_argument("--perturb", type=float, default=0.05)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config = LesChannelConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, steps=args.steps, warmup=args.warmup,
        sample=args.sample, re_tau=args.re_tau, u_tau=args.u_tau, cs=args.cs,
        lambda_=args.lambda_, perturb=args.perturb, perturb_modes=args.modes,
        seed=args.seed, threads=args.threads,
    )
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    result = run_native(config, out_dir, args.name, overwrite=args.overwrite)
    profile = wall_units(result)
    digest = {
        "case": args.name,
        "cells": config.cells,
        "eddy_turnovers": config.eddy_turnovers,
        "elapsed_seconds": result["provenance"]["elapsed_seconds"],
        "mega_cell_steps_per_second": result["provenance"]["mega_cell_steps_per_second"],
        "u_tau_routes": profile["u_tau"],
        "max_pairwise_u_tau_deviation_pct": profile["max_pairwise_u_tau_deviation_pct"],
        "wall_law": profile["wall_law"],
        "u_bulk_plus": profile["u_bulk_plus"],
        "cf": profile["cf"],
        "peaks": profile["peaks"],
        "anisotropy": profile["anisotropy"],
        "diagnostics": result["diagnostics"],
    }
    (out_dir / f"{args.name}_digest.json").write_text(json.dumps(digest, indent=2) + "\n")
    print(json.dumps(digest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())