#!/usr/bin/env python3
"""Run a single viscoplastic turbulent channel case.

This is the worker script for the cluster campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.viscoplastic_channel3d import ViscoplasticChannelConfig, run_viscoplastic_channel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tau-y", type=float, default=0.0)
    ap.add_argument("--K", type=float, default=0.02)
    ap.add_argument("--n", type=float, default=1.0)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.006)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ViscoplasticChannelConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        tau_y=args.tau_y, K=args.K, n=args.n,
        re_tau=args.re_tau, u_tau=args.u_tau,
        steps=args.steps, warmup_steps=args.warmup,
        output=out_dir / f"{args.name}_result.json",
    )

    print(f"Running case: {args.name}")
    print(f"  Y = {cfg.yield_number:.2f}")
    print(f"  n = {cfg.n}")
    print(f"  tau_y = {cfg.tau_y:.6e}")
    print(f"  Grid: {cfg.nx}x{cfg.ny}x{cfg.nz}")

    t0 = time.time()
    result = run_viscoplastic_channel(cfg)
    elapsed = time.time() - t0

    result["name"] = args.name
    result["elapsed_s"] = elapsed

    out_path = out_dir / f"{args.name}_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Done in {elapsed:.1f}s -> {out_path}")

    # Print digest
    stats = result["statistics"]
    digest = {
        "name": args.name,
        "Y": cfg.yield_number,
        "n": cfg.n,
        "u_tau_measured": result["derived"]["u_tau_measured"],
        "tke_max": float(stats["tke"].max()),
        "u_bulk": float(stats["u_mean"][cfg.ny // 2]),
        "elapsed_s": elapsed,
    }
    print("DIGEST " + json.dumps(digest, separators=(",", ":")))


if __name__ == "__main__":
    main()
