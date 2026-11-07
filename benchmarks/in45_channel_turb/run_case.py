#!/usr/bin/env python3
"""IN45-N01: wall-resolved turbulent channel case runner (production kernel).

Runs one D3Q19 wall-resolved channel case with the *production* solver in
``tensorlbm.boundary.wall_resolved_channel3d`` and dumps the reduced result
JSON (mean profile + Reynolds stresses + stationarity diagnostics).

Cost note (this node's key scheduling insight)
---------------------------------------------
At fixed Re_tau the lattice friction velocity ``u_tau`` is a *free* scale: it
only fixes the time step.  One outer (eddy-turnover) time is

    T_turn = h / u_tau      [lattice steps]

so the number of steps needed for a fixed amount of *wall-unit* averaging
time scales as ``1 / u_tau``.  The predecessor campaign (P20-N01) used
``u_tau = 0.003`` (T_turn = 20667 for h = 62) and therefore collected only
1.45 outer times in 30 000 statistics steps.  Here ``u_tau`` is raised until
the core velocity is at the edge of the incompressible LBM envelope
(``u_max = 18 u_tau <= 0.11``), which buys ~2.5x more averaging per unit of
CPU time at the same discretisation error budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from tensorlbm.boundary.wall_resolved_channel3d import (  # noqa: E402
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.006)
    ap.add_argument("--steps", type=int, default=26000)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--sample-interval", type=int, default=5)
    ap.add_argument("--collision", default="cumulant")
    ap.add_argument("--smagorinsky-cs", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}_result.json"
    ckpt = out_dir / f"{args.name}_ckpt.pt"

    config = WallResolvedChannel3DConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        steps=args.steps,
        warmup_steps=args.warmup,
        sample_interval=args.sample_interval,
        report_interval=250,
        checkpoint_interval=max(args.steps // 4, 1000),
        collision_model=args.collision,
        smagorinsky_cs=args.smagorinsky_cs,
        compile_natural_kbc=(args.collision == "natural_kbc"),
        device="cpu",
        output=out,
        checkpoint=ckpt,
    )
    started = time.time()
    result = run_wall_resolved_channel3d(config)
    elapsed = time.time() - started
    payload = {
        "schema": "in45-channel-case-v1",
        "name": args.name,
        "elapsed_seconds": elapsed,
        "lattice_ratings": {
            "cells": args.nx * args.ny * args.nz,
            "mega_cell_steps_per_second": args.nx
            * args.ny
            * args.nz
            * args.steps
            / max(elapsed, 1e-9)
            / 1e6,
        },
        "result": result,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[in45] {args.name} done in {elapsed:.1f}s -> {out}")
    # Compact, machine-readable digest on stdout: cluster jobs are fetched from
    # the job log first, so the numbers must not live only in the checkout.
    statistics = result["statistics"]
    digest = {
        "name": args.name,
        "elapsed_seconds": elapsed,
        "mega_cell_steps_per_second": args.nx
        * args.ny
        * args.nz
        * args.steps
        / max(elapsed, 1e-9)
        / 1e6,
        "configuration": result["configuration"],
        "derived": result["derived"],
        "mean_velocity_profile": statistics["mean_velocity_profile"],
        "reynolds_stress_profiles": statistics["reynolds_stress_profiles_uu_vv_ww_uv"],
        "statistics_steps": statistics["statistics_steps"],
        "profile_samples": statistics["profile_samples"],
        "statistics_eddy_turnovers": statistics["statistics_eddy_turnovers"],
        "stationarity_friction_velocity": statistics.get("stationarity_friction_velocity"),
        "recent_friction_velocity_error_pct": statistics.get(
            "recent_friction_velocity_error_pct"
        ),
        "stationarity_half_window_drift_fraction": statistics.get(
            "stationarity_half_window_drift_fraction"
        ),
    }
    print("IN45_DIGEST " + json.dumps(digest, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())