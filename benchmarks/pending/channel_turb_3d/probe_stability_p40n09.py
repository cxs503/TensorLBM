#!/usr/bin/env python3
"""Stability / cost probe for the P40-N09 case matrix (cluster-side, cheap).

The decomposition campaign needs a grid on which an unclosed (C_s = 0) D3Q19
cumulant channel at Re_tau = 180 actually *stays* turbulent for tens of eddy
turnovers.  Rather than guess, this probe sweeps the two knobs that control the
near-wall stiffness (initial perturbation amplitude, collision model) at the
production grid and reports, per variant, health, wall-shear closure, crossflow
activity and throughput.

Run with ``--threads N`` on the compute node; it is deliberately short (a few
hundred steps) so that a wrong parameter is rejected for the price of minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("TENSORLBM_SRC", "src"))

import torch  # noqa: E402

from tensorlbm.boundary.wall_resolved_channel3d import (  # noqa: E402
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)


def build(
    *,
    name: str,
    out_dir: Path,
    ny: int,
    u_tau: float,
    re_tau: float,
    steps: int,
    warmup: int,
    box_factor: float,
    cs: float,
    model: str,
    perturbation: float,
    noise: float,
    modes: int,
    batch_count: int,
) -> WallResolvedChannel3DConfig:
    delta = 0.5 * (ny - 2)
    nx = int(round(box_factor * 2.0 * math.pi * delta))
    nz = int(round(4.0 * math.pi * delta / 3.0))
    return WallResolvedChannel3DConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        re_tau=float(re_tau),
        u_tau=float(u_tau),
        steps=int(steps),
        warmup_steps=int(warmup),
        sample_interval=5,
        report_interval=max(20, steps // 100),
        checkpoint_interval=10**9,
        collision_model=model,
        smagorinsky_cs=cs,
        compile_natural_kbc=False,
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=perturbation,
        random_noise_fraction=noise,
        spectral_mode_count=modes,
        spectral_max_wavenumber=4,
        minimum_statistics_eddy_turnovers=0.01,
        stationarity_window_eddy_turnovers=0.2,
        seed=20261108,
        device="cpu",
        momentum_exchange_diagnostic=True,
        profile_batch_count=batch_count,
        wall_exchange_series_stride=5,
        output=out_dir / f"{name}_result.json",
        checkpoint=out_dir / f"{name}_state.ckpt",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs_probe"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("LBM_THREADS", "64")))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--u-tau", type=float, default=0.01)
    parser.add_argument("--re-tau", type=float, default=180.0)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="JSON file with an explicit variant list (overrides the defaults)",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        dict(name="p_noise10", cs=0.0, model="cumulant", perturbation=1.0, noise=0.5, modes=32),
        dict(name="p_soft", cs=0.0, model="cumulant", perturbation=0.3, noise=0.1, modes=16),
        dict(name="p_kbc", cs=0.0, model="natural_kbc", perturbation=1.0, noise=0.5, modes=32),
        dict(name="p_cs01", cs=0.10, model="cumulant", perturbation=1.0, noise=0.5, modes=32),
    ]
    if args.matrix:
        variants = json.loads(Path(args.matrix).read_text())
    records = []
    for variant in variants:
        config = build(
            out_dir=args.out_dir,
            ny=int(variant.pop("ny", args.ny)),
            u_tau=float(variant.pop("u_tau", args.u_tau)),
            re_tau=float(variant.pop("re_tau", args.re_tau)),
            steps=int(variant.pop("steps", args.steps)),
            warmup=int(variant.pop("warmup", args.warmup)),
            box_factor=float(variant.pop("box_factor", 2.0)),
            batch_count=int(variant.pop("batch_count", 4)),
            **variant,
        )
        cells = config.nx * config.ny * config.nz
        started = time.time()
        record = {
            "variant": variant["name"],
            "cells": cells,
            "steps": config.steps,
            "tau": config.tau,
            "nu": config.nu,
            "u_tau": config.u_tau,
            "re_tau": config.re_tau,
            "threads": args.threads,
        }
        try:
            result = run_wall_resolved_channel3d(config)
            elapsed = time.time() - started
            reports = result["reports"][-3:]
            record.update(
                ok=True,
                elapsed_s=elapsed,
                cell_updates_per_s=cells * config.steps / max(elapsed, 1e-9),
                minimum_population=min(float(item["minimum_population"]) for item in result["reports"]),
                maximum_speed=max(float(item["maximum_speed"]) for item in result["reports"]),
                crossflow_rms_over_u_tau=float(
                    result["statistics"]["recent_crossflow_rms_over_u_tau"]
                ),
                measured_u_tau=float(result["wall_exchange"]["mean_friction_velocity"]),
                friction_velocity_error_pct=float(
                    result["statistics"]["recent_friction_velocity_error_pct"]
                ),
                tail=reports,
            )
        except Exception as exc:  # honest failure, keep the probe record
            elapsed = time.time() - started
            record.update(ok=False, elapsed_s=elapsed, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        print("[probe] " + json.dumps(record, default=str)[:600], flush=True)

    (args.out_dir / "stability_probe.json").write_text(
        json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())