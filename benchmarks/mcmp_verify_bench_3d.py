#!/usr/bin/env python3
"""3D MCMP Young-Laplace calibration benchmark.

Produces a Young-Laplace calibration over >= 3 surface-tension levels (G_12)
and >= 3 radii per level, with the acceptance metric of
:func:`tensorlbm.mcmp_verification_3d.young_laplace_calibration_3d`.

Acceptance criteria (pre-registered):
- R^2 >= 0.99 for the free-line fit at each G_12 level
- sigma relative drift across radii < 5% (max_rel_err <= 0.05)
- >= 3 radii per level, >= 2 resolutions (grid sizes)

Usage::

    python benchmarks/mcmp_verify_bench_3d.py --out artifacts/mcmp_verify_bench_3d \
        --jobs 4 [--quick]

``--quick`` shrinks the protocol (fewer steps, smaller grids) for a smoke run;
the acceptance numbers in the report must come from a full run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tensorlbm.mcmp_verification_3d as mv3  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol (pre-registered: fixed before the numbers were produced)
# ---------------------------------------------------------------------------

#: Coupling levels (same window as 2D: G_12 = -2.2 dissolves, -2.3 inverts,
#: -3.5+ diverges).  Three levels inside the stable window.
G_LEVELS = [-2.5, -2.75, -3.0]

#: Droplet radii of the Laplace series (all <= 0.3 n for the n used).
RADII = [10.0, 14.0, 18.0]

#: Two resolutions for convergence check.
RESOLUTIONS = [64, 96]

FULL = {"n_laplace": 96, "steps_laplace": 12000, "steps_coex": 12000}
QUICK = {"n_laplace": 64, "steps_laplace": 4000, "steps_coex": 4000}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def _quiet_threads() -> None:
    torch.set_num_threads(1)


def job_coexistence_3d(args: tuple) -> dict:
    G, n, steps = args
    _quiet_threads()
    out = mv3.estimate_coexistence_3d(G_12=G, n=n, geometry="planar", steps=steps)
    return out


def job_laplace_3d(args: tuple) -> dict:
    G, n, radius, steps, coex = args
    _quiet_threads()
    run = mv3.simulate_static_droplet_3d(
        G_12=G,
        n=n,
        radius_init=radius,
        rho1_in=coex["rho1_liq"],
        rho2_in=coex["rho2_liq"],
        rho1_out=coex["rho1_gas"],
        rho2_out=coex["rho2_gas"],
        steps=steps,
    )
    case = mv3.measure_laplace_3d(run)
    case["history"] = run["history"]
    case["coexistence"] = {k: v for k, v in coex.items()}
    return case


def _run_all(jobs, fn, workers: int):
    if workers <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(processes=workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("tensorlbm")
    except Exception:
        return "dev"


def _json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def build(out_dir: Path, workers: int, quick: bool) -> dict:
    proto = QUICK if quick else FULL
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    results_by_resolution = {}

    for n_grid in RESOLUTIONS:
        n_lap = n_grid if quick else n_grid
        log(f"=== Resolution n={n_lap} ===")

        # 1. coexistence (one flat slab per coupling level)
        log(f"coexistence: {len(G_LEVELS)} planar slab runs at n={n_lap}")
        coex_rows = _run_all(
            [(G, n_lap, proto["steps_coex"]) for G in G_LEVELS], job_coexistence_3d, workers
        )
        coex = {row["G_12"]: row for row in coex_rows}
        for row in coex_rows:
            log(
                f"  G={row['G_12']}: rho1_liq={row['rho1_liq']:.4f} rho1_gas={row['rho1_gas']:.4f} "
                f"flat_dp={row['flat_interface_dp']:.2e} umax={row['u_max']:.4f}"
            )

        # 2. Young-Laplace series
        log(f"laplace: {len(G_LEVELS)} levels x {len(RADII)} radii at n={n_lap}")
        lap_rows = _run_all(
            [(G, n_lap, R, proto["steps_laplace"], coex[G]) for G in G_LEVELS for R in RADII],
            job_laplace_3d,
            workers,
        )
        levels = []
        for G in G_LEVELS:
            cases = [c for c in lap_rows if c["G_12"] == G]
            levels.append(mv3._fit_laplace_level_3d(G, cases, 0.05))
        laplace = {
            "protocol": {
                "n": n_lap,
                "steps": proto["steps_laplace"],
                "coexist_steps": proto["steps_coex"],
                "radii": RADII,
                "G_values": G_LEVELS,
                "threshold": 0.05,
                "coexistence_geometry": "planar",
                "metric": "3D Young-Laplace Delta p = 2 sigma / R, sigma "
                "fitted through the origin; per-case relative error",
            },
            "levels": levels,
            "cases": lap_rows,
            "coexistence": coex_rows,
            "pass": all(level["pass"] for level in levels),
        }
        for level in levels:
            log(
                f"  G={level['G_12']}: sigma_origin={level['sigma_origin']:.5f} "
                f"max_rel_err={level['max_rel_err']:.3%} pass={level['pass']} "
                f"r2={level['r2']:.6f} "
                f"guards={[g['code'] for g in level['guardrails']]}"
            )

        results_by_resolution[f"n{n_lap}"] = {
            "laplace": laplace,
            "coexistence": coex_rows,
        }

    # Cross-resolution comparison
    log("=== Cross-resolution comparison ===")
    comparison = {}
    for G in G_LEVELS:
        sigmas = {}
        for res_key in results_by_resolution:
            res_data = results_by_resolution[res_key]
            for level in res_data["laplace"]["levels"]:
                if level["G_12"] == G:
                    sigmas[res_key] = level["sigma_origin"]
                    break
        if len(sigmas) >= 2:
            vals = list(sigmas.values())
            mean_sigma = sum(vals) / len(vals)
            drift = max(abs(v - mean_sigma) / mean_sigma for v in vals) if mean_sigma else float("nan")
            comparison[f"G{str(G).replace('.', '_')}"] = {
                "sigmas": sigmas,
                "mean_sigma": mean_sigma,
                "relative_drift": drift,
                "drift_pass": drift < 0.05,
            }
            log(f"  G={G}: sigmas={sigmas}, drift={drift:.3%}, pass={drift < 0.05}")

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {"quick": quick, **proto},
        "library": {"version": _version(), "torch": torch.__version__},
        "resolutions": RESOLUTIONS,
        "results_by_resolution": results_by_resolution,
        "cross_resolution_comparison": comparison,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=_json_safe))
    log(f"wrote {out_dir / 'results.json'}")
    return results


def main():
    ap = argparse.ArgumentParser(description="3D MCMP Young-Laplace calibration benchmark")
    ap.add_argument("--out", default="artifacts/mcmp_verify_bench_3d")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="Quick smoke run")
    args = ap.parse_args()
    build(Path(args.out), args.jobs, args.quick)


if __name__ == "__main__":
    main()
