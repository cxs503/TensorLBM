#!/usr/bin/env python3
"""CASE-CYLINDER-DRAG-RE100 — case-pool entry for the Schäfer–Turek 2D-2 cylinder, Re = 100.

One command reproduces the benchmark: it drives the library case
(:mod:`tensorlbm.validation.cylinder_dfg`) on one or more grid resolutions,
compares Cd / Cl amplitude / St / Δp against the pinned reference
(:file:`refs.json`) and writes the run artifacts
(``results_D<k>.json``, ``series_D<k>.csv``, ``series_D<k>.npz``,
``summary.json``) into ``--out``.

The solver chain lives in the library (no physics here); this file owns only the
benchmark protocol: grid list, tolerances, artifact layout, provenance, verdict.

    PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re100/run_case.py \
        --grids 20 28 40 --steps 14000 20000 27200 --out runs/cylinder_drag_re100

    # fast smoke check (NOT a validation claim: not converged, D=10)
    PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re100/run_case.py \
        --grids 10 --steps 600 --out /tmp/cylinder_drag_re100_quick
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

METRICS = ("cd", "cl_amp", "st", "dp")


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _environment(n_threads: int) -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": n_threads,
        "git_rev": _git_rev(),
        "device": "cpu",
    }


def _write_series(out: Path, D: int, series: dict) -> None:
    cd = np.asarray(series["cd"], dtype=np.float64)
    cl = np.asarray(series["cl"], dtype=np.float64)
    dp = np.asarray(series["dp"], dtype=np.float64)
    start = int(series["window_start_step"])
    np.savez_compressed(out / f"series_D{D}.npz", cd=cd, cl=cl, dp=dp,
                        window_start_step=start,
                        rho_centerline_mean=np.asarray(series["rho_centerline_mean"],
                                                       dtype=np.float64))
    with (out / f"series_D{D}.csv").open("w") as fh:
        fh.write("step,cd,cl,dp_half_rho_u2\n")
        for i, (a, b, c) in enumerate(zip(cd, cl, dp)):
            fh.write(f"{start + i},{a:.10g},{b:.10g},{c:.10g}\n")


def _verdict(rows: list[dict], tol: float) -> dict:
    """Pass/fail on the finest grid, one line per metric (no aggregation hide)."""
    finest = max(rows, key=lambda r: r["D_cells"])
    per = {}
    for m in METRICS:
        err = finest[f"err_{m}_pct"]
        per[m] = {
            "ref_band": finest[f"ref_band_{m}"],
            "got": finest[m if m != "dp" else "dp"],
            "err_pct": err,
            "pass": bool(abs(err) <= tol),
        }
    cd_ok = per["cd"]["pass"]
    all_ok = all(v["pass"] for v in per.values())
    return {
        "tolerance_pct": tol,
        "finest_grid_D": finest["D_cells"],
        "per_metric": per,
        "primary_metric": "cd",
        "primary_pass": cd_ok,
        "all_metrics_pass": all_ok,
        "verdict": "verified" if all_ok else ("drag_verified" if cd_ok else "not_verified"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=int, nargs="+", default=[20, 28, 40],
                    help="cylinder diameters in lattice cells")
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="steps per grid (default: 24 shedding periods each)")
    ap.add_argument("--u-lb", type=float, default=0.0833, help="mean inlet velocity in lattice units")
    ap.add_argument("--avg-periods", type=float, default=12.0,
                    help="shedding periods averaged at the tail of the run")
    ap.add_argument("--tolerance", type=float, default=None, help="|err| band in %, default from refs.json")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    threads = a.threads or min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    refs = json.loads((HERE / "refs.json").read_text())
    tol = refs["tolerance_pct"] if a.tolerance is None else a.tolerance
    bands = cdfg.REFERENCE_BANDS["2D-2"]
    # refs.json and the frozen module register must not drift apart
    assert tuple(refs["ref"]["cd"]) == tuple(bands["cd"]), "refs.json Cd band != module register"
    assert tuple(refs["ref"]["dp_half_rho_u2"]) == tuple(bands["dp_half_rho_u2"]), "dp units drift"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_all = time.time()

    for i, D in enumerate(a.grids):
        cfg_probe = cdfg.CaseConfig(D_cells=D, u_mean_lb=a.u_lb, n_steps=1)
        steps = (a.steps[i] if a.steps else int(round(24 * cfg_probe.period_steps)))
        print(f"=== D={D} cells ({cfg_probe.nx}x{cfg_probe.ny}), steps={steps}, "
              f"tau={cfg_probe.tau:.5f} ===", flush=True)
        cfg = cdfg.CaseConfig(D_cells=D, u_mean_lb=a.u_lb, n_steps=steps,
                              avg_periods=a.avg_periods)
        res = cdfg.run_case(cfg, threads=threads, keep_series=True)
        series = res.pop("series", {})
        (out / f"results_D{D}.json").write_text(
            json.dumps(res, indent=2, default=cdfg._json_default))
        if series:
            _write_series(out, D, series)
        ref = bands
        row = {
            "D_cells": D,
            "nx": res["nx"],
            "ny": res["ny"],
            "cells": res["cells"],
            "steps": res["n_steps"],
            "avg_window": res["avg_window"],
            "tau": res["tau"],
            "nu_lb": res["nu_lb"],
            "cd": res["cd"],
            "cd_stderr": res["cd_stderr"],
            "cl_amp": res["cl_amp"],
            "st": res["st"],
            "dp": res["dp"],
            "dp_over_rho_u2": res["dp_over_rho_u2"],
            "ref_band_cd": list(ref["cd"]),
            "ref_band_cl_amp": list(ref["cl_amp"]),
            "ref_band_st": list(ref["st"]),
            "ref_band_dp": list(ref["dp_half_rho_u2"]),
            "err_cd_pct": res["err_pct"]["cd"],
            "err_cl_amp_pct": res["err_pct"]["cl_amp"],
            "err_st_pct": res["err_pct"]["st"],
            "err_dp_pct": res["err_pct"]["dp"],
            "wall_s": res["wall_s"],
        }
        rows.append(row)
        print(f"[D={D}] cd={row['cd']:.4f} ({row['err_cd_pct']:+.2f}%) "
              f"cl_amp={row['cl_amp']:.4f} ({row['err_cl_amp_pct']:+.2f}%) "
              f"st={row['st']:.4f} ({row['err_st_pct']:+.2f}%) "
              f"dp*={row['dp']:.4f} ({row['err_dp_pct']:+.2f}%) wall={row['wall_s']:.0f}s",
              flush=True)

    summary = {
        "anchor": refs["anchor"],
        "case": refs["case"],
        "reference": refs["ref"],
        "tolerance_pct": tol,
        "config": {"grids": a.grids, "u_mean_lb": a.u_lb, "avg_periods": a.avg_periods},
        "environment": _environment(threads),
        "rows": rows,
        "verdict": _verdict(rows, tol),
        "wall_s_total": round(time.time() - t_all, 1),
        "repro_cmd": refs["repro_cmd"],
        "library": refs["library"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    v = summary["verdict"]
    print(f"\nverdict: {v['verdict']} (finest D={v['finest_grid_D']}, tol={tol}%)")
    for m, d in v["per_metric"].items():
        print(f"  {m:7s} got={d['got']:.4f} ref={d['ref_band']} err={d['err_pct']:+.2f}% "
              f"{'PASS' if d['pass'] else 'FAIL'}")
    return 0 if v["primary_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())