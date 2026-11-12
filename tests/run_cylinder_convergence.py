#!/usr/bin/env python3
"""Convergence study runner for Schäfer–Turek cylinder cases.

Submitted to cluster via run_cluster; writes results to artifacts/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.cylinder_st2d1 import (
    REFERENCE as REF_RE20,
    compare_to_reference,
    run_cylinder_st2d1,
)
from tensorlbm.validation.cylinder_dfg import (
    REFERENCE_BANDS,
    CaseConfig,
    band_mid,
    run_case,
)


def richardson_extrapolation(h_vals, q_vals):
    """Richardson extrapolation: q = q_inf + C*h^p, three-point."""
    h1, h2, h3 = h_vals
    q1, q2, q3 = q_vals
    r = h1 / h2  # refinement ratio (assumed uniform)
    if abs(q2 - q3) > 1e-12 and abs(q1 - q2) > 1e-12:
        ratio = (q1 - q2) / (q2 - q3)
        p_obs = math.log(ratio) / math.log(r) if ratio > 0 else float("nan")
    else:
        p_obs = float("nan")
    if not math.isnan(p_obs) and p_obs > 0:
        rp = r ** p_obs
        q_inf = q3 + (q3 - q2) / (rp - 1) if abs(rp - 1) > 1e-12 else float("nan")
    else:
        q_inf = float("nan")
    return q_inf, p_obs


def run_re20(grids, steps, threads=16):
    torch.set_num_threads(threads)
    results = []
    for D in grids:
        print(f"\n=== Re=20, D={D} ===", flush=True)
        t0 = time.time()
        res = run_cylinder_st2d1(
            D_cells=D, steps=steps, tau=0.8, device="cpu",
            avg_window=min(20000, steps // 2), plateau_tol=1e-3,
            stop_on_plateau=True, min_steps=10000,
            progress=True, keep_fields=False,
        )
        wall_s = time.time() - t0
        results.append({
            "D_cells": D, "nx": res["geometry"]["nx"], "ny": res["geometry"]["ny"],
            "dx": res["geometry"]["dx"], "h": res["geometry"]["dx"],
            "tau": res["lattice_units"]["tau"], "u_mean_lb": res["lattice_units"]["u_mean"],
            "ma": res["lattice_units"]["ma"], "n_steps": res["n_steps"],
            "stop_reason": res["stop_reason"], "cd": res["cd"], "cl": res["cl"],
            "err_cd_pct": res["err_cd_pct"], "stationary": res["stationary"],
            "drift": res["cd_plateau_drift"], "wall_s": round(wall_s, 1),
        })
        print(f"  Cd={res['cd']:.6f} ({res['err_cd_pct']:+.3f}%) wall={wall_s:.0f}s", flush=True)
    return results


def run_re100(grids, steps_list, threads=8):
    torch.set_num_threads(threads)
    results = []
    bands = REFERENCE_BANDS["2D-2"]
    for i, D in enumerate(grids):
        steps = steps_list[i] if steps_list else None
        print(f"\n=== Re=100, D={D} ===", flush=True)
        cfg_probe = CaseConfig(D_cells=D, u_mean_lb=0.0833, n_steps=1)
        if steps is None:
            steps = int(round(24 * cfg_probe.period_steps))
        t0 = time.time()
        cfg = CaseConfig(D_cells=D, u_mean_lb=0.0833, n_steps=steps, avg_periods=12.0, seed_ampl=0.05)
        res = run_case(cfg, device="cpu", threads=threads, keep_series=True)
        wall_s = time.time() - t0
        cd_mid = band_mid(bands["cd"])
        err_cd = (res["cd"] - cd_mid) / cd_mid * 100.0
        results.append({
            "D_cells": D, "nx": res["nx"], "ny": res["ny"], "dx": res["dx"], "h": res["dx"],
            "tau": res["tau"], "u_mean_lb": res["u_mean_lb"], "n_steps": res["n_steps"],
            "cd": res["cd"], "cd_stderr": res["cd_stderr"], "cl_amp": res["cl_amp"],
            "st": res["st"], "dp": res["dp"], "err_cd_pct": err_cd,
            "err_cl_amp_pct": res["err_pct"]["cl_amp"], "err_st_pct": res["err_pct"]["st"],
            "err_dp_pct": res["err_pct"]["dp"], "wall_s": round(wall_s, 1),
        })
        print(f"  Cd={res['cd']:.4f} ({err_cd:+.2f}%) Cl={res['cl_amp']:.4f} St={res['st']:.4f} dp={res['dp']:.4f} wall={wall_s:.0f}s", flush=True)
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["re20", "re100", "both"], default="both")
    ap.add_argument("--re20-grids", type=int, nargs="+", default=[20, 40, 80])
    ap.add_argument("--re20-steps", type=int, default=200000)
    ap.add_argument("--re100-grids", type=int, nargs="+", default=[20, 28, 40])
    ap.add_argument("--re100-steps", type=int, nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"re20": None, "re100": None, "richardson": {}}

    if a.case in ("re20", "both"):
        re20 = run_re20(a.re20_grids, a.re20_steps, a.threads)
        summary["re20"] = re20
        h = [r["h"] for r in re20]
        cd = [r["cd"] for r in re20]
        cd_inf, p = richardson_extrapolation(h, cd)
        ref = REF_RE20["cd"]
        err = (cd_inf - ref) / ref * 100.0 if not math.isnan(cd_inf) else float("nan")
        summary["richardson"]["re20"] = {
            "h_vals": h, "cd_vals": cd, "cd_inf": cd_inf, "p_observed": p,
            "ref_cd": ref, "err_cd_inf_pct": err,
        }
        print(f"\nRe=20 Richardson: Cd_inf={cd_inf:.6f}, p={p:.3f}, err={err:+.3f}%")

    if a.case in ("re100", "both"):
        re100 = run_re100(a.re100_grids, a.re100_steps, a.threads)
        summary["re100"] = re100
        h = [r["h"] for r in re100]
        cd = [r["cd"] for r in re100]
        cd_mid = band_mid(REFERENCE_BANDS["2D-2"]["cd"])
        cd_inf, p = richardson_extrapolation(h, cd)
        err = (cd_inf - cd_mid) / cd_mid * 100.0 if not math.isnan(cd_inf) else float("nan")
        summary["richardson"]["re100"] = {
            "h_vals": h, "cd_vals": cd, "cd_inf": cd_inf, "p_observed": p,
            "ref_cd_mid": cd_mid, "ref_band": list(REFERENCE_BANDS["2D-2"]["cd"]),
            "err_cd_inf_pct": err,
        }
        print(f"\nRe=100 Richardson: Cd_inf={cd_inf:.6f}, p={p:.3f}, err={err:+.3f}%")

    (out / "convergence_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n→ {out}/convergence_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
