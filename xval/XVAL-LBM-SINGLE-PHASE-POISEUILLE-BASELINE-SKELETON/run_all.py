#!/usr/bin/env python3
"""Driver of ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``.

Runs every arm of ``cases.py`` through both codes (OSS binary + TensorLBM),
writes per-arm JSON, then the consolidated ``results/results.json`` with the
reference cross-check and the OSS-vs-TensorLBM differences.

    LETTUCE_PATH=/tmp/up/lettuce python3 run_all.py [--only V16,N32]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "TensorLBM" / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import arm_oss  # noqa: E402
import arm_tl  # noqa: E402
from cases import ARMS, OSS_DEFAULTS, STEADY_TOL, STEADY_WINDOW, WALL_CAP_S, dimensionless  # noqa: E402
from tensorlbm.verification.xval.poiseuille_skeleton import (  # noqa: E402
    analytic_profile,
    analytic_solution_object,
    oss_units,
    profile_kpis,
    two_code_diff_metrics,
    wall_normal_coordinates,
)

warnings.filterwarnings("ignore")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def reference_cross_check() -> dict:
    """Closed-form parabola vs the independent N-layer linear solve (same ODE)."""
    out = {}
    for res in (16, 32, 48):
        u = oss_units(res, **OSS_DEFAULTS)
        y, H = wall_normal_coordinates(res)
        closed = analytic_profile(y, H, u.acceleration_lu, u.viscosity_lu)
        lin = analytic_solution_object(H, u.acceleration_lu, u.viscosity_lu).u(y)
        out[str(res)] = {
            "max_rel_dev_closed_vs_linear_solve": float(np.max(np.abs(closed - lin))
                                                        / np.max(np.abs(lin))),
            "q_closed_form": float(np.trapezoid(closed, y)),
            "q_continuum": float(u.acceleration_lu * H ** 3 / (12.0 * u.viscosity_lu)),
            "q_linear_solve": float(analytic_solution_object(H, u.acceleration_lu, u.viscosity_lu).flow_rate()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated arm labels")
    ap.add_argument("--skip-oss", action="store_true")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    (HERE / "results").mkdir(exist_ok=True)
    oss_ok = arm_oss.available() and not args.skip_oss

    runs: list[dict] = []
    meta: dict = {
        "anchor": "XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO",
        "mission_id": 18375,
        "git_rev": git_rev(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "steady_tol": STEADY_TOL,
        "steady_window": STEADY_WINDOW,
        "wall_cap_s": WALL_CAP_S,
        "oss_available": oss_ok,
        "oss_checkout": arm_oss.oss_path(),
        "oss_version": arm_oss.version() if oss_ok else "unavailable",
        "reference_cross_check": reference_cross_check(),
        "arms": [],
    }

    for spec in ARMS:
        if only and spec["label"] not in only:
            continue
        for code in spec["codes"]:
            if code == "oss" and not oss_ok:
                meta["arms"].append({"label": spec["label"], "code": "lettuce",
                                     "status": "WITHHELD", "reason": "OSS import unavailable"})
                continue
            t0 = time.time()
            try:
                if code == "tl":
                    res = arm_tl.run(spec, STEADY_TOL, STEADY_WINDOW, WALL_CAP_S)
                else:
                    res = arm_oss.run(spec, STEADY_TOL, STEADY_WINDOW, WALL_CAP_S)
                status = "DONE"
            except Exception as exc:                        # honest FAIL, keep going
                res = {"code": code, "label": spec["label"], "status": "FAIL",
                       "failure_reason": f"{type(exc).__name__}: {exc}"}
                status = "FAIL"
            res["status"] = status
            res["elapsed_s"] = round(time.time() - t0, 2)
            runs.append(res)
            (HERE / "results" / f"arm_{spec['label']}_{code}.json").write_text(
                json.dumps(res, indent=2))
            k = res.get("kpis", {})
            print(f"[{status}] {spec['label']:>4}/{code:<7} steps={res.get('steps','-'):>7} "
                  f"eps_inf={k.get('eps_inf', float('nan')):.3e} "
                  f"wall={res['elapsed_s']:.1f}s", flush=True)
            meta["arms"].append({"label": spec["label"], "code": res["code"],
                                 "status": status, "wall_s": res["elapsed_s"]})

    # ---- consolidate: per-arm exact-reference KPIs already in the runs ----- #
    table = []
    by_key = {(r["label"], r["code"]): r for r in runs if r.get("status") == "DONE"}
    for spec in ARMS:
        if only and spec["label"] not in only:
            continue
        row = {"label": spec["label"], "family": spec["family"], "resolution": spec["resolution"],
               "force_scale": spec["force_scale"], "nu_mode": spec["nu_mode"],
               "dtype": spec["dtype"], "init": spec["init"],
               "dimensionless": dimensionless(spec)}
        for code, key in (("tl", "tensorlbm"), ("oss", "lettuce")):
            r = by_key.get((spec["label"], key))
            if r is None:
                row[key] = None
                continue
            row[key] = {k: r["kpis"][k] for k in r["kpis"]}
            row[key].update({
                "steps": r["steps"], "converged": r["converged"],
                "wall_cap_hit": r["wall_cap_hit"], "wall_s": r["wall_s"],
                "rho_min": r["rho_min"], "rho_max": r["rho_max"],
                "h_eff_mean": r["h_eff_mean"], "h_eff_max_dev": r["h_eff_max_dev"],
                "q_continuum": r["q_continuum"],
                "u_centerline": r["u_centerline"],
            })
            if key == "tensorlbm":
                row[key]["uy_max_abs"] = r["uy_max_abs"]
        tl, os_ = by_key.get((spec["label"], "tensorlbm")), by_key.get((spec["label"], "lettuce"))
        if tl and os_:
            row["cross_code"] = two_code_diff_metrics(tl["u_profile"], os_["u_profile"])
        table.append(row)

    out = {"meta": meta, "table": table, "runs": runs}
    (HERE / "results" / "results.json").write_text(json.dumps(out, indent=2))
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"wrote {HERE / 'results' / 'results.json'} ({len(runs)} arm runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())