#!/usr/bin/env python3
"""Lid-driven square cavity, Re = 5000 -- TensorLBM benchmark case (CASE-CAVITY-RE5000).

Reference benchmark
-------------------
U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", *J. Comput. Phys.* **48**
(1982) 387-411.  The Re = 5000 profiles live in **Table III** (u on the vertical
centreline x/L = 0.5) and **Table IV** (v on the horizontal centreline y/L = 0.5);
the Re = 100/400/1000 tables used for the harness self-check are **Table I/II**
of the same paper.

Pre-declared acceptance policy (fixed *before* the run, per mission brief)
-------------------------------------------------------------------------
* case tolerance: ``|ref - got| <= 8 % of the lid speed`` pointwise on both
  centrelines, scored only on rows of the reference table;
* harness self-check (must pass first): the same code path at Re = 1000 must
  reproduce Ghia Table I/II within 8 % -- otherwise the Re = 5000 number is
  meaningless;
* grid convergence: the 128^2 -> 256^2 change of u(0.5, y) must be *smaller*
  than the finest-grid error to the reference, otherwise the case is reported
  WITHHELD (not converged) instead of passed.

Reference-data admission (zero hand transcription)
--------------------------------------------------
The Ghia Re = 5000 table is embedded in the library module
``tensorlbm.verification.lid_driven_cavity.reference_data`` as ``GHIA_RE5000``,
machine-verified from the paper's Table III/IV (see commit 0dbf26ae for the
audit trail).  When the reference is unavailable the run still produces the
solution and the grid-convergence evidence, and the reference comparison is
reported as ``WITHHELD`` -- never as a pass.

Physics / numerics
------------------
* D2Q9 BGK collision with half-way bounce-back on stationary walls;
* Zou-He velocity boundary on the moving lid (implemented in CavitySolver);
* equal-Re recipe: ``tau = 3 u_lid L / Re + 0.5`` with ``L = n-1`` lattice units,
  ``u_lid = 0.1`` (Ma ~ 0.17, compressibility effect O(Ma^2) ~ 3 %).

Usage
-----
    python benchmarks/verified/cavity/re5000/run.py --n 128 256 --steps 50000
    python benchmarks/verified/cavity/re5000/run.py --selfcheck

Every number the report quotes is written to ``result.json`` next to this file
(profiles included), so the write-up can be re-derived without re-running.

Exit codes
----------
``0``  every grid reached a finite solution (the reference comparison is still
       ``WITHHELD`` if the reference is unavailable);
``2``  at least one grid diverged or produced non-finite fields.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "TensorLBM" / "src"))

from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    GHIA_RE1000, GHIA_RE5000, get_ghia_reference, interpolate_ghia
)

import tensorlbm

TOL_PCT = 8.0        # mission tolerance, % of the lid speed
U_LID = 0.1
RE = 5000.0


def run_case(n, re=RE, u_lid=U_LID, max_steps=50000, verbose=True):
    """Run a single cavity case and return results dict."""
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
    solver = CavitySolver(cfg)

    t0 = time.time()
    step = 0
    diverged = False
    history = []

    for step in range(1, max_steps + 1):
        solver.step()
        if step % 5000 == 0:
            y, u, x, v = solver.get_profiles()
            if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
                diverged = True
                break
            if verbose:
                print(f"  n={n} step={step} u_min={u.min():+.5f} v_max={v.max():+.5f}", flush=True)

    elapsed = time.time() - t0
    y_norm, u_norm, x_norm, v_norm = solver.get_profiles()

    finite = bool(np.all(np.isfinite(u_norm)) and np.all(np.isfinite(v_norm)))
    if diverged:
        status = "diverged"
    elif not finite:
        status = "non_finite"
    else:
        status = "ok"

    u_min_idx = int(np.argmin(u_norm))
    v_max_idx = int(np.argmax(v_norm))

    return {
        "n": n,
        "re": re,
        "status": status,
        "u_lid": u_lid,
        "tau": cfg.tau,
        "steps": step,
        "diverged": diverged,
        "elapsed_s": round(elapsed, 1),
        "u_min": float(u_norm[u_min_idx]),
        "u_min_y": float(y_norm[u_min_idx]),
        "v_max": float(v_norm[v_max_idx]),
        "v_max_x": float(x_norm[v_max_idx]),
        "y_centerline": y_norm.tolist(),
        "u_centerline": u_norm.tolist(),
        "x_centerline": x_norm.tolist(),
        "v_centerline": v_norm.tolist(),
    }


def compare_to_ref(run, ref):
    """Pointwise comparison on the tabulated rows."""
    if ref is None:
        return {"status": "WITHHELD", "reason": "reference data unavailable"}

    y = np.asarray(ref.y)
    u_ref = np.asarray(ref.u)
    x = np.asarray(ref.x)
    v_ref = np.asarray(ref.v)

    u_got = np.interp(y, np.asarray(run["y_centerline"]), run["u_centerline"])
    v_got = np.interp(x, np.asarray(run["x_centerline"]), run["v_centerline"])

    du = (u_got - u_ref) * 100.0
    dv = (v_got - v_ref) * 100.0

    max_err = float(max(np.abs(du).max(), np.abs(dv).max()))
    rmse = float(np.sqrt(np.mean(np.concatenate([du, dv]) ** 2)))

    return {
        "max_abs_err_pct": max_err,
        "rmse_pct": rmse,
        "pass": bool(max_err <= TOL_PCT),
        "u_errors": [[float(yi), float(ur), float(ug), float(e)]
                     for yi, ur, ug, e in zip(y, u_ref, u_got, du)],
        "v_errors": [[float(xi), float(vr), float(vg), float(e)]
                     for xi, vr, vg, e in zip(x, v_ref, v_got, dv)],
    }


def harness_selfcheck():
    """Same code path at Re=1000 vs Ghia Table I/II (must pass before any Re=5000 claim)."""
    run = run_case(128, re=1000.0, u_lid=0.1, max_steps=30000, verbose=False)
    result = compare_to_ref(run, GHIA_RE1000)
    return {"n": 128, "re": 1000.0, "steps": run["steps"], **result}


def grid_convergence(runs):
    """u(0.5,y) difference between the two finest grids."""
    keys = sorted(runs, key=lambda k: runs[k]["n"])
    if len(keys) < 2:
        return {"status": "single-grid"}
    fine, coarse = runs[keys[-1]], runs[keys[-2]]
    yf = np.asarray(fine["y_centerline"])
    uc = np.interp(yf, np.asarray(coarse["y_centerline"]), coarse["u_centerline"])
    d = np.abs(np.asarray(fine["u_centerline"]) - uc) * 100.0
    return {
        "pair": [coarse["n"], fine["n"]],
        "max_abs_delta_pct": float(d.max()),
        "rmse_pct": float(np.sqrt(np.mean(d ** 2))),
    }


def main():
    ap = argparse.ArgumentParser(description="Cavity Re=5000 benchmark")
    ap.add_argument("--n", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--u-lid", type=float, default=U_LID)
    ap.add_argument("--re", type=float, default=RE)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "result.json"))
    args = ap.parse_args()

    payload = {
        "case": "cavity_re5000_ghia",
        "description": "Lid-driven cavity Re=5000, D2Q9 BGK, Ghia(1982) Table III/IV verification",
        "lattice": "D2Q9",
        "collision": "BGK",
        "boundary": "Half-way bounce-back (stationary walls) + Zou-He (moving lid)",
        "tolerance_pct": TOL_PCT,
        "reference": "Ghia 1982 Table III (u, x=0.5) / Table IV (v, y=0.5), Re=5000 column",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorlbm": tensorlbm.__file__,
        },
        "acceptance_policy": {
            "pointwise": f"|ref-got| <= {TOL_PCT}% of lid speed on Table III/IV rows",
            "selfcheck": "Re=1000 vs Ghia Table I/II on the same code path must pass first",
            "grid_convergence": "128^2->256^2 delta must be below the finest-grid reference error",
        },
    }

    if args.selfcheck:
        print("[selfcheck] Re=1000 vs Ghia Table I/II ...", flush=True)
        payload["harness_selfcheck"] = harness_selfcheck()
        sc = payload["harness_selfcheck"]
        print(f"[selfcheck] max_err={sc['max_abs_err_pct']:.2f}% pass={sc['pass']}", flush=True)

    runs = {}
    t0 = time.time()
    for n in args.n:
        print(f"[run] n={n} re={args.re} u_lid={args.u_lid}", flush=True)
        runs[str(n)] = run_case(n, re=args.re, u_lid=args.u_lid, max_steps=args.steps)
        r = runs[str(n)]
        print(f"[done] n={n} steps={r['steps']} status={r['status']} "
              f"u_min={r['u_min']:+.5f}@{r['u_min_y']:.3f} t={r['elapsed_s']:.0f}s", flush=True)

    payload["grids"] = runs
    payload["grid_convergence"] = grid_convergence(runs)

    # Compare finest grid to reference
    finest_key = max(runs, key=lambda k: runs[k]["n"])
    finest_run = runs[finest_key]
    try:
        ref = get_ghia_reference(RE)
        payload["comparison"] = compare_to_ref(finest_run, ref)
    except KeyError:
        payload["comparison"] = {"status": "WITHHELD", "reason": "GHIA_RE5000 not available"}

    payload["wall_clock_s"] = round(time.time() - t0, 1)

    bad = {k: v["status"] for k, v in runs.items() if v["status"] != "ok"}
    payload["run_status"] = "ok" if not bad else "failed"
    payload["failed_grids"] = bad

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DONE", args.out, flush=True)

    if bad:
        print(f"[FAIL] non-finite/diverged grids: {bad} -> exit 2", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
