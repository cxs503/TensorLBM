"""Sphere drag-coefficient benchmark: BFL interpolated bounce-back (D3Q19).

This is the *executable* anchor of ``GAP-LBM-SPHERE``.  It drives
:mod:`tensorlbm.sphere_drag_bfl` (the mechanistic solver: D3Q19 TRT-LBM +
Bouzidi–Firdaouss–Lallemand interpolated bounce-back + momentum-exchange
force) over a Reynolds-number scan and a grid-refinement study, and writes
the raw per-case runs plus a machine-readable ``results.json``.

Reference data
--------------
``C_D`` is compared with the Clift–Grace–Weber (1978) compilation of
measurements for a smooth sphere (``REFERENCE_CD_CGW`` in the solver
module), i.e. an *external* anchor independent of the simulation.

Acceptance criteria (pre-declared, see ``ACCEPT`` below)
--------------------------------------------------------
* ``Re <= 200``  (laminar, steady): ``|C_D - C_D,ref| / C_D,ref <= 10 %``
* ``200 < Re <= 1000`` (transitional): same 10 % band used as a *trend*
  criterion, plus monotone decay of ``C_D(Re)``
* grid refinement ``8 -> 12 -> 16`` lattice diameters at ``Re = 20`` and
  ``Re = 100``: successive changes must shrink (monotone convergence) and
  the finest-grid change must be ``< 5 %``.

Usage::

    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py            # full study
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py --smoke    # quick check
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py --re 100 --diameters 12
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from tensorlbm.sphere_drag_bfl import (
    DEFAULT_DIAMETERS,
    DEFAULT_RE_SCAN,
    REFERENCE_CD_CGW,
    SphereBFLConfig,
    reference_cd,
    reference_source,
    run_sphere_case,
)

# --------------------------------------------------------------------------- #
# Study definition
# --------------------------------------------------------------------------- #

#: Reynolds numbers of the anchor scan (>= 5 required by the acceptance plan).
RE_SCAN: Tuple[float, ...] = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

#: main-scan lattice diameter and the grid-refinement ladder
D_MAIN: int = 12
D_GRID: Tuple[int, int, int] = (8, 12, 16)
RE_GRID: Tuple[float, ...] = (20.0, 100.0)

#: pre-declared acceptance bands
ACCEPT = {
    "laminar_re_max": 200.0,
    "cd_err_max_pct_laminar": 10.0,
    "cd_err_max_pct_transitional": 10.0,
    "grid_finest_change_max_pct": 5.0,
}

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "benchmarks" / "verified" / "sphere_drag_bfl"


def study_cases(
    re_scan: Sequence[float], d_main: int, d_grid: Sequence[int], re_grid: Sequence[float]
) -> List[Tuple[float, int]]:
    """Case list: (Re, D) with the main scan plus the refinement ladder."""
    cases: List[Tuple[float, int]] = [(re, d_main) for re in re_scan]
    for re in re_grid:
        for d in d_grid:
            if (re, d) not in cases:
                cases.append((re, d))
    return cases


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    re = float(job["re"])
    diameter = int(job["diameter"])
    cfg = SphereBFLConfig(
        re=re,
        diameter=diameter,
        u_lb=float(job["u_lb"]),
        max_steps=int(job["max_steps"]),
        window=int(job["window"]),
        tol=float(job["tol"]),
        sample_interval=int(job["sample_interval"]),
        threads=int(job["threads"]),
        output_dir=str(job["out_dir"]),
        save_fields=bool(job.get("save_fields", False)),
    )
    t0 = time.time()
    res = run_sphere_case(cfg, progress=False)
    res["job_wall_s"] = time.time() - t0

    case_dir = Path(job["out_dir"]).resolve() / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{cfg.tag}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    res["case_artifact"] = str(path.relative_to(ROOT))
    return res


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

def summarise(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for r in sorted(results, key=lambda r: (r["re"], r["diameter"])):
        status = "PASS" if (
            r["finite"] and r["cd_err_pct"] <= ACCEPT["cd_err_max_pct_laminar"]
        ) else "FAIL"
        rows.append(
            {
                "key": f"Re{r['re']:g}_D{r['diameter']}",
                "re": r["re"],
                "diameter": r["diameter"],
                "cd": r["cd"],
                "cd_ref": r["cd_ref"],
                "cd_err_pct": r["cd_err_pct"],
                "reference_source": r["reference_source"],
                "steps": r["steps"],
                "converged": r["converged"],
                "wall_s": r["wall_s"],
                "tau": r["tau"],
                "u_lb": r["u_lb"],
                "mach": r["mach"],
                "mass_drift_ppm": r["mass_drift_ppm"],
                "n_cut_links": r["n_cut_links"],
                "blockage_pct": r["blockage_pct"],
                "domain": r["domain"],
                "finite": r["finite"],
                "status": status,
            }
        )

    # --- grid refinement (per Re) -----------------------------------------
    grid: Dict[str, Any] = {}
    by_re: Dict[float, Dict[int, Dict[str, Any]]] = {}
    for r in results:
        by_re.setdefault(r["re"], {})[r["diameter"]] = r
    for re, per_d in by_re.items():
        ds = sorted(per_d)
        if len(ds) < 2:
            continue
        changes = []
        for d0, d1 in zip(ds[:-1], ds[1:]):
            c0, c1 = per_d[d0]["cd"], per_d[d1]["cd"]
            changes.append(abs(c1 - c0) / abs(c0) * 100.0 if c0 else float("nan"))
        grid[f"Re{re:g}"] = {
            "diameters": ds,
            "cd": [per_d[d]["cd"] for d in ds],
            "successive_change_pct": changes,
            "finest_change_pct": changes[-1],
            "monotone_convergence": all(
                changes[i] >= changes[i + 1] - 1e-9 for i in range(len(changes) - 1)
            ) if len(changes) > 1 else None,
            "pass_finest_5pct": changes[-1] <= ACCEPT["grid_finest_change_max_pct"],
        }

    # --- trend monotonicity over the scan ---------------------------------
    main = sorted((r for r in results if r["diameter"] == D_MAIN), key=lambda r: r["re"])
    trend = [
        {"re": a["re"], "re_next": b["re"], "cd": a["cd"], "cd_next": b["cd"],
         "decreasing": b["cd"] < a["cd"]}
        for a, b in zip(main[:-1], main[1:])
    ]

    def _err(vals):
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return max(vals) if vals else None

    laminar = [r for r in rows if r["re"] <= ACCEPT["laminar_re_max"] and r["diameter"] == D_MAIN]
    transitional = [r for r in rows if r["re"] > ACCEPT["laminar_re_max"] and r["diameter"] == D_MAIN]
    diverged = [r["key"] for r in rows if not r["finite"]]
    return {
        "rows": rows,
        "grid_convergence": grid,
        "trend": trend,
        "acceptance": {
            "criteria": ACCEPT,
            "laminar_max_err_pct": _err(r["cd_err_pct"] for r in laminar),
            "transitional_max_err_pct": _err(r["cd_err_pct"] for r in transitional),
            "all_finite": not diverged,
            "diverged_cases": diverged,
            "laminar_pass": all(
                r["finite"] and r["cd_err_pct"] <= ACCEPT["cd_err_max_pct_laminar"]
                for r in laminar
            ),
            "transitional_pass": all(
                r["finite"] and r["cd_err_pct"] <= ACCEPT["cd_err_max_pct_transitional"]
                for r in transitional
            ),
            "trend_monotone": all(t["decreasing"] for t in trend),
            "grid_pass": all(v.get("pass_finest_5pct") for v in grid.values()) if grid else None,
        },
    }


def print_table(summary: Dict[str, Any]) -> None:
    print("\n=== sphere drag: C_D vs Clift-Grace-Weber anchor ===")
    print(f"{'case':>14} {'Cd':>8} {'ref':>8} {'err%':>7} {'steps':>7} {'conv':>5} "
          f"{'wall_s':>7} {'Ma':>6} {'links':>6}")
    for r in summary["rows"]:
        print(f"{r['key']:>14} {r['cd']:8.4f} {r['cd_ref']:8.4f} {r['cd_err_pct']:7.2f} "
              f"{r['steps']:7d} {str(r['converged']):>5} {r['wall_s']:7.1f} "
              f"{r['mach']:6.3f} {r['n_cut_links']:6d}")
    if summary["grid_convergence"]:
        print("\n=== grid refinement ===")
        for k, v in summary["grid_convergence"].items():
            print(f"{k}: D={v['diameters']} Cd={['%.4f' % c for c in v['cd']]} "
                  f"successive change %={['%.2f' % c for c in v['successive_change_pct']]} "
                  f"monotone={v['monotone_convergence']} pass<5%={v['pass_finest_5pct']}")
    print("\n=== acceptance ===")
    for k, v in summary["acceptance"].items():
        if k != "criteria":
            print(f"  {k}: {v}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--re", type=float, nargs="*", default=None,
                    help="Reynolds numbers of the main scan")
    ap.add_argument("--diameters", type=int, nargs="*", default=None,
                    help="lattice diameters for the refinement ladder")
    ap.add_argument("--d-main", type=int, default=D_MAIN)
    ap.add_argument("--u", type=float, default=0.1, help="free-stream lattice velocity")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--threads", type=int, default=8, help="torch threads per worker")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--results", default=None, help="path of the combined results.json")
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run (D=8, Re=20, 400 steps)")
    ap.add_argument("--no-pool", action="store_true", help="serial execution (debugging)")
    args = ap.parse_args(argv)

    if args.smoke:
        args.re = [20.0]
        args.diameters = [8]
        args.d_main = 8
        args.max_steps = 400
        args.window = 100
        args.tol = 5.0
        args.save_fields = True

    re_scan = tuple(args.re) if args.re else RE_SCAN
    d_grid = tuple(args.diameters) if args.diameters else (
        D_GRID if not args.smoke else (8,)
    )
    if args.re and args.diameters is None:
        # a custom Re list: scan only, no refinement ladder
        d_grid = (args.d_main,)

    cases = study_cases(re_scan, args.d_main, d_grid, RE_GRID if not args.smoke else ())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "re": re, "diameter": d, "u_lb": args.u, "max_steps": args.max_steps,
            "window": args.window, "tol": args.tol, "sample_interval": args.sample_interval,
            "threads": args.threads, "out_dir": str(out_dir),
            "save_fields": args.save_fields,
        }
        for re, d in cases
    ]
    print(f"running {len(jobs)} cases with {args.workers} workers "
          f"({args.threads} threads each): {cases}", flush=True)

    t0 = time.time()
    if args.no_pool or args.workers <= 1:
        results = []
        for j in jobs:
            print(f"[case] Re={j['re']:g} D={j['diameter']}", flush=True)
            results.append(_run_one(j))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            results = []
            for res in pool.imap_unordered(_run_one, jobs):
                results.append(res)
                print(f"[done] Re={res['re']:g} D={res['diameter']} "
                      f"Cd={res['cd']:.4f} (ref {res['cd_ref']:.4f}, "
                      f"{res['cd_err_pct']:.2f}%) steps={res['steps']}", flush=True)
    elapsed = time.time() - t0

    summary = summarise(results)
    payload = {
        "study": "sphere_drag_bfl",
        "method": {
            "lattice": "D3Q19",
            "collision": "TRT (Lambda=3/16)",
            "boundary": "Bouzidi-Firdaouss-Lallemand interpolated bounce-back",
            "force": "momentum exchange (Ladd) on the cut links",
            "inlet": "equilibrium at free-stream velocity",
            "outlet": "zero-gradient extrapolation",
            "lateral": "free-slip specular reflection",
            "u_lb": args.u,
            "domain_diameters": {"upstream": 2.5, "downstream": 7.0, "lateral_half": 3.0},
        },
        "external_anchor": {
            "source": "Clift, Grace & Weber (1978) Table 5.2",
            "table": {str(k): v for k, v in REFERENCE_CD_CGW.items()},
        },
        "elapsed_s": elapsed,
        "n_cases": len(results),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary["rows"],
        "grid_convergence": summary["grid_convergence"],
        "trend": summary["trend"],
        "acceptance": summary["acceptance"],
        "cases": [
            {
                "key": f"Re{r['re']:g}_D{r['diameter']}",
                "artifact": r.get("case_artifact"),
                "config": r.get("config"),
                "cd_history": r.get("cd_history"),
                "fields": r.get("fields", {}),
            }
            for r in sorted(results, key=lambda r: (r["re"], r["diameter"]))
        ],
    }
    results_path = Path(args.results) if args.results else out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=float))
    print_table(summary)
    print(f"\nwrote {results_path}  ({elapsed:.1f}s total)")
    ok = bool(summary["acceptance"]["all_finite"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())