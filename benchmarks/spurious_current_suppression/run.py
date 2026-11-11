#!/usr/bin/env python3
"""Spurious-current suppression campaign (node p362w2-n06).

Runs, from one command, the four blocks of the node protocol and writes every
number to a single JSON artifact (the report is rendered from that JSON, nothing
is typed by hand):

A. **W scan** - static droplet R = 50, N = 200, interface parameter
   ``W in [3, 8]`` x 4 methods -> ``U_spurious`` and measured interface
   thickness (the Pareto trade-off: a wider interface damps the parasitic
   vortices but degrades interface fidelity).
B. **Laplace law** - radii R in {30, 40, 50} per method -> ``dp`` from the model
   equation of state, ``sigma`` from the ``dp = sigma/R`` fit, per-radius
   relative residual.
C. **Grid convergence** - R = 0.25 N for N in {100, 150, 200} at fixed interface
   thickness -> observed order of ``U_spurious``.
D. **sigma = 0 control** - uniform mixture, ``G12 = 0`` (no interface, no
   capillary force): ``U_spurious`` must be 0 to round-off.

Usage::

    python benchmarks/spurious_current_suppression/run.py --out artifacts/... [--quick]
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from tensorlbm.verification import spurious_current as sc  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered protocol
# ---------------------------------------------------------------------------

G12 = 2.5                      # repulsive coupling, = -G_12 of the library
N_MAIN = 200                   # grid; droplet radius R = 50 = 0.25 N
R_MAIN = 50.0
W_SCAN = [3.0, 5.0, 8.0]   # nominal 10-90 interface thickness (paper range 3..8)
LAPLACE_RADII = [30.0, 40.0, 50.0]
GRID_N = [100, 150, 200]       # R = 0.25 N, interface width fixed at W = 4
GRID_METHODS = ["baseline_bgk_lattice", "bgk_geometric"]
FULL = {"steps": 7000, "check_every": 250, "stop_tol": 1e-3}
QUICK = {"steps": 1200, "check_every": 100, "stop_tol": 2e-3}

W_TO_TANH = 1.0 / (2.0 * math.atanh(0.8))   # W_10-90 = 2.197 w_tanh


def _job(kind: str, method: str, **kw) -> dict:
    """One independent relaxation run (module level so it pickles)."""
    cfg = sc.METHODS[method]
    t0 = time.perf_counter()
    run = sc.run_static_droplet(
        G12=kw.get("G12", G12),
        n=kw["n"],
        radius=kw["radius"],
        width=kw["width"],
        steps=kw["steps"],
        collision=cfg["collision"],
        force=cfg["force"],
        forcing=cfg.get("forcing", "guo"),
        stop_tol=kw["stop_tol"],
        check_every=kw["check_every"],
        uniform=kw.get("uniform", False),
        keep_fields=True,
    )
    wall = time.perf_counter() - t0
    out: dict = {
        "kind": kind,
        "method": method,
        "method_label": cfg["label"],
        "n": kw["n"],
        "radius": kw["radius"],
        "width_tanh": kw["width"],
        "W_nominal": kw["width"] * 2.0 * math.atanh(0.8),
        "steps_run": run["steps_run"],
        "converged": run["converged"],
        "u_spurious": run["u_max"],
        "u_rms": run["u_rms"],
        "wall_s": wall,
        "ms_per_step": 1e3 * wall / max(run["steps_run"], 1),
        "rho1_centre": run["rho1_centre"],
        "rho1_far": run["rho1_far"],
        "rho_tot_drift": run["rho_tot_drift"],
    }
    if not kw.get("uniform", False):
        w = sc.interface_width_10_90(run)
        p = sc.droplet_pressure(run)
        out.update(
            {
                "width_10_90": w["width_10_90"],
                "w_eq": w["w_eq"],
                "radius_fit": w["radius_fit"],
                "dp": p["dp"],
                "p_in": p["p_in"],
                "p_out": p["p_out"],
            }
        )
    # Down-sampled velocity magnitude for the field figures (hero runs only).
    if kw.get("keep_field_npz"):
        umag = np.sqrt(run["ux"] ** 2 + run["uy"] ** 2)
        np.savez_compressed(
            kw["keep_field_npz"],
            umag=umag[::2, ::2],
            rho1=run["rho1"][::2, ::2],
            u_max=run["u_max"],
            method=method,
            n=kw["n"],
        )
    return out


def build_protocol(
    quick: bool,
    methods: list[str] | None = None,
    steps: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    p = dict(QUICK if quick else FULL)
    if steps is not None:
        p["steps"] = int(steps)
    n_main = 96 if quick else N_MAIN
    r_main = 24.0 if quick else R_MAIN
    w_scan = [3.0, 5.0, 8.0] if quick else W_SCAN
    radii = [15.0, 20.0, 24.0] if quick else LAPLACE_RADII
    grids = [48, 72, 96] if quick else GRID_N
    if methods is None:
        methods = list(sc.METHODS)

    scan, lap, grid = [], [], []
    for m in methods:
        for W in w_scan:
            scan.append(
                dict(kind="scan", method=m, n=n_main, radius=r_main,
                     width=W * W_TO_TANH, **p)
            )
        for R in radii:
            lap.append(
                dict(kind="laplace", method=m, n=n_main, radius=R,
                     width=4.0 * W_TO_TANH, **p)
            )
    for m in [g for g in GRID_METHODS if g in methods]:
        for n in grids:
            grid.append(
                dict(kind="grid", method=m, n=n, radius=0.25 * n,
                     width=4.0 * W_TO_TANH, **p)
            )
    return scan, lap, grid


def _summarise(results: list[dict], protocol: dict) -> dict:
    scan = [r for r in results if r["kind"] == "scan"]
    lap = [r for r in results if r["kind"] == "laplace"]
    grid = [r for r in results if r["kind"] == "grid"]

    # --- B. Laplace law per method -------------------------------------------
    laplace: dict[str, dict] = {}
    for m in sc.METHODS:
        rows = sorted([r for r in lap if r["method"] == m], key=lambda r: r["radius"])
        if len(rows) < 2:
            continue
        fit = sc.laplace_fit([r["radius"] for r in rows], [r["dp"] for r in rows])
        fit0 = sc.laplace_fit(
            [r["radius"] for r in rows], [r["dp"] for r in rows], through_origin=True
        )
        laplace[m] = {
            "label": sc.METHODS[m]["label"],
            "radii": fit["radii"],
            "dp": fit["dp"],
            "sigma": fit["sigma"],
            "sigma_through_origin": fit0["sigma"],
            "intercept": fit["intercept"],
            "r_squared": fit["r_squared"],
            "rel_error": fit["rel_error"],
            "max_rel_error": fit["max_rel_error"],
            "max_rel_error_through_origin": fit0["max_rel_error"],
            "dp_fit": fit["dp_fit"],
        }

    # --- A. Pareto front over (u_spurious, Laplace error, cost) --------------
    # Fidelity objective: the *worse* of the two Laplace-law residuals (free
    # intercept vs through-origin).  Taking the worse of the two is the
    # conservative choice: a non-zero intercept is itself a model defect
    # (unequal bulk pressures), so the pure `dp = sigma/R` residual may not be
    # traded away.  The front obtained from the forgiving (free-intercept)
    # residual alone is reported alongside as a sensitivity check.
    pts, meta, pts_fit = [], [], []
    for r in scan:
        lp = laplace.get(r["method"])
        if lp is None:
            continue
        worst = max(lp["max_rel_error"], lp["max_rel_error_through_origin"])
        # Cost is *not* a Pareto objective: the campaigns ran on a shared,
        # heavily loaded host, so ms/step varies by >30% between phases and is
        # not a property of the scheme.  It is reported as a column only.
        pts.append((r["u_spurious"], worst))
        pts_fit.append((r["u_spurious"], lp["max_rel_error"]))
        meta.append(r)
    front = sc.pareto_front(pts) if pts else []
    front_fit = sc.pareto_front(pts_fit) if pts_fit else []
    front_methods = sorted({meta[i]["method"] for i in front})
    sensitivity_methods = sorted({meta[i]["method"] for i in front_fit})
    pareto = {
        "objectives": [
            "u_spurious [lattice units] (min) = interface stability",
            "worst-case Laplace-law relative error, max(fit, through-origin) (min) "
            "= physical fidelity",
        ],
        "cost_note": "ms/step is reported but excluded: it varies by >30% between "
                     "campaign phases on the shared host and is not a scheme property",
        "sensitivity": {
            "objective_variant": "Laplace error from the free-intercept fit only",
            "methods_on_front": sensitivity_methods,
            "n_methods_on_front": len(sensitivity_methods),
        },
        "front": [
            {
                "method": meta[i]["method"],
                "W_nominal": meta[i]["W_nominal"],
                "width_10_90": meta[i].get("width_10_90"),
                "u_spurious": meta[i]["u_spurious"],
                "laplace_max_rel_error": pts[i][1],
                "ms_per_step": meta[i]["ms_per_step"],
            }
            for i in front
        ],
        "n_points": len(pts),
        "n_front": len(front),
        "methods_on_front": front_methods,
        "n_methods_on_front": len(front_methods),
    }

    # --- C. grid convergence -------------------------------------------------
    conv: dict[str, dict] = {}
    for m in GRID_METHODS:
        rows = sorted([r for r in grid if r["method"] == m], key=lambda r: r["n"])
        if len(rows) < 3:
            continue
        vals = [r["u_spurious"] for r in rows]
        hs = [1.0 / r["n"] for r in rows]
        conv[m] = {
            "label": sc.METHODS[m]["label"],
            "n": [r["n"] for r in rows],
            "radius": [r["radius"] for r in rows],
            "u_spurious": vals,
            "spacing": hs,
            **sc.convergence_order(vals, hs),
        }

    scan_by_method: dict[str, list[dict]] = {}
    for r in scan:
        scan_by_method.setdefault(r["method"], []).append(r)
    for m in scan_by_method:
        scan_by_method[m].sort(key=lambda r: r["W_nominal"])

    return {
        "scan": scan_by_method,
        "laplace": laplace,
        "grid_convergence": conv,
        "pareto": pareto,
    }


def _verdicts(summary: dict, control: dict, n_pytest: int, protocol: dict) -> dict:
    scan = summary["scan"]
    baseline = [r for m, rows in scan.items() if m == "baseline_bgk_lattice" for r in rows]
    baseline_u = {round(r["W_nominal"], 3): r["u_spurious"] for r in baseline}
    best = {}
    for m, rows in scan.items():
        best[m] = min(r["u_spurious"] for r in rows)
    pareto = summary["pareto"]
    conv = summary["grid_convergence"]
    lap = summary["laplace"]

    v = {}
    v["a_baseline_quantified"] = {
        "pass": len(baseline) > 0,
        "baseline_u_spurious_by_W": baseline_u,
        "evidence": "scan rows with method=baseline_bgk_lattice",
    }
    v["b_three_methods_on_pareto"] = {
        "pass": pareto["n_methods_on_front"] >= 3,
        "methods_on_front": pareto["methods_on_front"],
        "n_front": pareto["n_front"],
        "n_points": pareto["n_points"],
        "objectives": pareto["objectives"],
    }
    conv_best = {
        m: c.get("order") for m, c in conv.items()
    }
    finite = [o for o in conv_best.values() if isinstance(o, float) and math.isfinite(o)]
    v["c_grid_order_ge_1"] = {
        "pass": bool(finite) and max(finite) >= 1.0,
        "observed_order": conv_best,
        "spacing": {m: c["spacing"] for m, c in conv.items()},
        "u_spurious": {m: c["u_spurious"] for m, c in conv.items()},
    }
    v["d_sigma_zero_control"] = {
        "pass": control["u_spurious"] < 1e-10,
        "u_spurious": control["u_spurious"],
        "tolerance": 1e-10,
        "setup": "uniform mixture, G12=0 (no interface), 500 steps, float64",
    }
    worst_lap = max((l["max_rel_error"] for l in lap.values()), default=float("inf"))
    v["e_laplace_error_lt_3pct"] = {
        "pass": worst_lap < 0.03,
        "worst_method_max_rel_error": worst_lap,
        "per_method": {m: l["max_rel_error"] for m, l in lap.items()},
        "sigma_per_method": {m: l["sigma"] for m, l in lap.items()},
    }
    v["f_pytest_count"] = {"pass": n_pytest >= 16, "n_tests": n_pytest}
    v["all_pass"] = all(x["pass"] for x in v.values())
    v["protocol"] = protocol
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/spurious_current_p362w2_n06")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(40, os.cpu_count() or 4))
    ap.add_argument("--n-pytest", type=int, default=0)
    ap.add_argument("--methods", default=None,
                    help="comma separated subset of sc.METHODS (default: all)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the relaxation step cap")
    args = ap.parse_args()

    outdir = REPO_ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    method_list = (
        [m.strip() for m in args.methods.split(",")] if args.methods else None
    )
    scan, lap, grid = build_protocol(args.quick, method_list, args.steps)
    protocol_methods = method_list or list(sc.METHODS)
    jobs = scan + lap + grid
    for j in jobs:
        j["W_nominal"] = j["width"] * 2.0 * math.atanh(0.8)
    hero = [j for j in jobs if j["kind"] == "scan" and abs(j["W_nominal"] - 5.0) < 0.3]
    hero_paths = {}
    for j in hero:
        p = outdir / f"field_{j['method']}.npz"
        j["keep_field_npz"] = str(p)
        hero_paths[j["method"]] = str(p.relative_to(REPO_ROOT))

    print(f"[campaign] {len(jobs)} runs, jobs={args.jobs}", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(args.jobs) as pool:
        results = []
        for res in pool.imap_unordered(_job_apply, jobs):
            results.append(res)
            (outdir / "results_partial.json").write_text(
                json.dumps({"n_done": len(results), "n_total": len(jobs),
                            "runs": results}, indent=1)
            )
            print(
                f"[done {len(results)}/{len(jobs)}] {res['kind']:8s} {res['method']:22s} "
                f"n={res['n']:4d} R={res['radius']:5.1f} W={res['W_nominal']:4.2f} "
                f"u={res['u_spurious']:.3e} conv={res['converged']} "
                f"{res['wall_s']:.0f}s",
                flush=True,
            )
    wall = time.perf_counter() - t0

    # D. sigma = 0 control (cheap, run serially after the pool)
    ctrl_p = dict(kind="control", method="baseline_bgk_lattice", n=64, radius=16.0,
                  width=2.0, steps=500, stop_tol=1e-12, check_every=100,
                  G12=0.0, uniform=True)
    control = _job_apply(ctrl_p)

    protocol = {
        "G12": G12, "n_main": N_MAIN, "R_main": R_MAIN, "W_scan": W_SCAN,
        "laplace_radii": LAPLACE_RADII, "grid_n": GRID_N,
        "grid_methods": GRID_METHODS,
        "steps": (QUICK if args.quick else FULL)["steps"],
        "stop_tol": (QUICK if args.quick else FULL)["stop_tol"],
        "quick": args.quick, "jobs": args.jobs,
        "methods": protocol_methods,
    }
    summary = _summarise(results, protocol)
    verdicts = _verdicts(summary, control, args.n_pytest, protocol)

    payload = {
        "node": "p362w2-n06-lbm-spurious-current",
        "repo": "TensorLBM",
        "module": "tensorlbm.verification.spurious_current",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol,
        "wall_seconds": wall,
        "n_runs": len(results),
        "runs": results,
        "control_sigma_zero": control,
        "summary": summary,
        "verdicts": verdicts,
        "field_artifacts": hero_paths,
    }
    path = outdir / "results.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {path}  ({path.stat().st_size/1e3:.1f} kB)  wall={wall:.0f}s")
    print(json.dumps({k: v.get("pass") for k, v in verdicts.items() if isinstance(v, dict)}, indent=1))
    return 0


def _job_apply(kw: dict) -> dict:
    kw = dict(kw)
    kind = kw.pop("kind")
    method = kw.pop("method")
    return _job(kind, method, **kw)


if __name__ == "__main__":
    raise SystemExit(main())
