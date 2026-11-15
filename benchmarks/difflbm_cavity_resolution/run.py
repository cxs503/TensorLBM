#!/usr/bin/env python3
"""GAP-DIFFLBM-CAV-RES — lid-driven cavity resolution-convergence study.

Predeclared question (fixed before the n>=65 runs, see
``docs/DIFFLBM_CAVITY_RESOLUTION.md`` §1)
-----------------------------------------
Q1: does the max-point error of the cavity anchor against Ghia et al. (1982)
    fall below 3 % for every n >= 97 ?
Q2: is the scheme second order, i.e. is the observed order p >= 1.8 ?

Primary metric for Q1: ``max_abs_dev_pct`` = 100 * max |u_sim - u_Ghia| over
the 34 tabulated centreline points (simulation interpolated onto the Ghia
points), identical to the already-verified benchmark convention.

Primary metric for Q2: observed order from **self-convergence (Richardson)** on
the vertical-centreline profile u(y), L2 norm, grids n = 33/65/129 (cell size
halving, r = 2).  The vs-Ghia error is *not* used as the order metric because
the reference table has finite precision and the error saturates near ~0.7 %
(the verified nx=128/192 pair already moves 0.7514 % -> 0.7263 %); that
saturation is reported separately, not hidden.

Zero re-implementation
----------------------
Solver, boundary conditions and the Ghia Table I/II transcription are the
*verified* ones; this driver only adds the resolution sweep, the profile
capture hook and the order fits:

  * ``benchmarks/verified/cavity/re100/run.py`` -> ``run_case(..., capture=)``
    (D2Q9 MRT + V3 pre-streaming half-way bounce-back on the three stationary
    walls + Zou-He moving lid + GHIA_RE100 17-point table)
  * ``benchmarks/verified/cavity/re400/run.py`` -> same, GHIA_RE400

Cases (7 shards, one resolution each)
-------------------------------------
  Re=100 : n = 33, 65, 97, 129   (60000 steps)
  Re=400 : n = 65, 97, 129       (100000 steps, the verified setting)

Pre-registered extension (declared 2026-11-15, before any n >= 97 case was
launched; never a post-hoc addition)
---------------------------------------------------------------------
The self-convergence order needs three grids with exact cell-size halving
(``ORDERSET = 33/65/129``).  For Re=400 the predeclared sweep above starts at
n=65, so the order criterion would be unanswerable.  One extra case,
``Re=400, n=33`` (same 100000 steps, run through ``--cases 400:33``), is
therefore declared *a priori* to complete the Re=400 Richardson triple.  It
does not touch any threshold and does not change the Q1/Q2 metrics.  If that
run turns out to be linearly unstable (tau = 0.5149 at n=33 is close to the
SRT stability limit), the outcome is reported as a limitation, not dropped.

Usage
-----
    PYTHONPATH=src:benchmarks python3 benchmarks/difflbm_cavity_resolution/run.py --shard i --shards 7
    PYTHONPATH=src:benchmarks python3 benchmarks/difflbm_cavity_resolution/run.py --aggregate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # <repo>
OUTDIR = REPO / "artifacts" / "gap_difflbm_cav_res"

#: Predeclared acceptance thresholds (never edited after inspecting results).
THRESH_ORDER_P = 1.8            # observed order of accuracy
THRESH_MAX_DEV_PCT = 3.0        # max-point error vs Ghia
THRESH_N = 97                   # resolution at/above which <3 % must hold
STEADY_RESID = 1e-5             # steady-state residual gate
ORDERSET = (33, 65, 129)        # exact cell-size halving for Richardson

#: (re, n, steps) — the sweep.
CASES = [
    (100.0, 33, 60000),
    (100.0, 65, 60000),
    (100.0, 97, 60000),
    (100.0, 129, 60000),
    (400.0, 65, 100000),
    (400.0, 97, 100000),
    (400.0, 129, 100000),
]


def _ensure_matplotlib():
    """Headless nodes without matplotlib: satisfy ``tensorlbm.lid_driven_cavity``.

    The library imports ``matplotlib`` at module scope only to draw optional
    flow snapshots.  This study needs no plot on the compute node, so when the
    dependency is absent install a no-op stub instead of failing the run
    (observed on the loong64 cluster nodes: ``ModuleNotFoundError: matplotlib``).
    """
    try:
        import matplotlib  # noqa: F401
        return
    except Exception:
        pass
    import types
    mpl = types.ModuleType("matplotlib")
    mpl.use = lambda *a, **k: None
    mpl.rcParams = {}
    plt = types.ModuleType("matplotlib.pyplot")
    for nm in ("figure", "subplots", "plot", "scatter", "savefig", "close",
               "colorbar", "xlabel", "ylabel", "title", "legend", "grid",
               "tight_layout", "show", "imshow", "contourf", "streamplot"):
        setattr(plt, nm, lambda *a, **k: None)
    mpl.pyplot = plt
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = plt


_ensure_matplotlib()

def _load(rel: str, name: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module_for(re: float):
    if int(round(re)) == 100:
        return _load("benchmarks/verified/cavity/re100/run.py", "_cav100")
    if int(round(re)) == 400:
        return _load("benchmarks/verified/cavity/re400/run.py", "_cav400")
    raise SystemExit(f"no verified benchmark for Re={re}")


def run_one(re: float, n: int, steps: int, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    mod = _module_for(re)
    cap: dict = {}
    t0 = time.time()
    res = mod.run_case(n, re, 0.06, steps, torch.device("cpu"), capture=cap)
    res["wall_s"] = round(time.time() - t0, 1)
    res["h"] = 1.0 / (n - 1)
    res["steady"] = bool(res["last_resid"] is not None and res["last_resid"] < STEADY_RESID)
    res["u_cl"] = [float(v) for v in cap["u_cl"]]
    res["v_cl"] = [float(v) for v in cap["v_cl"]]
    res["y"] = [float(v) for v in cap["y"]]
    res["x"] = [float(v) for v in cap["x"]]
    return res


def fit_order(nx: list[int], err: list[float]) -> dict:
    """Least-squares slope of ln(err) vs ln(h), h = 1/(nx-1)."""
    h = np.array([1.0 / (n - 1) for n in nx], dtype=float)
    e = np.array(err, dtype=float)
    ok = (e > 0) & np.isfinite(e)
    if ok.sum() < 2:
        return {"p": None, "r2": None, "n_points": int(ok.sum())}
    x, y = np.log(h[ok]), np.log(e[ok])
    p, c = np.polyfit(x, y, 1)
    yhat = p * x + c
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"p": float(p), "r2": float(r2), "n_points": int(ok.sum())}


def pairwise_orders(nx: list[int], err: list[float]) -> list[dict]:
    out = []
    for i in range(len(nx) - 1):
        e0, e1 = err[i], err[i + 1]
        if e0 > 0 and e1 > 0:
            out.append({"from": nx[i], "to": nx[i + 1],
                        "p": math.log(e0 / e1) / math.log(nx[i + 1] / nx[i])})
    return out


def _on_common(case_coarse: dict, case_fine: dict, key: str, coord: str):
    """Interpolate the coarse profile onto the fine grid's node positions."""
    xp = np.asarray(case_coarse[coord])
    fp = np.asarray(case_coarse[key])
    xf = np.asarray(case_fine[coord])
    return np.interp(xf, xp, fp), np.asarray(case_fine[key])


def richardson(cases: dict[int, dict]) -> dict:
    """Observed order from three grids with cell size halving (r = 2)."""
    out = {}
    n0, n1, n2 = ORDERSET
    if not all(k in cases for k in ORDERSET):
        return {"note": "orderset incomplete", "have": sorted(cases)}
    for key, coord, label in (("u_cl", "y", "vertical_centreline_u"),
                              ("v_cl", "x", "horizontal_centreline_v")):
        c_lo, c_mid, c_hi = cases[n0], cases[n1], cases[n2]
        mid_on_hi, hi = _on_common(c_mid, c_hi, key, coord)
        lo_on_hi, _ = _on_common(c_lo, c_hi, key, coord)
        # interior window: skip the lid/wall nodes where u is imposed exactly
        m = (np.asarray(c_hi[coord]) > 1.5 / (n2 - 1)) & \
            (np.asarray(c_hi[coord]) < 1.0 - 1.5 / (n2 - 1))
        d_coarse = lo_on_hi[m] - mid_on_hi[m]
        d_fine = mid_on_hi[m] - hi[m]
        e_coarse = {"l2": float(np.sqrt(np.mean(d_coarse ** 2))),
                    "max": float(np.max(np.abs(d_coarse)))}
        e_fine = {"l2": float(np.sqrt(np.mean(d_fine ** 2))),
                  "max": float(np.max(np.abs(d_fine)))}
        r = 2.0
        p = {}
        for norm in ("l2", "max"):
            p[norm] = (math.log(e_coarse[norm] / e_fine[norm]) / math.log(r)
                       if e_fine[norm] > 0 else None)
        out[label] = {"e_coarse_33_65": e_coarse, "e_fine_65_129": e_fine,
                      "r": r, "p": p}
    return out


def aggregate() -> dict:
    files = sorted(OUTDIR.glob("case_re*.json"))
    cases = [json.loads(f.read_text()) for f in files]
    cases.sort(key=lambda c: (c["re"], c["nx"]))
    summary = {
        "gap": "GAP-DIFFLBM-CAV-RES",
        "reference": "Ghia, Ghia & Shin (1982) JCP 48:387-411, Tables I & II "
                     "(17-point vertical/horizontal centreline, velocities / U_lid)",
        "solver": "D2Q9 MRT, V3 pre-streaming half-way bounce-back (3 stationary "
                  "walls) + Zou-He moving lid; u_lid = 0.06",
        "metric": "max_abs_dev_pct = 100 * max |u_sim-u_Ghia| over the 34 tabulated "
                  "centreline points, simulation interpolated onto the Ghia points; "
                  "secondary: rmse_u / rmse_v over 17 points each",
        "order_metric": "Richardson self-convergence on n=33/65/129 (r=2), "
                        "vertical-centreline u, L2 norm (primary)",
        "thresholds": {
            "order_p_min": THRESH_ORDER_P,
            "max_dev_pct_max": THRESH_MAX_DEV_PCT,
            "max_dev_pct_n_at_least": THRESH_N,
            "steady_resid_max": STEADY_RESID,
        },
        "cases": cases,
        "by_re": {},
    }
    for re in sorted({c["re"] for c in cases}):
        sub = sorted([c for c in cases if c["re"] == re], key=lambda c: c["nx"])
        nx = [c["nx"] for c in sub]
        maxdev = [c["max_abs_dev_pct"] for c in sub]
        f_ghia_max = fit_order([n for n in nx if n >= 65],
                               [c["max_abs_dev_pct"] for c in sub if c["nx"] >= 65])
        f_ghia_l2 = fit_order([n for n in nx if n >= 65],
                              [c["rmse_u"] for c in sub if c["nx"] >= 65])
        rich = richardson({c["nx"]: c for c in sub})
        p_self = (rich.get("vertical_centreline_u", {}).get("p", {}) or {}).get("l2")
        p_self_max = (rich.get("vertical_centreline_u", {}).get("p", {}) or {}).get("max")
        above = [c["max_abs_dev_pct"] for c in sub if c["nx"] >= THRESH_N]
        crit = {
            "order_p_self_l2_ge_1.8": bool(p_self is not None and p_self >= THRESH_ORDER_P),
            "order_p_self_max_ge_1.8": bool(p_self_max is not None and p_self_max >= THRESH_ORDER_P),
            "max_dev_lt_3pct_for_all_n_ge_97": bool(
                above and all(d < THRESH_MAX_DEV_PCT for d in above)),
            "max_dev_monotone_decrease": bool(all(
                sub[i + 1]["max_abs_dev_pct"] <= sub[i]["max_abs_dev_pct"]
                for i in range(len(sub) - 1))),
            "all_steady": bool(all(c["steady"] for c in sub)),
        }
        crit["all_criteria_met"] = all(v for k, v in crit.items())
        summary["by_re"][str(int(re))] = {
            "nx": nx,
            "h": [round(1.0 / (n - 1), 6) for n in nx],
            "tau": [c["tau"] for c in sub],
            "steps": [c["steps"] for c in sub],
            "last_resid": [c["last_resid"] for c in sub],
            "max_abs_dev_pct": [round(v, 4) for v in maxdev],
            "rmse_u": [round(c["rmse_u"], 5) for c in sub],
            "rmse_v": [round(c["rmse_v"], 5) for c in sub],
            "u_mid": [round(c["u_mid"], 5) for c in sub],
            "u_mid_ghia": sub[0]["u_mid_ghia"],
            "vortex": [c["vortex"] for c in sub],
            "fit_vs_ghia_max_point": f_ghia_max,
            "fit_vs_ghia_l2": f_ghia_l2,
            "pairwise_order_vs_ghia": pairwise_orders(nx, maxdev),
            "richardson": rich,
            "criteria": crit,
        }
    for c in cases:                      # keep the JSON readable
        for k in ("u_cl", "v_cl", "y", "x"):
            c.pop(k, None)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=len(CASES))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CAV_THREADS", "8")))
    ap.add_argument("--all", action="store_true", help="run every case, then aggregate")
    ap.add_argument("--cases", default=None,
                    help="explicit subset, e.g. 100:129,400:65 (steps follow the per-Re default)")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        s = aggregate()
        for re, blk in s["by_re"].items():
            print(f"Re={re}: n={blk['nx']} max_dev={blk['max_abs_dev_pct']} "
                  f"rmse_u={blk['rmse_u']}")
            print(f"   p_ghia={blk['fit_vs_ghia_max_point']['p']} "
                  f"p_self_l2={(blk['richardson'].get('vertical_centreline_u',{}).get('p') or {}).get('l2')}")
            print(f"   criteria={json.dumps(blk['criteria'])}")
        return

    shard = args.shard % args.shards
    if args.all:
        my = list(CASES)
    elif args.cases:
        steps_by_re = {re: st for re, _n, st in CASES}
        my = []
        for tok in args.cases.split(","):
            re_s, n_s = tok.split(":")
            my.append((float(re_s), int(n_s), steps_by_re[float(re_s)]))
    else:
        my = [c for i, c in enumerate(CASES) if i % args.shards == shard]
    for re, n, steps in my:
        res = run_one(re, n, steps, args.threads)
        out = OUTDIR / f"case_re{int(re)}_nx{n}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"[shard {args.shard}] Re={re} n={n} max_dev={res['max_abs_dev_pct']:.3f}% "
              f"rmse_u={res['rmse_u']:.5f} resid={res['last_resid']:.2e} "
              f"steady={res['steady']} wall={res['wall_s']}s -> {out}", flush=True)
    if args.all:
        print(json.dumps(aggregate()["by_re"], indent=2), flush=True)


if __name__ == "__main__":
    main()