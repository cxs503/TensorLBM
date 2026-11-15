"""Squeeze-film LBM vs an independent first-principles Navier-Stokes reference.

Produces ``results.json`` (every number in the report is injected from it - no hand
copying) and prints the tables.

What is compared
----------------
Three calibers are reported separately and must never be mixed up again:

======================================  ==============================================
``rey``      ``8 mu V a^3 / h^3``       leading-order lubrication ***asymptote*** (pressure
                                        only); only exact as ``h/a -> 0``
``ref``      ``reference_ns.solve``     independent 2D incompressible N-S (staggered MAC +
                                        artificial compressibility) at the *same* finite
                                        ``h/a``; reported as pressure load and as total
                                        normal-stress load
``lbm``      ``simulate_squeeze``       the module under test; ``force_gauge_*`` is the
                                        momentum-exchange load (= total normal stress) and
                                        ``force_pressure_integral`` the wall pressure sum
======================================  ==============================================

The verification question is ``lbm`` vs ``ref``.  ``ref`` vs ``rey`` is the *caliber*
diagnostic: it measures how much of the historical "12 % - 37 % error" was the
asymptote's own ``O((h/a)^2)`` truncation.

Usage
-----
    python run_compare.py --out results.json            # full sweep (~6 min)
    python run_compare.py --out results.json --quick    # reduced grid levels
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reference_ns as REF  # noqa: E402

from tensorlbm.squeeze_film_lbm import simulate_squeeze  # noqa: E402

NU = 0.1
V = 0.005
A = 16
H_LIST = [3, 4, 6, 8, 12]


def _rel(x: float, base: float) -> float:
    return (x / base - 1.0) * 100.0


def run_lbm(a: int, h: int) -> dict:
    t0 = time.time()
    r = simulate_squeeze(half_width=a, gap=h, nu=NU, V=V, tol=1e-11,
                         max_steps=200_000, check_every=500)
    return {
        "h": h, "a": a, "h_over_a": h / a,
        "steps": r["steps"], "converged": bool(r["converged"]),
        "force_gauge_top": r["force_gauge_top"],
        "force_gauge_bottom": r["force_gauge_bottom"],
        "force_pressure_integral": r["force_pressure_integral"],
        "force_pressure_integral_top_row": r["force_pressure_integral_top_row"],
        "baseline_static_force": r["baseline_static_force"],
        "ledger_residual_rel": r["ledger"]["residual_rel"],
        "wall_clock_s": round(time.time() - t0, 2),
        "_fields": {"x": r["x"].tolist(), "y": r["y"].tolist(),
                    "p": r["p"].tolist(), "ux": r["ux"].tolist(), "uy": r["uy"].tolist()},
    }


def run_ref(a: float, h: float, ny: int, stokes: bool = False, seed: float = 1e-3) -> dict:
    t0 = time.time()
    r = REF.solve(a=a, h=h, nu=NU, V=V, ny=ny, stokes=stokes, seed=seed,
                  max_iter=400_000)
    return {
        "a": a, "h": h, "ny": ny, "stokes": stokes, "seed": seed,
        "nx": r["nx"], "dx": r["dx"], "dy": r["dy"], "dt": r["dt"], "c2": r["c2"],
        "iterations": r["iterations"], "converged": bool(r["converged"]),
        "load_residual": r["load_residual"], "div_scaled": r["div_scaled"],
        "mass_imbalance_rel": r["mass_imbalance_rel"],
        "normal_load_top": r["normal_load_top"],
        "normal_load_bottom": r["normal_load_bottom"],
        "pressure_load_top": r["pressure_load_top"],
        "pressure_load_bottom": r["pressure_load_bottom"],
        "reynolds_load": r["reynolds_load"],
        "viscous_normal_top": r["normal_load_top"] - r["pressure_load_top"],
        "wall_clock_s": round(time.time() - t0, 2),
        "_fields": {"x": r["x"].tolist(), "p_top": r["p_top"].tolist(),
                    "u_mid": r["u_mid_profile"].tolist(), "y_u": r["y_u"].tolist()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-lbm", action="store_true")
    ap.add_argument("--ny-ref", type=int, default=0)
    ap.add_argument("--h-list", default="")
    ap.add_argument("--mode", default="all", choices=["all", "selfcheck", "sweep"])
    args = ap.parse_args()

    ny_ref = args.ny_ref or (16 if args.quick else 32)
    h_list = [int(x) for x in args.h_list.split(",")] if args.h_list else H_LIST
    out: dict = {
        "meta": {
            "nu": NU, "V": V, "fidelity": "lattice units throughout",
            "reference": "staggered MAC + artificial compressibility (Chorin 1967), "
                         "independent of the LBM and of the Reynolds asymptote",
            "lbm_caliber": "momentum-exchange gauge load (= total normal stress); "
                           "force_pressure_integral is the wall pressure sum",
            "reynolds_caliber": "8 mu V a^3/h^3 (pressure only, h/a -> 0)",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "reference_selfcheck": {},
        "sweep": [],
    }

    do_self = args.mode in ("all", "selfcheck")
    do_sweep = args.mode in ("all", "sweep")

    # ---------------- 1. the reference must validate itself -----------------
    # (a) thin-limit: at h/a = 0.047 the lubrication asymptote is accurate to O((h/a)^2)
    #     = 0.2 %, so a 2 % agreement proves the whole setup (BCs, load integral, gauge).
    thin = run_ref(32.0, 3.0, ny=12) if do_self else None
    if do_self:
     out["reference_selfcheck"]["thin_limit"] = {
        "a": 32.0, "h": 3.0, "h_over_a": 3.0 / 32.0, **{k: thin[k] for k in (
             "ny", "nx", "converged", "iterations", "div_scaled", "mass_imbalance_rel",
             "normal_load_top", "pressure_load_top", "reynolds_load", "viscous_normal_top")},
        "err_pressure_vs_rey_pct": _rel(thin["pressure_load_top"], thin["reynolds_load"]),
        "err_normal_vs_rey_pct": _rel(thin["normal_load_top"], thin["reynolds_load"]),
    }

    # (b) grid convergence of the reference itself at the case of record h/a = 0.375
    levels = ([16, 24] if args.quick else [16, 24, 32]) if do_self else []
    grid = []
    for ny in levels:
        r = run_ref(16.0, 6.0, ny=ny)
        grid.append({"ny": ny, **{k: r[k] for k in (
            "nx", "converged", "iterations", "mass_imbalance_rel", "div_scaled",
            "normal_load_top", "pressure_load_top", "reynolds_load")},
            "err_normal_vs_rey_pct": _rel(r["normal_load_top"], r["reynolds_load"])})
    if do_self:
        out["reference_selfcheck"]["grid_convergence"] = grid

    # (c) inertia: Stokes (no convection) vs full N-S at h/a = 0.375
    st = run_ref(16.0, 6.0, ny=ny_ref, stokes=True) if do_self else None
    ns = run_ref(16.0, 6.0, ny=ny_ref) if do_self else None
    if do_self:
      out["reference_selfcheck"]["inertia"] = {
        "h_over_a": 0.375, "ny": ny_ref,
        "stokes_normal_load": st["normal_load_top"],
        "ns_normal_load": ns["normal_load_top"],
        "inertia_share_pct": _rel(ns["normal_load_top"], st["normal_load_top"]),
        "re_h": V * 6.0 / NU,
    }

    # (d) seed independence (the symmetry-breaking seed must not set the answer)
    s2 = run_ref(16.0, 6.0, ny=ny_ref, seed=3e-3) if do_self else None
    if do_self:
      out["reference_selfcheck"]["seed_independence"] = {
        "seed_a": 1e-3, "seed_b": 3e-3,
        "load_a": ns["normal_load_top"], "load_b": s2["normal_load_top"],
        "rel_diff_pct": _rel(s2["normal_load_top"], ns["normal_load_top"]),
    }

    # ---------------- 2. the sweep -----------------------------------------
    for h in (h_list if do_sweep else []):
        row: dict = {"h": h, "h_over_a": h / A}
        if not args.skip_lbm:
            row["lbm"] = run_lbm(A, h)
        row["ref"] = run_ref(float(A), float(h), ny=ny_ref)
        rey = row["ref"]["reynolds_load"]
        row["reynolds_load"] = rey
        if "lbm" in row:
            row["err_pct"] = {
                "lbm_me_vs_reynolds": _rel(row["lbm"]["force_gauge_top"], rey),
                "lbm_me_vs_ref_normal": _rel(row["lbm"]["force_gauge_top"],
                                             row["ref"]["normal_load_top"]),
                "lbm_me_vs_ref_pressure": _rel(row["lbm"]["force_gauge_top"],
                                               row["ref"]["pressure_load_top"]),
                "lbm_pint_vs_ref_pressure": _rel(row["lbm"]["force_pressure_integral"],
                                                 row["ref"]["pressure_load_top"]),
                "ref_normal_vs_reynolds": _rel(row["ref"]["normal_load_top"], rey),
                "ref_pressure_vs_reynolds": _rel(row["ref"]["pressure_load_top"], rey),
                "viscous_normal_share_of_ref_pct":
                    100.0 * row["ref"]["viscous_normal_top"] / row["ref"]["normal_load_top"],
            }
        out["sweep"].append(row)
        print(f"h={h:2d} h/a={h/A:.3f} "
              + (f"lbm_me={row['lbm']['force_gauge_top']:.6e} " if "lbm" in row else "")
              + f"ref_normal={row['ref']['normal_load_top']:.6e} rey={rey:.6e} "
              + (f"lbm_vs_ref={row['err_pct']['lbm_me_vs_ref_normal']:+.2f}% "
                 f"ref_vs_rey={row['err_pct']['ref_normal_vs_reynolds']:+.2f}%"
                 if "lbm" in row else ""), flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())