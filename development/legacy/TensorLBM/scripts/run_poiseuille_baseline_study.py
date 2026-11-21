#!/usr/bin/env python
"""Baseline study for ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``.

Runs the OSS baseline case on the in-repo D2Q9 channel solver and writes the
JSON that the tests, the figures and the case report consume.  Three stages,
each a self-contained JSON file (the stages are separate so that a single
process stays inside the short-test budget):

``--stage oss``      OSS unit convention (Re, Ma fixed) at resolutions 8..128.
``--stage fixed``    textbook refinement: lattice ``nu`` and ``g`` held fixed.
``--stage tau``      relaxation-time scan of the wall slip at resolution 32.

Usage::

    python scripts/run_poiseuille_baseline_study.py --stage oss
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.verification.xval.poiseuille_skeleton import (  # noqa: E402
    run_baseline_case,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "xval_poiseuille_baseline_skeleton"
OUT.mkdir(parents=True, exist_ok=True)

# the OSS res=16 lattice pair, used as the fixed point of the textbook refinement
NU_FIX = 0.17320508075688773
G_FIX = 8.888888888888893e-09


def _row(res: int, run, wall_s: float) -> dict:
    return {
        "resolution": res,
        "height_lu": float(run.params.height_lu),
        "tau": float(run.params.relaxation_parameter_lu),
        "nu_lu": float(run.params.viscosity_lu),
        "force_lu": float(run.params.acceleration_lu),
        "steps": int(run.steps),
        "converged": bool(run.converged),
        "wall_s": wall_s,
        "mass_drift": float(run.mass_drift),
        "u_peak": float(np.max(run.u_num)),
        "u_peak_exact": float(np.max(run.u_exact)),
        "kpis": {k: float(v) for k, v in run.kpis.items()},
        "slip_kpis": {k: float(v) for k, v in run.slip_kpis.items()},
    }


def stage_oss(resolutions=(8, 16, 32, 64, 128)) -> dict:
    rows = []
    for res in resolutions:
        t0 = time.time()
        run = run_baseline_case(res, max_steps=600_000)
        wall = time.time() - t0
        rows.append(_row(res, run, wall))
        print(f"  oss res={res:4d} tau={run.params.relaxation_parameter_lu:5.3f} "
              f"eps_inf={run.kpis['eps_inf']:.3e} shape={run.slip_kpis['eps_inf_shape']:.2e} "
              f"({wall:.1f}s)", flush=True)
    return {"mode": "oss_unit_convention", "rows": rows}


def stage_fixed(resolutions=(16, 32, 64)) -> dict:
    rows = []
    for res in resolutions:
        t0 = time.time()
        run = run_baseline_case(res, nu_lu=NU_FIX, force_lu=G_FIX, max_steps=600_000)
        wall = time.time() - t0
        rows.append(_row(res, run, wall))
        print(f"  fixed res={res:4d} eps_inf={run.kpis['eps_inf']:.3e} "
              f"shape={run.slip_kpis['eps_inf_shape']:.2e} ({wall:.1f}s)", flush=True)
    rows.sort(key=lambda r: r["resolution"])
    orders = []
    for a, b in zip(rows, rows[1:]):
        p = float(np.log(a["kpis"]["eps_inf"] / b["kpis"]["eps_inf"])
                  / np.log(b["height_lu"] / a["height_lu"]))
        orders.append({"pair": (a["resolution"], b["resolution"]), "order_eps_inf": p})
    return {"mode": "fixed_lattice_nu_force", "nu_lu": NU_FIX, "force_lu": G_FIX,
            "rows": rows, "orders": orders}


def stage_tau(reynolds=(8.0, 4.0, 2.0, 1.4, 1.0, 0.8, 0.65, 0.55, 0.45, 0.35)) -> dict:
    rows = []
    for re in reynolds:
        t0 = time.time()
        run = run_baseline_case(32, reynolds_number=re, max_steps=400_000)
        wall = time.time() - t0
        rows.append({
            "reynolds_number": float(re),
            "tau": float(run.params.relaxation_parameter_lu),
            "nu_lu": float(run.params.viscosity_lu),
            "slip_over_g": float(run.slip_kpis["slip"] / run.params.acceleration_lu),
            "slip_rel": float(run.slip_kpis["slip_rel"]),
            "slip_rel_predicted": float(8.0 * run.params.viscosity_lu
                                        * (run.slip_kpis["slip"] / run.params.acceleration_lu)
                                        / run.params.height_lu ** 2),
            "wall_s": wall,
        })
        print(f"  tau res=32 Re={re:5.2f} tau={run.params.relaxation_parameter_lu:6.4f} "
              f"slip/g={rows[-1]['slip_over_g']:+.5f} ({wall:.1f}s)", flush=True)
    rows.sort(key=lambda r: r["tau"])
    tau = np.array([r["tau"] for r in rows])
    f = np.array([r["slip_over_g"] for r in rows])
    # zero crossing of the slip law, linear interpolation between bracketing points
    zero = None
    for a, b in zip(rows, rows[1:]):
        if a["slip_over_g"] * b["slip_over_g"] < 0.0:
            t = a["tau"] + (b["tau"] - a["tau"]) * abs(a["slip_over_g"]) / (
                abs(a["slip_over_g"]) + abs(b["slip_over_g"]))
            zero = float(t)
            break
    # large-tau asymptote: least squares on slip/g = a*tau + b for tau >= 2
    mask = tau >= 2.0
    slope, intercept = np.polyfit(tau[mask], f[mask], 1)
    return {"mode": "relaxation_time_scan", "resolution": 32, "rows": rows,
            "slip_zero_tau": zero,
            "asymptote": {"slope": float(slope), "intercept": float(intercept),
                          "tau_min": 2.0}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("oss", "fixed", "tau"))
    args = ap.parse_args()
    result = {"oss": stage_oss, "fixed": stage_fixed, "tau": stage_tau}[args.stage]()
    path = OUT / f"study_{args.stage}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())