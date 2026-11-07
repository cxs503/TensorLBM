#!/usr/bin/env python3
"""P49-N02 deliverable 4 -- physical invariance of the added mass / damping.

Three independent measurements are cross-validated at ``rho_s/rho_f = 1.0``:

A. **potential flow** (incompressible, same staircase body mask): the unit-velocity
   perturbation potential gives ``A = rho * int |grad psi|^2``, whose unbounded
   value for a circular body is exactly ``rho pi R^2``  (i.e. ``A/M = 1`` when
   ``rho_s = rho_f``).  Resolution/blockage series quantify the discretisation
   and confinement error.
B. **LBM interface operator**: the impulse-response added mass of the coupling
   window (central difference of the momentum-exchange impulse), reported as a
   function of the window length.
C. **coupled free decay** (stabilised Aitken sub-iterations): identified
   ``omega``, ``zeta`` and ``A = C/omega^2 - m_s`` from the campaign artifacts.

B is the operator that enters the sub-iteration stability factor, A is the
physical reference in the same geometry, C is the physical-invariance test in
the full coupled system (its acoustic limitation is reported explicitly).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.fsi_added_mass import (  # noqa: E402
    HeavingDiskPartitioned,
    potential_added_mass,
)

REFERENCE_A_RATIO = 1.0  # verified/floating_body_free_decay heave: A_inf / M


def job_potential(spec):
    return {"kind": "potential", **spec, **potential_added_mass(**spec)}


def job_operator(spec):
    nx, ny, radius, window, tau = spec
    case = HeavingDiskPartitioned(
        nx=nx, ny=ny, radius=radius, window=window, tau=tau, density_ratio=1.0, spring_c=1.0
    )
    res = case.measure_impulse_added_mass(window=window)
    return {
        "kind": "lbm_operator",
        "nx": nx,
        "ny": ny,
        "radius": radius,
        "window": window,
        "tau": tau,
        "c_a": res["c_a"],
        "K": res["K"],
        "nonlinearity": res["nonlinearity"],
        "volume": res["volume"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/fsi_added_mass")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--artifacts", default="artifacts/fsi_added_mass")
    args = ap.parse_args(argv)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    pot_specs = [
        dict(nx=48, ny=48, radius=6.0),
        dict(nx=96, ny=96, radius=6.0),
        dict(nx=192, ny=192, radius=6.0),
        dict(nx=192, ny=192, radius=12.0),
        dict(nx=384, ny=384, radius=12.0),
        dict(nx=192, ny=192, radius=24.0),
    ]
    op_specs = [(48, 48, 6.0, w, 0.55) for w in (1, 2, 3, 5, 10, 20, 30, 40)]

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        potential = list(pool.map(job_potential, pot_specs))
        operator = list(pool.map(job_operator, op_specs))

    ref = potential[0]
    # unbounded extrapolation: c_a - 1 is first order in h/R (staircase) and
    # decreases with R at fixed box aspect; fit c_a = 1 + a/R  over the fixed
    # aspect-ratio series (nx/R = 16) to isolate the discretisation error
    series = [p for p in potential if abs(p["nx"] / p["radius"] - 16.0) < 1e-9]
    extrap = None
    if len(series) >= 2:
        inv = np.array([1.0 / p["radius"] for p in series])
        ca = np.array([p["c_a"] for p in series])
        slope, intercept = np.polyfit(inv, ca, 1)
        extrap = {"c_a_infinite_R": float(intercept), "slope": float(slope), "n": int(len(series))}

    # free-decay (from the campaign artifacts) -- the coupled-system measurement
    camp_path = os.path.join(os.path.abspath(args.artifacts), "campaign.json")
    decay = {}
    if os.path.exists(camp_path):
        with open(camp_path, encoding="utf-8") as fh:
            camp = json.load(fh)
        rows = [d for d in camp.get("free_decay", []) if "c_a_measured" in d]
        decay = {
            "rows": [
                {
                    "nx": d["domain"]["nx"],
                    "blockage": d["blockage"],
                    "omega": d["identification"]["omega"],
                    "acoustic_transit_steps": 2.0 * d["domain"]["nx"] / math.sqrt(1.0 / 3.0),
                    "omega_times_acoustic_time": d["identification"]["omega"]
                    * (2.0 * d["domain"]["nx"] / math.sqrt(1.0 / 3.0)),
                    "c_a_from_frequency": d["c_a_measured"],
                    "zeta": d["identification"]["zeta"],
                }
                for d in rows
            ]
        }

    op10 = next(o for o in operator if o["window"] == 10)
    verdict = {
        "reference": {
            "benchmark": "benchmarks/verified/floating_body_free_decay",
            "case": "heave",
            "added_mass_ratio": REFERENCE_A_RATIO,
            "damping_ratio": 0.0,
        },
        "A_potential_flow": {
            "criterion": "incompressible potential added mass on the same body mask "
            "vs A_inf/M = 1.0 (unbounded extrapolation)",
            "rows": potential,
            "best_c_a": ref["c_a"],
            "best_err_pct": abs(ref["c_a"] - REFERENCE_A_RATIO) / REFERENCE_A_RATIO * 100.0,
            "extrapolation": extrap,
            "pass": bool(
                abs(ref["c_a"] - REFERENCE_A_RATIO) / REFERENCE_A_RATIO * 100.0 <= 5.0
                and (extrap is None or abs(extrap["c_a_infinite_R"] - 1.0) <= 0.05)
            ),
        },
        "B_lbm_operator": {
            "criterion": "LBM momentum-exchange impulse operator at the stability-study "
            "window (10 steps) vs A_inf/M = 1.0",
            "rows": operator,
            "c_a_window10": op10["c_a"],
            "err_pct_window10": abs(op10["c_a"] - REFERENCE_A_RATIO) * 100.0,
            "window_dependence": {str(o["window"]): o["c_a"] for o in operator},
            "pass": bool(abs(op10["c_a"] - REFERENCE_A_RATIO) <= 0.05),
        },
        "C_coupled_free_decay": decay,
        "verdicts": {},
    }
    d_rows = decay.get("rows", [])
    if d_rows:
        verdict["C_coupled_free_decay"]["criterion"] = (
            "stabilised coupled free decay: identified added mass and damping vs the "
            "verified free-decay reference"
        )
        verdict["C_coupled_free_decay"]["added_mass_err_pct"] = [
            abs(r["c_a_from_frequency"] - 1.0) * 100.0 for r in d_rows
        ]
        verdict["C_coupled_free_decay"]["zeta"] = [r["zeta"] for r in d_rows]
        verdict["C_coupled_free_decay"]["pass"] = bool(
            all(abs(r["c_a_from_frequency"] - 1.0) <= 0.05 for r in d_rows)
            and all(r["zeta"] <= 0.05 for r in d_rows)
        )
    verdict["verdicts"] = {
        "A_added_mass_potential": verdict["A_potential_flow"]["pass"],
        "B_added_mass_lbm_operator": verdict["B_lbm_operator"]["pass"],
        "C_coupled_free_decay": verdict["C_coupled_free_decay"].get("pass", False),
    }
    verdict["pass"] = bool(
        verdict["A_potential_flow"]["pass"]
        and verdict["B_lbm_operator"]["pass"]
        and verdict["C_coupled_free_decay"].get("pass", False)
    )
    with open(os.path.join(outdir, "invariance.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1, default=float)
    print(json.dumps(verdict["verdicts"], indent=1))
    print(
        "A: best c_a=%.4f  extrap=%s" % (ref["c_a"], verdict["A_potential_flow"]["extrapolation"])
    )
    print("B: window series %s" % verdict["B_lbm_operator"]["window_dependence"])
    print(
        "C: %s"
        % [
            (
                r["nx"],
                round(r["c_a_from_frequency"], 3),
                round(r["zeta"], 4),
                round(r["omega_times_acoustic_time"], 2),
            )
            for r in d_rows
        ]
    )
    return verdict


if __name__ == "__main__":
    main()
