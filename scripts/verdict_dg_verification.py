#!/usr/bin/env python3
"""Aggregate the acceptance criteria of IN70-N01 into VERDICT.json.

Every number is read from the per-benchmark JSON written by
``run_dg_verification.py`` -- nothing is typed by hand.

    PYTHONPATH=src python scripts/verdict_dg_verification.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "IN70_N01")


def load(name):
    path = os.path.join(ART, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    cone = load("rotating_cone")
    gauss = load("rotating_gaussian")
    mms = load("mms")
    sod = load("sod")
    iso = load("sod_isothermal")
    square = load("square_wave")
    mass = load("mass_ledger")

    crit = {}

    # 1) rotating cone convergence order
    if cone:
        p1 = cone["orders"]["P1"]["L2"]["order_last_pair"]
        p2 = cone["orders"]["P2"]["L2"]["order_last_pair"]
        crit["cone_order_P1_ge_1.8"] = {
            "value": p1,
            "gate": 1.8,
            "pass": bool(p1 >= 1.8),
            "evidence": "artifacts/IN70_N01/rotating_cone.json",
        }
        crit["cone_order_P2_ge_2.7"] = {
            "value": p2,
            "gate": 2.7,
            "pass": bool(p2 >= 2.7),
            "evidence": "artifacts/IN70_N01/rotating_cone.json",
        }
        crit["cone_L2_table"] = {
            f"P{r['degree']}_n{r['n_cells']}": r["L2"] for r in cone["rows"]
        }
    if gauss:
        g1 = gauss["orders"]["P1"]["L2"]["order_last_pair"]
        g2 = gauss["orders"]["P2"]["L2"]["order_last_pair"]
        crit["smooth_rotating_advection_order"] = {
            "P1": g1,
            "P2": g2,
            "note": "smooth Gaussian IC: isolates the kink of the cone IC",
            "evidence": "artifacts/IN70_N01/rotating_gaussian.json",
        }

    # 2) MMS order vs theory
    if mms:
        rows = {}
        for p in (1, 2, 3):
            key = f"P{p}"
            if key not in mms["orders"]:
                continue
            got = mms["orders"][key]["L2"]["order_last_pair"]
            rows[key] = {
                "measured": got,
                "theory": p + 1,
                "abs_dev": abs(got - (p + 1)),
                "pass": bool(abs(got - (p + 1)) < 0.3),
            }
        crit["mms_order_within_0.3"] = {
            "rows": rows,
            "pass": all(v["pass"] for v in rows.values()),
            "evidence": "artifacts/IN70_N01/mms.json",
        }

    # 3) Sod shock position
    if iso:
        rows = [r for r in iso["rows"] if r["t"] <= 200.0]
        crit["sod_shock_position_lt_2pct"] = {
            "rows": [
                {
                    "degree": r["degree"],
                    "t": r["t"],
                    "err_rel": r["shock_pos_err_rel"],
                    "lbm_delta_cells": (r.get("lbm") or {}).get("shock_x_offset_cells"),
                }
                for r in iso["rows"]
            ],
            "pass_up_to_t200": all(r["shock_pos_err_rel"] < 0.02 for r in rows),
            "t400_degraded": [
                r["shock_pos_err_rel"] for r in iso["rows"] if r["t"] > 200.0
            ],
            "evidence": "artifacts/IN70_N01/sod_isothermal.json",
        }
    if sod:
        crit["sod_ideal_gas"] = {
            "pass": bool(sod["verdict"]["pass"]),
            "max_shock_err_rel": max(r["shock_pos_err_rel_pressure"] for r in sod["rows"]),
            "L1_rho": {f"P{r['degree']}_n{r['n_cells']}": r["L1_rho"] for r in sod["rows"]},
            "status": "HONEST FAIL (see verdict note)",
            "evidence": "artifacts/IN70_N01/sod.json",
        }

    # 4) square-wave TVD
    if square:
        tv = square["tvd_check"]
        crit["square_wave_tvd"] = {
            "max_u_limited": tv["max_u_limited"],
            "min_u_limited": tv["min_u_limited"],
            "overshoot_unlimited": tv["worst_overshoot_unlimited"],
            "undershoot_unlimited": tv["worst_undershoot_unlimited"],
            "pass": bool(
                tv["worst_overshoot_limited"] < tv["tol"]
                and tv["worst_undershoot_limited"] < tv["tol"]
            ),
            "evidence": "artifacts/IN70_N01/square_wave.json",
        }

    # 5) mass ledger
    if mass:
        crit["mass_conservation_lt_1e-10"] = {
            "value": mass["worst_rel_drift"],
            "gate": 1e-10,
            "pass": bool(mass["worst_rel_drift"] < 1e-10),
            "evidence": "artifacts/IN70_N01/mass_ledger.json",
        }

    verdict = {
        "node": "IN70-N01",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/dg_verification",
        "criteria": crit,
        "n_pass": sum(1 for v in crit.values() if isinstance(v, dict) and v.get("pass")),
        "n_fail": sum(
            1 for v in crit.values() if isinstance(v, dict) and v.get("pass") is False
        ),
        "honest_failures": [
            "rotating-cone convergence order (kink-limited + P2 error floor)",
            "ideal-gas (gamma=1.4) Sod shock position (Rusanov contact diffusion)",
            "isothermal Sod at t=400 (limiter degradation at long time)",
        ],
        "artifacts": sorted(
            os.path.relpath(os.path.join(ART, f), HERE)
            for f in os.listdir(ART)
            if f.endswith((".json", ".npz"))
        ),
        "figures": sorted(
            os.path.relpath(os.path.join(ART, "figures", f), HERE)
            for f in os.listdir(os.path.join(ART, "figures"))
        )
        if os.path.isdir(os.path.join(ART, "figures"))
        else [],
    }
    out = os.path.join(ART, "VERDICT.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1, sort_keys=True)
    print(json.dumps(verdict["criteria"], indent=1, sort_keys=True))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())