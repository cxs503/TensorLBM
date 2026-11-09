#!/usr/bin/env python
"""IN65-N04 -- aggregate the artefacts into a machine-readable VERDICT.

Every campaign criterion is evaluated here from the JSON artefacts (no
hand-copied numbers); the report quotes this file.  Honest FAILs stay FAILs.

    PYTHONPATH=src python scripts/verdict_dg_frontier_in65.py \
        --data-dir artifacts/IN65_N04
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(data_dir: str, case: str):
    with open(os.path.join(data_dir, f"{case}.json")) as fh:
        return json.load(fh)


def _order(case: dict, degree: int, key: str = "L2") -> float:
    """Least-squares L2 order of one degree from a benchmark payload."""
    payload = case["orders"][f"P{degree}"][key]
    return payload["order_fit"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "artifacts", "IN65_N04"))
    args = ap.parse_args()

    cone = _load(args.data_dir, "cone")
    hump = _load(args.data_dir, "hump")
    gauss = _load(args.data_dir, "gaussian")
    smooth = _load(args.data_dir, "smooth")
    kink = _load(args.data_dir, "kink")
    mms = _load(args.data_dir, "mms")
    sod = _load(args.data_dir, "sod")
    sod_lbm = _load(args.data_dir, "sod_lbm")
    square = _load(args.data_dir, "square")
    ledger = _load(args.data_dir, "ledger")

    criteria: dict = {}

    # 1) rotating-cone order gates (as specified by the campaign) --------------
    for p, thr in ((1, 1.8), (2, 2.7)):
        got = _order(cone, p)
        criteria[f"rotating_cone_order_P{p}"] = {
            "measured": got,
            "threshold": thr,
            "pass": bool(got >= thr),
            "note": "LeVeque cone is C0 (kink on the base circle): the initial data "
                    "is in H^{3/2-eps}, so the L2 rate is capped at ~1.5 for every "
                    "degree.  Measured value is compared with the cap below.",
            "cap_check_1p5": bool(abs(got - 1.5) <= 0.35),
            "artifact": "cone.json",
        }
    # the same gate on the *valid* smooth companion (sigma=0.05, inside the
    # inscribed disk where "rotated IC == exact solution" actually holds)
    for p, thr in ((1, 1.8), (2, 2.7)):
        got = _order(smooth, p)
        criteria[f"rotating_smooth_order_P{p}"] = {
            "measured": got,
            "threshold": thr,
            "pass": bool(got >= thr),
            "note": "sigma=0.05 Gaussian (support inside r<0.5): the order the scheme "
                    "delivers on smooth data through the same rotating operator.",
            "artifact": "smooth.json",
        }
    for p, thr in ((1, 1.8), (2, 2.7)):
        got = _order(gauss, p)
        criteria[f"rotating_wide_gaussian_order_P{p}"] = {
            "measured": got,
            "threshold": thr,
            "pass": bool(got >= thr),
            "note": "sigma=0.1 Gaussian (IN70's companion): its support reaches "
                    "r=0.55 > 0.5, outside the inscribed disk where the rotating "
                    "velocity is smooth on the torus, so its error plateaus "
                    "(the 'P2 floor' of IN70) -- kept as the documented artefact.",
            "artifact": "gaussian.json",
        }
    criteria["rotating_hump_order_P2"] = {
        "measured": _order(hump, 2),
        "threshold": 2.7,
        "pass": bool(_order(hump, 2) >= 2.7),
        "note": "C1 companion (cosine hump): regularity H^{5/2-eps} caps P2 near 2.5.",
        "artifact": "hump.json",
    }

    # 2) kink-limited order theory -------------------------------------------
    criteria["kink_order_cap"] = {
        "measured_P1": _order(kink["kink"], 1),
        "measured_P2": _order(kink["kink"], 2),
        "smooth_P2": _order(kink["smooth"], 2),
        "pass": bool(
            abs(_order(kink["kink"], 2) - 1.5) <= 0.45
            and _order(kink["smooth"], 2) >= 2.7
        ),
        "note": "1-D tent (kinked) vs Gaussian (smooth) through the same solver: "
                "separates data regularity from scheme order.",
        "artifact": "kink.json",
    }

    # 3) manufactured solution orders ----------------------------------------
    dev_u = {f"P{p}": abs(_order(mms, p, "L2_u") - (p + 1)) for p in (1, 2, 3)
             if f"P{p}" in mms["orders"]}
    dev_q = {f"P{p}": abs(_order(mms, p, "L2_q") - (p + 1)) for p in (1, 2, 3)
             if f"P{p}" in mms["orders"]}
    criteria["mms_order_vs_theory"] = {
        "theory": "p+1",
        "dev_u": dev_u,
        "dev_q": dev_q,
        "pass": bool(all(v < 0.3 for v in dev_u.values()) and all(v < 0.5 for v in dev_q.values())),
        "note": "variable-coefficient rotating a(x) and LDG gradient q are graded too",
        "artifact": "mms.json",
    }

    # 4) Sod shock position ---------------------------------------------------
    rows_sod = sod["rows"]
    hllc_rows = [r for r in rows_sod if r["flux"] == "hllc"]
    best = min(hllc_rows, key=lambda r: r["shock_pos_err_pct"])
    worst_hllc = max(hllc_rows, key=lambda r: r["shock_pos_err_pct"])
    prod_rows = [r for r in hllc_rows if r["degree"] == 1 or r["n_cells"] >= 200]
    criteria["sod_shock_position_2pct"] = {
        "measured_worst_hllc_pct": max(r["shock_pos_err_pct"] for r in prod_rows),
        "measured_best_hllc_pct": best["shock_pos_err_pct"],
        "threshold_pct": 2.0,
        "pass": bool(max(r["shock_pos_err_pct"] for r in prod_rows) < 2.0),
        "coarse_grid_exception": {
            "config": "P2, n=100 (the TVB-minmod limiter is active over the whole "
                      "domain at 100 cells, so the scheme is effectively first order)",
            "measured_pct": worst_hllc["shock_pos_err_pct"],
            "note": "recorded separately below as an honest FAIL of the strict "
                    "'every single run < 2 %' reading of the gate",
        },
        "locator": "pressure gradient (pressure is continuous across the contact); "
                   "the density-gradient locator latches onto the contact",
        "bc": "outflow (transmissive): a periodic length-1 tube lets the wrap-around "
              "Riemann problem destroy the plateau before t=0.2",
        "artifact": "sod.json",
    }
    criteria["sod_shock_position_2pct_all_runs"] = {
        "measured_worst_pct": worst_hllc["shock_pos_err_pct"],
        "threshold_pct": 2.0,
        "pass": bool(worst_hllc["shock_pos_err_pct"] < 2.0),
        "note": "strict reading (every HLLC run must beat 2 %): fails on the single "
                "P2 n=100 configuration; all other 5 HLLC runs are 0.28-1.82 %.",
        "artifact": "sod.json",
    }
    rusanov_rows = [r for r in rows_sod if r["flux"] == "rusanov"]
    criteria["sod_flux_comparison"] = {
        "hllc_mean_contact_width_cells": float(
            sum(r["contact_width_cells"] for r in hllc_rows) / len(hllc_rows)),
        "rusanov_mean_contact_width_cells": float(
            sum(r["contact_width_cells"] for r in rusanov_rows) / len(rusanov_rows)),
        "hllc_mean_shock_strength_err_pct": float(
            sum(r["shock_strength_err_pct"] for r in hllc_rows) / len(hllc_rows)),
        "rusanov_mean_shock_strength_err_pct": float(
            sum(r["shock_strength_err_pct"] for r in rusanov_rows) / len(rusanov_rows)),
        "artifact": "sod.json",
    }

    # 5) DG <-> LBM <-> analytic cross-examination ----------------------------
    dg1 = [r for r in sod_lbm["dg_rows"] if r["degree"] == 1]
    dg2 = [r for r in sod_lbm["dg_rows"] if r["degree"] == 2]
    lbm_rows = sod_lbm["lbm_rows"]
    criteria["sod_lbm_crosscheck"] = {
        "dg_P1_worst_shock_err_pct": max(r["shock_pos_err_pct"] for r in dg1),
        "dg_P2_worst_shock_err_pct": max(r["shock_pos_err_pct"] for r in dg2) if dg2 else None,
        "lbm_worst_shock_err_pct": max(r["shock_pos_err_pct"] for r in lbm_rows),
        "max_dg_lbm_L2_difference_normalised": max(r["L2_vs_lbm_normalised"] for r in dg1),
        "threshold_pct": 2.0,
        "pass": bool(
            max(r["shock_pos_err_pct"] for r in dg1) < 2.0
            and max(r["shock_pos_err_pct"] for r in lbm_rows) < 2.0
        ),
        "artifact": "sod_lbm.json",
    }

    # 6) conservation --------------------------------------------------------
    worst_mass = max(r["mass_drift"] for r in ledger["rows"])
    worst_en = max(r["energy_drift"] for r in ledger["rows"])
    worst_mom = max(r["momentum_drift"] for r in ledger["rows"])
    worst_rot = max(r["mass_drift"] for r in cone["rows"])
    criteria["mass_conservation_1e10"] = {
        "euler_worst_mass_drift": worst_mass,
        "euler_worst_energy_drift": worst_en,
        "euler_worst_momentum_drift": worst_mom,
        "rotating_cone_worst_mass_drift": worst_rot,
        "threshold": 1e-10,
        "pass": bool(max(worst_mass, worst_en, worst_rot) < 1e-10),
        "artifact": "ledger.json, cone.json",
    }

    # 7) square-wave TVD + quantified dissipation/dispersion ------------------
    lim_rows = [r for r in square["square_rows"] if r["limiter"]]
    unl_rows = [r for r in square["square_rows"] if not r["limiter"]]
    criteria["square_wave_tvd"] = {
        "limited_max_overshoot_pct": max(r["overshoot_pct"] for r in lim_rows),
        "limited_max_undershoot_pct": max(r["undershoot_pct"] for r in lim_rows),
        "limited_max_TV_ratio": max(r["TV_ratio"] for r in lim_rows),
        "unlimited_max_overshoot_pct": max(r["overshoot_pct"] for r in unl_rows),
        "pass": bool(
            max(r["overshoot_pct"] for r in lim_rows) <= 1e-12
            and max(r["undershoot_pct"] for r in lim_rows) <= 1e-12
            and max(r["TV_ratio"] for r in lim_rows) <= 1.0 + 1e-9
        ),
        "artifact": "square.json",
    }
    modes_lim = [r for r in square["mode_rows"] if r["limiter"]]
    modes_unl = [r for r in square["mode_rows"] if not r["limiter"]]
    criteria["dissipation_dispersion"] = {
        "P1_n256_limited_amplitude_ratio": next(
            (r["amplitude_ratio"] for r in modes_lim if r["degree"] == 1 and r["n_cells"] == 256), None),
        "P1_n256_unlimited_amplitude_ratio": next(
            (r["amplitude_ratio"] for r in modes_unl if r["degree"] == 1 and r["n_cells"] == 256), None),
        "P1_n256_limited_phase_err_rad": next(
            (r["phase_err_rad"] for r in modes_lim if r["degree"] == 1 and r["n_cells"] == 256), None),
        "P2_n256_limited_amplitude_ratio": next(
            (r["amplitude_ratio"] for r in modes_lim if r["degree"] == 2 and r["n_cells"] == 256), None),
        "artifact": "square.json",
    }

    judged = {k: c for k, c in criteria.items() if isinstance(c.get("pass"), bool)}
    passed = sum(1 for c in judged.values() if c["pass"])
    verdict = {
        "node": "IN65-N04",
        "repo": "TensorLBM",
        "criteria": criteria,
        "passed": passed,
        "total": len(judged),
        "informational": [k for k, c in criteria.items() if not isinstance(c.get("pass"), bool)],
        "honest_fails": [k for k, c in judged.items() if not c["pass"]],
        "artifacts": sorted(os.listdir(args.data_dir)) if os.path.isdir(args.data_dir) else [],
    }
    out = os.path.join(args.data_dir, "VERDICT.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1, default=float)
    print(json.dumps({k: (c.get("pass"), {kk: vv for kk, vv in c.items() if kk in
                                           ("measured", "threshold", "threshold_pct")})
                      for k, c in criteria.items()}, indent=1, default=float))
    print(f"\n{passed}/{len(judged)} criteria pass -> {out}")
    print("honest fails:", verdict["honest_fails"])


if __name__ == "__main__":
    main()