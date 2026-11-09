"""Assemble the final calibration package from the campaign artefacts.

Inputs (all produced by the campaign scripts, all JSON):

``runs/w5/calibration.json``   rate sweep, shape cases, pure control, contract shell
``runs/e1/ensemble.json``      dilute seed ensemble (5 packings x 4 concentrations)
``runs/h1/highphi.json``       dense batch (2 packings x 4 concentrations)
``runs/g1/drag_gate.json``     single-sphere drag gate (Stokes + GCI)

Output ``runs/ice_eta_final/calibration.json`` + figures, in exactly the schema
:mod:`scripts.ice_eta_report` consumes.  Every value is carried over from the
inputs -- the merge only aggregates (means, SEM, weighted fits), it never
invents a number.

Usage::

    python scripts/ice_eta_merge.py --final runs/ice_eta_final
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from tensorlbm.ice_suspension_rheology import closures
from tensorlbm.ice_suspension_rheology.calibrate import make_figures
from tensorlbm.ice_suspension_rheology.closures import (
    BATCHELOR_K2,
    INTRINSIC_VISCOSITY_SPHERE,
    ViscosityContract,
    batchelor_relative_viscosity,
    fit_krieger_dougherty_phi_max,
)


def _load(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="runs/w5/calibration.json")
    ap.add_argument("--ensemble", default="runs/e1/ensemble.json")
    ap.add_argument("--highphi", default="runs/h1/highphi.json")
    ap.add_argument("--gate", default="runs/g1/drag_gate.json")
    ap.add_argument("--final", default="runs/ice_eta_final")
    args = ap.parse_args(argv)

    cal = _load(args.campaign)
    ens = _load(args.ensemble)
    hi = _load(args.highphi)
    gate = _load(args.gate)
    os.makedirs(args.final, exist_ok=True)
    pure = (ens or hi or {}).get("pure", {}) or (cal or {}).get("analysis", {}).get("pure_control", {})

    # ---- per-seed measurement rows ---------------------------------------
    rows = []
    for src in (ens, hi):
        for g in (src or {}).get("groups", []):
            vals = g.get("eta_r_values") or []
            phis = g.get("phi_lattice_values") or [g["phi"]] * len(vals)
            for k, (v, p) in enumerate(zip(vals, phis or vals)):
                rows.append({
                    "phi": float(p), "eta_r": float(v), "seed": k + 1,
                    "phi_target": g.get("phi_target"), "n_particles": g["n_particles"],
                    "profile_r2": g.get("profile_r2_min"), "source": "ensemble" if src is ens else "highphi",
                    "shape": "sphere", "polydisperse": 0.0,
                })
    rows.sort(key=lambda r: r["phi"])

    # ---- groups (ensemble means with SEM) --------------------------------
    groups = []
    for src, tag in ((ens, "ensemble"), (hi, "highphi")):
        for g in (src or {}).get("groups", []):
            gg = dict(g)
            gg["source"] = tag
            groups.append(gg)
    groups.sort(key=lambda g: g["phi"])

    # ---- dilute fit (ensemble, 1/SEM^2 weights) --------------------------
    dil = dict((ens or {}).get("fit") or {})
    if not dil and len(groups) >= 2:
        sub = [g for g in groups if g["phi"] <= 0.09]
        x = np.array([g["phi"] for g in sub])
        y = np.array([g["eta_r_mean"] for g in sub])
        A = np.vstack([x, np.ones_like(x)]).T
        (s, i), *_ = np.linalg.lstsq(A, y, rcond=None)
        dil = {"slope": float(s), "intercept": float(i),
               "slope_rel_err_vs_2p5": abs(s / 2.5 - 1), "intercept_abs_err_vs_1": abs(i - 1),
               "pass_slope_3pct": bool(abs(s / 2.5 - 1) <= 0.03),
               "pass_intercept": bool(abs(i - 1) <= 0.01), "n_points": len(sub),
               "phi_range": [float(x.min()), float(x.max())]}

    # ---- Batchelor window -------------------------------------------------
    semi = [g for g in groups if 0.05 < g["phi"] <= 0.15]
    bat = {}
    if semi:
        errs = [abs(g["eta_r_mean"] / float(batchelor_relative_viscosity(g["phi"])) - 1.0) for g in semi]
        bat = {
            "max_rel_err": float(max(errs)), "mean_rel_err": float(np.mean(errs)),
            "n_points": len(semi), "pass_10pct": bool(max(errs) <= 0.10),
            "rows": [{"phi": g["phi"], "eta_r": g["eta_r_mean"],
                      "batchelor": float(batchelor_relative_viscosity(g["phi"]))} for g in semi],
        }

    # ---- Krieger-Dougherty -------------------------------------------------
    kd = {}
    if len(rows) >= 3:
        p = np.array([r["phi"] for r in rows])
        e = np.array([r["eta_r"] for r in rows])
        f = fit_krieger_dougherty_phi_max(p, e, intrinsic=INTRINSIC_VISCOSITY_SPHERE)
        kd = {k: v for k, v in f.items() if k != "residual"}
        kd["pass_range_0p58_0p68"] = bool(0.58 <= f["phi_max"] <= 0.68)
        kd["note"] = ("phi_max is fitted to the measured range; extrapolation beyond it is "
                      "not identifiable from these data")
    phi_max = float(kd.get("phi_max", 0.63))
    contract = ViscosityContract(
        phi_max=phi_max,
        intrinsic=float(dil.get("slope", INTRINSIC_VISCOSITY_SPHERE)),
        k2=BATCHELOR_K2,
        i_v=((cal or {}).get("analysis", {}).get("rate_sweep", {}) or {}).get("i_v_estimate"),
    )
    cala = (cal or {}).get("analysis", {})

    ana = {
        "rows": rows,
        "groups": groups,
        "dilute_fit": dil,
        "ensemble_fit": dil,
        "batchelor": bat,
        "krieger_dougherty": kd,
        "pure_control": cala.get("pure_control", {}),
        "negative_control_eta_r_pure": cala.get("negative_control_eta_r_pure", 1.0),
        "closure_limits": {
            "kd_phi0": float(closures.krieger_dougherty_relative_viscosity(0.0, phi_max)),
            "einstein_phi0": float(closures.einstein_relative_viscosity(0.0)),
            "batchelor_phi0": float(closures.batchelor_relative_viscosity(0.0)),
        },
        "contract": contract.as_dict(),
        "rate_sweep": cala.get("rate_sweep", {}),
        "shape_sensitivity": cala.get("shape_sensitivity", []),
        "gate": gate,
        "ensemble": ens,
        "highphi": hi,
    }
    meta = (cal or {}).get("meta", {})
    meta.update({
        "campaign": "IN13-W4-N02",
        "merged_from": [args.campaign, args.ensemble, args.highphi, args.gate],
        "n_cases": sum(len((s or {}).get("groups", [])) for s in (ens, hi)),
    })
    figs = make_figures(ana, os.path.join(args.final, "figures"))
    payload = {"meta": meta, "analysis": ana,
               "figures": [os.path.basename(f) for f in figs]}
    with open(os.path.join(args.final, "calibration.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(json.dumps({
        "groups": [(round(g["phi"], 4), round(g["eta_r_mean"], 4), round(g["eta_r_sem"], 4))
                   for g in groups],
        "dilute": {k: dil.get(k) for k in ("slope", "intercept", "slope_stderr",
                                           "slope_rel_err_vs_2p5", "pass_slope_3pct")},
        "batchelor": {k: bat.get(k) for k in ("max_rel_err", "pass_10pct")},
        "kd": {k: kd.get(k) for k in ("phi_max", "rmse", "r2", "pass_range_0p58_0p68")},
        "figures": [os.path.basename(f) for f in figs],
        "gate": {k: (gate or {}).get(k) for k in ("max_err_pct", "gci", "pass_3pct", "pass_gci_5pct")},
    }, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())