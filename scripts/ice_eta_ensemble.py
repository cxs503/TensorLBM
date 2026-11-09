"""Seed ensemble for the dilute-limit regression (IN13-W4-N02).

A frozen particle configuration makes the measured ``eta_r`` configuration
dependent: at ``phi ~ 0.01-0.08`` a 32x32x45 cell holds 2-9 spheres, so the
configuration-to-configuration scatter is comparable to the signal.  This
script averages several independent random packings per ``phi`` and reports the
mean, the standard error of the mean (SEM) and the per-seed values, so the
dilute-limit regression uses an *ensemble* rather than one lucky packing.

Usage::

    python scripts/ice_eta_ensemble.py --out runs/ice_eta_ensemble --jobs 10

Writes ``ensemble_raw.json`` (per-case) and ``ensemble.json`` (per-phi summary
plus the weighted linear fit ``eta_r = 1 + [eta] phi``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np



def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ice_eta_ensemble")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--phis", default="0.015,0.03,0.05,0.08")
    ap.add_argument("--reuse", default=None,
                    help="existing sweep_raw.json to seed the ensemble with")
    args = ap.parse_args(argv)
    from tensorlbm.ice_suspension_rheology.calibrate import _dispatch as dispatch

    os.makedirs(args.out, exist_ok=True)
    base = dict(nx=32, ny=32, nz=45, radius=4.0, nu=0.1, u_wall=0.02,
                n_steps=1700, wall_margin=7)
    phis = [float(x) for x in args.phis.split(",")]
    seeds = list(range(1, args.seeds + 1))
    cases = [dict(**base, phi=0.0, seed=0, tag="pure")]
    for p in phis:
        for s in seeds:
            cases.append(dict(**base, phi=p, seed=s, tag=f"p{p}_s{s}"))
    print(f"[ensemble] {len(cases)} cases, phi={phis}, seeds={seeds}", flush=True)
    raw = dispatch(cases, jobs=args.jobs, threads=args.threads,
                   workdir=os.path.join(args.out, "cases"))
    for c, r in zip(cases, raw):
        r["tag"] = c["tag"]
    json.dump(raw, open(os.path.join(args.out, "ensemble_raw.json"), "w"),
              indent=2, default=float)
    errs = [r for r in raw if "error" in r]
    if errs:
        print(f"[ensemble] {len(errs)} failed cases: {errs[0]['error']}", flush=True)
    pure = next(r for r in raw if r.get("phi_target", 1) == 0)
    t0, g0 = pure["tau_corrected"], pure["gamma_dot_measured"]

    groups = []
    for p in phis:
        rs = [r for r in raw if abs(r.get("phi_target", -1) - p) < 1e-12
              and "error" not in r]
        if not rs:
            continue
        eta = np.array([(r["tau_corrected"] / t0) * (g0 / r["gamma_dot_measured"]) for r in rs])
        phi = np.array([r["phi_lattice"] for r in rs])
        fpp = np.array([abs(r["f_particle_per_area"] / r["tau_corrected"]) for r in rs])
        groups.append(
            {
                "phi": float(phi.mean()),
                "phi_target": float(p),
                "n_seeds": int(len(rs)),
                "n_particles": int(rs[0]["n_particles"]),
                "eta_r_mean": float(eta.mean()),
                "eta_r_std": float(eta.std(ddof=1)) if len(eta) > 1 else 0.0,
                "eta_r_sem": float(eta.std(ddof=1) / np.sqrt(len(eta))) if len(eta) > 1 else 0.0,
                "eta_r_values": eta.tolist(),
                "phi_lattice_values": phi.tolist(),
                "relative_particle_force_max": float(fpp.max()),
                "profile_r2_min": float(min(r["profile_r2"] for r in rs)),
            }
        )
    fit = {}
    if len(groups) >= 2:
        x = np.array([g["phi"] for g in groups])
        y = np.array([g["eta_r_mean"] for g in groups])
        w = np.array([1.0 / max(g["eta_r_sem"], 1e-6) ** 2 for g in groups])
        A = np.vstack([x, np.ones_like(x)]).T
        W = np.diag(w)
        cov = np.linalg.inv(A.T @ W @ A)
        beta = cov @ (A.T @ W @ y)
        pred = A @ beta
        ss_res = float(np.sum(w * (y - pred) ** 2))
        ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        fit = {
            "slope": float(beta[0]),
            "intercept": float(beta[1]),
            "slope_stderr": float(np.sqrt(cov[0, 0])),
            "intercept_stderr": float(np.sqrt(cov[1, 1])),
            "slope_rel_err_vs_2p5": float(abs(beta[0] / 2.5 - 1.0)),
            "intercept_abs_err_vs_1": float(abs(beta[1] - 1.0)),
            "weighted_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pass_slope_3pct": bool(abs(beta[0] / 2.5 - 1.0) <= 0.03),
            "pass_intercept": bool(abs(beta[1] - 1.0) <= 0.01),
            "weighting": "1/sem^2 of the seed ensemble",
        }
    out = {"base": base, "phis": phis, "seeds": seeds,
           "pure": {k: pure[k] for k in ("eta_eff", "gamma_dot_measured", "profile_r2",
                                         "tau_corrected", "nu", "rel_err_vs_nu")
                    if k in pure},
           "groups": groups, "fit": fit,
           "pure_rel_err_vs_nu": float(abs(pure["eta_eff"] / pure["nu"] - 1.0))}
    json.dump(out, open(os.path.join(args.out, "ensemble.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k in ("groups", "fit", "pure_rel_err_vs_nu")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())