"""High-concentration batch: constrain the Krieger-Dougherty phi_max.

    python scripts/ice_eta_highphi.py --out runs/h1 --jobs 8 --threads 2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/h1")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--phis", default="0.12,0.18,0.25,0.32")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)
    from tensorlbm.ice_suspension_rheology.calibrate import _dispatch

    os.makedirs(args.out, exist_ok=True)
    base = dict(nx=32, ny=32, nz=45, radius=4.0, nu=0.1, u_wall=0.02,
                n_steps=1700, wall_margin=7)
    phis = [float(x) for x in args.phis.split(",")]
    cases = [dict(**base, phi=0.0, seed=0, tag="pure")]
    for p in phis:
        for s in range(1, args.seeds + 1):
            cases.append(dict(**base, phi=p, seed=s, tag=f"p{p}_s{s}"))
    print(f"[highphi] {len(cases)} cases", flush=True)
    raw = _dispatch(cases, jobs=args.jobs, threads=args.threads,
                    workdir=os.path.join(args.out, "cases"))
    for c, r in zip(cases, raw):
        r["tag"] = c["tag"]
    json.dump(raw, open(os.path.join(args.out, "highphi_raw.json"), "w"), indent=2, default=float)
    pure = next(r for r in raw if r.get("phi_target", 1) == 0)
    t0, g0 = pure["tau_corrected"], pure["gamma_dot_measured"]
    groups = []
    for p in phis:
        rs = [r for r in raw if abs(r.get("phi_target", -1) - p) < 1e-12 and "error" not in r]
        if not rs:
            continue
        eta = np.array([(r["tau_corrected"] / t0) * (g0 / r["gamma_dot_measured"]) for r in rs])
        phi = np.array([r["phi_lattice"] for r in rs])
        groups.append({
            "phi": float(phi.mean()), "phi_target": float(p), "n_seeds": len(rs),
            "n_particles": int(rs[0]["n_particles"]),
            "eta_r_mean": float(eta.mean()),
            "eta_r_std": float(eta.std(ddof=1)) if len(eta) > 1 else 0.0,
            "eta_r_sem": float(eta.std(ddof=1) / np.sqrt(len(eta))) if len(eta) > 1 else 0.0,
            "eta_r_values": eta.tolist(),
            "relative_particle_force_max": float(max(abs(r["f_particle_per_area"] / r["tau_corrected"]) for r in rs)),
            "profile_r2_min": float(min(r["profile_r2"] for r in rs)),
        })
    out = {"base": base, "groups": groups,
           "pure": {k: pure[k] for k in ("eta_eff", "gamma_dot_measured", "profile_r2", "nu")}}
    json.dump(out, open(os.path.join(args.out, "highphi.json"), "w"), indent=2, default=float)
    for g in groups:
        print("phi=%.4f n=%d mean=%.3f std=%.3f fpp=%.3f" % (
            g["phi"], g["n_particles"], g["eta_r_mean"], g["eta_r_std"], g["relative_particle_force_max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
