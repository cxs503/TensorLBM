"""Exploratory diagnostic (post-hoc, declared as such): where is the stress measured?

Motivation
----------
The campaign's near-wall stress estimator samples rows ``1..wall_band`` of the
Couette cell, i.e. a region at least ``wall_margin = 7`` lattice units away from
any particle surface.  Momentum released by the particles has to diffuse to that
band before the wall stress "knows" about it: with ``nu = 0.1`` the diffusive
length after ``t`` steps is ``sqrt(nu t)`` -- only ``11.8`` cells at 1400 steps
and ``16.1`` at 2600 steps, versus a particle-free gap of 7 cells plus a 43-cell
core.  The *row-resolved* stress ``tau_rows_mean`` returned by
:meth:`ShearCell.run` makes the question measurable instead of speculative:
compare the stress carried in the particle-free near-wall band with the stress
in the core (where the particles actually are).

This script is **not** part of the predeclared campaign; it is the diagnostic
that explains its residual bias, and it is reported as such.

Usage::

    PYTHONPATH=src python benchmarks/suspension/probe_core_stress.py \
        --out benchmarks/suspension/artifacts/probe_core_stress.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))


def run_one(phi: float, radius: float, seed: int, *, nx: int, ny: int, nz: int,
            nu: float, u_wall: float, n_steps: int, sample_every: int,
            wall_margin: int, threads: int) -> dict:
    import torch

    torch.set_num_threads(max(1, threads))
    from tensorlbm.ice_suspension_rheology import shapes
    from tensorlbm.ice_suspension_rheology.couette import ShearCell, ShearCellConfig

    cfg = ShearCellConfig(nx=nx, ny=ny, nz=nz, nu=nu, u_wall=u_wall)
    ps = []
    if phi > 0:
        ps = shapes.place_particles(nx, ny, nz, phi, radius, seed=seed,
                                    wall_margin=wall_margin)
    cell = ShearCell(cfg, ps, wall_band=max(2, wall_margin - 1))
    r = cell.run(n_steps, n_warmup=n_steps // 2, sample_every=sample_every)
    rows = np.array(r["tau_rows_mean"], dtype=float)   # per-z mean streamwise flux
    prof = cell.velocity_profile()
    return {
        "phi_target": phi,
        "phi_lattice": float(cell.phi_lattice),
        "radius": radius,
        "seed": seed,
        "n_particles": int(cell.n_particles),
        "tau_rows": rows.tolist(),
        "tau_near": float(r["tau_near"]),
        "tau_corrected": float(r["tau_corrected"]),
        "gamma_dot_measured": float(prof["slope"]),
        "profile_r2": float(prof["r2"]),
        "nu": nu,
        "nz": nz,
        "n_steps": n_steps,
        "wall_margin": wall_margin,
        "solid_rows": sorted({int(z) for p in ps for z in np.nonzero(p.mask)[0].tolist()}),
    }


def _band_mean(rows: np.ndarray, lo: int, hi: int) -> float:
    return float(np.mean(rows[lo:hi + 1]))


def analyse(pure: dict, susp: dict, cfg: dict) -> dict:
    nz = cfg["nz"]
    band = (1, max(2, cfg["wall_margin"] - 1))
    core = (cfg["wall_margin"] + 2, nz - 3 - cfg["wall_margin"])
    out = {"wall_band": list(band), "core_band": list(core)}
    for name, (lo, hi) in (("near_wall", band), ("core", core)):
        tp = _band_mean(np.array(pure["tau_rows"]), lo, hi)
        ts = _band_mean(np.array(susp["tau_rows"]), lo, hi)
        eta = (ts / tp) * (pure["gamma_dot_measured"] / susp["gamma_dot_measured"])
        out[name] = {
            "tau_pure": tp,
            "tau_susp": ts,
            "eta_r": float(eta),
        }
    phi_mask = susp["phi_lattice"]
    phi_core = phi_mask * nz / (nz - 2.0 * cfg["wall_margin"])
    out["phi_mask"] = float(phi_mask)
    out["phi_core_estimate"] = float(phi_core)
    for name in ("near_wall", "core"):
        inc = out[name]["eta_r"] - 1.0
        out[name]["slope_vs_phi_mask"] = float(inc / phi_mask) if phi_mask else float("nan")
        out[name]["slope_vs_phi_core"] = float(inc / phi_core) if phi_core else float("nan")
    out["profile_r2"] = [pure["profile_r2"], susp["profile_r2"]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nz", type=int, default=57)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--u-wall", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--wall-margin", type=int, default=7)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--phi", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts",
                                                 "probe_core_stress.json"))
    args = ap.parse_args(argv)
    base = dict(nx=args.nx, ny=args.ny, nz=args.nz, nu=args.nu, u_wall=args.u_wall,
                n_steps=args.steps, sample_every=args.sample_every,
                wall_margin=args.wall_margin, threads=args.threads)
    pure = run_one(0.0, args.radius, args.seed, **base)
    susp = run_one(args.phi, args.radius, args.seed, **base)
    res = {
        "config": {k: v for k, v in base.items() if k != "threads"},
        "pure": pure,
        "susp": susp,
        "comparison": analyse(pure, susp, base),
        "note": "exploratory post-hoc diagnostic; not part of the predeclared campaign",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, default=float)
    c = res["comparison"]
    for name in ("near_wall", "core"):
        print(f"[{name}] eta_r={c[name]['eta_r']:.4f} "
              f"slope(phi_mask)={c[name]['slope_vs_phi_mask']:.3f} "
              f"slope(phi_core)={c[name]['slope_vs_phi_core']:.3f}")
    print(f"phi_mask={c['phi_mask']:.5f} phi_core={c['phi_core_estimate']:.5f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())