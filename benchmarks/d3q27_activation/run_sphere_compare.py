#!/usr/bin/env python3
"""Sphere drag: D3Q19 vs D3Q27 through GeneralSimEngine — gates (c)(d)(e)(h).

Cases (predeclared in ``prereg.json``; ``--case-index i`` runs case i):

  idx  lattice  Re   extrap      steps
  0    d3q19    100  quadratic   6000
  1    d3q27    100  quadratic   6000
  2    d3q19    100  none        6000   (secondary: shipped-default force path)
  3    d3q27    100  none        6000
  4    d3q19    200  quadratic  10000
  5    d3q27    200  quadratic  10000   (also the gate-(h) 10 000-step MRT run)

Everything else is identical between the two lattices: D = 20 lattice cells,
auto domain (geometry bbox × 3 with the engine's streamwise pad → 70×60×60),
MRT collision, half-way bounce-back, mass correction every 200 steps,
force sample every 10 steps.  Drag reference: Schiller-Naumann
Cd = 24/Re (1 + 0.15 Re^0.687) → 1.0917 (Re=100), 0.8064 (Re=200);
cross-reference Clift-Gauvin is stored in the JSON for the report.

Units: ρ=1000 kg/m³, μ=1e-6 m²/s, L=1 m (sphere diameter), U chosen so that
Re = U L / ν matches; the engine reports Cd = F_x / (½ ρ U² A), A = πR².
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

from tensorlbm.general_sim import (
    CollisionModel,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

CASES = [
    {"lattice": "d3q19", "re": 100, "extrap": "quadratic", "steps": 6000},
    {"lattice": "d3q27", "re": 100, "extrap": "quadratic", "steps": 6000},
    {"lattice": "d3q19", "re": 100, "extrap": "none", "steps": 6000},
    {"lattice": "d3q27", "re": 100, "extrap": "none", "steps": 6000},
    {"lattice": "d3q19", "re": 200, "extrap": "quadratic", "steps": 10000},
    {"lattice": "d3q27", "re": 200, "extrap": "quadratic", "steps": 10000},
]


def cd_schiller_naumann(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def cd_clift_gauvin(re: float) -> float:
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def run_case(case: dict, threads: int) -> dict:
    torch.set_num_threads(threads)
    re = case["re"]
    lat = LatticeModel.D3Q27 if case["lattice"] == "d3q27" else LatticeModel.D3Q19
    # U = Re * nu / L with nu = 1e-6, L = 1 m
    u_phys = re * 1.0e-6 / 1.0
    tag = f"sphere_{case['lattice']}_re{re}_{case['extrap']}"
    cfg = GeneralSimConfig(
        name=tag,
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=u_phys,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=lat,
            collision=CollisionModel.MRT,
            resolution=20,  # D = 20 lattice cells
            max_steps=case["steps"],
            device="cpu",
            force_sample_interval=10,
            snapshot_interval=10**9,
            pressure_extrap=case["extrap"],
            mass_correction=True,
            mass_correction_interval=200,
            divergence_check_interval=50,
        ),
        output=OutputConfig(directory=f"/tmp/{tag}", save_macroscopic=False),
    )
    eng = GeneralSimEngine(cfg)
    t0 = time.time()
    setup = eng.setup()
    out = eng.run(case["steps"])
    wall = time.time() - t0

    hist = [
        {k: e[k] for k in ("step", "cd_pressure", "cd_friction", "cd_total", "cl")}
        for e in eng.forces_log
    ]
    steps = case["steps"]
    tail = [e for e in hist if e["step"] > 0.75 * steps]
    prev = [e for e in hist if 0.5 * steps < e["step"] <= 0.75 * steps]
    mean_tail = sum(e["cd_total"] for e in tail) / len(tail)
    mean_prev = sum(e["cd_total"] for e in prev) / len(prev) if prev else float("nan")
    drift_pct = abs(mean_tail - mean_prev) / abs(mean_tail) * 100.0 if prev else float("nan")
    mass_drift_ppm = (float(eng.f.sum().item()) - eng._initial_mass) / eng._initial_mass * 1e6

    # mid-plane fields for figures / independent audits
    from tensorlbm.drag_pressure import _macroscopic

    rho, ux, uy, uz = _macroscopic(eng.f)
    mid = rho.shape[1] // 2
    fields = {
        "rho": rho[:, mid, :].numpy(),
        "ux": ux[:, mid, :].numpy(),
        "uy": uy[:, mid, :].numpy(),
        "uz": uz[:, mid, :].numpy(),
        "solid_mid": eng.solid[:, mid, :].numpy(),
    }
    os.makedirs(RUNS, exist_ok=True)
    npz = os.path.join(RUNS, f"{tag}_fields.npz")
    import numpy as np

    np.savez_compressed(npz, **fields)

    cd_ref = cd_schiller_naumann(re)
    return {
        "case": tag,
        "lattice": case["lattice"],
        "re": re,
        "extrap": case["extrap"],
        "steps_requested": steps,
        "steps_run": eng.step_count,
        "diverged": out["diverged"],
        "finite": not out["diverged"],
        "resolution_D": 20,
        "domain_lu": setup["domain_lu"],
        "tau": setup["tau"],
        "u_lb": setup["u_lb"],
        "nu_lb": setup["nu_lb"],
        "Ma": setup["Ma"],
        "obstacle_cells": setup["obstacle_cells"],
        "collision": "mrt",
        "wall_treatment": "bounce_back(half-way)",
        "cd_tail_mean": mean_tail,
        "cd_prev_mean": mean_prev,
        "plateau_drift_pct": drift_pct,
        "mass_drift_ppm": mass_drift_ppm,
        "cd_ref_schiller_naumann": cd_ref,
        "cd_ref_clift_gauvin": cd_clift_gauvin(re),
        "cd_error_pct": abs(mean_tail - cd_ref) / cd_ref * 100.0,
        "wall_s": wall,
        "force_history": hist,
        "fields_npz": os.path.relpath(npz, HERE),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--case-index", type=int, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TL_SPHERE_THREADS", "16")))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(json.dumps(CASES, indent=2))
        return 0
    idxs = range(len(CASES)) if a.case_index is None else [a.case_index]
    rc = 0
    for i in idxs:
        case = CASES[i]
        print(f"[sphere] case {i}: {case}", flush=True)
        payload = run_case(case, a.threads)
        os.makedirs(RUNS, exist_ok=True)
        out = os.path.join(RUNS, f"{payload['case']}.json")
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(
            f"[sphere] {payload['case']}: Cd={payload['cd_tail_mean']:.4f} "
            f"ref={payload['cd_ref_schiller_naumann']:.4f} "
            f"err={payload['cd_error_pct']:.2f}% drift={payload['plateau_drift_pct']:.2f}% "
            f"finite={payload['finite']} wall={payload['wall_s']:.0f}s",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())