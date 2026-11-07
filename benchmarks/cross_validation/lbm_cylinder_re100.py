#!/usr/bin/env python3
"""D2Q9 LBM free-stream cylinder at Re=100 — tileable cross-validation driver.

Same physical case as TensorFVM (``fvm_cylinder_re100.py``): uniform free
stream over a circular cylinder, Re = U D / nu = 100, vortex shedding
regime.  Drag is measured with the Ladd momentum-exchange (MEM) wall force
integral from :func:`tensorlbm.boundaries.compute_obstacle_forces`, which is a
*direct* force measurement (not a control-volume momentum budget).

Numerical ingredients (identical in spirit to the repository-verified case
``benchmarks/verified/cylinder/re100``):

* MRT collision with a per-cell relaxation field for the downstream sponge
  (``tau_eff = tau (1 + alpha sigma)``), ``tensorlbm.solver.collide_mrt``;
* ``tensorlbm.boundaries.far_field_bc_2d`` free-stream inlet / lateral
  boundaries, zero-gradient outlet, half-way bounce-back on the cylinder;
* two-period inlet seeding of a transverse velocity wave at ``St_seed`` to
  trigger shedding deterministically instead of waiting for round-off growth;
* force history reduced to cycle-averaged Cd and spectral St.

Usage::

    python3 benchmarks/cross_validation/lbm_cylinder_re100.py \
        --diameter-cells 20 --steps 30000 --out artifacts/lbm_re100_d20.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundaries import (
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_bgk, collide_mrt, stream

REF_CD = 1.35  # Henderson (1995) doi:10.1063/1.868459
# D2Q9 lattice velocity components (columns match tensorlbm.d2q9.C)
_CX = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
_CY = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
REF_ST = 0.1647


def strouhal_from_lift(lift: np.ndarray, dt: float, diameter: float, u_inf: float):
    signal = lift - lift.mean()
    if signal.size < 256:
        return float("nan")
    spec = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    k = int(np.argmax(spec[1:])) + 1
    delta = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        den = a - 2.0 * b + c
        if abs(den) > 1e-30:
            delta = float(0.5 * (a - c) / den)
    return float((k + delta) / signal.size * dt * diameter / u_inf)



def _json_safe(obj):
    """Recursively replace non-finite floats with None so strict JSON stays valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def run(args) -> dict:
    D = args.diameter_cells
    nx = int(args.domain_length_d * D)
    ny = int(args.domain_height_d * D)
    nu = args.u * D / args.re
    tau = 0.5 + 3.0 * nu
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)

    mask = cylinder_mask(nx, ny, args.cylinder_x_d * D, ny / 2.0, D / 2.0, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - args.sponge_d * D), int(args.sponge_d * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + args.sponge_alpha * sigma)

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, args.u), torch.zeros_like(rho0))

    if args.collision.upper() == "MRT":
        def collide_fn(f):
            return collide_mrt(f, tau, tau_field=tau_field)
    else:
        def collide_fn(f):
            return collide_bgk(f, tau)

    def step_plain(f):
        f = stream(collide_fn(f))
        fx, fy = compute_obstacle_forces(f, mask)
        return far_field_bc_2d(f, args.u, mask), fx, fy

    omega_seed = 2.0 * math.pi * args.st_seed * args.u / D
    seed_steps = int(round(args.seed_periods / (args.st_seed * args.u / D)))

    def step_seeded(f, uy_val):
        f = stream(collide_fn(f))
        fx, fy = compute_obstacle_forces(f, mask)
        f = far_field_bc_2d(f, args.u, mask)
        col = torch.full((ny, 1), uy_val, device=device, dtype=f.dtype)
        f[:, :, 0] = equilibrium(
            torch.ones((ny, 1), device=device, dtype=f.dtype),
            torch.full((ny, 1), args.u, device=device, dtype=f.dtype),
            col,
        )[:, :, 0]
        return f, fx, fy

    q_dyn = 0.5 * 1.0 * args.u * args.u * D
    fx_hist: list[float] = []
    fy_hist: list[float] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step <= seed_steps:
            f, fx, fy = step_seeded(f, 0.10 * args.u * math.sin(omega_seed * step))
        else:
            f, fx, fy = step_plain(f)
        fx_hist.append(float(fx) / q_dyn)
        fy_hist.append(float(fy) / q_dyn)
        if args.report_interval and step % args.report_interval == 0:
            print(
                f"  step {step}/{args.steps} Cd={np.mean(fx_hist[-500:]):.4f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError("non-finite populations")

    if args.snapshot_out:
        rho = f.sum(dim=0)
        ux = (f * _CX.view(9, 1, 1)).sum(dim=0) / rho
        uy = (f * _CY.view(9, 1, 1)).sum(dim=0) / rho
        # vorticity by central differences in lattice units
        w = torch.zeros_like(ux)
        w[1:-1, 1:-1] = (uy[1:-1, 2:] - uy[1:-1, :-2]) / 2.0 - (ux[2:, 1:-1] - ux[:-2, 1:-1]) / 2.0
        np.savez(
            args.snapshot_out,
            vorticity=w.cpu().numpy(),
            ux=ux.cpu().numpy(),
            uy=uy.cpu().numpy(),
            mask=mask.cpu().numpy(),
            diameter=D,
            u_in=args.u,
        )

    cd = np.asarray(fx_hist)
    cl = np.asarray(fy_hist)
    warm = int(args.warmup_frac * cd.size)
    cd_w, cl_w = cd[warm:], cl[warm:]
    st = strouhal_from_lift(cl_w, 1.0, D, args.u)
    cd_mean = float(cd_w.mean())
    return {
        "solver": "TensorLBM D2Q9 %s + Ladd MEM wall force" % args.collision.upper(),
        "case": "free-stream circular cylinder Re=100 (cross-validation with TensorFVM)",
        "mesh": {
            "lattice": "D2Q9",
            "diameter_lattice_units": D,
            "nx": nx,
            "ny": ny,
            "cells": int(nx * ny),
            "domain_length_D": args.domain_length_d,
            "domain_height_D": args.domain_height_d,
            "cylinder_x_D": args.cylinder_x_d,
            "lateral_clearance_D": args.domain_height_d - 1.0,
            "blockage_ratio_D_over_H": 1.0 / args.domain_height_d,
            "upstream_D": args.cylinder_x_d,
            "downstream_D": args.domain_length_d - args.cylinder_x_d,
            "sponge": {
                "width_D": args.sponge_d,
                "alpha": args.sponge_alpha,
                "power": 2.0,
            },
        },
        "numerics": {
            "collision": args.collision.upper(),
            "reynolds": args.re,
            "u_in_lattice": args.u,
            "mach": args.u * math.sqrt(3.0),
            "nu_lattice": nu,
            "tau": tau,
            "steps": args.steps,
            "seed": {"st_seed": args.st_seed, "periods": args.seed_periods,
                     "amplitude_fraction": 0.10, "steps": seed_steps},
            "device": str(device),
            "torch_threads": args.threads,
        },
        "sampling": {
            "warmup_fraction": args.warmup_frac,
            "averaging_window_steps": int(cd.size - warm),
            "averaging_window_periods": float((cd.size - warm) * args.st_seed * args.u / D),
        },
        "cd_mean": cd_mean,
        "cd_min": float(cd_w.min()),
        "cd_max": float(cd_w.max()),
        "cd_rms_fluctuation": float(cd_w.std()),
        "cl_amplitude": float(0.5 * (cl_w.max() - cl_w.min())),
        "st": st,
        "cd_reference": REF_CD,
        "st_reference": REF_ST,
        "cd_relative_error_pct": 100.0 * (cd_mean - REF_CD) / REF_CD,
        "st_relative_error_pct": 100.0 * (st - REF_ST) / REF_ST if math.isfinite(st) else None,
        "elapsed_s": round(elapsed, 1),
        "snapshot_npz": str(args.snapshot_out) if args.snapshot_out else None,
        "drag_lift_history": {
            "drag": [round(v, 6) for v in cd],
            "lift": [round(v, 6) for v in cl],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TensorLBM cylinder Re=100 cross-validation run")
    ap.add_argument("--diameter-cells", type=int, default=20)
    ap.add_argument("--domain-length-d", type=float, default=30.0)
    ap.add_argument("--domain-height-d", type=float, default=18.0)
    ap.add_argument("--cylinder-x-d", type=float, default=8.0)
    ap.add_argument("--sponge-d", type=float, default=8.0)
    ap.add_argument("--sponge-alpha", type=float, default=10.0)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--collision", default="MRT")
    ap.add_argument("--st-seed", type=float, default=0.165)
    ap.add_argument("--seed-periods", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup-frac", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report-interval", type=int, default=2000)
    ap.add_argument("--snapshot-out", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"[lbm D={args.diameter_cells}] steps={args.steps} Cd={result['cd_mean']:.4f} "
        f"({result['cd_relative_error_pct']:+.2f}%) St={result['st']:.4f} "
        f"elapsed={result['elapsed_s']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
