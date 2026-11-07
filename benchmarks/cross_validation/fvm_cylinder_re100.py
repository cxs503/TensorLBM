#!/usr/bin/env python3
"""Unsteady FVM (SIMPLE) cylinder at Re=100 — cross-validation partner for TensorLBM.

Runs TensorFVM's body-fitted O-grid finite-volume solver in transient mode
(``time_step`` + ``inner_iterations``) on the same physical case as
``benchmarks/verified/cylinder/re100/run.py`` (free-stream cylinder, Re=100)
and reduces the force history to cycle-averaged Cd and the Strouhal number.

TensorFVM is a separate repository; point ``TENSORFVM_SRC`` at its ``src``
directory (or have it importable) — nothing in TensorFVM is modified here.

Usage::

    TENSORFVM_SRC=/path/to/TensorFVM/src python3 \
        benchmarks/cross_validation/fvm_cylinder_re100.py \
        --nx 48 --ny 24 --steps 1600 --out artifacts/fvm_re100_48x24.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_SRC = os.environ.get("TENSORFVM_SRC")
if _SRC and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorfvm.solver import SimpleSolver, SolverConfig  # noqa: E402

REF_CD = 1.35  # Henderson (1995) doi:10.1063/1.868459, Table I fits at Re=100
REF_ST = 0.1647


def strouhal_from_lift(lift: np.ndarray, dt: float, diameter: float, u_inf: float):
    """Spectral Strouhal number with parabolic peak interpolation."""
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
    cfg = SolverConfig(
        mesh_type="body-fitted",
        outer_boundary="far-field",
        body_fitted_stretching=args.stretch,
        nx=args.nx,
        ny=args.ny,
        length=args.length,
        height=args.height,
        cylinder_x=args.cx,
        cylinder_y=args.cy,
        cylinder_radius=args.radius,
        inlet_velocity=args.u,
        reynolds=args.re,
        density=1.0,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
        turbulence_model="laminar",
        time_step=args.dt,
        inner_iterations=args.inner,
        initial_perturbation=args.perturbation,
        tolerance=args.tol,
        max_iterations=args.steps,
    )
    solver = SimpleSolver(cfg)
    if args.seed_dipole:
        # Localised counter-rotating vortex pair behind the cylinder: a strong
        # antisymmetric seed that cannot decay before the (unstable) shedding
        # mode amplifies.  The built-in ``initial_perturbation`` global sinusoid
        # was observed to be damped out and the solver then locks onto the
        # (linearly unstable) symmetric steady branch, see docs report.
        import torch as _torch

        centres = solver.mesh.centers.reshape(-1, 2)
        x, y = centres[:, 0], centres[:, 1]
        D = 2.0 * args.radius
        sigma = 0.5 * D
        amp = args.seed_dipole * args.u
        for dy, sign in ((0.6 * D, 1.0), (-0.6 * D, -1.0)):
            g = _torch.exp(-((x - (args.cx + 1.2 * D)) ** 2 + (y - (args.cy + dy)) ** 2) / (2 * sigma ** 2))
            solver.velocity[:, 1] = solver.velocity[:, 1] + sign * amp * g
        print(f"  seeded dipole amplitude={amp:.3f} behind cylinder", flush=True)
    if args.seed_shear:
        # Antisymmetric streamwise shear pair straddling the cylinder: the
        # discrete analogue of a slightly asymmetric inflow, which is the
        # perturbation that actually amplifies at Re=100 (a purely transverse
        # vortex-pair seed is annihilated by the first pressure projection).
        import torch as _torch

        centres = solver.mesh.centers.reshape(-1, 2)
        x, y = centres[:, 0], centres[:, 1]
        D = 2.0 * args.radius
        amp = args.seed_shear * args.u
        lobe = _torch.exp(-((x - args.cx) ** 2 + (y - (args.cy + 1.3 * D)) ** 2) / (D * D))
        lobe = lobe - _torch.exp(-((x - args.cx) ** 2 + (y - (args.cy - 1.3 * D)) ** 2) / (D * D))
        solver.velocity[:, 0] = solver.velocity[:, 0] + amp * lobe
        print(f"  seeded antisymmetric shear amplitude={amp:.3f}", flush=True)
    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        solver.step()
        history = solver.force_history
        if args.report_interval and step % args.report_interval == 0:
            np.savez(
                args.out.with_suffix(".partial.npz"),
                drag=np.array([h["drag"] for h in history], dtype=np.float64),
                lift=np.array([h["lift"] for h in history], dtype=np.float64),
            )
            print(
                f"  step {step}/{args.steps} Cd={np.mean([h['drag'] for h in history[-200:]]):.4f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    elapsed = time.time() - t0
    converged = bool(solver.converged)

    if len(history) < 32:
        raise RuntimeError(f"only {len(history)} force samples; increase --steps")
    drag = np.array([h["drag"] for h in history], dtype=np.float64)
    lift = np.array([h["lift"] for h in history], dtype=np.float64)
    warm = int(args.warmup_frac * drag.size)
    drag_w, lift_w = drag[warm:], lift[warm:]
    diameter = 2.0 * args.radius
    st = strouhal_from_lift(lift_w, args.dt, diameter, args.u)
    cd = float(drag_w.mean())
    summary = {
        "solver": "TensorFVM body-fitted SIMPLE (transient)",
        "case": "free-stream circular cylinder Re=100 (cross-validation with TensorLBM)",
        "mesh": {
            "type": "body-fitted O-grid",
            "nx": args.nx,
            "ny": args.ny,
            "cells": int(args.nx * args.ny),
            "domain_length_D": args.length / diameter,
            "domain_height_D": args.height / diameter,
            "cylinder_center": [args.cx, args.cy],
            "cylinder_radius": args.radius,
            "diameter": diameter,
            "stretching_to_cylinder": args.stretch,
            "blockage_ratio_D_over_H": diameter / args.height,
        },
        "numerics": {
            "time_step": args.dt,
            "inner_iterations": args.inner,
            "velocity_relaxation": 0.7,
            "pressure_relaxation": 0.3,
            "tolerance": args.tol,
            "reynolds": args.re,
            "inlet_velocity": args.u,
            "kinematic_viscosity": args.u * diameter / args.re,
            "initial_perturbation": args.perturbation,
        },
        "sampling": {
            "time_steps": int(drag.size),
            "physical_time": float(drag.size * args.dt),
            "warmup_fraction": args.warmup_frac,
            "averaging_window_steps": int(drag.size - warm),
            "averaging_window_periods": float(
                (drag.size - warm) * args.dt * st / 1.0
            ) if math.isfinite(st) else None,
        },
        "cd_mean": cd,
        "cd_min": float(drag_w.min()),
        "cd_max": float(drag_w.max()),
        "cd_rms_fluctuation": float(drag_w.std()),
        "cl_amplitude": float(0.5 * (lift_w.max() - lift_w.min())),
        "st": st,
        "cd_reference": REF_CD,
        "st_reference": REF_ST,
        "cd_relative_error_pct": 100.0 * (cd - REF_CD) / REF_CD,
        "st_relative_error_pct": 100.0 * (st - REF_ST) / REF_ST if math.isfinite(st) else None,
        "converged": converged,
        "elapsed_s": round(elapsed, 1),
        "drag_lift_history": {
            "drag": [round(float(v), 6) for v in drag],
            "lift": [round(float(v), 6) for v in lift],
        },
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TensorFVM cylinder Re=100 cross-validation run")
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--stretch", type=float, default=7.0)
    ap.add_argument("--length", type=float, default=30.0)
    ap.add_argument("--height", type=float, default=20.0)
    ap.add_argument("--cx", type=float, default=10.0)
    ap.add_argument("--cy", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--u", type=float, default=1.0)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--perturbation", type=float, default=0.02)
    ap.add_argument("--seed-dipole", type=float, default=0.0,
                    help="amplitude (fraction of U) of a localised vortex-pair seed")
    ap.add_argument("--seed-shear", type=float, default=0.0,
                    help="amplitude (fraction of U) of an antisymmetric streamwise shear seed")
    ap.add_argument("--warmup-frac", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--report-interval", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    summary = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"[fvm {args.nx}x{args.ny}] steps={summary['sampling']['time_steps']} "
        f"Cd={summary['cd_mean']:.4f} ({summary['cd_relative_error_pct']:+.2f}%) "
        f"St={summary['st']:.4f} elapsed={summary['elapsed_s']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
