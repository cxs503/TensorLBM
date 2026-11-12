"""Unconfined 2-D cylinder drag at Re=100 — Schäfer & Turek (1996) anchor.

Target (CASE-CYLINDER-DRAG-RE100, mission m15432)
--------------------------------------------------
The canonical "cylinder drag Re=100" benchmark is the *unconfined* 2-D cylinder.
Schäfer, Rannacher & Turek (1996), "Flow control by means of fictitious bodies",
report for the 2-D cylinder at Re=100:

    Cd = 1.3796   (time-averaged drag coefficient)
    Cl = 0.0005   (time-averaged lift coefficient, ~0 by symmetry)
    St = 0.164    (Strouhal number)

This module runs the repo's own D2Q9 BGK solver on a large box with the
``far_field_bc_2d`` boundary set (free-stream equilibrium on the inlet and both
lateral faces, zero-gradient outlet, halfway bounce-back on the cylinder, Ladd
momentum-exchange force).  Using free-stream lateral faces (instead of solid
channel walls) removes the blockage that made the channel-BC runs drift, so the
body sees effectively unbounded flow and the result can be compared directly
with the unconfined anchor.

Why a new module (not the channel harness)
------------------------------------------
``tests/validation/cylinder_drag_benchmark.py`` targets the *confined* DFG 2D-2
geometry with solid channel walls and a plug inlet; its channel boundary set was
shown (docs/cylinder_drag_benchmark_r1.md) to carry a momentum imbalance that
prevents a quotable Cd.  The unconfined case needs open lateral faces, which is
exactly what ``far_field_bc_2d`` provides.  This module reuses the same solver
primitives (``collide_bgk_matmul`` / ``stream`` / ``bounce_back_cells`` /
``compute_obstacle_forces``) so the evidence stays on the same verified kernel.

Usage
-----
    python -m tests.validation.cylinder_re100_unconfined --pilot
    python -m tests.validation.cylinder_re100_unconfined --full --emit-dir artifacts/cylinder_re100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from tensorlbm.boundary.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
)
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk_matmul, stream

# Schäfer, Rannacher & Turek (1996), 2-D cylinder, Re = 100 (unconfined).
REF = {
    "cd": 1.3796,
    "cl": 0.0005,
    "st": 0.164,
    "source": (
        "Schäfer, Rannacher, Turek (1996), 'Flow control by means of fictitious "
        "bodies', in Numerical Methods in Fluid Dynamics, Springer — 2-D cylinder "
        "Re=100: Cd=1.3796, Cl=0.0005, St=0.164"
    ),
}


@dataclass
class Config:
    name: str
    re: float = 100.0
    diameter: int = 20
    upstream_d: float = 5.0
    downstream_d: float = 20.0
    height_d: float = 10.0  # transverse extent in diameters (open lateral faces)
    u_in: float = 0.1
    min_warmup_steps: int = 8000
    stats_steps: int = 18000
    max_steps: int = 30000
    sponge_cols: int = 0  # >0: relax last N columns toward free-stream equilibrium
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class Measurement:
    config: dict
    derived: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _period_steps(cl_hist):
    """Shedding period in lattice steps from interpolated up-crossings of Cl-mean."""
    if len(cl_hist) < 64:
        return float("nan")
    mean = _mean(cl_hist)
    sig = (_mean([(c - mean) ** 2 for c in cl_hist])) ** 0.5
    if sig < 1e-9:
        return float("nan")
    crossings = []
    prev = cl_hist[0] - mean
    for i in range(1, len(cl_hist)):
        cur = cl_hist[i] - mean
        if prev < 0.0 <= cur:
            crossings.append(i - 1 + (-prev / (cur - prev)))
        prev = cur
    if len(crossings) < 3:
        return float("nan")
    return (crossings[-1] - crossings[0]) / (len(crossings) - 1)


def _sponge_outlet(f: torch.Tensor, u_in: float, n_cols: int) -> torch.Tensor:
    """Quadratic sponge: relax the last n_cols columns toward free-stream equilibrium.

    Absorbs the wake before it can reflect off the zero-gradient outlet, which
    is the momentum-imbalance source identified in docs/cylinder_drag_benchmark_r1.md.
    """
    nx = f.shape[2]
    feq = equilibrium(
        torch.ones(f.shape[1], f.shape[2], dtype=f.dtype, device=f.device),
        torch.full((f.shape[1], f.shape[2]), u_in, dtype=f.dtype, device=f.device),
        torch.zeros(f.shape[1], f.shape[2], dtype=f.dtype, device=f.device),
    )
    f = f.clone()
    for j in range(n_cols):
        col = nx - 1 - j
        beta = ((j + 1) / n_cols) ** 2
        f[:, :, col] = (1.0 - beta) * f[:, :, col] + beta * feq[:, :, col]
    return f


def run_cfg(cfg: Config, verbose: bool = True) -> Measurement:
    dev = torch.device(cfg.device)
    dt = getattr(torch, cfg.dtype)
    torch.set_num_threads(1)  # small tensors: thread pool is pure overhead (see r1 report)

    radius = cfg.diameter / 2.0
    ny = int(round(cfg.height_d * cfg.diameter))
    nx = int(round((cfg.upstream_d + cfg.downstream_d) * cfg.diameter))
    cx = int(round(cfg.upstream_d * cfg.diameter))
    cy = ny // 2
    nu = cfg.u_in * cfg.diameter / cfg.re
    tau = 3.0 * nu + 0.5

    mask = cylinder_mask(nx, ny, cx, cy, radius, device=dev)
    rho = torch.ones(ny, nx, device=dev, dtype=dt)
    ux = torch.full_like(rho, cfg.u_in)
    f = equilibrium(rho, ux, torch.zeros_like(rho), device=dev).to(dt)
    q = 0.5 * cfg.u_in**2 * cfg.diameter  # dynamic pressure x projected area (rho=1)

    cd_hist: list[float] = []
    cl_hist: list[float] = []
    t0 = time.time()
    stop_reason = "max_steps"
    drift = float("nan")
    with torch.inference_mode():
        for step in range(1, cfg.max_steps + 1):
            f = collide_bgk_matmul(f, tau=tau)
            f = stream(f)
            fx, fy = compute_obstacle_forces(f, mask)
            f = far_field_bc_2d(f, cfg.u_in, mask)
            if cfg.sponge_cols > 0:
                f = _sponge_outlet(f, cfg.u_in, cfg.sponge_cols)
            if step > cfg.min_warmup_steps:
                cd_hist.append(float(fx.item()) / q)
                cl_hist.append(float(fy.item()) / q)
            if verbose and step % 2000 == 0:
                cd = cd_hist[-1] if cd_hist else float("nan")
                print(f"  [{cfg.name}] {step}/{cfg.max_steps} Cd~{cd:.4f} "
                      f"samples={len(cd_hist)} ({time.time() - t0:.0f}s)", flush=True)
            if len(cd_hist) >= cfg.stats_steps and len(cd_hist) % 2000 == 0:
                half = len(cd_hist) // 2
                drift = 100.0 * abs(_mean(cd_hist[half:]) - _mean(cd_hist[:half])) / _mean(cd_hist)
                if drift < 1.0:
                    stop_reason = "stationary"
                    break
    wall_s = time.time() - t0
    rho_f, ux_f, uy_f = macroscopic(f)
    # vorticity (2D): wz = dux/dy - duy/dx, central differences
    wz = torch.zeros_like(ux_f)
    wz[1:-1, 1:-1] = (ux_f[2:, 1:-1] - ux_f[:-2, 1:-1]) / 2.0 - (uy_f[1:-1, 2:] - uy_f[1:-1, :-2]) / 2.0
    # wake centerline velocity profile (y = cy) for comparison
    prof = ux_f[cy, :].detach().cpu().tolist()
    period_steps = _period_steps(cl_hist)
    st = (cfg.diameter / (period_steps * cfg.u_in)) if period_steps == period_steps else float("nan")
    half = len(cd_hist) // 2
    return Measurement(
        config=asdict(cfg),
        derived={
            "nx": nx, "ny": ny, "cells": nx * ny,
            "nu": nu, "tau": tau,
            "mach": cfg.u_in / (1.0 / 3.0) ** 0.5,
            "blockage": cfg.diameter / ny,
            "cd_mean": _mean(cd_hist),
            "cl_mean": _mean(cl_hist),
            "cd_half_drift_pct": 100.0 * abs(_mean(cd_hist[half:]) - _mean(cd_hist[:half])) / _mean(cd_hist) if cd_hist else float("nan"),
            "cd_amp": 0.5 * (max(cd_hist) - min(cd_hist)) if cd_hist else float("nan"),
            "cl_amp": 0.5 * (max(cl_hist) - min(cl_hist)) if cl_hist else float("nan"),
            "strouhal": st,
            "period_steps": period_steps,
            "samples": len(cd_hist),
            "wall_seconds": wall_s,
            "stop_reason": stop_reason,
            "stationarity_drift_pct": drift,
        },
        samples={
            "cd_tail": cd_hist,
            "cl_tail": cl_hist,
            "ux": ux_f.detach().cpu().tolist(),
            "uy": uy_f.detach().cpu().tolist(),
            "wz": wz.detach().cpu().tolist(),
            "centerline_ux": prof,
            "cx": cx,
            "cy": cy,
        },
    )


def default_configs(device: str = "cpu") -> list[Config]:
    return [
        Config(name="unconfined_D20", diameter=20, upstream_d=5.0, downstream_d=20.0,
               height_d=10.0, u_in=0.1, min_warmup_steps=8000, stats_steps=18000,
               max_steps=30000, device=device),
    ]


def build_registry(measurements: list[Measurement]) -> dict:
    reg = {
        "schema": "tensorlbm-cylinder-re100-unconfined-v1",
        "generated_by": "tests/validation/cylinder_re100_unconfined.py",
        "reference": REF,
        "runs": [{"config": m.config, "derived": m.derived} for m in measurements],
        "comparisons": [],
    }
    for m in measurements:
        for key, ref_key in (("cd_mean", "cd"), ("cl_mean", "cl"), ("strouhal", "st")):
            val = m.derived[key]
            if val != val:
                continue
            ref = REF[ref_key]
            reg["comparisons"].append({
                "quantity": f"{ref_key.upper()} ({m.config['name']})",
                "measured": val,
                "reference": ref,
                "rel_err_pct": 100.0 * (val - ref) / ref if ref != 0 else float("nan"),
                "source": REF["source"],
            })
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="short run for smoke test")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--emit-dir", default=None)
    ap.add_argument("--diameter", type=int, default=None, help="override cylinder diameter (cells)")
    ap.add_argument("--name", default=None, help="override run name")
    ap.add_argument("--height-d", type=float, default=None, help="override transverse extent (diameters)")
    ap.add_argument("--u-in", type=float, default=None, help="override lattice inlet velocity")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--sponge-cols", type=int, default=None)
    args = ap.parse_args()

    device = "cpu"
    cfgs = default_configs(device)
    if args.diameter is not None or args.name is not None or args.height_d is not None or args.u_in is not None or args.max_steps is not None:
        cfgs = [cfgs[0]]
        for c in cfgs:
            if args.diameter is not None:
                c.diameter = args.diameter
            if args.name is not None:
                c.name = args.name
            if args.height_d is not None:
                c.height_d = args.height_d
            if args.u_in is not None:
                c.u_in = args.u_in
            if args.max_steps is not None:
                c.max_steps = args.max_steps
            if args.sponge_cols is not None:
                c.sponge_cols = args.sponge_cols
    if args.pilot:
        for c in cfgs:
            c.diameter = 10
            c.upstream_d = 3.0
            c.downstream_d = 8.0
            c.height_d = 6.0
            c.min_warmup_steps = 1500
            c.stats_steps = 1500
            c.max_steps = 4000

    measurements = []
    for cfg in cfgs:
        m = run_cfg(cfg)
        measurements.append(m)

    reg = build_registry(measurements)
    print("\n=== comparisons ===")
    for c in reg["comparisons"]:
        print(f"  {c['quantity']}: measured={c['measured']:.4f} ref={c['reference']} "
              f"err={c['rel_err_pct']:+.2f}%")

    if args.emit_dir:
        os.makedirs(args.emit_dir, exist_ok=True)
        out = {k: v for k, v in reg.items()}
        out["samples"] = [m.samples for m in measurements]
        with open(os.path.join(args.emit_dir, "cylinder_re100_unconfined.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"emitted {args.emit_dir}/cylinder_re100_unconfined.json")


if __name__ == "__main__":
    main()
