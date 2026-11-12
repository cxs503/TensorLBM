#!/usr/bin/env python3
"""CASE-BACKWARD-STEP-LBM - laminar backward-facing-step reattachment length (Armaly et al. 1983).

Solver = TensorLBM library kernels only (d2q9.equilibrium/macroscopic, solver.collide_*/stream,
boundaries.zou_he_inlet_velocity / zou_he_outlet_pressure / make_sponge_strength).
Scalar loop = half-way bounce-back (pre-stream), Zou/He parabolic velocity inlet,
Zou/He pressure outlet, non-equilibrium sponge at the outflow (acoustic damping).

Caliber
    geometry  step height S = m cells, inlet height = m, outlet height H = 2m  (ER = 2)
              (Armaly et al. 1983 measured ER = 1.9423 -> documented deviation)
    inlet     fully developed parabola, row mean = u_in
    Re        Re = u_in * H / nu,  H = 2m = 2*S = D_h of the inlet channel  -> Armaly caliber
    observable X1/S : lower-wall reattachment, - -> + zero crossing of the near-wall
              least-squares shear slope, measured on a time-averaged window.

Usage: run.py --re 100 --m 16 --steps 60000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.boundaries import (  # noqa: E402
    make_sponge_strength,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_rlbm, stream  # noqa: E402

X_STEP_UPSTREAM = 8  # cells upstream of the step face


def build_solid(ny: int, nx: int, m: int, x_step: int) -> torch.Tensor:
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[-1, :] = True                  # top wall
    solid[0, x_step:] = True             # bottom wall downstream of the step
    solid[0 : m + 1, :x_step] = True     # step block / upstream lower wall
    return solid


def inlet_profile(ny: int, m: int, u_in: float) -> torch.Tensor:
    a = float(m)
    y = np.arange(ny, dtype=np.float64)
    s = y - (m + 0.5)
    rows = np.arange(m + 1, ny - 1)
    raw = 6.0 * s[rows] * (a - s[rows]) / (a * a)
    prof = np.zeros(ny, dtype=np.float64)
    prof[rows] = raw / raw.mean() * u_in
    return torch.tensor(prof, dtype=torch.float32)


def shear_weights(k: int) -> torch.Tensor:
    y = np.arange(k, dtype=np.float64)
    w = y - y.mean()
    return torch.tensor(w / (w * w).sum(), dtype=torch.float32)


def first_upcrossing(prof: np.ndarray, start: int) -> float:
    for c in range(start, len(prof) - 1):
        a, b = prof[c], prof[c + 1]
        if a <= 0.0 < b:
            return c + (0.0 - a) / (b - a)
    return float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--m", type=int, default=16, help="step height in cells (inlet height = m -> ER=2)")
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--avg-frac", type=float, default=0.25, help="last fraction of steps used for time average")
    ap.add_argument("--avg-stride", type=int, default=200)
    ap.add_argument("--l-over-h", type=float, default=12.0)
    ap.add_argument("--collision", default="bgk", choices=["bgk", "rlbm"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    m = args.m
    ny = 2 * m + 1
    H = ny - 1                      # = 2m
    nx = X_STEP_UPSTREAM + int(round(args.l_over_h * H))
    nu = args.u * H / args.re
    tau = 0.5 + 3.0 * nu
    if tau <= 0.5:
        raise SystemExit(f"tau={tau} <= 0.5")

    solid = build_solid(ny, nx, m, X_STEP_UPSTREAM)
    opp = OPPOSITE
    solid3 = solid.unsqueeze(0)
    u_prof = inlet_profile(ny, m, args.u)
    collide = collide_bgk if args.collision == "bgk" else collide_rlbm

    sw = int(round(3.0 * H))
    sigma = 0.3 * make_sponge_strength(ny, nx, nx - sw, sw, power=2.0,
                                       device=torch.device("cpu")).unsqueeze(0)

    rho = torch.ones((ny, nx))
    ux = torch.zeros((ny, nx))
    ux[:, :X_STEP_UPSTREAM] = u_prof.view(ny, 1)
    ux[:, X_STEP_UPSTREAM:] = args.u / 2.0
    ux[solid] = 0.0
    f = equilibrium(rho, ux, torch.zeros_like(ux))

    k = max(2, min(6, m // 2))
    rows = torch.arange(1, k + 1)
    wshear = shear_weights(k)

    n_avg_start = int(args.steps * (1.0 - args.avg_frac))
    acc = np.zeros(nx)
    n_acc = 0
    out_interval = max(2000, args.steps // 12)
    t0 = time.time()
    x1_series = []

    for step in range(1, args.steps + 1):
        fp = f
        fc = collide(f, tau=tau)
        fc = torch.where(solid3, fp[opp], fc)
        f = stream(fc)
        f = zou_he_inlet_velocity(f, u_prof, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        feq = equilibrium(*macroscopic(f))
        f = f - sigma * (f - feq)

        if step >= n_avg_start and step % args.avg_stride == 0:
            ux_now = macroscopic(f)[1].masked_fill(solid, 0.0)
            acc += (wshear[:, None] * ux_now[rows]).sum(0).numpy()
            n_acc += 1

        if step % out_interval == 0 or step == args.steps:
            ux_now = macroscopic(f)[1].masked_fill(solid, 0.0)
            inst = (wshear[:, None] * ux_now[rows]).sum(0).numpy()
            xc_inst = first_upcrossing(inst, X_STEP_UPSTREAM)
            x1_inst = (xc_inst - (X_STEP_UPSTREAM - 0.5)) / m if xc_inst == xc_inst else float("nan")
            x1_series.append({"step": step, "x1_over_s_inst": x1_inst})
            print(f"[Re={args.re:.0f} m={m} H={H} nx={nx} tau={tau:.4f}] step={step} "
                  f"X1/S(inst)={x1_inst:.3f} t={time.time()-t0:.0f}s", flush=True)

    x1_final = float("nan")
    x1_row1 = float("nan")
    if n_acc > 0:
        shear_avg = acc / n_acc
        xc = first_upcrossing(shear_avg, X_STEP_UPSTREAM)
        if xc == xc:
            x1_final = (xc - (X_STEP_UPSTREAM - 0.5)) / m
    ux_now = macroscopic(f)[1].masked_fill(solid, 0.0).numpy()
    xc1 = first_upcrossing(ux_now[1], X_STEP_UPSTREAM)
    if xc1 == xc1:
        x1_row1 = (xc1 - (X_STEP_UPSTREAM - 0.5)) / m

    rho_now = macroscopic(f)[0]
    payload = {
        "case": "CASE-BACKWARD-STEP-LBM",
        "solver": "TensorLBM library kernels (d2q9/solver/boundaries) + scalar BFS loop",
        "config": vars(args),
        "derived": {"ny": ny, "nx": nx, "H": H, "step_h_cells": m, "expansion_ratio": H / float(m),
                     "nu": nu, "tau": tau, "D_h": float(2 * m), "l_over_h": nx / H},
        "result": {
            "x1_over_s": x1_final,
            "x1_over_s_row1": x1_row1,
            "x1_over_H": x1_final / 2.0 if x1_final == x1_final else float("nan"),
            "n_averaged_samples": n_acc,
            "time_average_window_steps": args.steps - n_avg_start,
            "rho_min": float(rho_now.min()), "rho_max": float(rho_now.max()),
            "max_speed": float(torch.sqrt(sum(v * v for v in macroscopic(f)[1:])).max()),
            "series": x1_series,
        },
        "wall_time_s": time.time() - t0,
    }
    out = Path(args.out) if args.out else (REPO / "runs" / "case_backward_step_armaly" / f"re{int(args.re)}_m{m}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"RESULT Re={args.re:.0f} m={m}: X1/S={x1_final:.4f} (row1 {x1_row1:.4f}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
