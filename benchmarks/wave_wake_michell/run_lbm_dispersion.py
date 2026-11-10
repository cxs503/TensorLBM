#!/usr/bin/env python3
"""LBM free-surface leg (2-D): wave-keeping, dispersion and mass audit.

What this measures (all from the simulation, no hard-coded expectations):

``C7-like``  relative drift of the total liquid inventory (Körner mass ledger)
             over ``n_steps`` free-surface steps.
``C6-like``  decay of the surface-elevation amplitude over the same window,
             resolved at ``lambda/dx = 32 / 64 / 128`` cells per wavelength --
             this is the quantity that decides whether a ship wake survives
             long enough to be measurable, so it is the honest 2-D surrogate for
             the campaign's grid-convergence criterion.
``dispersion``  dominant period from the probe record (autocorrelation of the
             surface elevation), compared with the linear-theory period
             ``T = 2 pi / sqrt(g k tanh(k h))`` at the same ``k``.

Why 2-D: the 3-D Körner kernel moves ~4e5 cells at ~1e1 s/step on the control
plane (measured, see the mission report) -> O(1e5 s) for one Wigley case at
``lambda/dx >= 20``.  The 2-D module has the same free-surface mechanics
(same Körner mass exchange, conversion rules and ledger) at a cost that fits a
mission budget, so it is the right place to certify the *mechanism*.

Run (one resolution per invocation, writes one JSON each)::

    PYTHONPATH=src python benchmarks/wave_wake_michell/run_lbm_dispersion.py \
        --lam 64 --nx 256 --out runs/wave_wake_michell/lbm_disp_lam64.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.free_surface_lbm_2d import (
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)

G_LAT = 6.0e-4  # lattice gravity; hydrostatic rho variation ~6% at h=32
CS2 = 1.0 / 3.0


def build(nx: int, nz: int, h: float, lam: float, amp: float, slip: bool):
    """Still water + sinusoidal surface displacement, hydrostatic density."""
    x = np.arange(nx)
    z = np.arange(nz).reshape(-1, 1)
    eta = amp * np.cos(2.0 * math.pi * x / lam)[None, :]
    fill = np.clip(h + eta - z + 0.5, 0.0, 1.0)  # 0 = gas, 1 = liquid
    fill_t = torch.tensor(fill, dtype=torch.float32)
    solid = torch.zeros(nz, nx, dtype=torch.bool)
    solid[0, :] = True  # bottom wall
    flags = init_flags_from_fill_2d(fill_t, solid)
    mass = init_mass_from_fill_2d(fill_t, flags)
    rho = np.where(fill > 0.0, 1.0 - G_LAT * (z - h) / CS2, 0.0)
    rho_t = torch.tensor(rho, dtype=torch.float32)
    zero = torch.zeros_like(rho_t)
    f = equilibrium(rho_t, zero, zero)
    f = torch.where(fill_t[None, :, :] > 0.0, f, torch.zeros_like(f))
    slip_mask = torch.zeros(nz, nx, dtype=torch.bool)
    if slip:
        slip_mask[:, 0] = True
        slip_mask[:, -1] = True
    return f, mass, flags, solid, slip_mask


def analyse(eta_trace: np.ndarray, t: np.ndarray, k: float, h: float, g: float) -> dict:
    """Dominant period from the probe record + linear-theory comparison."""
    s = eta_trace - np.mean(eta_trace)
    n = s.size
    ac = np.correlate(s, s, mode="full")[n - 1:]
    ac /= ac[0] if ac[0] != 0 else 1.0
    # first interior local maximum of the autocorrelation = one period
    period_meas = None
    for i in range(2, len(ac) - 2):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2:
            period_meas = float(t[i] - t[0])
            break
    omega_theory = math.sqrt(g * k * math.tanh(k * h))
    period_theory = 2.0 * math.pi / omega_theory
    return {
        "period_measured_steps": period_meas,
        "period_theory_steps": period_theory,
        "period_error_pct": (
            None if period_meas is None else 100.0 * (period_meas - period_theory) / period_theory
        ),
        "record_length_steps": float(t[-1] - t[0]),
        "record_over_theory_period": float((t[-1] - t[0]) / period_theory),
        "autocorr_first_peak": float(np.max(ac[2:])) if ac.size > 2 else None,
    }


def main() -> int:
    global G_LAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=64.0, help="wavelength in cells")
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--h", type=float, default=32.0)
    ap.add_argument("--headroom", type=int, default=12)
    ap.add_argument("--amp", type=float, default=1.0, help="initial crest height [cells]")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.5025)
    ap.add_argument("--g", type=float, default=G_LAT, help="lattice gravity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--slip", action="store_true", help="free-slip lateral walls")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default="runs/wave_wake_michell/lbm_disp.json")
    args = ap.parse_args()

    G_LAT = float(args.g)
    torch.set_num_threads(int(args.threads))
    nz = int(args.h + args.headroom)
    f, mass, flags, solid, slip_mask = build(
        args.nx, nz, args.h, args.lam, args.amp, args.slip)
    m0 = float(total_liquid_inventory_2d(f, mass, flags))
    probe = args.nx // 2

    t_list, eta_list = [], []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        f, mass, flags = koerner_step_2d(
            f, mass, flags, solid, tau=args.tau, gy=-G_LAT,
            free_slip_mask=slip_mask if args.slip else None)
        if step % args.every == 0:
            fill = torch.where(flags > 0, mass, torch.zeros_like(mass))
            col = fill.clamp(0.0, 1.0).sum(dim=0)
            t_list.append(float(step))
            eta_list.append(float(col[probe].item()) - args.h)
        if not torch.isfinite(f).all():
            return _fail(args, step, "non-finite populations")
    wall = time.time() - t0

    eta = np.asarray(eta_list)
    t = np.asarray(t_list)
    k = 2.0 * math.pi / args.lam
    res = {
        "config": vars(args),
        "grid": {"nx": args.nx, "nz": nz},
        "lambda_over_dx": float(args.lam),
        "mass_initial": m0,
        "mass_final": float(total_liquid_inventory_2d(f, mass, flags)),
        "mass_relative_drift": float(
            (total_liquid_inventory_2d(f, mass, flags) - m0) / m0),
        "eta_probe_initial": float(args.amp),
        "eta_probe_max": float(np.max(eta)),
        "eta_probe_min": float(np.min(eta)),
        "amplitude_ratio_final_over_initial": float(
            np.max(np.abs(eta[t > 0.75 * t[-1]])) / max(args.amp, 1e-12)),
        "dispersion": analyse(eta, t, k, args.h, G_LAT),
        "wall_clock_s": wall,
        "s_per_step": wall / max(args.steps, 1),
        "eta_trace": eta.tolist(),
        "t": t.tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({kk: res[kk] for kk in (
        "lambda_over_dx", "mass_relative_drift", "eta_probe_max", "eta_probe_min",
        "amplitude_ratio_final_over_initial", "dispersion", "s_per_step")}, indent=2))
    print(f"[done] -> {out}")
    return 0


def _fail(args, step: int, why: str) -> int:
    print(json.dumps({"status": "failed", "step": step, "reason": why,
                      "config": vars(args)}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())