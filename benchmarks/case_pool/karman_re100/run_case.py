#!/usr/bin/env python3
"""CASE-KARMAN-RE100 — free-stream circular cylinder, Kármán vortex street, Re = U·D/ν = 100.

Pool entry (benchmarks/case_pool/karman_re100).  Same solver chain as the
repository-verified case ``benchmarks/verified/cylinder/re100`` (library
primitives only, zero hand-written collide/stream/equilibrium):

    collide_mrt(tau_field)  →  stream  →  compute_obstacle_forces (Ladd MEM)
    →  far_field_bc_2d(inlet/lateral Dirichlet, zero-gradient outlet, obstacle
       half-way bounce-back)

with a downstream viscous sponge ``tau_field = tau·(1 + α·σ)``
(``make_sponge_strength``) and a two-period transverse-velocity seeding at the
inlet so that shedding starts deterministically instead of from round-off.

Reduction (identical estimator to the verified case):
  * Cd = mean(Fx) / (½ ρ U² D) over the second half of the record,
  * St = spectral peak of the lift signal (Hann window + parabolic peak
    interpolation), cross-checked by hysteretic zero crossings.

Usage::

    PYTHONPATH=src python benchmarks/case_pool/karman_re100/run_case.py \
        --D 20 --domain 24x20 --u-in 0.1 --steps 14000 --out runs/karman_re100
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
from tensorlbm.solver import collide_mrt, stream

SPONGE_D = 6.0  # absorbing-layer width, in diameters
SPONGE_ALPHA = 10.0  # tau_eff = tau (1 + alpha sigma)
ST_SEED = 0.165  # seeding frequency
SEED_AMPL = 0.10  # transverse velocity amplitude / U
SEED_PERIODS = 2.0


def strouhal_from_lift(lift: np.ndarray, dt: float, diameter: float, u_inf: float):
    """Spectral Strouhal number (Hann window + parabolic peak) and zero-crossing check."""
    signal = lift - lift.mean()
    if signal.size < 256:
        return float("nan"), float("nan")
    spec = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    k = int(np.argmax(spec[1:])) + 1
    delta = 0.0
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        den = a - 2.0 * b + c
        if abs(den) > 1e-30:
            delta = float(0.5 * (a - c) / den)
    st = float((k + delta) / signal.size * dt * diameter / u_inf)
    # hysteretic crossings for an independent period estimate
    thr = 0.25 * float(np.abs(signal).max())
    sig = np.where(signal >= thr, 1, np.where(signal <= -thr, -1, 0))
    idx = np.where(sig != 0, np.arange(sig.size), -1)
    last = np.maximum.accumulate(idx)
    state = np.where(last >= 0, sig[np.maximum(last, 0)], 0)
    cross = np.where((state[:-1] < 0) & (state[1:] > 0))[0] + 1
    st_cross = float("nan")
    if len(cross) >= 3:
        st_cross = float(diameter / (u_inf * float(np.median(np.diff(cross)))))
    return st, st_cross


def run(D: int, ld: float, hd: float, u_in: float, re: float, steps: int,
        warmup_frac: float, out_dir: Path, nthreads: int = 16, snapshot: bool = True):
    torch.set_num_threads(nthreads)
    device = torch.device("cpu")
    nx, ny = int(ld * D), int(hd * D)
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu
    x_cyl, y_cyl = 8.0 * D, ny / 2.0

    mask = cylinder_mask(nx, ny, x_cyl, y_cyl, D / 2.0, device)
    sigma = make_sponge_strength(ny, nx, int(nx - SPONGE_D * D), int(SPONGE_D * D),
                                 power=2.0, device=device)
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    omega_seed = 2.0 * math.pi * ST_SEED * u_in / D
    seed_steps = int(round(SEED_PERIODS * D / (ST_SEED * u_in)))
    uy_amp = SEED_AMPL * u_in

    fx_hist, fy_hist = [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        f = collide_mrt(f, tau, tau_field=tau_field)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)  # post-stream, pre-bounce-back
        f = far_field_bc_2d(f, u_in, mask)
        if step <= seed_steps:
            uy_val = uy_amp * math.sin(omega_seed * step)
            uy_col = torch.full((ny, 1), uy_val, device=device, dtype=f.dtype)
            f[:, :, 0] = equilibrium(
                torch.ones((ny, 1), device=device, dtype=f.dtype),
                torch.full((ny, 1), u_in, device=device, dtype=f.dtype),
                uy_col,
            )[:, :, 0]
        fx_hist.append(fx.item())
        fy_hist.append(fy.item())
    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"D={D}: non-finite populations after {steps} steps")

    fx_np = np.asarray(fx_hist, dtype=np.float64)
    fy_np = np.asarray(fy_hist, dtype=np.float64)
    q_dyn = 0.5 * u_in * u_in * D  # rho = 1, D in lattice units
    w0 = int(warmup_frac * steps)
    cd_series, cl_series = fx_np / q_dyn, fy_np / q_dyn
    cd_mean = float(cd_series[w0:].mean())
    cl_amp = float(np.abs(cl_series[w0:] - cl_series[w0:].mean()).max())
    st, st_cross = strouhal_from_lift(cl_series[w0:], 1.0, D, u_in)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"series_D{D}.npz", cd=cd_series, cl=cl_series, analyze_from=w0)

    if snapshot:
        # vorticity and speed from the final field (post-processing only)
        rho = f.sum(0)
        mx = (f * torch.tensor([0., 1, 0, -1, 0, 1, -1, -1, 1]).view(9, 1, 1)).sum(0)
        my = (f * torch.tensor([0., 0, 1, 0, -1, 1, 1, -1, -1]).view(9, 1, 1)).sum(0)
        ux = (mx / rho).numpy()
        uy = (my / rho).numpy()
        uy[0, :] = u_in  # inlet column is the seeded equilibrium
        dvdx = np.gradient(uy, axis=1)
        dudy = np.gradient(ux, axis=0)
        np.savez_compressed(out_dir / f"field_D{D}.npz", ux=ux, uy=uy, vort=dvdx - dudy,
                            x_cyl=x_cyl, y_cyl=y_cyl, D=D, u_in=u_in)

    res = {
        "case": "CASE-KARMAN-RE100",
        "D": D, "nx": nx, "ny": ny, "steps": steps,
        "domain_D": [ld, hd], "blockage": D / (hd * D),
        "Re": re, "u_in": u_in, "nu_lb": nu, "tau": round(tau, 6),
        "ma": u_in / math.sqrt(1.0 / 3.0),
        "sponge": {"x0": int(nx - SPONGE_D * D), "width": int(SPONGE_D * D),
                   "alpha": SPONGE_ALPHA, "power": 2.0},
        "seed": {"st_seed": ST_SEED, "ampl_frac": SEED_AMPL, "periods": SEED_PERIODS,
                 "steps": seed_steps},
        "warmup_frac": warmup_frac, "analyze_from": w0,
        "cd": round(cd_mean, 4), "cl_amp": round(cl_amp, 4),
        "st": round(st, 4) if math.isfinite(st) else None,
        "st_crossing": round(st_cross, 4) if math.isfinite(st_cross) else None,
        "steps_per_shedding_period": round(D / (ST_SEED * u_in), 1),
        "finite": True, "elapsed_s": round(elapsed, 1),
    }
    (out_dir / f"result_D{D}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="Kármán vortex street, circular cylinder, Re=100")
    ap.add_argument("--D", type=int, default=20, help="cylinder diameter in lattice cells")
    ap.add_argument("--domain", default="24x20", help="domain size in diameters LxH")
    ap.add_argument("--u-in", type=float, default=0.1)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="runs/karman_re100")
    args = ap.parse_args()
    ld, hd = (float(v) for v in args.domain.lower().split("x"))
    run(args.D, ld, hd, args.u_in, args.re, args.steps, args.warmup_frac,
        Path(args.out), nthreads=args.threads)


if __name__ == "__main__":
    main()