#!/usr/bin/env python3
"""Reference audit for the 3D Taylor-Green vortex used by XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED.

The round-1 (mission 18720) campaign compared the decomposed LBM field against a
reference that decayed as ``exp(-2 nu k^2 t)``.  This script decides, *without*
using that reference, which exponent the LBM solution actually follows:

E1  spectral eigen-check: for the discrete mode u = sin(kx)cos(ky)cos(kz) the
    Laplacian eigenvalue is -3k^2 (2D TGV would be -2k^2).
E2  effective decay rate: fit log(amplitude) ~ t from the solver output only.
    Two amplitudes (u0 and u0/10) separate the linear viscous rate from the
    O(Ma^2) nonlinear correction.
E3  rel-L2 of the same field against both candidate references.

Predeclared decision rule (fixed before the numbers were seen):
  * the exponent 3 nu k^2 is accepted when |lambda_fit - 3 nu k^2| / (3 nu k^2)
    < 0.08 and shrinks with resolution, while |lambda_fit - 2 nu k^2| / (2 nu k^2)
    stays > 0.20 at every resolution and every amplitude;
  * a residual that scales with u0^2 is attributed to the nonlinear TGV
    harmonics, a residual independent of u0^2 to numerics.

Usage:
  PYTHONPATH=src python scripts/xval_tgv_reference_audit.py --n 16 32 --steps 400 \
      --out xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/audit/reference_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

from tensorlbm.xval_mpi_ghost_fused.campaign import stream_periodic
from tensorlbm.xval_mpi_ghost_fused.case import TGV, macroscopics


# ---------------------------------------------------------------------------
# E1 -- discrete Laplacian eigenvalue of the analytic mode
# ---------------------------------------------------------------------------
def e1_laplacian_eigenvalue(n: int) -> dict:
    """3-point central difference of the analytic field on the n^3 periodic grid.

    For a pure Fourier mode the central difference is exact, so the ratio
    Lap(u)/u must be -k^2 per axis and therefore -3k^2 in total.
    """
    x = torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)
    S = torch.sin(x)
    C = torch.cos(x)
    u = S.view(1, 1, -1) * C.view(1, -1, 1) * C.view(-1, 1, 1)  # sin(kx)cos(ky)cos(kz)
    d2 = torch.zeros_like(u)
    for axis in range(3):
        d2 += torch.roll(u, 1, axis) - 2.0 * u + torch.roll(u, -1, axis)
    mask = u.abs() > 1e-8
    ratio = (d2[mask] / u[mask])
    k = 2.0 * math.pi / n
    return {
        "n": n,
        "k": k,
        "lap_over_u_mean": float(ratio.mean()),
        "lap_over_u_spread": float(ratio.max() - ratio.min()),
        "minus_2k2": float(-2.0 * k * k),
        "minus_3k2": float(-3.0 * k * k),
        "rel_err_vs_3k2": float((ratio.mean().item() + 3.0 * k * k) / (3.0 * k * k)),
        "rel_err_vs_2k2": float((ratio.mean().item() + 2.0 * k * k) / (2.0 * k * k)),
    }


# ---------------------------------------------------------------------------
# mode projection (no reference solution involved)
# ---------------------------------------------------------------------------
def mode_amplitude(f: torch.Tensor, case: TGV) -> float:
    """Project (ux,uy) onto the fundamental TGV mode; returns the amplitude U0*f(t)."""
    _, ux, uy, _ = macroscopics(f, case.dtype)
    n = case.n
    x = torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)
    Sx, Cx = torch.sin(x), torch.cos(x)
    Sf, Cf = Sx.view(1, 1, -1), Cx.view(1, 1, -1)
    Sy, Cy = Sx.view(1, -1, 1), Cx.view(1, -1, 1)
    Sz, Cz = Sx.view(-1, 1, 1), Cx.view(-1, 1, 1)
    m1 = Sf * Cy * Cz
    m2 = -Cf * Sy * Cz
    a1 = float((ux.to(torch.float64) * m1).sum()) / float((m1 * m1).sum())
    a2 = float((uy.to(torch.float64) * m2).sum()) / float((m2 * m2).sum())
    return 0.5 * (a1 + a2)


def rel_l2(f: torch.Tensor, case: TGV, t: float, exponent: float) -> float:
    """rel-L2 of (ux,uy) against the analytic mode with decay exp(-exponent nu k^2 t)."""
    _, ux, uy, _ = macroscopics(f, case.dtype)
    ax, ay, _ = case.analytic_velocity(0.0)
    scale = math.exp(-exponent * case.nu * case.k**2 * t)
    refx = (ax * scale).to(torch.float64)
    refy = (ay * scale).to(torch.float64)
    num = torch.cat([ux.flatten().to(torch.float64), uy.flatten().to(torch.float64)])
    ref = torch.cat([refx.flatten(), refy.flatten()])
    return float((num - ref).norm() / ref.norm())


def fit_rate(sample, t_lo: float):
    pts = [(t, math.log(a)) for t, a in sample if t >= t_lo and a > 0.0]
    m = len(pts)
    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mt = sum(ts) / m
    my = sum(ys) / m
    sxx = sum((t - mt) ** 2 for t in ts)
    sxy = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
    slope = sxy / sxx
    intercept = my - slope * mt
    resid = [y - (intercept + slope * t) for t, y in zip(ts, ys)]
    return -slope, intercept, math.sqrt(sum(r * r for r in resid) / max(m - 1, 1)), m


def run_arm(n: int, steps: int, u0: float, tau: float, sample_every: int) -> dict:
    torch.set_num_threads(1)
    case = TGV(n=n, u0=u0, tau=tau, dtype=torch.float64)
    f = case.initial_field()
    sample = [(0, mode_amplitude(f, case))]
    for step in range(1, steps + 1):
        f = stream_periodic(case.collide(f))
        if step % sample_every == 0 or step == steps:
            sample.append((step, mode_amplitude(f, case)))
    lam, icpt, rms, m = fit_rate(sample, 0.25 * steps)
    k, nu = case.k, case.nu
    amp0 = math.exp(icpt)
    return {
        "n": n,
        "u0": u0,
        "ma": u0 * math.sqrt(3.0),
        "tau": tau,
        "steps": steps,
        "nu": nu,
        "k": k,
        "lambda_2nuk2": 2.0 * nu * k * k,
        "lambda_3nuk2": 3.0 * nu * k * k,
        "lambda_fit": lam,
        "fit_rms_log": rms,
        "fit_points": m,
        "rel_dev_vs_2nuk2": (lam - 2.0 * nu * k * k) / (2.0 * nu * k * k),
        "rel_dev_vs_3nuk2": (lam - 3.0 * nu * k * k) / (3.0 * nu * k * k),
        "amp0_over_u0_fit": amp0 / u0,
        "rel_l2_final_vs_2nuk2": rel_l2(f, case, steps, 2.0),
        "rel_l2_final_vs_3nuk2": rel_l2(f, case, steps, 3.0),
        "rel_l2_mid_vs_2nuk2": rel_l2(f, case, steps, 2.0),
        "decay_at_final_2nuk2": math.exp(-2.0 * nu * k * k * steps),
        "decay_at_final_3nuk2": math.exp(-3.0 * nu * k * k * steps),
        "amp_final_measured": sample[-1][1],
        "sample": sample,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--u0", type=float, default=0.04)
    ap.add_argument("--tau", type=float, default=0.65)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = {
        "case": "3D Taylor-Green vortex, D3Q19 BGK, periodic, u = U0 sin(kx)cos(ky)cos(kz)",
        "predeclared_rule": (
            "accept 3 nu k^2 if |dev| < 8% and shrinking with n and |dev vs 2 nu k^2| > 20% always; "
            "u0^2-scaling residual = nonlinear harmonics, u0-independent residual = numerics"
        ),
        "E1_discrete_laplacian": [e1_laplacian_eigenvalue(n) for n in args.n],
        "E2_E3_arms": [],
    }
    for n in args.n:
        for u0 in (args.u0, args.u0 / 10.0):
            rec = run_arm(n, args.steps, u0, args.tau, args.sample_every)
            out["E2_E3_arms"].append(rec)
            print(
                f"n={n} u0={u0:g} lambda_fit={rec['lambda_fit']:.6f} "
                f"2nuk2={rec['lambda_2nuk2']:.6f} 3nuk2={rec['lambda_3nuk2']:.6f} "
                f"dev2={100*rec['rel_dev_vs_2nuk2']:+.1f}% dev3={100*rec['rel_dev_vs_3nuk2']:+.2f}% "
                f"relL2(2nuk2)={rec['rel_l2_final_vs_2nuk2']:.4f} relL2(3nuk2)={rec['rel_l2_final_vs_3nuk2']:.4f}",
                flush=True,
            )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())