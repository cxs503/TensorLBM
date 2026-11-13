#!/usr/bin/env python3
"""XVAL-LBM-BENCHMARK-METHOD-SELECTION — one arm of the method-selection matrix.

Case (fixed, predeclared): 2-D square-cylinder vortex shedding, Re = U D / nu = 200,
blockage D/H = 1/20, domain 20D x 20D, square centre at 6D, U = 0.05 (Ma = 0.087),
nu = U D / Re, tau = 0.5 + 3 nu, downstream 6D quadratic sponge (unless --no-sponge),
inlet seeding for 2 periods of St_seed = 0.13 at 10 % of U.

Arms (only the *method-selection* variables change; kernels, BC, force sampler and
post-processing are shared so the comparison isolates the selection decision):

  A1  MRT (recorded TensorLBM selection, oss_solutions id=417)
  A2  BGK   (OSS collision choice: Maarten-vd-Sande/lbm lbm/streamcollide.py::collide)
  A3  TRT   (OSS collision choice: magic Lambda=3/16, Ginzburg 2008 / lbmpy / lettuce)
  A4  MRT without the absorbing sponge (OSS examples keep the plain outlet)

Force sampling must happen post-stream / pre-bounce-back (Ladd momentum exchange).
Sampling the pre-stream state flips the sign of Cd (recorded defect, oss_solutions 417).

Usage:
  python xval/benchmark_method_selection/run_arm.py --arm A1 --D 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.boundary.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_mrt, collide_trt, stream  # noqa: E402

# A4 is the bridge arm: it shares everything with A1 except the sponge, and
# everything with A2/A3 except the collision operator, so the two selection
# variables are separated without confounders.
ARMS = {
    "A1": dict(collision="mrt", sponge=True),   # recorded TensorLBM selection
    "A2": dict(collision="bgk", sponge=False),  # OSS single-relaxation selection
    "A3": dict(collision="trt", sponge=False),  # OSS magic-parameter TRT selection
    "A4": dict(collision="mrt", sponge=False),  # sponge-isolated control for A1
    # A5 = A3 with the anti-symmetric relaxation clamped into the stable TRT
    # window (tau_minus <= 2).  Needed because the OSS magic parameter
    # Lambda = 3/16 requires tau_minus = 0.5 + Lambda/(tau_plus-0.5) = 25.5 at
    # this lattice viscosity (tau_plus = 0.5075), which is far outside the
    # recommended tau_minus in [0.5, 2] window and diverges (measured blow-up).
    "A5": dict(collision="trt_clamped", sponge=False),
}


def square_mask(nx, ny, cx, cy, side, device):
    """Upright square mask; half-way bounce-back puts the wall at x0-0.5/x1+0.5,
    so the effective body size equals ``side`` lattice cells exactly."""
    x0 = int(round(cx - side / 2.0))
    x1 = x0 + int(round(side)) - 1
    y0 = int(round(cy - side / 2.0))
    y1 = y0 + int(round(side)) - 1
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def strouhal_fft(cl: np.ndarray, dt: float = 1.0) -> float:
    """Strouhal from the lift spectrum: mean removal + Hann window + parabolic peak."""
    x = cl - cl.mean()
    n = x.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    return float((freqs[k] + delta * df))


def strouhal_zero_crossing(cl: np.ndarray) -> float:
    """Strouhal from the mean zero-crossing period (upward crossings, linear interp)."""
    x = cl - cl.mean()
    idx = np.where((x[:-1] < 0.0) & (x[1:] >= 0.0))[0]
    if idx.size < 3:
        return float("nan")
    t = idx + (-x[idx]) / (x[idx + 1] - x[idx])
    return float(1.0 / np.mean(np.diff(t)))


def analyze(fx: np.ndarray, fy: np.ndarray, *, D: int, u_in: float, warmup_frac: float):
    n = fx.size
    i0 = int(warmup_frac * n)
    cd_all = 2.0 * fx / (u_in * u_in * D)
    cl_all = 2.0 * fy / (u_in * u_in * D)
    sel = slice(i0, n)
    cd, cl = cd_all[sel], cl_all[sel]
    half = slice(i0 + (n - i0) // 2, n)  # pre-registered "last 50 %" window
    return {
        "cd_mean": float(cd.mean()),
        "cd_mean_last50": float(cd_all[half].mean()),
        "cd_std_frac": float(cd.std() / abs(cd.mean())) if cd.mean() != 0 else float("nan"),
        "cl_amp": float(0.5 * (cl.max() - cl.min())),
        "cl_mean": float(cl.mean()),
        "st_fft": strouhal_fft(cl) * D / u_in,
        "st_zc": strouhal_zero_crossing(cl) * D / u_in,
        "n_samples": int(n),
        "warmup_samples": int(i0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument("--domain-x", type=float, default=20.0, help="streamwise length in D")
    ap.add_argument("--domain-y", type=float, default=20.0, help="height in D (blockage 1/20)")
    ap.add_argument("--x-center", type=float, default=6.0, help="square centre, x/D")
    ap.add_argument("--sponge-d", type=float, default=6.0)
    ap.add_argument("--sponge-alpha", type=float, default=10.0)
    ap.add_argument("--st-seed", type=float, default=0.13)
    ap.add_argument("--st-estimate", type=float, default=0.15, help="period-count estimate, fixed a priori")
    ap.add_argument("--seed-ampl", type=float, default=0.10)
    ap.add_argument("--seed-periods", type=float, default=2.0)
    ap.add_argument("--periods", type=float, default=20.0, help="shedding periods after seeding")
    ap.add_argument("--warmup-frac", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0, help="override (0 = predeclared schedule)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    spec = ARMS[args.arm]
    D = args.D
    nx, ny = int(args.domain_x * D), int(args.domain_y * D)
    nu = args.u_in * D / args.re
    tau = 0.5 + 3.0 * nu

    seed_steps = int(round(args.seed_periods / (args.st_seed * args.u_in / D)))
    period_steps = args.st_estimate * args.u_in / D  # shedding frequency
    n_period_steps = int(round(args.periods / period_steps))
    total_steps = args.steps or (seed_steps + n_period_steps)

    mask = square_mask(nx, ny, args.x_center * D, ny / 2.0, D, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - args.sponge_d * D), int(args.sponge_d * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + args.sponge_alpha * sigma) if spec["sponge"] else None

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, args.u_in), torch.zeros_like(rho0))

    omega_seed = 2.0 * math.pi * args.st_seed * args.u_in / D
    uy_amp = args.seed_ampl * args.u_in
    collision = spec["collision"]

    def collide(f):
        if collision == "mrt":
            return collide_mrt(f, tau, tau_field=tau_field)
        if collision == "bgk":
            return collide_bgk(f, tau)
        if collision == "trt":
            return collide_trt(f, tau)
        if collision == "trt_clamped":
            lam = min(3.0 / 16.0, 1.5 * (tau - 0.5))
            return collide_trt(f, tau, lambda_trt=lam)
        raise ValueError(collision)

    fx_hist: list[float] = []
    fy_hist: list[float] = []
    t0 = time.time()
    for step in range(1, total_steps + 1):
        f = collide(f)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)
        f = far_field_bc_2d(f, args.u_in, mask)
        if step <= seed_steps:
            uy_col = torch.full((ny, 1), uy_amp * math.sin(omega_seed * step), device=device)
            f[:, :, 0] = equilibrium(
                torch.ones((ny, 1), device=device),
                torch.full((ny, 1), args.u_in, device=device),
                uy_col,
            )[:, :, 0]
        fx_hist.append(float(fx))
        fy_hist.append(float(fy))
        if not bool(torch.isfinite(f).all()):
            print(json.dumps({"status": "blowup", "step": step}), flush=True)
            return 2

    wall = time.time() - t0
    fx_np, fy_np = np.asarray(fx_hist), np.asarray(fy_hist)
    res = {
        "arm": args.arm,
        "collision": collision,
        "sponge": spec["sponge"],
        "D": D,
        "re": args.re,
        "u_in": args.u_in,
        "nu": nu,
        "tau": tau,
        "ma": args.u_in / math.sqrt(1.0 / 3.0),
        "blockage": 1.0 / args.domain_y,
        "nx": nx,
        "ny": ny,
        "seed_steps": seed_steps,
        "period_steps_est": n_period_steps,
        "total_steps": total_steps,
        "wall_s": round(wall, 1),
        "ms_per_step": round(1000.0 * wall / total_steps, 3),
        "stability_ok": bool(np.isfinite(fx_np).all() and np.isfinite(fy_np).all()),
        "cl_max_abs": float(np.abs(2.0 * fy_np / (args.u_in ** 2 * D)).max()),
        **analyze(fx_np, fy_np, D=D, u_in=args.u_in, warmup_frac=args.warmup_frac),
    }
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "runs" / f"{args.arm}_D{D}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    np.savez_compressed(
        out_dir / "trace.npz",
        fx=fx_np,
        fy=fy_np,
        meta=np.asarray(json.dumps({k: res[k] for k in ("arm", "D", "u_in", "tau", "total_steps")})),
    )
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())