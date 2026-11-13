#!/usr/bin/env python3
"""XVAL-LBM-BENCHMARK-METHOD-SELECTION — one arm of the method-selection matrix.

Case (fixed, predeclared in ``predeclared_v2.json``): 2-D square-cylinder vortex
shedding, Re = U D / nu = 200, blockage D/H = 1/20, domain 20D x 20D, square
centre at 6D, U = 0.05 (Ma = 0.087), nu = U D / Re, tau = 0.5 + 3 nu, downstream
6D quadratic sponge (unless the arm switches it off), inlet seeding for 2
periods of St_seed = 0.13 at 10 % of U.

Arms (only the *method-selection* variables change; kernels, BC, force sampler
and post-processing are shared so the comparison isolates the selection
decision):

  A1  MRT (recorded TensorLBM selection, oss_solutions id=417)
  A2  BGK   (OSS collision choice: Maarten-vd-Sande/lbm lbm/streamcollide.py::collide)
  A3  TRT   (OSS collision choice: magic Lambda=3/16, Ginzburg 2008 / lbmpy / lettuce)
  A4  MRT without the absorbing sponge (OSS examples keep the plain outlet)
  A5  TRT with Lambda re-derived from the stable tau_minus window (remediation
      of A3): Lambda = 1.5*(tau-0.5) gives tau_minus = 2.0, the upper end of the
      recommended window, because the OSS magic Lambda transplanted at
      tau_plus = 0.5075 would need tau_minus = 25.5 and diverges.
  A6  BGK with the absorbing sponge kept on (round 2): completes the
      2x2 factorial {MRT, BGK} x {sponge on, off} = A1, A4, A6, A2 so the two
      method-selection knobs and their interaction are separately estimable.

Force sampling must happen post-stream / pre-bounce-back (Ladd momentum exchange).
Sampling the pre-stream state flips the sign of Cd (recorded defect, oss_solutions 417).

Artifacts written per arm (programmatic provenance for every reported number):

  runs/<arm>_D<D>/result.json   scalar KPIs + case parameters
  runs/<arm>_D<D>/trace.npz     fx/fy history + final |u| / vorticity fields
  runs/<arm>_D<D>/fail.json     present *instead* of result.json on a measured
                                stability failure (step, reason, diagnostics)

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
    # this lattice viscosity (tau_plus = 0.5075), far outside the recommended
    # tau_minus in [0.5, 2] window and diverges (measured blow-up).
    "A5": dict(collision="trt_clamped", sponge=False),
    # Round-2 cell that completes the 2x2 factorial {MRT, BGK} x {sponge on, off}:
    # A1 = MRT+on, A4 = MRT+off, A6 = BGK+on, A2 = BGK+off.  Without A6 the two
    # main effects could not be separated from their interaction.
    "A6": dict(collision="bgk", sponge=True),
}

CL_FAIL_ABS = 10.0  # predeclared stability rule: |Cl| > 10 == 50x the O(1) amplitude


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
        # Secondary, period-count-robust companion to cl_amp: the extreme-value
        # estimator is sensitive to the sampled record length, the std is not.
        "cl_rms": float(cl.std()),
        "cl_mean": float(cl.mean()),
        "st_fft": strouhal_fft(cl) * D / u_in,
        "st_zc": strouhal_zero_crossing(cl) * D / u_in,
        "n_samples": int(n),
        "warmup_samples": int(i0),
    }


def scheduling(*, D: int, u_in: float, st_seed: float, seed_periods: float,
               st_estimate: float, periods: float) -> tuple[int, int]:
    """Step schedule fixed *before* the run (no wall-clock truncation)."""
    seed_steps = int(round(seed_periods / (st_seed * u_in / D)))
    period_steps = st_estimate * u_in / D  # shedding frequency estimate
    n_period_steps = int(round(periods / period_steps))
    return seed_steps, n_period_steps


def run_arm(
    arm: str,
    D: int = 10,
    *,
    re: float = 200.0,
    u_in: float = 0.05,
    domain_x: float = 20.0,
    domain_y: float = 20.0,
    x_center: float = 6.0,
    sponge_d: float = 6.0,
    sponge_alpha: float = 10.0,
    st_seed: float = 0.13,
    st_estimate: float = 0.165,
    seed_ampl: float = 0.10,
    seed_periods: float = 2.0,
    periods: float = 15.0,
    warmup_frac: float = 0.30,
    threads: int = 4,
    steps: int = 0,
    out: str | Path | None = None,
    keep_fields: bool = True,
    return_trace: bool = False,
    time_budget: float = 0.0,
    resume: bool = True,
) -> dict:
    """Run one arm of the matrix and persist its artifacts.  Library entry point
    used by ``benchmarks/xval_method_selection_bench.py`` and the pytest in
    ``tests/test_xval_benchmark_method_selection.py``."""
    torch.set_num_threads(max(1, threads))
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    spec = ARMS[arm]
    nx, ny = int(domain_x * D), int(domain_y * D)
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    seed_steps, n_period_steps = scheduling(
        D=D, u_in=u_in, st_seed=st_seed, seed_periods=seed_periods,
        st_estimate=st_estimate, periods=periods,
    )
    total_steps = steps or (seed_steps + n_period_steps)

    mask = square_mask(nx, ny, x_center * D, ny / 2.0, D, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - sponge_d * D), int(sponge_d * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + sponge_alpha * sigma) if spec["sponge"] else None

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    omega_seed = 2.0 * math.pi * st_seed * u_in / D
    uy_amp = seed_ampl * u_in
    collision = spec["collision"]
    lam_used = None
    if collision == "trt":
        lam_used = 3.0 / 16.0
    elif collision == "trt_clamped":
        lam_used = min(3.0 / 16.0, 1.5 * (tau - 0.5))
    tau_minus = None if lam_used is None else 0.5 + lam_used / (tau - 0.5)

    def collide(f):
        if collision == "mrt":
            return collide_mrt(f, tau, tau_field=tau_field)
        if collision == "bgk":
            # tau_field is None unless this arm turns the sponge on, so A2 stays
            # the literal OSS single-relaxation arm.
            return collide_bgk(f, tau, tau_field=tau_field)
        if collision == "trt":
            return collide_trt(f, tau, lambda_trt=3.0 / 16.0)
        if collision == "trt_clamped":
            return collide_trt(f, tau, lambda_trt=lam_used)
        raise ValueError(collision)

    out_dir = Path(out) if out else (Path(__file__).resolve().parent / "runs" / f"{arm}_D{D}")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.npz"
    start_step = 0
    fx_hist: list[float] = []
    fy_hist: list[float] = []
    cl_abs_hist: list[float] = []
    if resume and state_path.exists():
        # chunked execution: this container kills long-lived background processes,
        # so an arm is run in wall-clock-bounded chunks and resumes from state.npz
        st = np.load(state_path)
        f = torch.tensor(st["f"])
        start_step = int(st["step"])
        fx_hist = [float(v) for v in st["fx"]]
        fy_hist = [float(v) for v in st["fy"]]
        print(json.dumps({"status": "resume", "step": start_step, "arm": arm, "D": D}), flush=True)
    deadline = (time.time() + time_budget) if time_budget > 0 else None
    t0 = time.time()
    fail: dict | None = None
    for step in range(start_step + 1, total_steps + 1):
        f = collide(f)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)
        f = far_field_bc_2d(f, u_in, mask)
        if step <= seed_steps:
            uy_col = torch.full((ny, 1), uy_amp * math.sin(omega_seed * step), device=device)
            f[:, :, 0] = equilibrium(
                torch.ones((ny, 1), device=device),
                torch.full((ny, 1), u_in, device=device),
                uy_col,
            )[:, :, 0]
        fx_v, fy_v = float(fx), float(fy)
        fx_hist.append(fx_v)
        fy_hist.append(fy_v)
        cl_abs = abs(2.0 * fy_v / (u_in * u_in * D))
        cl_abs_hist.append(cl_abs)
        if step % 50 == 0 or step == total_steps:
            if not bool(torch.isfinite(f).all()):
                fail = dict(reason="non_finite_population", step=step)
            elif cl_abs > CL_FAIL_ABS:
                fail = dict(reason="cl_amplitude_runaway", step=step, cl_abs=cl_abs)
        if fail is not None:
            break
        if step % 1000 == 0 or step == total_steps:
            np.savez_compressed(state_path, f=f.numpy(), step=np.asarray(step),
                                fx=np.asarray(fx_hist), fy=np.asarray(fy_hist))
            if deadline is not None and time.time() > deadline:
                print(json.dumps({"status": "partial", "arm": arm, "D": D, "step": step,
                                  "total_steps": total_steps, "wall_s": round(time.time() - t0, 1)}),
                      flush=True)
                return {"status": "partial", "step": step, "record": {"arm": arm, "D": D},
                        "fx": np.asarray(fx_hist), "fy": np.asarray(fy_hist)}

    wall = time.time() - t0
    if state_path.exists():
        state_path.unlink()

    if fail is not None:
        rec = {
            "arm": arm,
            "D": D,
            "collision": collision,
            "sponge": spec["sponge"],
            "status": "blowup",
            "fail_stage": fail["reason"],
            "step": fail["step"],
            "total_steps_planned": total_steps,
            "wall_s": round(wall, 1),
            "tau": tau,
            "nu": nu,
            "u_in": u_in,
            "lambda_trt": lam_used,
            "tau_minus": tau_minus,
            "cl_max_abs": float(max(cl_abs_hist)),
            "nx": nx,
            "ny": ny,
            "seed_steps": seed_steps,
        }
        (out_dir / "fail.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2), flush=True)
        return {"status": "blowup", "record": rec, "fx": np.asarray(fx_hist), "fy": np.asarray(fy_hist)}

    fx_np, fy_np = np.asarray(fx_hist), np.asarray(fy_hist)
    res = {
        "arm": arm,
        "collision": collision,
        "sponge": spec["sponge"],
        "D": D,
        "re": re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "lambda_trt": lam_used,
        "tau_minus": tau_minus,
        "ma": u_in / math.sqrt(1.0 / 3.0),
        "blockage": 1.0 / domain_y,
        "nx": nx,
        "ny": ny,
        "seed_steps": seed_steps,
        "period_steps_est": n_period_steps,
        "total_steps": total_steps,
        "wall_s": round(wall, 1),
        "ms_per_step": round(1000.0 * wall / total_steps, 3),
        "status": "ok",
        "stability_ok": bool(np.isfinite(fx_np).all() and np.isfinite(fy_np).all()),
        "cl_max_abs": float(np.abs(2.0 * fy_np / (u_in ** 2 * D)).max()),
        **analyze(fx_np, fy_np, D=D, u_in=u_in, warmup_frac=warmup_frac),
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2) + "\n")
    payload = {
        "fx": fx_np,
        "fy": fy_np,
        "meta": np.asarray(json.dumps({k: res[k] for k in ("arm", "D", "u_in", "tau", "total_steps")})),
    }
    if keep_fields:
        # final instantaneous flow field (last 50 % averaging-free snapshot) for figures
        rho = f.sum(dim=0)
        ux = (f * torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=f.dtype).view(9, 1, 1)).sum(0) / rho
        uy = (f * torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=f.dtype).view(9, 1, 1)).sum(0) / rho
        umag = torch.sqrt(ux * ux + uy * uy)
        omega = torch.zeros_like(ux)
        omega[1:-1, 1:-1] = 0.5 * (
            (uy[2:, 1:-1] - uy[:-2, 1:-1]) - (ux[1:-1, 2:] - ux[1:-1, :-2])
        )
        payload.update(
            umag=umag.cpu().numpy(),
            vort=omega.cpu().numpy(),
            solid=mask.cpu().numpy(),
        )
    np.savez_compressed(out_dir / "trace.npz", **payload)
    print(json.dumps(res, indent=2), flush=True)
    out = {"status": "ok", "record": res, "fx": fx_np, "fy": fy_np}
    if return_trace:
        out["fields"] = payload
    return out


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
    ap.add_argument("--st-estimate", type=float, default=0.165, help="period-count estimate, fixed a priori")
    ap.add_argument("--seed-ampl", type=float, default=0.10)
    ap.add_argument("--seed-periods", type=float, default=2.0)
    ap.add_argument("--periods", type=float, default=15.0, help="shedding periods after seeding")
    ap.add_argument("--warmup-frac", type=float, default=0.30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0, help="override (0 = predeclared schedule)")
    ap.add_argument("--out", default="")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="wall-clock seconds for this chunk (0 = run to completion)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    r = run_arm(
        args.arm, args.D,
        re=args.re, u_in=args.u_in, domain_x=args.domain_x, domain_y=args.domain_y,
        x_center=args.x_center, sponge_d=args.sponge_d, sponge_alpha=args.sponge_alpha,
        st_seed=args.st_seed, st_estimate=args.st_estimate, seed_ampl=args.seed_ampl,
        seed_periods=args.seed_periods, periods=args.periods,
        warmup_frac=args.warmup_frac, threads=args.threads, steps=args.steps,
        out=(args.out or None), time_budget=args.time_budget, resume=not args.no_resume,
    )
    return 0 if r["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())