#!/usr/bin/env python3
"""Gartling 1990 backward-facing step Re=800 — single-segment runner with checkpoint I/O.

This script runs ONE segment of the simulation. It can:
  - Start fresh (no --resume) or resume from a checkpoint (--resume PATH)
  - Save a checkpoint at the end (--save-ckpt PATH)
  - Output convergence data to JSON

Designed to be called multiple times in a loop (shell script) to accumulate
many steps across wall-clock-limited SDAA sessions.

Usage:
  # Segment 1 (fresh start):
  python run_segment.py --m 32 --steps 30000 --save-ckpt ckpt/seg0.pt --out out/seg1.json

  # Segment 2 (resume):
  python run_segment.py --m 32 --steps 30000 --resume ckpt/seg0.pt --save-ckpt ckpt/seg1.pt --out out/seg2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from compile_route import (
    add_compile_mode_arg,
    compile_mode_from_args,
    route_step,
)

from tensorlbm.boundaries import (
    make_sponge_strength,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_mrt, collide_rlbm, stream

# Reference values
REF_X1_H = 6.10
ERR_TOL_PCT = 3.0
X_STEP = 8
DEFAULT_SPONGE_PEAK = 0.3
DEFAULT_SPONGE_WIDTH_H = 3.0


def _default_device() -> str:
    try:
        import torch_sdaa
        if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
            return "sdaa:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def build_solid(ny, nx, m, x_step, device):
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[-1, :] = True
    solid[0, x_step:] = True
    solid[0 : m + 1, :x_step] = True
    return solid


def inlet_profile(ny, m, u_mean, device, dtype):
    a = float(m)
    y = np.arange(ny, dtype=np.float64)
    s = y - (m + 0.5)
    prof = np.zeros(ny, dtype=np.float64)
    rows = np.arange(m + 1, ny - 1)
    ss = s[rows]
    raw = 6.0 * ss * (a - ss) / (a * a)
    raw_mean = raw.mean()
    prof[rows] = raw / raw_mean * u_mean
    return torch.tensor(prof, dtype=dtype, device=device)


def _shear_weights(K):
    y = np.arange(K, dtype=np.float64)
    ym = y.mean()
    w = y - ym
    return w / float((w * w).sum())


def _crossings(profile, start_col):
    out = []
    p = np.asarray(profile, dtype=np.float64)
    for c in range(start_col, len(p) - 1):
        a, b = p[c], p[c + 1]
        if a == b:
            continue
        if a <= 0.0 < b:
            out.append((c + (0.0 - a) / (b - a), +1))
        elif a >= 0.0 > b:
            out.append((c + (0.0 - a) / (b - a), -1))
    return out


def _x1_from_shear(shear, x_step, H):
    for xc, d in _crossings(shear, x_step):
        if d == +1:
            x1_cells = xc - (x_step - 0.5)
            return x1_cells / H, x1_cells
    return float("nan"), float("nan")


def _rows_near_walls(ny, m, K):
    lo_rows = np.arange(1, K + 1)
    up_rows = np.arange(ny - 2, ny - 2 - K, -1)
    w = _shear_weights(K)
    return lo_rows, w, up_rows, w[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--u", type=float, default=0.06)
    ap.add_argument("--re", type=float, default=800.0)
    ap.add_argument("--steps", type=int, default=30000, help="steps THIS segment")
    ap.add_argument("--avg-steps", type=int, default=10000)
    ap.add_argument("--avg-stride", type=int, default=20)
    ap.add_argument("--L-over-H", type=float, default=15.0)
    ap.add_argument("--collision", choices=["mrt", "rlbm"], default="mrt")
    ap.add_argument("--sponge-peak", type=float, default=DEFAULT_SPONGE_PEAK)
    ap.add_argument("--sponge-width-H", type=float, default=DEFAULT_SPONGE_WIDTH_H)
    ap.add_argument("--no-prestream-bb", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    ap.add_argument("--save-ckpt", type=str, default=None, help="save checkpoint here at end")
    ap.add_argument("--out", type=str, required=True, help="output JSON path")
    ap.add_argument("--seg-id", type=int, default=0, help="segment index (for logging)")
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else _default_device())
    compile_mode = compile_mode_from_args(args)

    m = args.m
    ny = 2 * m + 1
    H = ny - 1
    L = int(round(args.L_over_H * H))
    nx = X_STEP + L
    x_step = X_STEP
    nu = args.u * H / args.re
    tau = 0.5 + 3.0 * nu

    print(
        f"[seg {args.seg_id}] m={m} H={H} ny={ny} nx={nx} tau={tau:.4f} "
        f"steps={args.steps} device={device} resume={args.resume}",
        flush=True,
    )

    # Build geometry
    solid = build_solid(ny, nx, m, x_step, device)
    u_prof = inlet_profile(ny, m, args.u, device, torch.float32)

    sponge_w = int(round(args.sponge_width_H * H))
    sponge_x0 = nx - sponge_w
    if args.sponge_peak > 0.0 and sponge_w > 0:
        sigma = args.sponge_peak * make_sponge_strength(
            ny, nx, sponge_x0, sponge_w, power=2.0, device=device
        )
    else:
        sigma = torch.zeros((ny, nx), device=device)

    K = max(2, min(6, m // 2))
    lo_rows, w_lo, up_rows, w_up = _rows_near_walls(ny, m, K)
    K = len(lo_rows)

    collide = {"mrt": collide_mrt, "rlbm": collide_rlbm}[args.collision]
    opp = OPPOSITE.to(device)
    solid3 = solid.unsqueeze(0)
    prestream_bb = not args.no_prestream_bb

    def _noneq_sponge(f):
        if args.sponge_peak <= 0.0:
            return f
        rho, ux, uy = macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        return f - sigma.unsqueeze(0) * (f - feq)

    if prestream_bb:
        def _step(f):
            fp = f.clone()
            fc = collide(f, tau=tau)
            fc = torch.where(solid3, fp[opp], fc)
            f = stream(fc)
            f = zou_he_inlet_velocity(f, u_prof, 0.0)
            f = zou_he_outlet_pressure(f, 1.0)
            return _noneq_sponge(f)
    else:
        def _step(f):
            f = collide(f, tau=tau)
            f = stream(f)
            f = zou_he_inlet_velocity(f, u_prof, 0.0)
            f = zou_he_outlet_pressure(f, 1.0)
            f = torch.where(solid3, f[opp], f)
            return _noneq_sponge(f)

    step_fn = route_step(_step, compile_mode, name=f"bfs_seg{args.seg_id}[m{m}]")

    # Initialize or resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        f = ckpt["f"]
        start_step = ckpt["step"]
        print(f"[seg {args.seg_id}] resumed from step {start_step}", flush=True)
    else:
        rho0 = torch.ones((ny, nx), device=device)
        ux0 = torch.zeros((ny, nx), device=device)
        ux0[:, :x_step] = u_prof.view(ny, 1)
        ux0[:, x_step:] = args.u / 2.0
        ux0[solid] = 0.0
        f = equilibrium(rho0, ux0, torch.zeros_like(ux0))
        start_step = 0
        print(f"[seg {args.seg_id}] fresh start", flush=True)

    # Measurement
    n_win = max(1, int(round(args.avg_steps / max(1, args.avg_stride))))
    win = deque(maxlen=n_win)
    series = []
    t0 = time.time()
    step = start_step

    for i in range(args.steps):
        step += 1
        f = step_fn(f)

        if step % args.avg_stride == 0:
            ux = macroscopic(f)[1]
            uxm = ux.masked_fill(solid, 0.0)
            lo = uxm[lo_rows].detach().cpu().numpy().astype(np.float64)
            s_lo = (w_lo[:, None] * lo).sum(axis=0)
            x1h, _ = _x1_from_shear(s_lo, x_step, H)
            rho, _, _ = macroscopic(f)
            win.append(x1h)
            x1h_win = float(np.mean(win))
            series.append({
                "step": step,
                "X1_H_inst": float(x1h),
                "X1_H_win": x1h_win,
                "max_speed": float(ux.abs().max()),
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
            })

            log_interval = max(1, args.avg_stride * max(1, args.steps // 10 // args.avg_stride))
            if step % log_interval == 0 or i == args.steps - 1:
                elapsed = time.time() - t0
                rate = (step - start_step) / elapsed if elapsed > 0 else 0
                print(
                    f"[seg {args.seg_id}] step={step:7d} X1/H(win)={x1h_win:.4f} "
                    f"X1/H(inst)={x1h:.4f} max|u|={series[-1]['max_speed']:.4f} "
                    f"rho=[{series[-1]['rho_min']:.4f},{series[-1]['rho_max']:.4f}] "
                    f"rate={rate:.0f}st/s t={elapsed:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    final_x1 = series[-1]["X1_H_win"] if series else float("nan")

    # Save checkpoint
    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"f": f.cpu(), "step": step}, ckpt_path)
        print(f"[seg {args.seg_id}] checkpoint saved: {ckpt_path} (step={step})", flush=True)

    # Save result
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "seg_id": args.seg_id,
        "start_step": start_step,
        "end_step": step,
        "steps_run": args.steps,
        "final_X1_H": final_x1,
        "err_pct": (final_x1 - REF_X1_H) / REF_X1_H * 100.0,
        "elapsed_s": elapsed,
        "rate_steps_s": (step - start_step) / elapsed if elapsed > 0 else 0,
        "series": series,
        "checkpoint_saved": args.save_ckpt,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[seg {args.seg_id}] result: {out_path}", flush=True)
    print(f"[seg {args.seg_id}] DONE: X1/H={final_x1:.4f} err={result['err_pct']:+.2f}% "
          f"elapsed={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
