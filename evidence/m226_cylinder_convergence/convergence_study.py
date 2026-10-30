#!/usr/bin/env python3
"""SDAA cylinder family convergence study — m226.

Runs re20_st / re100 / re200 on a D × steps ladder, saving incremental
checkpoints per sub-case.  Wall-clock budget: 20 min per sub-case.

Grid ladder:
  re20_st:  D = 10, 20, 40, 80   (steps = 8000, 16000, 60000, 120000)
  re100:    D = 10, 20, 40, 64   (steps = 6000, 12000, 40000, 70000)
  re200:    D = 10, 20, 40, 64   (steps = 6000, 12000, 40000, 70000)

Each run saves a checkpoint JSON after completion.  The final summary
aggregates all checkpoints into a convergence table.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Bootstrap tensorlbm
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch

# ---------------------------------------------------------------------------
# re20_st — Schäfer-Turek 2D-1 channel
# ---------------------------------------------------------------------------
REF_CD_RE20 = 5.57953523384
REF_CL_RE20 = 0.010618948146


def run_re20_st(D_cells: int, n_steps: int, device: str, out_path: str | None = None) -> dict:
    """Run Re=20 steady channel case."""
    from tensorlbm.boundaries import (
        bounce_back_cells,
        compute_obstacle_forces,
        cylinder_mask,
        make_channel_wall_mask,
        zou_he_inlet_velocity,
        zou_he_outlet_pressure,
    )
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_mrt, stream

    dev = torch.device(device)
    lx, ly = 2.2, 0.41
    dx = 0.1 / D_cells
    nx = int(round(lx / dx))
    ny = int(round(ly / dx))
    cx = round(0.2 / dx)
    cy = round(0.2 / dx)
    radius = 0.05 / dx
    mask_radius = radius  # shift=0 (verified config)

    tau = 0.8
    nu_lb = (tau - 0.5) / 3.0
    U_char_lb = 20.0 * nu_lb / D_cells
    y_phys = torch.arange(ny, device=dev) * dx
    u_profile = 4.0 * 1.5 * y_phys * (0.41 - y_phys) / 0.41**2
    u_lb = U_char_lb * u_profile

    solid = cylinder_mask(nx, ny, cx, cy, mask_radius, dev)
    wall = make_channel_wall_mask(ny, nx, solid, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0) | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1) | torch.roll(fluid, -1, 1)
    )

    dyn_p = 0.5 * U_char_lb**2 * D_cells

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.zeros_like(rho0)
    ux0[:, :] = u_lb[:, None].expand(ny, nx)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)

    t0 = time.time()
    cd_list, cl_list = [], []
    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        f = zou_he_inlet_velocity(f, u_lb, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, wall)
        fx, fy = compute_obstacle_forces(f, surface)
        f = bounce_back_cells(f, solid)
        cd_list.append(float(fx.item()) / dyn_p)
        cl_list.append(float(fy.item()) / dyn_p)

    win = min(len(cd_list), 50000)
    cd = sum(cd_list[-win:]) / win
    cl = sum(cl_list[-win:]) / win
    elapsed = time.time() - t0

    result = {
        "case": "re20_st",
        "D_cells": D_cells,
        "nx": nx, "ny": ny,
        "n_steps": n_steps,
        "cd": cd, "cl": cl,
        "err_cd_pct": (cd - REF_CD_RE20) / REF_CD_RE20 * 100,
        "err_cl_pct": (cl - REF_CL_RE20) / REF_CL_RE20 * 100,
        "ref_cd": REF_CD_RE20,
        "finite": True,
        "wall_s": round(elapsed, 1),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[re20_st D={D_cells}] steps={n_steps} t={elapsed:.0f}s "
          f"Cd={cd:.4f} ({result['err_cd_pct']:+.2f}%)", flush=True)
    return result


# ---------------------------------------------------------------------------
# re100 — Free-stream vortex shedding (Braza 1986)
# ---------------------------------------------------------------------------
REF_CD_RE100 = 1.35
REF_ST_RE100 = 0.1645


def run_re100(D: int, n_steps: int, device: str, out_path: str | None = None) -> dict:
    """Run Re=100 free-stream vortex shedding."""
    import numpy as np
    from tensorlbm.boundaries import (
        compute_obstacle_forces,
        cylinder_mask,
        far_field_bc_2d,
        make_sponge_strength,
    )
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_mrt, stream

    dev = torch.device(device)
    DOMAIN_D = 40.0
    CYL_X_D = 10.0
    SPONGE_D = 10.0
    SPONGE_ALPHA = 10.0
    ST_SEED = 0.165
    SEED_AMPL = 0.10
    SEED_PERIODS = 2.0

    nx = int(DOMAIN_D * D)
    ny = nx
    u_in = 0.05
    re = 100.0
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    mask = cylinder_mask(nx, ny, CYL_X_D * D, ny / 2.0, D / 2.0, dev)
    sigma = make_sponge_strength(ny, nx, int(nx - SPONGE_D * D), int(SPONGE_D * D), power=2.0, device=dev)
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=dev)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    omega_seed = 2.0 * math.pi * ST_SEED * u_in / D
    seed_steps = int(round(SEED_PERIODS / (ST_SEED * u_in / D)))
    uy_amp = SEED_AMPL * u_in

    from tensorlbm.d2q9 import C as C2D
    c2d = C2D.to(dev).float()
    cy_w = c2d[:, 1].view(9, 1, 1)

    t0 = time.time()
    fx_hist, fy_hist = [], []
    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau, tau_field=tau_field)
        f = torch.where(mask.unsqueeze(0), before, collided)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)
        f = far_field_bc_2d(f, u_in, obstacle_mask=mask)

        if step <= seed_steps:
            phase = 2 * math.pi * step / (D / (ST_SEED * u_in))
            rho_col = f.sum(0)[:, :4]
            ux_col = (f * c2d[:, 0].view(9, 1, 1)).sum(0)[:, :4] / rho_col.clamp(min=1e-12)
            uy_col = (f * cy_w).sum(0)[:, :4] / rho_col.clamp(min=1e-12)
            uy_col += uy_amp * math.sin(phase)
            feq_col = equilibrium(rho_col, ux_col, uy_col)
            f[:, :, :4] = feq_col

        fx_hist.append(fx)
        fy_hist.append(fy)

    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        return {"case": "re100", "D": D, "finite": False, "wall_s": round(elapsed, 1)}

    fx_np = torch.stack(fx_hist).detach().cpu().numpy().astype(np.float64)
    fy_np = torch.stack(fy_hist).detach().cpu().numpy().astype(np.float64)

    q_dyn = 0.5 * 1.0 * u_in * u_in * D
    w0 = int(0.5 * n_steps)
    cd_series = fx_np / q_dyn
    cd_mean = float(cd_series[w0:].mean())
    cl_series = fy_np / q_dyn

    cl_w = cl_series[w0:] - cl_series[w0:].mean()
    st_val = float("nan")
    if cl_w.size >= 256:
        spec = np.abs(np.fft.rfft(cl_w * np.hanning(cl_w.size)))
        k = int(np.argmax(spec[1:])) + 1
        delta = 0.0
        if 0 < k < spec.size - 1:
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-30:
                delta = float(0.5 * (a - c) / den)
        st_val = float((k + delta) / cl_w.size * D / u_in)

    err_cd = (cd_mean - REF_CD_RE100) / REF_CD_RE100 * 100.0
    err_st = (st_val - REF_ST_RE100) / REF_ST_RE100 * 100.0 if math.isfinite(st_val) else float("nan")

    result = {
        "case": "re100",
        "D": D,
        "nx": nx, "ny": ny,
        "n_steps": n_steps,
        "cd": round(cd_mean, 4),
        "st": round(st_val, 4) if math.isfinite(st_val) else None,
        "err_cd_pct": round(err_cd, 2),
        "st_err_pct": round(err_st, 2) if math.isfinite(err_st) else None,
        "ref_cd": REF_CD_RE100,
        "ref_st": REF_ST_RE100,
        "finite": True,
        "wall_s": round(elapsed, 1),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[re100 D={D}] steps={n_steps} t={elapsed:.0f}s "
          f"Cd={cd_mean:.4f} ({err_cd:+.2f}%) St={st_val:.4f}", flush=True)
    return result


# ---------------------------------------------------------------------------
# re200 — Free-stream vortex shedding (Braza 1986)
# ---------------------------------------------------------------------------
REF_CD_RE200 = 1.33
REF_ST_RE200 = 0.195


def run_re200(D: int, n_steps: int, device: str, out_path: str | None = None) -> dict:
    """Run Re=200 free-stream vortex shedding."""
    import numpy as np
    from tensorlbm.boundaries import (
        compute_obstacle_forces,
        cylinder_mask,
        far_field_bc_2d,
        make_sponge_strength,
    )
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_mrt, stream

    dev = torch.device(device)
    nx_D, ny_D = 40, 40
    u_in = 0.08
    re = 200.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    nx, ny = nx_D * D, ny_D * D
    cx, cy = 12 * D, ny / 2.0
    radius = D / 2.0

    solid = cylinder_mask(nx, ny, cx, cy, radius, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0) | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1) | torch.roll(fluid, -1, 1)
    )
    dyn_p = 0.5 * u_in**2 * D

    SPONGE_D = 10.0
    SPONGE_ALPHA = 10.0
    x0 = int(nx - SPONGE_D * D)
    W = nx - x0
    sigma = make_sponge_strength(ny, nx, x0, W, power=2.0, device=dev)
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    ST_SEED = 0.195
    T_seed = D / (ST_SEED * u_in)
    n_seed = int(2 * T_seed)
    seed_amp = 0.10

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)

    from tensorlbm.d2q9 import C as C2D
    c2d = C2D.to(dev).float()
    cx_w = c2d[:, 0].view(9, 1, 1)
    cy_w = c2d[:, 1].view(9, 1, 1)

    t0 = time.time()
    cd_list, cl_list = [], []
    cl_series = []
    steps_done = 0
    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau, tau_field=tau_field)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, surface)
        f = far_field_bc_2d(f, u_in, obstacle_mask=solid)

        if step <= n_seed:
            phase = 2 * math.pi * step / T_seed
            rho_col = f.sum(0)[:, :4]
            ux_col = (f * cx_w).sum(0)[:, :4] / rho_col.clamp(min=1e-12)
            uy_col = (f * cy_w).sum(0)[:, :4] / rho_col.clamp(min=1e-12)
            uy_col += seed_amp * u_in * math.sin(phase)
            feq_col = equilibrium(rho_col, ux_col, uy_col)
            f[:, :, :4] = feq_col

        if step > int(n_steps * 0.5):
            cd_list.append(float(fx.item()) / dyn_p)
            cl_list.append(float(fy.item()) / dyn_p)
        if step % 10 == 0:
            cl_series.append(float(fy.item()) / dyn_p)

        if step % 1000 == 0 and torch.isnan(f).any():
            print(f"  *** NaN at step {step} ***", flush=True)
            break
        steps_done = step

    elapsed = time.time() - t0
    cd_avg = sum(cd_list) / max(len(cd_list), 1)

    st_val = float("nan")
    if len(cl_series) > 64:
        sig = np.array(cl_series) - np.mean(cl_series)
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        freqs = np.fft.rfftfreq(len(sig), d=10.0)
        k = int(np.argmax(spec[1:])) + 1
        st_val = freqs[k] * D / u_in
        if 0 < k < len(spec) - 1:
            lp = np.log(np.maximum(spec[k - 1:k + 2], 1e-30))
            denom = lp[0] - 2.0 * lp[1] + lp[2]
            if abs(denom) > 1e-12:
                delta = 0.5 * (lp[0] - lp[2]) / denom
                delta = max(-0.5, min(0.5, delta))
                st_val = (freqs[k] + delta * (freqs[1] - freqs[0])) * D / u_in

    err_cd = (cd_avg - REF_CD_RE200) / REF_CD_RE200 * 100
    err_st = (st_val - REF_ST_RE200) / REF_ST_RE200 * 100 if math.isfinite(st_val) else float("nan")

    result = {
        "case": "re200",
        "D": D,
        "nx": nx, "ny": ny,
        "n_steps": n_steps,
        "steps_done": steps_done,
        "cd": round(cd_avg, 4),
        "st": round(st_val, 4) if math.isfinite(st_val) else None,
        "err_cd_pct": round(err_cd, 2),
        "st_err_pct": round(err_st, 2) if math.isfinite(err_st) else None,
        "ref_cd": REF_CD_RE200,
        "ref_st": REF_ST_RE200,
        "finite": True,
        "wall_s": round(elapsed, 1),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[re200 D={D}] steps={n_steps} t={elapsed:.0f}s "
          f"Cd={cd_avg:.4f} ({err_cd:+.2f}%) St={st_val:.4f}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Convergence ladder
# ---------------------------------------------------------------------------
LADDER = {
    "re20_st": [
        (10, 8000),
        (20, 16000),
        (40, 60000),
        (80, 120000),
    ],
    "re100": [
        (10, 6000),
        (20, 12000),
        (40, 40000),
        (64, 70000),
    ],
    "re200": [
        (10, 6000),
        (20, 12000),
        (40, 40000),
        (64, 70000),
    ],
}

RUNNERS = {
    "re20_st": run_re20_st,
    "re100": run_re100,
    "re200": run_re200,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["re20_st", "re100", "re200", "all"], default="all")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out-dir", default=str(REPO / "evidence" / "m226_cylinder_convergence"))
    ap.add_argument("--budget-min", type=int, default=20, help="wall-clock budget per sub-case (min)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = ["re20_st", "re100", "re200"] if args.case == "all" else [args.case]
    all_results = {}

    for case_name in cases:
        print(f"\n{'='*60}")
        print(f"  CASE: {case_name}")
        print(f"{'='*60}", flush=True)

        runner = RUNNERS[case_name]
        ladder = LADDER[case_name]
        case_results = []
        t_case_start = time.time()
        budget_s = args.budget_min * 60

        for D, steps in ladder:
            elapsed_case = time.time() - t_case_start
            if elapsed_case > budget_s:
                print(f"  [budget exceeded: {elapsed_case/60:.1f}min > {args.budget_min}min, stopping]", flush=True)
                break

            ckpt_path = str(out_dir / f"{case_name}_D{D}.json")
            try:
                r = runner(D, steps, args.device, out_path=ckpt_path)
                case_results.append(r)
            except Exception as e:
                print(f"  [ERROR D={D}]: {e}", flush=True)
                case_results.append({
                    "case": case_name, "D": D, "n_steps": steps,
                    "finite": False, "error": str(e),
                })

        all_results[case_name] = case_results

    # Summary
    summary = {
        "mission": "m226",
        "title": "SDAA cylinder family convergence study",
        "device": args.device,
        "budget_min": args.budget_min,
        "cases": all_results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")

    # Print convergence table
    print(f"\n{'='*60}")
    print("  CONVERGENCE TABLE")
    print(f"{'='*60}")
    for case_name, results in all_results.items():
        print(f"\n--- {case_name} ---")
        print(f"  {'D':>4}  {'steps':>7}  {'Cd':>8}  {'err%':>7}  {'St':>6}  {'St_err%':>8}  {'finite':>6}  {'wall_s':>7}")
        for r in results:
            d = r.get("D", r.get("D_cells", "?"))
            steps = r.get("n_steps", "?")
            cd = r.get("cd", "N/A")
            err = r.get("err_cd_pct", "N/A")
            st = r.get("st", "N/A")
            st_err = r.get("st_err_pct", "N/A")
            finite = r.get("finite", "?")
            wall = r.get("wall_s", "?")
            print(f"  {d:>4}  {steps:>7}  {str(cd):>8}  {str(err):>7}  {str(st):>6}  {str(st_err):>8}  {str(finite):>6}  {str(wall):>7}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
