#!/usr/bin/env python3
"""B14-ellipse: 3D elliptical-pipe Poiseuille — TWO-OBSERVABLE hydraulic inversion.

Verification-debt closure for the pending benchmark
`benchmarks/pending/poiseuille_3d_ellipse/` (s^Q single-scale comparison failed
at a=20: 5.11% > 3%).  The root cause recorded in the pending README is that the
digital (staircase) ellipse wall is ANISOTROPIC, so a single area-equivalent
scale s^Q cannot represent its hydraulic geometry.

This script closes the debt with the promotion path named in benchmarks/TODO.md:
a two-observable (Q + fully-developed pressure gradient G) hydraulic inversion
of the effective semi-axes (a_e, b_e), followed by a per-cell profile comparison
against the analytic elliptic parabola.

Analytic relations (steady, fully developed, no-slip, continuous ellipse a_e,b_e):
    u(y,z) = U_c * (1 - (y/a_e)^2 - (z/b_e)^2)
    Q      = pi * a_e * b_e * U_c / 2
    G      = 2*nu*U_c*(a_e^2 + b_e^2) / (a_e^2 * b_e^2)      [G = dp/L]
Inversion (U_c = 2*u_in imposed by the uniform-velocity inlet):
    P = (a_e*b_e)^2 = (2*Q/(pi*U_c))^2          <- from Q  (integral observable)
    S = a_e^2 + b_e^2 = G*P/(2*nu*U_c)          <- from G  (integral observable)
    a_e = sqrt((S + sqrt(S^2 - 4P))/2),  b_e = sqrt((S - sqrt(S^2 - 4P))/2)
Mass-conservation renormalization (geometry-only, no profile fit):
    M_d(a_e,b_e) = sum_{digital fluid cells} (1 - (y/a_e)^2 - (z/b_e)^2)
    U_c' = Q / M_d          (exact if the profile is the parabola; a pure
                             geometric sum over the digital cross-section)
Primary metric (same philosophy as the s^Q primary metric): max relative error
of the lambda_e-binned (elliptically-averaged) profile in the central region
(u_ana > 0.2*U_c'), analytic value evaluated per cell then bin-averaged.

Nothing in (a_e, b_e, U_c') is fitted to the velocity profile: Q and G are
global integral observables, M_d is a geometric count.  The profile is used
ONLY as the quantity being tested.

Usage:
    run_hydraulic.py a out.json [--ratio 2.0] [--tau 0.8] [--u-in 0.02]
        [--min-steps 20000] [--max-steps 60000] [--device cpu] [--seed 0]
        [--threads N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

CS2 = 1.0 / 3.0


def ellipse_setup(a: int, b: int, L_over_a: int, device: torch.device):
    ny = 2 * a + 3
    nz = 2 * b + 3
    nx = L_over_a * a
    yc = a + 1
    zc = b + 1
    iz = torch.arange(nz, device=device, dtype=torch.float32).view(-1, 1)
    iy = torch.arange(ny, device=device, dtype=torch.float32).view(1, -1)
    lam = ((iy - yc) / a) ** 2 + ((iz - zc) / b) ** 2  # (nz, ny)
    fluid2d = lam <= 1.0
    wall_mask = (~fluid2d).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    return ny, nz, nx, yc, zc, lam, fluid2d, wall_mask


def invert_two_observable(Q: float, G: float, U_c: float, nu: float) -> dict:
    """Invert (a_e, b_e) from (Q, G) with U_c imposed."""
    P = (2.0 * Q / (math.pi * U_c)) ** 2
    S = G * P / (2.0 * nu * U_c)
    disc = S * S - 4.0 * P
    out = {"P": P, "S": S, "disc": disc}
    if disc <= 0:
        out.update(ok=False, a_e=float("nan"), b_e=float("nan"))
        return out
    a2 = 0.5 * (S + math.sqrt(disc))
    b2 = 0.5 * (S - math.sqrt(disc))
    out.update(ok=True, a_e=math.sqrt(a2), b_e=math.sqrt(b2))
    return out


def digital_M(fluid: np.ndarray, a_e: float, b_e: float, yc: int, zc: int) -> float:
    """M_d = sum over digital fluid cells of (1 - (y/a_e)^2 - (z/b_e)^2)."""
    nz_p, ny_p = fluid.shape
    dy = (np.arange(ny_p, dtype=np.float64) - yc) / a_e
    dz = (np.arange(nz_p, dtype=np.float64) - zc) / b_e
    ly2 = np.broadcast_to(dy[np.newaxis, :] ** 2, fluid.shape)
    lz2 = np.broadcast_to(dz[:, np.newaxis] ** 2, fluid.shape)
    val = 1.0 - ly2 - lz2
    return float(val[fluid].sum())


def profile_compare(u_np, fluid, a_e, b_e, U_c, yc, zc) -> dict:
    """Bin by lambda_e = (y/a_e)^2 + (z/b_e)^2, compare per-cell analytic."""
    nz_p, ny_p = u_np.shape
    dy = (np.arange(ny_p, dtype=np.float64) - yc) / a_e
    dz = (np.arange(nz_p, dtype=np.float64) - zc) / b_e
    ly2 = np.broadcast_to(dy[np.newaxis, :] ** 2, u_np.shape)
    lz2 = np.broadcast_to(dz[:, np.newaxis] ** 2, u_np.shape)
    lam_e = ly2 + lz2
    u_ana_cell = U_c * (1.0 - lam_e)
    nb = max(1, int(round(b_e)))
    bin_idx = np.clip(np.floor(lam_e * b_e).astype(int), 0, nb - 1)
    u_num, u_ana, cells, lam_avg = [], [], [], []
    for k in range(nb):
        m = fluid & (bin_idx == k)
        if not m.any():
            continue
        u_num.append(float(u_np[m].mean()))
        u_ana.append(float(u_ana_cell[m].mean()))
        cells.append(int(m.sum()))
        lam_avg.append(float(lam_e[m].mean()))
    u_num = np.array(u_num)
    u_ana = np.array(u_ana)
    central = np.abs(u_ana) > 0.2 * abs(U_c)
    max_rel_bin = (
        float(np.max(np.abs(u_num[central] - u_ana[central]) / np.abs(u_ana[central])) * 100.0)
        if central.any() else float("nan")
    )
    m_c = fluid & (np.abs(u_ana_cell) > 0.2 * abs(U_c))
    max_rel_cell = (
        float(np.max(np.abs(u_np[m_c] - u_ana_cell[m_c]) / np.abs(u_ana_cell[m_c])) * 100.0)
        if m_c.any() else float("nan")
    )
    w = np.where(fluid, 1.0, 0.0)
    l2 = float(np.linalg.norm(np.sqrt(w) * (u_np - u_ana_cell)) / np.linalg.norm(np.sqrt(w) * u_ana_cell))
    return {
        "max_rel_bin_central_pct": max_rel_bin,
        "max_rel_err_central_pct": max_rel_cell,
        "l2_rel_err": l2,
        "u_center": float(u_np[fluid].max()),
        "bins": [round(float(v), 8) for v in u_num],
        "bins_ana": [round(float(v), 8) for v in u_ana],
        "bins_lam": [round(float(v), 4) for v in lam_avg],
        "bins_cells": cells,
    }


def run_case(a, b, tau, u_in, min_steps, max_steps, out_path, device, seed=0, L_over_a=6):
    torch.manual_seed(seed)
    nu = (tau - 0.5) / 3.0
    ny, nz, nx, yc, zc, lam, fluid2d, wall_mask = ellipse_setup(a, b, L_over_a, device)
    rho_out = 1.0
    U_max_init = 2.0 * u_in
    lam3 = lam.unsqueeze(-1)
    ux0 = torch.where(fluid2d.unsqueeze(-1), U_max_init * (1.0 - lam3), torch.zeros_like(lam3))
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = ux0.expand(nz, ny, nx)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)

    x_meas = nx // 2
    x_dev = nx - 8
    t0 = time.time()
    umax_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        f = bounce_back_cells_3d(f, wall_mask)
        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            umax_hist.append(float(ux[:, :, x_meas].max().item()))
            if step >= min_steps and len(umax_hist) >= 10:
                recent = umax_hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    acc_meas = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho0 = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_m = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_d = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    for _ in range(400):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        f = bounce_back_cells_3d(f, wall_mask)
        rho, ux, _, _ = macroscopic3d(f)
        acc_meas += ux[:, :, x_meas]
        acc_rho0 += rho[:, :, 0]
        acc_rho_m += rho[:, :, x_meas]
        acc_rho_d += rho[:, :, x_dev]
    for t in (acc_meas, acc_rho0, acc_rho_m, acc_rho_d):
        t /= 400.0

    fluid_np = fluid2d.cpu().numpy()
    u_np = acc_meas.cpu().numpy().astype(np.float64)
    Q = float(u_np[fluid_np].sum())
    rho0_m = float(acc_rho0[fluid2d].mean().item())
    rho_m = float(acc_rho_m[fluid2d].mean().item())
    rho_d = float(acc_rho_d[fluid2d].mean().item())
    G_fd = (rho_m - rho_d) * CS2 / (x_dev - x_meas)
    G_in = (rho0_m - rho_d) * CS2 / x_dev
    G_full = (rho0_m - rho_out) * CS2 / nx

    U_c = 2.0 * u_in
    inv = invert_two_observable(Q, G_fd, U_c, nu)
    a_e, b_e = inv["a_e"], inv["b_e"]
    M_d = digital_M(fluid_np, a_e, b_e, yc, zc) if inv["ok"] else float("nan")
    U_c_p = Q / M_d if inv["ok"] and M_d > 0 else float("nan")

    prof = profile_compare(u_np, fluid_np, a_e, b_e, U_c_p, yc, zc) if inv["ok"] else {}

    # s^Q reference (reproduce the original pending result for comparison)
    s_Q = math.sqrt(2.0 * Q / (math.pi * a * b * U_c))
    prof_sq = profile_compare(u_np, fluid_np, a * s_Q, b * s_Q, U_c / s_Q**2, yc, zc)

    result = {
        "case": "poiseuille_3d_ellipse_hydraulic",
        "a": a, "b": b, "nx": nx, "ny": ny, "nz": nz,
        "tau": tau, "nu": nu, "u_in": u_in, "rho_out": rho_out,
        "n_steps": step, "steady": steady, "elapsed_s": round(elapsed, 1),
        "Q": Q,
        "rho_in": rho0_m, "rho_meas": rho_m, "rho_dev": rho_d,
        "G_fd": G_fd, "G_in": G_in, "G_full": G_full,
        "U_c_imposed": U_c,
        "inversion": inv,
        "a_e": a_e, "b_e": b_e,
        "a_e_minus_a": a_e - a if inv["ok"] else float("nan"),
        "b_e_minus_b": b_e - b if inv["ok"] else float("nan"),
        "M_d": M_d, "U_c_prime": U_c_p,
        "u_center": float(u_np[fluid_np].max()),
        "s_Q": s_Q,
        "hydraulic": prof,
        "sQ_reference": {
            "s_Q": s_Q,
            "max_rel_bin_central_pct": prof_sq.get("max_rel_bin_central_pct"),
            "max_rel_err_central_pct": prof_sq.get("max_rel_err_central_pct"),
            "l2_rel_err": prof_sq.get("l2_rel_err"),
        },
        "finite": bool(torch.isfinite(f).all().item()),
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=int)
    ap.add_argument("out_json", type=str)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u-in", type=float, default=0.02)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    b = max(1, int(round(args.a / args.ratio)))
    r = run_case(args.a, b, args.tau, args.u_in, args.min_steps, args.max_steps,
                 args.out_json, device, args.seed)
    print(json.dumps({k: r[k] for k in
                      ["a", "b", "nx", "n_steps", "steady", "Q", "G_fd", "G_in", "G_full",
                       "a_e", "b_e", "a_e_minus_a", "b_e_minus_b", "M_d", "U_c_prime",
                       "u_center", "s_Q"]}, indent=2), flush=True)
    print("HYDRAULIC_MAX_BIN=" + str(r["hydraulic"].get("max_rel_bin_central_pct")), flush=True)
    print("HYDRAULIC_MAX_CELL=" + str(r["hydraulic"].get("max_rel_err_central_pct")), flush=True)
    print("SQ_MAX_BIN=" + str(r["sQ_reference"].get("max_rel_bin_central_pct")), flush=True)
    print("FULL_RESULT_JSON_START", flush=True)
    print(json.dumps(r), flush=True)
    print("FULL_RESULT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
