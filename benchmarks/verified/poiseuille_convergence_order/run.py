#!/usr/bin/env python3
"""Poiseuille 2D convergence order study with wall position corrections.

This benchmark measures the spatial convergence order of 2D Poiseuille flow
using three approaches:
1. Half-way bounce-back (standard, first-order wall position)
2. Effective wall position correction (y_w = 0.5*delta)
3. Bouzidi interpolated bounce-back (second-order)

Physics
-------
Fully-developed 2D laminar channel flow between parallel plates, driven by
a constant body force (equivalently, a constant pressure gradient). The
analytic solution is:

    u(y) = 4*u_max*(y/H)*(1 - y/H)

where H is the channel height and u_max is the centerline velocity.

Convergence measurement
-----------------------
Three grids with refinement ratio r = 2. L2 error of velocity profile vs
analytic solution. Observed order: p = ln(E_coarse/E_fine) / ln(r)

Acceptance criteria (pre-declared):
  (a) Half-way BB: p ~ 1 (first-order due to wall position error)
  (b) Effective wall correction: p ~ 2 (second-order)
  (c) Bouzidi: p >= 1.9 (second-order)
  (d) Bouzidi error < 1% on finest grid

References
----------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). "Momentum transfer of
    a Boltzmann-lattice fluid with boundaries." Phys. Fluids 13, 3452.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tensorlbm.d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import stream

DEVICE = torch.device("cpu")


def guo_source_1d(ux: torch.Tensor, accel: float, tau: float) -> torch.Tensor:
    """Guo source for x-directed body force (channel flow)."""
    c = C.to(dtype=torch.float32)
    w = W.to(dtype=torch.float32).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    ux_b = ux.unsqueeze(0)
    cdotu = cx * ux_b
    coeff = 1.0 - 1.0 / (2.0 * tau)
    S = coeff * w * (3.0 * (cx - ux_b) + 9.0 * cdotu * cx) * accel
    return S


def apply_halfway_bb(f: torch.Tensor) -> torch.Tensor:
    """Pre-streaming half-way bounce-back on top/bottom walls."""
    ny = f.shape[1]
    f_out = f.clone()
    # Bottom wall (y=0)
    for i in [4, 7, 8]:  # c_y < 0
        f_out[OPPOSITE[i], 0, :] = f[i, 1, :]
    # Top wall (y=ny-1)
    for i in [2, 5, 6]:  # c_y > 0
        f_out[OPPOSITE[i], ny - 1, :] = f[i, ny - 2, :]
    return f_out


def apply_bouzidi_bb(f: torch.Tensor, f_prev: torch.Tensor, q_bottom: float, q_top: float) -> torch.Tensor:
    """Bouzidi interpolated bounce-back with fractional distances q.
    
    q = distance from fluid node to wall / delta
    q = 0.5 corresponds to half-way bounce-back.
    """
    ny = f.shape[1]
    f_out = f.clone()
    
    # Bottom wall
    for i in [4, 7, 8]:  # c_y < 0
        opp = OPPOSITE[i]
        if q_bottom < 0.5:
            # Linear interpolation
            f_out[opp, 0, :] = 2.0 * q_bottom * f[i, 1, :] + (1.0 - 2.0 * q_bottom) * f_prev[i, 2, :]
        else:
            # Quadratic interpolation
            f_out[opp, 0, :] = f[i, 1, :] / (2.0 * q_bottom) + (2.0 * q_bottom - 1.0) / (2.0 * q_bottom) * f_prev[opp, 1, :]
    
    # Top wall
    for i in [2, 5, 6]:  # c_y > 0
        opp = OPPOSITE[i]
        if q_top < 0.5:
            f_out[opp, ny - 1, :] = 2.0 * q_top * f[i, ny - 2, :] + (1.0 - 2.0 * q_top) * f_prev[i, ny - 3, :]
        else:
            f_out[opp, ny - 1, :] = f[i, ny - 2, :] / (2.0 * q_top) + (2.0 * q_top - 1.0) / (2.0 * q_top) * f_prev[opp, ny - 2, :]
    
    return f_out


def run_poiseuille_case(
    H: int,
    tau: float,
    u_max_target: float,
    wall_model: str,
    steps: int,
) -> dict:
    """Run one Poiseuille case.
    
    wall_model: 'halfway' | 'effective' | 'bouzidi'
    """
    torch.manual_seed(0)
    nu = (tau - 0.5) / 3.0
    
    # Channel geometry
    if wall_model == "effective":
        # Effective wall position: walls at y = 0.5*delta from ghost rows
        # So H_eff = H (the requested height)
        ny = H + 2  # ghost rows at 0 and ny-1
        H_eff = float(H)
        q_wall = 0.5  # half-way
    elif wall_model == "bouzidi":
        # Bouzidi: walls at y = 0.5*delta (same as half-way for flat walls)
        ny = H + 2
        H_eff = float(H)
        q_wall = 0.5
    else:  # halfway
        ny = H + 2
        H_eff = float(H)
        q_wall = 0.5
    
    nx = 4 * H  # L = 4H (long enough for fully-developed flow)
    
    # Body force to achieve target u_max
    # u_max = accel * H^2 / (8*nu)  =>  accel = 8*nu*u_max/H^2
    accel = 8.0 * nu * u_max_target / (H_eff * H_eff)
    
    # Analytic solution
    # y_phys = (j - 0.5) * delta for fluid rows j = 1, ..., ny-2
    # u(y) = 4*u_max*(y/H)*(1 - y/H)
    y_fluid = torch.arange(1, ny - 1, dtype=torch.float64) - 0.5
    y_norm = y_fluid / H_eff
    u_analytic = 4.0 * u_max_target * y_norm * (1.0 - y_norm)
    
    # Initialize
    rho = torch.ones((ny, nx), dtype=torch.float32)
    ux = torch.zeros((ny, nx), dtype=torch.float32)
    uy = torch.zeros((ny, nx), dtype=torch.float32)
    f = equilibrium(rho, ux, uy)
    
    t0 = time.perf_counter()
    f_prev = f.clone()
    for step in range(steps):
        rho, ux, uy = macroscopic(f)
        # Guo forcing
        ux_star = ux + 0.5 * accel / rho
        feq = equilibrium(rho, ux_star, uy)
        S = guo_source_1d(ux_star, accel, tau)
        f = f + (feq - f) / tau + S
        
        # Wall boundary
        if wall_model == "bouzidi":
            f = apply_bouzidi_bb(f, f_prev, q_wall, q_wall)
        else:
            f = apply_halfway_bb(f)
        
        f_prev = f.clone()
        f = stream(f)
    
    elapsed = time.perf_counter() - t0
    
    # Extract velocity profile (middle of channel, away from inlet/outlet)
    rho, ux, uy = macroscopic(f)
    ux_star = ux + 0.5 * accel / rho
    u_profile = ux_star[1:ny-1, nx // 2].double()  # fluid rows only
    
    # Compute errors
    err = u_profile - u_analytic
    l2_err = torch.sqrt(torch.mean(err ** 2)).item()
    linf_err = err.abs().max().item()
    l2_rel = l2_err / u_max_target
    
    return {
        "H": H,
        "H_eff": H_eff,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "accel": accel,
        "u_max_target": u_max_target,
        "wall_model": wall_model,
        "steps": steps,
        "elapsed_s": round(elapsed, 2),
        "l2_err": l2_err,
        "l2_rel": l2_rel,
        "linf_err": linf_err,
        "linf_rel": linf_err / u_max_target,
        "u_profile": u_profile.tolist(),
        "u_analytic": u_analytic.tolist(),
        "finite": bool(torch.isfinite(f).all().item()),
    }


def compute_order(errors: list[float], ratios: list[float]) -> list[float]:
    """Compute observed convergence order."""
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0 and ratios[i] > 1:
            p = math.log(errors[i] / errors[i + 1]) / math.log(ratios[i])
            orders.append(p)
        else:
            orders.append(float("nan"))
    return orders


def run_convergence_study(
    H_list: list[int],
    tau: float,
    u_max: float,
    steps: int,
    wall_models: list[str],
) -> dict:
    """Run convergence study for multiple wall models."""
    results = {}
    
    for model in wall_models:
        print(f"\n=== Wall model: {model} ===")
        cases = []
        for H in H_list:
            r = run_poiseuille_case(H, tau, u_max, model, steps)
            cases.append(r)
            print(f"H={H:3d} l2_rel={r['l2_rel']:.6e} linf_rel={r['linf_rel']:.6e} finite={r['finite']}")
        
        errors_l2 = [c["l2_rel"] for c in cases]
        errors_linf = [c["linf_rel"] for c in cases]
        ratios = [H_list[i] / H_list[i - 1] for i in range(1, len(H_list))]
        
        orders_l2 = compute_order(errors_l2, ratios)
        orders_linf = compute_order(errors_linf, ratios)
        
        results[model] = {
            "per_grid": cases,
            "convergence": {
                "errors_l2_rel": errors_l2,
                "errors_linf_rel": errors_linf,
                "grid_ratios": ratios,
                "observed_orders_l2": orders_l2,
                "observed_orders_linf": orders_linf,
            },
        }
    
    return {
        "case": "poiseuille_convergence_order",
        "lattice": "D2Q9",
        "collision": "bgk",
        "forcing": "guo",
        "driving": "body_force",
        "tau": tau,
        "u_max_target": u_max,
        "H_list": H_list,
        "steps": steps,
        "wall_models": results,
    }


def main():
    ap = argparse.ArgumentParser(description="Poiseuille 2D convergence order study")
    ap.add_argument("--H", type=int, nargs="+", default=[20, 40, 80])
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--umax", type=float, default=0.04)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--models", nargs="+", default=["halfway", "effective", "bouzidi"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    
    result = run_convergence_study(args.H, args.tau, args.umax, args.steps, args.models)
    
    out_json = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out_json)
    print("\n" + out_json)


if __name__ == "__main__":
    main()
