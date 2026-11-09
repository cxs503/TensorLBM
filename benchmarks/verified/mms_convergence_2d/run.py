#!/usr/bin/env python3
"""B-MMS: 2D MMS (Method of Manufactured Solutions) convergence for LBM.

Physics
-------
Manufactured exact solution of the incompressible Navier–Stokes on a
doubly-periodic square domain [0, L] x [0, L]:

    u(x,y) = U0 * sin(2*pi*x/L) * cos(2*pi*y/L)
    v(x,y) = -U0 * cos(2*pi*x/L) * sin(2*pi*y/L)

This velocity field is exactly divergence-free.  Substituting into the
steady incompressible N-S with body force gives the analytic source term
F(x,y) that we inject via the Guo (2002) second-order forcing scheme.

Convergence measurement
-----------------------
Three grids N = {N1, N2, N3} with refinement ratio r = 2.
L2 error of (u, v) vs the manufactured solution at each grid.
Observed order:  p = ln(E_coarse / E_fine) / ln(r)

References
----------
Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the
    forcing term in the lattice Boltzmann method." Phys. Rev. E 65, 046308.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from tensorlbm.d2q9 import C, W, equilibrium, macroscopic

DEVICE = torch.device("cpu")
TWO_PI = 2.0 * math.pi


def stream_periodic(f: torch.Tensor) -> torch.Tensor:
    """Fast periodic streaming using torch.roll."""
    out = torch.empty_like(f)
    for i in range(9):
        cy, cx = int(C[i, 1]), int(C[i, 0])
        out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return out


def guo_source_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Guo (2002) discrete source term for arbitrary 2D body force (fx, fy)."""
    device = ux.device
    c = C.to(device=device, dtype=torch.float32)
    w = W.to(device=device, dtype=torch.float32).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    ux_b = ux.unsqueeze(0)
    uy_b = uy.unsqueeze(0)
    fx_b = fx.unsqueeze(0)
    fy_b = fy.unsqueeze(0)

    cdotu = cx * ux_b + cy * uy_b
    cdotf = cx * fx_b + cy * fy_b

    coeff = 1.0 - 1.0 / (2.0 * tau)
    S = coeff * w * (
        3.0 * ((cx - ux_b) * fx_b + (cy - uy_b) * fy_b)
        + 9.0 * cdotu * cdotf
    )
    return S


def manufactured_velocity(x: torch.Tensor, y: torch.Tensor, U0: float, L: float):
    """Return (u_exact, v_exact) for the manufactured solution."""
    kx = TWO_PI * x / L
    ky = TWO_PI * y / L
    u_exact = U0 * torch.sin(kx) * torch.cos(ky)
    v_exact = -U0 * torch.cos(kx) * torch.sin(ky)
    return u_exact, v_exact


def manufactured_force(x: torch.Tensor, y: torch.Tensor, U0: float, nu: float, L: float):
    """Analytic body force F = (u.grad)u - nu*laplacian(u)."""
    k = TWO_PI / L
    kx = k * x
    ky = k * y

    # Convective: (u.grad)u
    conv_x = 0.5 * U0 * U0 * k * torch.sin(2.0 * kx)
    conv_y = 0.5 * U0 * U0 * k * torch.sin(2.0 * ky)

    # Viscous: -nu*laplacian(u) = +nu*k^2*u for sin/cos
    visc_x = nu * k * k * U0 * torch.sin(kx) * torch.cos(ky)
    visc_y = -nu * k * k * U0 * torch.cos(kx) * torch.sin(ky)

    fx = conv_x + visc_x
    fy = conv_y + visc_y
    return fx, fy


def run_mms_case(
    N: int,
    tau: float,
    U0: float,
    steps: int,
    enable_forcing: bool = True,
) -> dict:
    """Run MMS on one grid of size N x N (doubly periodic)."""
    torch.manual_seed(0)
    L = 1.0
    nu = (tau - 0.5) / 3.0
    ny = N
    nx = N

    # Physical coordinates at cell centers
    dx = L / N
    x_1d = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y_1d = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    yy, xx = torch.meshgrid(y_1d, x_1d, indexing="ij")

    # Exact solution
    u_exact, v_exact = manufactured_velocity(xx, yy, U0, L)

    # Source term
    fx, fy = manufactured_force(xx, yy, U0, nu, L)

    # Convert to float32
    fx32 = fx.float()
    fy32 = fy.float()
    u_exact32 = u_exact.float()
    v_exact32 = v_exact.float()

    # Initialize from REST (more stable than from exact solution)
    rho = torch.ones((ny, nx), dtype=torch.float32)
    ux = torch.zeros((ny, nx), dtype=torch.float32)
    uy = torch.zeros((ny, nx), dtype=torch.float32)
    f = equilibrium(rho, ux, uy)

    t0 = time.perf_counter()
    for step in range(steps):
        rho, ux, uy = macroscopic(f)
        if enable_forcing:
            # Guo velocity correction
            ux_star = ux + 0.5 * fx32 / rho
            uy_star = uy + 0.5 * fy32 / rho
            feq = equilibrium(rho, ux_star, uy_star)
            S = guo_source_2d(ux_star, uy_star, fx32, fy32, tau)
            f = f + (feq - f) / tau + S
        else:
            # No forcing: just collide
            feq = equilibrium(rho, ux, uy)
            f = f + (feq - f) / tau
        f = stream_periodic(f)

    elapsed = time.perf_counter() - t0

    # Final macroscopic
    rho, ux, uy = macroscopic(f)
    if enable_forcing:
        ux = ux + 0.5 * fx32 / rho
        uy = uy + 0.5 * fy32 / rho

    # L2 error
    err_u = (ux - u_exact32).double()
    err_v = (uy - v_exact32).double()
    l2_u = torch.sqrt(torch.mean(err_u ** 2)).item()
    l2_v = torch.sqrt(torch.mean(err_v ** 2)).item()
    l2_combined = math.sqrt(l2_u ** 2 + l2_v ** 2)

    u_ref = U0
    l2_rel = l2_combined / u_ref if u_ref > 0 else l2_combined

    linf_u = err_u.abs().max().item()
    linf_v = err_v.abs().max().item()
    linf_combined = max(linf_u, linf_v)

    return {
        "N": N,
        "dx": dx,
        "tau": tau,
        "nu": nu,
        "U0": U0,
        "steps": steps,
        "elapsed_s": round(elapsed, 2),
        "l2_u": l2_u,
        "l2_v": l2_v,
        "l2_combined": l2_combined,
        "l2_rel": l2_rel,
        "linf_u": linf_u,
        "linf_v": linf_v,
        "linf_combined": linf_combined,
        "finite": bool(torch.isfinite(ux).all().item() and torch.isfinite(uy).all().item()),
    }


def compute_observed_order(errors: list[float], ratios: list[float]) -> list[float]:
    """Compute observed order from errors at consecutive grids."""
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0 and ratios[i] > 1:
            p = math.log(errors[i] / errors[i + 1]) / math.log(ratios[i])
            orders.append(p)
        else:
            orders.append(float("nan"))
    return orders


def run_scan(
    grids: list[int],
    tau: float,
    U0: float,
    steps_per_grid: int | None = None,
    enable_forcing: bool = True,
) -> dict:
    """Run MMS on multiple grids and compute convergence order."""
    results = []
    for N in grids:
        if steps_per_grid is not None:
            steps = steps_per_grid
        else:
            # Diffusive scaling: steps ~ N^2 to ensure convergence
            steps = max(2000, N * N)
        r = run_mms_case(N, tau, U0, steps, enable_forcing=enable_forcing)
        results.append(r)
        print(
            f"N={N:4d} dx={r['dx']:.5f} l2_rel={r['l2_rel']:.6e} "
            f"linf={r['linf_combined']:.6e} steps={r['steps']} "
            f"finite={r['finite']} elapsed={r['elapsed_s']}s",
            flush=True,
        )

    # Compute observed orders
    errors_l2 = [r["l2_rel"] for r in results]
    errors_linf = [r["linf_combined"] for r in results]
    ratios = [grids[i] / grids[i - 1] for i in range(1, len(grids))]

    orders_l2 = compute_observed_order(errors_l2, ratios)
    orders_linf = compute_observed_order(errors_linf, ratios)

    valid_orders_l2 = [p for p in orders_l2 if not math.isnan(p)]
    avg_order_l2 = sum(valid_orders_l2) / len(valid_orders_l2) if valid_orders_l2 else float("nan")

    valid_orders_linf = [p for p in orders_linf if not math.isnan(p)]
    avg_order_linf = sum(valid_orders_linf) / len(valid_orders_linf) if valid_orders_linf else float("nan")

    return {
        "case": "mms_convergence_2d",
        "lattice": "D2Q9",
        "collision": "bgk",
        "forcing": "guo" if enable_forcing else "none",
        "boundaries": "doubly-periodic",
        "manufactured_solution": "sinusoidal_divergence_free",
        "tau": tau,
        "U0": U0,
        "grids": grids,
        "per_grid": results,
        "convergence": {
            "errors_l2_rel": errors_l2,
            "errors_linf": errors_linf,
            "grid_ratios": ratios,
            "observed_orders_l2": orders_l2,
            "observed_orders_linf": orders_linf,
            "avg_order_l2": avg_order_l2,
            "avg_order_linf": avg_order_linf,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="MMS 2D convergence (D2Q9 BGK + Guo forcing)")
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 32, 48])
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--U0", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=None, help="Fixed steps (default: auto)")
    ap.add_argument("--no-forcing", action="store_true", help="Disable Guo forcing (negative control)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = run_scan(
        args.grids, args.tau, args.U0, args.steps,
        enable_forcing=not args.no_forcing,
    )

    out_json = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out_json)
    print(out_json)


if __name__ == "__main__":
    main()
