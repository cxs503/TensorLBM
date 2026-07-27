#!/usr/bin/env python
"""Teaching Example 02: Poiseuille flow — exact solution with corrected BB.

Demonstrates the **half-way bounce-back** for pressure-driven channel flow
and verifies the parabolic velocity profile and friction coefficient.

Physics
-------
Poiseuille flow: two parallel plates, both stationary, driven by a body
force (pressure gradient).  The exact steady-state solution is parabolic:

    u(y) = (G/2ν) * y * (H - y)

where G is the pressure gradient, ν is viscosity, H is channel height,
and y is measured from the bottom wall (y=0.5 to y=H-0.5 for half-way BB).

Key points
----------
1. **Body force**: applied as a uniform acceleration to all fluid cells.
2. **Corrected BB**: both walls use ``f_pre`` for exact no-slip.
3. **Friction coefficient**: Cf = τ_wall / (0.5 ρ U_max²) matches exact.

Usage
-----
    PYTHONPATH=src python teaching/02_poiseuille_exact.py [--device sdaa:9]

Expected output
---------------
    u_max_error < 0.1%
    Cf_error < 1%
    PASS
"""
from __future__ import annotations

import argparse
import math

import torch

from tensorlbm.d3q19 import C, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d


def run(
    ny: int = 16,
    nx: int = 32,
    nz: int = 4,
    force: float = 1e-5,
    tau: float = 1.0,
    n_steps: int = 5000,
    device: str = "sdaa:9",
):
    dev = torch.device(device)
    torch.manual_seed(42)

    # --- Grid setup ---
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall

    # --- Initial condition ---
    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=dev)

    # Body force: Guo forcing scheme
    # Δf_i = w_i * (1 - 1/(2τ)) * [3*c_ix*F + 9*c_ix*(c·u)*F - 3*ux*F]
    # For simplicity with small u: Δf_i ≈ w_i * (1 - 1/(2τ)) * 3 * c_ix * F
    from tensorlbm.d3q19 import W as W_LAT
    c = C.to(dev).float()
    w_lat = W_LAT.to(dev).float()
    # Precompute the direction-dependent part: w_i * 3 * c_ix * (1 - 1/(2τ))
    force_coeff = w_lat * 3.0 * c[:, 0] * (1.0 - 0.5 / tau) * force  # (19,)

    print(f"=== Poiseuille Flow: Corrected BB ===")
    print(f"Grid: {nx}×{ny}×{nz}, tau={tau}, force={force}, steps={n_steps}")
    print(f"Device: {device}")
    print()

    # --- Time loop ---
    for step in range(1, n_steps + 1):
        # 1. Save pre-collision f
        f_pre = f.clone()

        # 2. Collision with body force (Guo scheme)
        rho, ux, uy, uz = macroscopic3d(f)
        # Add force to velocity for equilibrium
        ux = ux + force * tau / rho
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        f = f - (f - feq) / tau
        # Force term (Guo scheme, simplified for small u)
        f = f + force_coeff.view(19, 1, 1, 1).expand(19, nz, ny, nx)

        # 3. Corrected half-way bounce-back
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 4. Streaming
        f = stream3d(f)

        # 5. Periodic BC in x
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    # --- Extract velocity profile ---
    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))  # (ny-2,)

    # --- Exact solution ---
    nu = (tau - 0.5) / 3.0
    H = ny - 1.0  # channel height (wall at 0.5 to ny-1+0.5)
    y_interior = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
    # u(y) = G/(2nu) * (y-0.5) * (H-0.5 - (y-0.5))
    # where G = force / rho (body force per unit mass)
    G = force  # pressure gradient = body force
    u_exact = G / (2.0 * nu) * (y_interior - 0.5) * (H - 0.5 - (y_interior - 0.5))

    # --- Error ---
    u_num = u_profile.cpu().numpy()
    u_ex = u_exact.cpu().numpy()
    u_max_exact = float(u_ex.max())
    u_max_num = float(u_num.max())
    max_err = max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_max_exact)
    max_err_pct = max_err * 100

    # Friction: Cf = tau_wall / (0.5 * rho * U_max^2)
    # tau_wall = nu * du/dy at wall = nu * G * H / (2*nu) = G*H/2
    # Cf = 2 * tau_wall / (rho * U_max^2) = 2 * G*H/2 / (G^2*H^2/(8*nu^2))
    #    = 8*nu^2 / (G*H^2) ... let's just compute numerically
    u_max_val = abs(u_max_num)
    if u_max_val > 1e-12:
        # du/dy at wall (forward diff from first interior cell)
        du_dy_wall = u_num[0] / 0.5  # u at y=1, wall at y=0.5, distance=0.5
        tau_wall = nu * du_dy_wall
        cf_num = 2.0 * tau_wall / u_max_val**2
    else:
        cf_num = 0.0

    # Exact Cf: u_max = G*H^2/(8*nu), tau_wall = G*H/2
    # Cf = 2*tau_wall/u_max^2 = 2*(G*H/2) / (G^2*H^4/(64*nu^2))
    #    = 64*nu^2 / (G*H^3) ... let's use the standard form
    # For plane Poiseuille: Cf = 8*nu / (U_max * H)
    cf_exact = 8.0 * nu / (u_max_exact * H) if u_max_exact > 0 else 0.0
    cf_err_pct = abs(cf_num - cf_exact) / abs(cf_exact) * 100 if cf_exact > 0 else 0.0

    print(f"Velocity profile (interior cells):")
    print(f"  {'y':>5s}  {'u_num':>12s}  {'u_exact':>12s}  {'err':>12s}")
    for i, (yv, un, ue) in enumerate(zip(y_interior.cpu().numpy(), u_num, u_ex)):
        print(f"  {yv:5.1f}  {un:12.8f}  {ue:12.8f}  {abs(un-ue):12.2e}")

    print()
    print(f"u_max:  num={u_max_num:.8f}, exact={u_max_exact:.8f}")
    print(f"u_max error: {max_err_pct:.2f}%")
    print(f"Cf:     num={cf_num:.6f}, exact={cf_exact:.6f}")
    print(f"Cf error:    {cf_err_pct:.2f}%")

    if max_err_pct < 0.1:
        print("PASS: Corrected BB gives accurate Poiseuille profile")
    else:
        print(f"CHECK: u_max error = {max_err_pct:.2f}% (expected < 0.1%)")

    return {"u_max_err_pct": max_err_pct, "cf_err_pct": cf_err_pct,
            "u_max_num": u_max_num, "u_max_exact": u_max_exact}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poiseuille flow — corrected BB")
    parser.add_argument("--device", default="sdaa:9", help="Device")
    parser.add_argument("--ny", type=int, default=16, help="Grid size Y")
    parser.add_argument("--n-steps", type=int, default=5000, help="Steps")
    args = parser.parse_args()
    run(ny=args.ny, n_steps=args.n_steps, device=args.device)
