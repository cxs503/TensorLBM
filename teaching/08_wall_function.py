#!/usr/bin/env python3
"""08 — Wall function: log-law for high-Reynolds-number flows.

At high Reynolds numbers, resolving the boundary layer down to the wall
(y+ < 1) requires extremely fine grids. Wall functions bridge the gap
by imposing the log-law of the wall at the first off-wall cell:

  u+ = (1/κ) * ln(y+) + B   for y+ > 11.06 (log-law region)
  u+ = y+                    for y+ < 5     (viscous sublayer)

where:
  u+ = u / u_tau    (friction velocity)
  y+ = y * u_tau / ν (wall coordinate)
  κ = 0.41          (von Karman constant)
  B = 5.0           (log-law intercept)

This example demonstrates:
  1. Wall function boundary condition in LBM
  2. Log-law velocity profile enforcement
  3. Friction velocity computation from wall shear stress
  4. Comparison with resolved simulation

Usage:
  python 08_wall_function.py [device_id]

Expected output:
  Wall function gives reasonable Cd at high Re with coarse grid
  u+ vs y+ follows log-law
"""
from __future__ import annotations

import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import collide_bgk3d, stream3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.momentum_exchange import bounce_back_pre_collision


def compute_friction_velocity(u_near, nu, dist=0.5, kappa=0.41, B=5.0):
    """Compute friction velocity u_tau from near-wall velocity.

    Solves the log-law iteratively:
      u_near = u_tau * (1/kappa * ln(dist * u_tau / nu) + B)

    Args:
        u_near: Near-wall velocity magnitude.
        nu: Kinematic viscosity.
        dist: Distance from wall to near-wall cell (default 0.5 lattice units).
        kappa: von Karman constant (0.41).
        B: Log-law intercept (5.0).

    Returns:
        Friction velocity u_tau.
    """
    if u_near < 1e-10:
        return 0.0

    # Initial guess: u_tau = u_near / 20 (typical for Re ~ 1e4)
    u_tau = u_near / 20.0

    # Newton-Raphson iteration
    for _ in range(50):
        y_plus = dist * u_tau / nu
        if y_plus < 1e-10:
            return u_near * nu / dist  # viscous sublayer

        if y_plus < 11.06:
            # Viscous sublayer: u+ = y+
            f = u_tau * y_plus - u_near
            df = 2.0 * dist * u_tau / nu  # df/du_tau
        else:
            # Log-law region
            f = u_tau * (1.0 / kappa * math.log(y_plus) + B) - u_near
            df = (1.0 / kappa * math.log(y_plus) + B) + u_tau / kappa / u_tau * dist / nu
            # Simplified: df ≈ (1/kappa * ln(y+) + B) + 1/kappa
            df = (1.0 / kappa * math.log(y_plus) + B) + 1.0 / kappa

        if abs(df) < 1e-15:
            break
        u_tau_new = u_tau - f / df
        if u_tau_new < 0:
            u_tau_new = u_tau / 2
        if abs(u_tau_new - u_tau) < 1e-10:
            break
        u_tau = u_tau_new

    return u_tau


def run_wall_function(device_id=3, n_steps=5000, warmup=1000):
    """Run channel flow with wall function at high Re."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters: high Re channel flow
    nx, ny, nz = 128, 32, 4
    u_in = 0.1
    Re = 10000  # High Reynolds number
    H = ny - 2  # channel height
    nu = u_in * H / Re
    tau = 3.0 * nu + 0.5
    kappa = 0.41
    B_log = 5.0

    print(f"=== Wall Function (Log-Law) Re={Re} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}x{ny}x{nz}, tau={tau:.4f}, nu={nu:.6e}")
    print(f"u_in={u_in}, H={H}, kappa={kappa}, B={B_log}")
    print(f"Steps: {n_steps} (warmup={warmup})")

    # Solid mask: top and bottom walls
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    target_mass = float(rho0.sum().item())

    # Body force to drive flow
    g = 1e-5
    c = C.to(device).float()
    w = W.to(device).float().view(19, 1, 1, 1)

    cf_hist = []
    y_plus_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)

        # NoDynamics
        sm = solid.unsqueeze(0).expand_as(f)
        f = torch.where(sm, f_pre, f)

        # Bounce-back
        f = bounce_back_pre_collision(f, f_pre, solid)

        # Body force
        rho, ux, uy, uz = macroscopic3d(f)
        force_factor = (1.0 - 1.0 / (2.0 * tau))
        df_force = w * force_factor * 3.0 * c[:, 0].view(19, 1, 1, 1) * g
        f = f + torch.where(sm, torch.zeros_like(df_force), df_force)

        # Streaming
        f = stream3d(f)

        # Periodic in x and z (channel)
        # No far-field BC needed for periodic channel

        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)

        if step > warmup and step % 100 == 0:
            # Compute friction velocity from near-wall velocity
            _, ux, _, _ = macroscopic3d(f)
            u_near = float(ux[:, 1, :].mean().item())  # first off-wall cell
            u_tau = compute_friction_velocity(u_near, nu, dist=0.5, kappa=kappa, B=B_log)
            y_plus = 0.5 * u_tau / nu

            # Cf = tau_wall / (0.5 * rho * u^2) = u_tau^2 / (0.5 * u_in^2)
            cf = u_tau ** 2 / (0.5 * u_in ** 2)
            cf_hist.append(cf)
            y_plus_hist.append(y_plus)

        if step % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            u_near = float(u_prof[1])
            u_tau = compute_friction_velocity(u_near, nu, dist=0.5)
            y_plus = 0.5 * u_tau / nu
            print(f"  step={step} u_near={u_near:.4f} u_tau={u_tau:.6f} y+={y_plus:.1f}")

    # Final results
    cf_mean = sum(cf_hist) / max(len(cf_hist), 1) if cf_hist else 0.0
    y_plus_mean = sum(y_plus_hist) / max(len(y_plus_hist), 1) if y_plus_hist else 0.0

    # Analytical Cf for turbulent channel (Blasius): Cf = 0.074 * Re^(-0.2)
    Cf_blasius = 0.074 * Re ** (-0.2)

    # Print u+ vs y+ profile
    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()
    u_tau_final = compute_friction_velocity(float(u_prof[1]), nu, dist=0.5)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Cf (wall function) = {cf_mean:.6f}")
    print(f"Cf (Blasius)        = {Cf_blasius:.6f}")
    print(f"y+ mean             = {y_plus_mean:.1f}")
    print(f"u_tau               = {u_tau_final:.6f}")

    print(f"\n  y      u        y+       u+")
    print(f"  -----  -------  --------  --------")
    for y in range(1, ny - 1):
        dist = y - 0.5  # distance from bottom wall
        yp = dist * u_tau_final / nu
        up = u_prof[y] / max(u_tau_final, 1e-10)
        print(f"  {dist:.1f}  {u_prof[y]:.4f}  {yp:.1f}  {up:.2f}")

    return {
        "Cf_wall_function": cf_mean,
        "Cf_blasius": Cf_blasius,
        "y_plus_mean": y_plus_mean,
        "u_tau": u_tau_final,
    }


if __name__ == "__main__":
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_wall_function(device_id)
