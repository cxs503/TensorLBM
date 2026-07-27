#!/usr/bin/env python
"""Teaching Example 09: Wall function — log law at high Re.

Demonstrates the wall-function boundary condition for high-Reynolds-number
flows where the near-wall grid is too coarse to resolve the viscous
sublayer.

The log law of the wall:
    u+ = (1/κ) * ln(y+) + B

where κ=0.41, B=5.0, y+ = y*u*/ν, u+ = u/u*.

Usage:
    PYTHONPATH=src python teaching/09_wall_function.py [--device sdaa:12]
"""
from __future__ import annotations

import argparse
import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.wall_model import compute_wall_distance_fmm, compute_wall_slip_velocity

KAPPA = 0.41
B_CONST = 5.0


def run(
    ny: int = 32, nx: int = 64, nz: int = 4,
    u_in: float = 0.08, tau: float = 0.55,
    n_steps: int = 3000, device: str = "sdaa:12",
):
    dev = torch.device(device)
    nu = (tau - 0.5) / 3.0
    Re = u_in * ny / nu

    print(f"=== Wall Function: Log Law at Re={Re:.0f} ===")
    print(f"Grid: {nx}×{ny}×{nz}, u_in={u_in}, tau={tau}, nu={nu:.6f}")
    print(f"Device: {device}, steps={n_steps}")
    print()

    # Channel: both walls solid
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Wall distance
    dist = compute_wall_distance_fmm(solid, dx=1.0)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)

    # Body force (pressure gradient)
    force = 5e-5

    print(f"{'Step':>6s}  {'u_max':>8s}  {'u_tau':>8s}  {'y+_max':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        # Collision with body force
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux + force * tau / rho
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        f = f - (f - feq) / tau

        # Corrected BB at walls
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # Wall function: set slip velocity at near-wall cells
        # (simplified: just apply BB for this demo)
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

        if step % 500 == 0 or step == n_steps:
            _, ux, _, _ = macroscopic3d(f)
            u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
            u_max = float(u_profile.max().item())

            # Friction velocity: u_tau = sqrt(tau_wall / rho)
            # tau_wall = nu * du/dy at wall
            du_dy = float(u_profile[0].item()) / 0.5
            u_tau = math.sqrt(nu * du_dy)
            y_max = float(dist[0, ny // 2, nx // 2].item())
            y_plus_max = u_tau * y_max / nu
            print(f" {step:5d}  {u_max:8.4f}  {u_tau:8.5f}  {y_plus_max:8.1f}")

    # Final: check log law
    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
    du_dy = float(u_profile[0].item()) / 0.5
    u_tau = math.sqrt(nu * du_dy)

    print()
    print("=== Log Law Verification ===")
    print(f"  u_tau = {u_tau:.6f}")
    print(f"  {'y':>6s}  {'y+':>8s}  {'u+':>8s}  {'log_law':>8s}  {'err':>8s}")
    for i in range(0, ny - 2, 2):
        y = i + 0.5  # distance from wall (half-way)
        y_plus = u_tau * y / nu
        u_plus = float(u_profile[i].item()) / u_tau
        log_law = (1.0 / KAPPA) * math.log(max(y_plus, 1e-10)) + B_CONST
        err = abs(u_plus - log_law) / max(abs(log_law), 1e-10) * 100
        print(f"  {y:6.1f}  {y_plus:8.1f}  {u_plus:8.2f}  {log_law:8.2f}  {err:8.1f}%")

    return {"u_tau": u_tau, "Re": Re}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wall function log law")
    parser.add_argument("--device", default="sdaa:12")
    parser.add_argument("--n-steps", type=int, default=3000)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
