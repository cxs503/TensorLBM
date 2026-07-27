#!/usr/bin/env python
"""Teaching Example 09: Wall function — log law at high Re.

Demonstrates the wall-function boundary condition for high-Reynolds-number
flows where the near-wall grid is too coarse to resolve the viscous
sublayer.

The log law of the wall:
    u+ = (1/κ) * ln(y+) + B
where κ=0.41, B=5.0, y+ = y*u*/ν, u+ = u/u*.

Uses the COMMON INTERFACE ONLY:
  - wall_function() from tensorlbm.wall_function_common
  - compute_u_tau() / compute_y_plus() from tensorlbm.wall_function_common
  - bounce_back_cells_3d(f_pre) for half-way BB
  - SurfaceMesh.from_gradient() for surface normals
  - drag_friction_integration() for friction drag

Usage:
    PYTHONPATH=src python teaching/09_wall_function.py [device_id]
"""
from __future__ import annotations

import sys
import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.wall_function_common import (
    wall_function, compute_u_tau, compute_y_plus,
)

KAPPA = 0.41
B_CONST = 5.0


def run(
    ny: int = 32, nx: int = 64, nz: int = 4,
    u_in: float = 0.08, tau: float = 0.55,
    n_steps: int = 4000, device_id: int = 19,
):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)
    nu = (tau - 0.5) / 3.0
    Re = u_in * ny / nu

    print(f"=== Wall Function: Log Law at Re={Re:.0f} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}x{ny}x{nz}, u_in={u_in}, tau={tau}, nu={nu:.6f}")
    print(f"Steps: {n_steps}")
    print()

    # Channel: both walls solid
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Near-wall mesh for friction computation
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)

    # Body force (pressure gradient)
    force = 5e-5
    c = C.to(dev).float()
    w_lat = W.to(dev).float()
    force_coeff = w_lat * 3.0 * c[:, 0] * (1.0 - 0.5 / tau) * force

    dpS = 0.5 * u_in ** 2 * ny * nz

    print(f"{'Step':>6s}  {'u_max':>8s}  {'u_tau':>8s}  {'y+_max':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision f (BB fix)
        f_pre = f.clone()

        # 2. Collision with body force (Guo scheme)
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux + force * tau / rho
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        f = f - (f - feq) / tau
        f = f + force_coeff.view(19, 1, 1, 1).expand(19, nz, ny, nx)

        # 3. Wall function correction (common interface)
        u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz)
        u_tau_field = compute_u_tau(u_mag, nu, y_val=0.5, wall_law="log")
        y_plus_field = compute_y_plus(u_tau_field, nu, y_val=0.5)
        f = wall_function(f, solid, u_tau_field, y_plus_field,
                          lattice="D3Q19", nu=nu, y_val=0.5,
                          rho=rho, ux=ux, uy=uy, uz=uz)

        # 4. Corrected half-way BB (common interface)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Streaming
        f = stream3d(f)

        # 6. Periodic BC in x
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

        if step % 500 == 0 or step == n_steps:
            _, ux, _, _ = macroscopic3d(f)
            u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
            u_max = float(u_profile.max().item())

            # Friction velocity from near-wall velocity
            du_dy = float(u_profile[0].item()) / 0.5
            u_tau_val = math.sqrt(nu * du_dy)
            y_max = (ny - 2) / 2.0
            y_plus_max = u_tau_val * y_max / nu
            print(f" {step:5d}  {u_max:8.4f}  {u_tau_val:8.5f}  {y_plus_max:8.1f}")

    # Final: check log law
    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
    du_dy = float(u_profile[0].item()) / 0.5
    u_tau_val = math.sqrt(nu * du_dy)

    # Also compute friction via common interface
    ffx, _, _ = drag_friction_integration(f, mesh, dpS, nu)
    Cf = ffx / dpS

    print()
    print("=== Log Law Verification ===")
    print(f"  u_tau = {u_tau_val:.6f}")
    print(f"  Cf (drag_friction_integration) = {Cf:.6f}")
    print(f"  {'y':>6s}  {'y+':>8s}  {'u+':>8s}  {'log_law':>8s}  {'err':>8s}")
    log_errs = []
    for i in range(0, ny - 2, 2):
        y = i + 0.5  # distance from wall (half-way)
        y_plus = u_tau_val * y / nu
        u_plus = float(u_profile[i].item()) / u_tau_val
        log_law = (1.0 / KAPPA) * math.log(max(y_plus, 1e-10)) + B_CONST
        err = abs(u_plus - log_law) / max(abs(log_law), 1e-10) * 100
        log_errs.append(err)
        print(f"  {y:6.1f}  {y_plus:8.1f}  {u_plus:8.2f}  {log_law:8.2f}  {err:8.1f}%")

    avg_err = sum(log_errs) / len(log_errs) if log_errs else 999
    passed = avg_err < 20.0
    print(f"\n  Avg log-law error: {avg_err:.1f}%")
    print(f"  PASS (avg err <20%): {passed}")

    return {"u_tau": u_tau_val, "Cf": Cf, "Re": Re, "avg_log_err": avg_err,
            "passed": passed}


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    run(device_id=dev)
