#!/usr/bin/env python
"""Teaching Example 02: Poiseuille flow — exact solution with corrected BB.

Physics: two parallel plates, both stationary, driven by a body force.
         Parabolic velocity profile: u(y) = (G/2ν) * y' * (H - y')

Key concepts:
  - Half-way bounce-back (BB) with f_pre for exact no-slip
  - Body force via velocity shift (Guo scheme, no double-counting)
  - Friction coefficient Cf = 8ν / (U_max * H)
  - lbm_step_correct() for verified-correct main loop

Usage:
  PYTHONPATH=src python teaching/02_poiseuille_exact.py [device_id]
"""
from __future__ import annotations
import sys, math
sys.path.insert(0, 'src')
import torch
from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)


def run(device_id=0, ny=16, nx=32, nz=4, force=1e-5, tau=1.0, n_steps=5000):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)

    print(f"=== Poiseuille Flow (SDAA:{device_id}) ===")
    print(f"Grid: {nx}×{ny}×{nz}, tau={tau}, force={force}, steps={n_steps}")
    print()

    # --- Solid mask: walls at y=0 and y=ny-1 ---
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall

    # --- Near-wall mesh ---
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)

    # --- Initial condition ---
    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=dev)
    initial_mass = float(rho0.sum().item())

    nu = (tau - 0.5) / 3.0
    H = float(ny - 2)  # channel height: walls at y=0.5 and y=ny-1-0.5, so H=ny-2
    dpS = 0.5 * 1.0 * force * H * nz  # for pressure normalization

    # --- Custom collide_fn: BGK + body force via velocity shift ---
    # Velocity shift method: u' = u + τ*F/ρ in equilibrium provides the
    # correct body force G.  No explicit force term needed — the simplified
    # Guo force term Δf = w*3*c_x*(1-1/2τ)*G combined with velocity shift
    # double-counts the force (gives 1.5×G instead of G).
    def collide_bgk_force(f, tau, **kwargs):
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux + force * tau / rho  # velocity shift (Guo)
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        return f - (f - feq) / tau

    # --- Far-field BC: periodic in x (handled by torch.roll) ---
    def periodic_x_bc(f, u_in):
        return f

    print(f"nu={nu:.6f}, H={H}, u_max_exact={force*H**2/(8*nu):.8f}")
    print(f"{'Step':>6s}  {'u_max':>8s}  {'Cf':>8s}  {'Cd_p':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_bgk_force, tau, solid, 0.0, periodic_x_bc,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200,
        )

        if step % 500 == 0 or step == n_steps:
            _, ux, _, _ = macroscopic3d(f)
            u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))
            u_max = float(u_profile.max().item())
            # drag_friction_integration already returns F/dpS
            ffx, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            Cf = ffx  # already normalized
            fpx, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            print(f" {step:5d}  {u_max:8.5f}  {Cf:8.6f}  {fpx:8.6f}")

    # --- Extract velocity profile ---
    _, ux, _, _ = macroscopic3d(f)
    u_profile = ux[:, 1:-1, :].mean(dim=(0, 2))  # (ny-2,)

    # --- Exact solution: u(y) = G/(2ν) * y' * (H - y') ---
    # where y' = y - 0.5 (distance from bottom wall), H = ny - 2
    y_interior = torch.arange(1, ny - 1, device=dev, dtype=torch.float32)
    G = force
    u_exact = G / (2.0 * nu) * (y_interior - 0.5) * (H - (y_interior - 0.5))

    u_num = u_profile.cpu().numpy()
    u_ex = u_exact.cpu().numpy()
    u_max_exact = float(u_ex.max())
    u_max_num = float(u_num.max())
    max_err = max(abs(a - b) for a, b in zip(u_num, u_ex)) / abs(u_max_exact)
    max_err_pct = max_err * 100

    # Friction: Cf = 8ν / (U_max * H)  (plane Poiseuille)
    cf_exact = 8.0 * nu / (u_max_exact * H) if u_max_exact > 0 else 0.0
    du_dy_wall = u_num[0] / 0.5  # forward diff: u(y=1) / 0.5
    tau_wall = nu * du_dy_wall
    cf_num = 2.0 * tau_wall / u_max_num**2 if u_max_num > 1e-12 else 0.0
    cf_err_pct = abs(cf_num - cf_exact) / abs(cf_exact) * 100 if cf_exact > 0 else 0.0

    print(f"\n=== FINAL RESULTS ===")
    print(f"u_max:  num={u_max_num:.8f}, exact={u_max_exact:.8f}")
    print(f"u_max error: {max_err_pct:.2f}%")
    print(f"Cf:     num={cf_num:.6f}, exact={cf_exact:.6f}")
    print(f"Cf error:    {cf_err_pct:.2f}%")

    print(f"\n  {'y':>5s}  {'u_num':>12s}  {'u_exact':>12s}  {'err':>12s}")
    for yv, un, ue in zip(y_interior.cpu().numpy(), u_num, u_ex):
        print(f"  {yv:5.1f}  {un:12.8f}  {ue:12.8f}  {abs(un-ue):12.2e}")

    passed = max_err_pct < 1.0
    print(f"\n{'PASS' if passed else 'FAIL'}: u_max error={max_err_pct:.2f}% (target < 1%)")
    return passed


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(device_id=dev)
