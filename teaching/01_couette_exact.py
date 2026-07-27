#!/usr/bin/env python3
"""Teaching Example 01: Couette Flow — Exact Solution

Physics: Two parallel plates, top moving at u_top, bottom stationary.
         Linear velocity profile: u(y) = u_top * y / H
         Wall shear stress: τ = ν * u_top / H
         Friction coefficient: Cf = 2ν / (H * u_top)

Key concepts:
  - Half-way bounce-back (BB) for no-slip walls
  - Bug 27 fix: BB uses pre-collision f (not post-collision)
  - Friction formula: τ = 2ν · u_t (standard, exact for linear profile)
  - Pressure integration: Cd_p = 0 (no pressure drag in Couette)

Usage:
  PYTHONPATH=src python teaching/01_couette_exact.py [device_id]
"""
import sys, math
sys.path.insert(0, 'src')
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import stream3d, collide_bgk3d, correct_mass3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.drag_pressure import get_near_wall_3d, SurfaceMesh, drag_pressure_integration, drag_friction_integration

def run(device_id=0):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)

    # Parameters
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # ν = 1/6
    u_top = 0.05
    H = ny - 2  # channel height (fluid cells)
    Cf_exact = 2.0 * nu / (H * u_top)
    n_steps = 3000

    print(f'=== Couette Flow (SDAA:{device_id}) ===')
    print(f'Grid: {nx}x{ny}x{nz}, tau={tau}, nu={nu:.6f}')
    print(f'u_top={u_top}, H={H}, Cf_exact={Cf_exact:.6f}')
    print(f'Steps: {n_steps}')
    print()

    # Solid mask: walls at y=0 (bottom) and y=ny-1 (top)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True    # bottom (stationary)
    solid[:, -1, :] = True   # top (moving)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mesh for friction computation
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f'Near-wall cells: {int(near.sum().item())}')

    # Initial condition: fluid at rest
    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)
    initial_mass = float(rho0.sum().item())

    # Precompute lattice constants
    c = C.to(dev).float()  # (19, 3)
    w = W.to(dev).float()   # (19,)

    dpS = 0.5 * 1.0 * u_top**2 * H * nz  # dynamic pressure × area

    print('\n--- Time loop ---')
    for step in range(1, n_steps + 1):
        # 1. Save pre-collision f (Bug 27 fix)
        f_pre = f.clone()

        # 2. Collision (all cells)
        f = collide_bgk3d(f, tau=tau)

        # 3. NoDynamics: restore solid cells to pre-collision
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Bounce-back with pre-collision f (Bug 27 fix)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 5. Moving wall correction: top wall moves at u_top in x-direction
        #    For downward-pointing populations (c[1]<0): Δf = 6*ρ*u_top*w*c_x
        rho, _, _, _ = macroscopic3d(f)
        rho_top = rho[:, -1, :]  # (nz, nx)
        for q in range(19):
            if c[q, 1] < 0:  # downward-pointing (into fluid)
                correction = 6.0 * rho_top * u_top * w[q] * c[q, 0]  # c_x!
                f[q, :, -1, :] += correction

        # 6. Streaming
        f = stream3d(f)

        # 7. Periodic BC in x
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

        # 8. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Output
        if step % 500 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            u_mid = float(ux[:, ny//2, :].mean().item())
            # Friction on bottom wall (stationary)
            ffx, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            Cf = ffx / dpS
            # Pressure (should be ~0)
            fpx, _, _ = drag_pressure_integration(f, mesh, dpS)
            print(f'  step={step:4d} u_mid={u_mid:.5f} Cf={Cf:.6f} Cd_p={fpx:.6f} (exact Cf={Cf_exact:.6f})')

    # Final results
    rho, ux, uy, uz = macroscopic3d(f)
    print('\n=== FINAL RESULTS ===')

    # Velocity profile
    u_profile = [float(ux[0, y, nx//2].item()) for y in range(ny)]
    u_exact = [u_top * (y - 0.5) / (ny - 1 - 0.5) if 0 < y < ny-1 else (u_top if y == ny-1 else 0.0) for y in range(ny)]
    u_err = max(abs(u_profile[i] - u_exact[i]) / max(abs(u_exact[i]), 1e-10) for i in range(1, ny-1))

    print(f'u_err_max = {u_err*100:.4f}%')
    print(f'Cf        = {Cf:.6f}  (exact={Cf_exact:.6f}, err={abs(Cf-Cf_exact)/Cf_exact*100:.4f}%)')
    print(f'Cd_p      = {fpx:.6f}  (should be ~0)')
    print()
    print('  y     u_sim     u_exact   err%')
    for y in range(ny):
        err = abs(u_profile[y] - u_exact[y]) / max(abs(u_exact[y]), 1e-10) * 100
        print(f'  {y:2d}  {u_profile[y]:+.6f}  {u_exact[y]:+.6f}  {err:.2f}%')

    print(f'\nTeaching points:')
    print(f'  1. Half-way BB: f[solid] = f_pre[opp] (pre-collision, Bug 27)')
    print(f'  2. Moving wall: add 2*ρ*u*w*c to downward populations')
    print(f'  3. Friction: τ=2ν·u_t (exact for linear profile)')
    print(f'  4. Pressure: Cd_p≈0 (no pressure drag in Couette)')

if __name__ == '__main__':
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(dev)
