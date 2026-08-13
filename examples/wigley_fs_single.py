"""Wigley hull free-surface using single-phase free-surface LBM (GALS).

The single-phase free-surface model (Körner et al. 2005) tracks a fill field
(0=void/gas, 1=liquid) and applies gravity as a simple body force on the
liquid — far more stable than CG multiphase for free-surface ship flows.

Combines:
- free_surface_step (single-phase fill tracking + gravity)
- Wall function (log-law body force on hull)
- CV momentum integral for drag
- Far-field lateral BC

    PYTHONPATH=src python examples/wigley_fs_single.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import free_surface_step, LIQUID, INTERFACE, GAS, SOLID, init_flags_from_fill
from tensorlbm.obstacles import wigley_hull_mask
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import stream3d
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient


def run(fn=0.25, nx=240, ny=96, nz=96, u_in=0.05,
        fill_fraction=0.55, n_steps=5000, warmup=1500, device="cuda"):
    fill_height = int(fill_fraction * nz)
    hull_length = max(6.0, 0.35 * nx)
    g_lu = u_in**2 / (fn**2 * hull_length) if fn > 0.001 else 0.0
    re = 5000
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5

    # Hull
    hull = wigley_hull_mask(nx=nx, ny=ny, nz=nz, cx=int(0.4*nx), cy=0.5*(ny-1),
                            cz_keel=1.0, length=hull_length,
                            beam=max(3.0, 0.25*ny), draft=fill_height+4, device=device)

    # Free-surface flags: LIQUID below fill, GAS above, INTERFACE at the boundary
    zz = torch.arange(nz, device=device).view(nz, 1, 1).expand(nz, ny, nx)
    flags = init_flags_from_fill(fill, solid_mask)

    # Fill field: 1.0 below waterline, 0.0 above
    fill = torch.where(zz < fill_height, torch.ones_like(zz, dtype=torch.float32),
                       torch.zeros_like(zz, dtype=torch.float32))

    # Initialize f with uniform flow in liquid, zero in gas
    rho0 = torch.where(zz < fill_height, torch.ones((nz,ny,nx), device=device),
                       torch.zeros((nz,ny,nx), device=device))
    ux0 = torch.where(zz < fill_height, torch.full((nz,ny,nx), u_in, device=device),
                      torch.zeros((nz,ny,nx), device=device))
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))

    # Solid mask: hull only (domain boundaries handled by far-field BC)
    solid_mask = hull.clone()

    S_wet = float((hull & (zz < fill_height)).sum().item())
    dyn_p_S = 0.5 * u_in**2 * max(S_wet, 1.0)
    cf_ref = _ittc57_friction_coefficient(re) * 1.15  # form factor for Wigley

    # CV integral box (water region)
    x0_cv = int(0.2 * nx); x1_cv = int(0.65 * nx)

    cv_samples = []
    print(f"Wigley free-surface (single-phase GALS): Fn={fn} Re={re} g={g_lu:.5f}")
    print(f"  grid={nx}x{ny}x{nz}  fill_z={fill_height}  S_wet={S_wet:.0f}  tau={tau:.4f}")
    print(f"  Cf_ref(ITTC×1.15)={cf_ref:.5f}\n")

    for step in range(1, n_steps + 1):
        # Gravity ramp
        ramp = min(1.0, step / 500.0) if fn > 0.001 else 0.0
        gz = g_lu * ramp

        # 1. Free-surface step (collision + stream + fill update + gravity)
        f = free_surface_step(f, fill, flags, solid_mask,
                              tau=tau, gz=-gz, C_s=0.1, rho_liquid=1.0, rho_gas=0.0)

        # 2. Wall function on hull (water-side only)
        rho_now, ux, uy, uz = macroscopic3d(f)
        water_now = fill > 0.5
        if water_now.any() and hull.any():
            f, df, dp = wall_function_3d(f, solid_mask, nu, y_val=0.5, wall_law="reichardt")

        # 3. Far-field BC (inlet/outlet/lateral)
        f = far_field_bc_3d(f, u_in=u_in)
        # Zero out gas region above fill (maintain vacuum)
        gas_cells = fill < 0.01
        f[:, gas_cells] = 0.0

        # 4. Force measurement
        if step > warmup:
            rho_now = f.sum(dim=0)
            _, ux, uy, uz = macroscopic3d(f)
            water_mask = fill > 0.5
            # CV integral in water region only
            y0, y1 = 2, ny - 3
            z0, z1 = 1, fill_height - 1
            M_in = (rho_now[z0:z1+1, y0:y1+1, x0_cv] * ux[z0:z1+1, y0:y1+1, x0_cv]**2).sum().item()
            M_out = (rho_now[z0:z1+1, y0:y1+1, x1_cv] * ux[z0:z1+1, y0:y1+1, x1_cv]**2).sum().item()
            My0 = (rho_now[z0:z1+1, y0, x0_cv:x1_cv+1] * ux[z0:z1+1, y0, x0_cv:x1_cv+1] *
                   uy[z0:z1+1, y0, x0_cv:x1_cv+1]).sum().item()
            My1 = (rho_now[z0:z1+1, y1, x0_cv:x1_cv+1] * ux[z0:z1+1, y1, x0_cv:x1_cv+1] *
                   uy[z0:z1+1, y1, x0_cv:x1_cv+1]).sum().item()
            Mz0 = (rho_now[z0, y0:y1+1, x0_cv:x1_cv+1] * ux[z0, y0:y1+1, x0_cv:x1_cv+1] *
                   uz[z0, y0:y1+1, x0_cv:x1_cv+1]).sum().item()
            Mz1 = (rho_now[z1, y0:y1+1, x0_cv:x1_cv+1] * ux[z1, y0:y1+1, x0_cv:x1_cv+1] *
                   uz[z1, y0:y1+1, x0_cv:x1_cv+1]).sum().item()
            cv = M_in - M_out + My0 - My1 + Mz0 - Mz1
            if math.isfinite(cv):
                cv_samples.append(cv / dyn_p_S)

        if step % 1000 == 0 or step == n_steps:
            ct = sum(cv_samples)/max(len(cv_samples),1) if cv_samples else 0.0
            _, ux, uy, uz = macroscopic3d(f)
            ms = float(torch.sqrt(ux*ux+uy*uy+uz*uz).max().item())
            stable = math.isfinite(ms) and ms < 0.5
            print(f"  step={step:5d}  Ct_CV={ct:.5f}  (ref {cf_ref:.5f})  max|u|={ms:.4f}  "
                  f"{'OK' if stable else 'UNSTABLE'}")

    ct = sum(cv_samples)/max(len(cv_samples),1) if cv_samples else 0.0
    err = abs(ct - cf_ref)/cf_ref*100 if cf_ref > 0 else 0
    print(f"\nFinal Ct_CV={ct:.5f}  vs Cf_ref={cf_ref:.5f}  (err {err:.1f}%)")
    return {"Fn": fn, "Ct": ct, "Cf_ref": cf_ref}


if __name__ == "__main__":
    run()
