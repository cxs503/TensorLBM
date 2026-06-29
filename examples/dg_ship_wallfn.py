"""Ship hull viscous resistance via wall function (double-body, high Re).

Wigley hull at model-scale Re, fully submerged (double-body approximation —
no free surface, isolates the viscous resistance).  Same wall-function
approach as SUBOFF: log-law body force + far-field BC + friction/pressure drag.

Reference: ITTC-57 Cf × form factor (1+k≈1.15 for Wigley).

    PYTHONPATH=src python examples/dg_ship_wallfn.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.ship_cad import build_hull_mask, ShipHullType
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


def run(re=1e6, hull_type="wigley", nx=320, ny=96, nz=96, u_in=0.06,
        n_steps=4000, warmup=1000, device="cuda"):
    nu_lat = u_in * (nx * 0.5) / re          # L = nx*0.5 (hull length default)
    tau = 3.0 * nu_lat + 0.5
    cx, cy, cz = nx * 0.3, ny * 0.5, nz * 0.5
    solid, stats = build_hull_mask(hull_type, nx, ny, nz, cx=cx, cy=cy, cz_keel=cz,
                                   device=device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S
    # Reference: ITTC Cf × form factor (1+k)
    cf_ittc = _ittc57_friction_coefficient(re)
    ff = 1.15
    ct_ref = cf_ittc * ff

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    print(f"Ship hull ({hull_type}) double-body: Re={re:.0e} tau={tau:.5f} nu={nu_lat:.2e}")
    print(f"wetted_area={S:.0f}  Cf_ITTC={cf_ittc:.5f}  (1+k)={ff}  Ct_ref={ct_ref:.5f}\n")

    fric, pres = [], []
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.1)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)
        if step % 1000 == 0 or step == n_steps:
            cf = (sum(fric)/max(len(fric),1))/dyn_p_S
            cp = (sum(pres)/max(len(pres),1))/dyn_p_S
            print(f"  step {step:4d}: Ct_fric={cf:.5f} Ct_pres={cp:.5f} Ct_tot={cf+cp:.5f} (ref {ct_ref:.5f})")
    cf = (sum(fric)/max(len(fric),1))/dyn_p_S
    cp = (sum(pres)/max(len(pres),1))/dyn_p_S
    ct = cf + cp
    print(f"\nFinal: Ct_fric={cf:.5f}  Ct_pres={cp:.5f}  Ct_total={ct:.5f}  "
          f"vs ITTC×ff {ct_ref:.5f}  (ratio {ct/ct_ref:.2f}, err {abs(ct-ct_ref)/ct_ref*100:.1f}%)")


if __name__ == "__main__":
    run()
