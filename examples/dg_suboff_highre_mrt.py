"""High-Re SUBOFF with MRT + LES + far-field BC + wall model.

Tries to rescue the real-Re (Re=2M) SUBOFF drag from the ~320x over-prediction
seen with BGK + channel walls at tau~0.5.  Levers: MRT (stable at low tau),
far-field BC (no blockage), log-law wall model (avoids resolving the
near-wall at Re=2M).  Reports the total resistance coefficient Ct vs the
experimental AFF-8 ~0.004.

    PYTHONPATH=src python examples/dg_suboff_highre_mrt.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.wall_model import apply_wall_model_bounce_back
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.rans_ke import KESolver, collide_rans_ke


def run(re=2e6, hull_length=96.0, nx=240, ny=96, nz=96, u_in=0.06, cs=0.1,
        n_steps=1200, warmup=300, device="cuda", use_rans=True):
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _stats = build_suboff_mask(hull_type="full", nx=nx, ny=ny, nz=nz,
                                       cx=cx, cy=cy, cz=cz, length=hull_length, device=device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    # Wall-model viscosity: use the true lattice nu (NOT a floor).  A floor pushes
    # y+ < 11.6 and activates the wall model's *laminar* branch, defeating the
    # log-law.  With nu_lat the first off-wall cell sits at y+ ~ 100-1000 (deep
    # log-law region) — the correct high-Re wall treatment.
    nu_wm = nu_lat
    # k-epsilon RANS solver (3D, eddy viscosity from k/eps transport equations).
    ke_solver = None
    if use_rans:
        rho_i, ux_i, uy_i, uz_i = macroscopic3d(f)
        ke_solver = KESolver(nu=nu_lat)
        ke_solver.initialize(ux_i, uy_i, uz_i)
    print(f"Re={re:.0e} tau={tau:.6f} nu_lat={nu_lat:.2e} RANS={'k-epsilon' if use_rans else 'no'} S={S:.0f}")
    print(f"experimental AFF-8 Ct ~ 0.004 ; prior BGK+channel Ct ~ 1.29 (320x)\n")

    drag_samples = []
    for step in range(1, n_steps + 1):
        if use_rans:
            f = collide_rans_ke(f, tau=tau, ke_solver=ke_solver, mask=solid)  # k-epsilon RANS collision
        else:
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)          # MRT + LES (stable at low tau)
        f = stream3d(f)
        rho, ux, uy, uz = macroscopic3d(f)
        f = apply_wall_model_bounce_back(f, solid, ux, uy, uz, nu_wm)   # log-law wall model on hull
        f = far_field_bc_3d(f, u_in=u_in)                           # far-field lateral (no blockage)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)
        fx, _fy, _fz = compute_obstacle_forces_3d(f, solid)
        drag = float(fx.item())
        if step > warmup and math.isfinite(drag):
            drag_samples.append(drag)
        if step % 100 == 0 or step == n_steps:
            ms = float(torch.sqrt(ux*ux+uy*uy+uz*uz).max().item())
            ct = (sum(drag_samples)/max(len(drag_samples),1))/dyn_p_S if drag_samples else float('nan')
            print(f"  step {step:4d}: drag_lu={drag:8.3f}  Ct={ct:.4f}  max|u|={ms:.4f}  {'UNSTABLE' if (not math.isfinite(ms) or ms>1.0) else ''}")
    ct = (sum(drag_samples)/max(len(drag_samples),1))/dyn_p_S if drag_samples else float('nan')
    print(f"\nFinal Ct = {ct:.4f}  (experimental ~0.004, ratio {ct/0.004:.1f}x)")


if __name__ == "__main__":
    run()
