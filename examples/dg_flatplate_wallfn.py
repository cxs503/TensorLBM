"""Flat-plate turbulent boundary layer — wall-function validation.

Canonical high-Re wall-function test: a flat plate in a uniform stream develops
a turbulent boundary layer; the total skin-friction coefficient Cf should follow
the Schlichting turbulent correlation Cf = 0.074 / Re_L^0.2 (for Re_L ~ 1e5-1e7).

Uses the same τ-decoupled log-law wall_function_3d as the SUBOFF high-Re
solver.  This validates the wall function on a DEVELOPING boundary layer
(local Cf varies with x), an independent check from the SUBOFF hull case.

    PYTHONPATH=src python examples/dg_flatplate_wallfn.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.ibm import ibm_apply_body_force_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


def run(re_L=1e6, L=128.0, nx=256, ny=96, nz=8, u_in=0.06, n_steps=3000, warmup=800, device="cuda"):
    nu_lat = u_in * L / re_L
    tau = 3.0 * nu_lat + 0.5
    # Flat plate: solid on the bottom wall (y=0), from x=4 to nx (full length L).
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, 4:] = True                 # plate on the bottom row, starting 4 cells in
    plate_area = (nx - 4) * nz             # plate area (lattice cells)
    dyn_p_A = 0.5 * 1.0 * u_in ** 2 * plate_area

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())
    cf_ref = 0.074 / re_L ** 0.2
    print(f"Flat plate Re_L={re_L:.0e} L={L:.0f} tau={tau:.5f} nu={nu_lat:.2e}")
    print(f"Schlichting turbulent Cf = {cf_ref:.5f}\n")

    samples = []
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.1)
        f = stream3d(f)
        f, drag_f, drag_p = wall_function_3d(f, solid, nu_lat, y_val=0.5)
        # Top boundary: far-field (free stream); inlet/outlet far-field; bottom=plate (wall fn handled it)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)   # keep the plate solid (undo far-field on y=0)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)
        if step > warmup and math.isfinite(drag_f):
            samples.append(drag_f)
        if step % 500 == 0 or step == n_steps:
            cf = (sum(samples) / max(len(samples), 1)) / dyn_p_A if samples else float('nan')
            print(f"  step {step:4d}: Cf={cf:.5f} (ref {cf_ref:.5f}, ratio {cf/cf_ref:.2f})")
    cf = (sum(samples) / max(len(samples), 1)) / dyn_p_A
    print(f"\nFinal Cf = {cf:.5f}  vs Schlichting {cf_ref:.5f}  (ratio {cf/cf_ref:.2f}, err {abs(cf-cf_ref)/cf_ref*100:.1f}%)")


if __name__ == "__main__":
    run()
