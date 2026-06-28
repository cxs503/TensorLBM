"""Decisive test: does lifting τ_eff off the 0.5 cliff fix the high-Re drag?

At Re=2M the laminar τ→0.5 (the LBM accuracy/stability cliff).  PowerFlow's
trick is that the *effective* τ (including turbulent ν_t) stays safely >0.5.
This run FORCES a per-cell ν_t floor so τ_eff ≥ 0.6 everywhere, and checks
whether the SUBOFF drag comes down from the ~10x over-prediction.  If it does,
the τ-cliff is confirmed as the root cause and the path is "a turbulence model
that keeps τ_eff safe" (k-ε once bootstrapped).

    PYTHONPATH=src python examples/dg_suboff_taueff_test.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import C as C3D, equilibrium3d, macroscopic3d
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import _get_d3q19_mrt_matrices, _neq_stress_norm_3d


def collide_mrt_taueff_floor(f, tau_lam, nu_floor):
    """MRT collision with per-cell τ_eff = 3(ν_lam + max(ν_t_smag, nu_floor)) + 0.5."""
    device = f.device
    M, M_inv = _get_d3q19_mrt_matrices(device)
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_3d(f_neq)                       # strain measure
    # Smagorinsky ν_t = (C_s·Δ)²·|S|, |S| ∝ pi_norm; floor it.
    nu_t = (0.1 ** 2) * pi_norm.clamp(min=0.0)
    nu_t = torch.clamp(nu_t, min=nu_floor)                    # force off the τ=0.5 cliff
    nu_lam = (tau_lam - 0.5) / 3.0
    tau_eff = (3.0 * (nu_lam + nu_t) + 0.5).clamp(0.55, 3.0)
    s_nu_field = 1.0 / tau_eff
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(19, -1); feq_flat = feq.reshape(19, -1)
    s_nu_flat = s_nu_field.reshape(-1)
    m = M @ f_flat; m_eq = M @ feq_flat; dm = m - m_eq
    s_fixed = torch.tensor([0,1.19,1.4, 0,1.2,0,1.2,0,1.2, 0,0,0,0,0, 1.19,1.19, 1,1,1], dtype=f.dtype, device=device)
    m_star = m - s_fixed.unsqueeze(1)*dm
    for k in (9,10,11,12,13): m_star[k] = m[k] - s_nu_flat*dm[k]
    return (M_inv @ m_star).reshape(19, nz, ny, nx)


def run(re=2e6, hull_length=96.0, nx=240, ny=96, nz=96, u_in=0.06, nu_floor=0.033,
        n_steps=800, warmup=200, device="cuda"):
    nu_lat = u_in * hull_length / re; tau = 3.0*nu_lat + 0.5
    cx, cy, cz = nx*0.35, ny/2.0, nz/2.0
    solid,_ = build_suboff_mask(hull_type="full", nx=nx,ny=ny,nz=nz,cx=cx,cy=cy,cz=cz,length=hull_length,device=device)
    S = _voxel_wetted_area(solid,1.0); dyn_p_S = 0.5*1.0*u_in**2*S
    rho0 = torch.ones(nz,ny,nx,device=device); ux0=torch.full((nz,ny,nx),u_in,device=device); ux0[solid]=0
    f = equilibrium3d(rho0,ux0,torch.zeros_like(ux0),torch.zeros_like(ux0),device=device)
    initial_mass=float(rho0.sum().item())
    print(f"Re={re:.0e} tau_lam={tau:.5f} nu_floor={nu_floor} -> tau_eff>= {3*(nu_lat+nu_floor)+0.5:.3f}  S={S:.0f}")
    samples=[]
    for step in range(1,n_steps+1):
        f = collide_mrt_taueff_floor(f, tau, nu_floor)
        f = stream3d(f)
        f = bounce_back_cells_3d(f, solid)            # plain obstacle bounce-back (no wall model)
        f = far_field_bc_3d(f, u_in=u_in)
        if step%100==0: f = correct_mass3d(f, initial_mass)
        fx,_,_ = compute_obstacle_forces_3d(f, solid); drag=float(fx.item())
        if step>warmup and math.isfinite(drag): samples.append(drag)
        if step%200==0 or step==n_steps:
            _,ux,uy,uz=macroscopic3d(f); ms=float(torch.sqrt(ux*ux+uy*uy+uz*uz).max().item())
            ct=(sum(samples)/max(len(samples),1))/dyn_p_S if samples else float('nan')
            print(f"  step {step:4d}: drag={drag:8.3f} Ct={ct:.4f} max|u|={ms:.4f} {'UNSTABLE' if (not math.isfinite(ms) or ms>1.0) else ''}")
    ct=(sum(samples)/max(len(samples),1))/dyn_p_S
    print(f"\nFinal Ct={ct:.4f}  (exp ~0.004, ratio {abs(ct)/0.004:.1f}x)")


if __name__=="__main__":
    run()
