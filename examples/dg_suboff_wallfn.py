"""High-Re SUBOFF with a log-law wall function (body force) — decoupled from τ.

The wall shear τ_w is computed from the log-law at the first off-wall cell and
applied as a Guo body force on the near-wall fluid cells.  This DECOUPLES the
wall shear from the bulk τ (which stays at the high-Re τ≈0.5), exactly what
PowerFlow-style wall functions do.  The drag is the integrated wall shear
(Σ τ_w·t̂_x), NOT the τ≈0.5-unreliable momentum exchange.

    PYTHONPATH=src python examples/dg_suboff_wallfn.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.ibm import ibm_apply_body_force_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

KAPPA = 0.41
B_CONST = 5.0


def wall_function_3d(f, solid, nu, y_val=1.0):
    """Log-law wall function: body force on near-wall cells + total drag.

    Returns (f_with_force, drag_friction_x, drag_pressure_x).
    drag_friction = Σ τ_w·(u_x/|u|);  drag_pressure = Σ p·n̂_x over hull faces
    (form/pressure drag, p=(ρ-1)/3 gauge).
    """
    device = f.device
    fluid = ~solid
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    near = nbrs

    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
    u_tau = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu
    turb = (y_plus > 11.6) & near
    if bool(turb.any()):
        ut = u_tau[turb].clone(); um = u_mag[turb]
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu)
            fv = ut * (lyp/KAPPA + B_CONST) - um
            fp = (lyp/KAPPA + B_CONST) + 1.0/KAPPA
            ut = (ut - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)
    fx = coef * (ux * inv_umag); fy = coef * (uy * inv_umag); fz = coef * (uz * inv_umag)
    f = ibm_apply_body_force_3d(f, fx, fy, fz)
    drag_fric = (tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item()
    # pressure (form) drag: Σ p·n̂_x over hull faces = Σ_F p_F·(#solid +x nbr − #solid −x nbr)
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)   # solid at +x neighbour of F
    sm = torch.roll(solid, -1, dims=2)  # solid at -x neighbour of F
    drag_pres = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, drag_fric, drag_pres


def run(re=2e6, hull_length=128.0, nx=320, ny=128, nz=128, u_in=0.06, cs=0.1,
        n_steps=1000, warmup=300, y_val=0.5, device="cuda"):
    nu_lat = u_in * hull_length / re; tau = 3.0*nu_lat + 0.5
    cx, cy, cz = nx*0.35, ny/2.0, nz/2.0
    solid,_ = build_suboff_mask(hull_type="full", nx=nx,ny=ny,nz=nz,cx=cx,cy=cy,cz=cz,length=hull_length,device=device)
    S = _voxel_wetted_area(solid,1.0); dyn_p_S = 0.5*1.0*u_in**2*S
    rho0 = torch.ones(nz,ny,nx,device=device); ux0=torch.full((nz,ny,nx),u_in,device=device); ux0[solid]=0
    f = equilibrium3d(rho0,ux0,torch.zeros_like(ux0),torch.zeros_like(ux0),device=device)
    initial_mass=float(rho0.sum().item())
    print(f"Re={re:.0e} tau_lam={tau:.5f} nu_lat={nu_lat:.2e} grid={nx}x{ny}x{nz} hull={hull_length:.0f} S={S:.0f}")
    print(f"experimental AFF-8 Ct ~ 0.004\n")
    fric=[]; pres=[]
    for step in range(1,n_steps+1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=y_val)
        f = far_field_bc_3d(f, u_in=u_in)
        if step%100==0: f = correct_mass3d(f, initial_mass)
        if step>warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)
        if step%200==0 or step==n_steps:
            _,ux,uy,uz=macroscopic3d(f); ms=float(torch.sqrt(ux*ux+uy*uy+uz*uz).max().item())
            cf=(sum(fric)/max(len(fric),1))/dyn_p_S; cp=(sum(pres)/max(len(pres),1))/dyn_p_S
            print(f"  step {step:4d}: Ct_fric={cf:.4f} Ct_pres={cp:.4f} Ct_tot={cf+cp:.4f} max|u|={ms:.4f} {'UNSTABLE' if (not math.isfinite(ms) or ms>1.0) else ''}")
    cf=(sum(fric)/max(len(fric),1))/dyn_p_S; cp=(sum(pres)/max(len(pres),1))/dyn_p_S
    print(f"\nFinal: Ct_fric={cf:.4f}  Ct_pres={cp:.4f}  Ct_total={cf+cp:.4f}  (exp ~0.004, ratio {(cf+cp)/0.004:.2f}x)")


if __name__=="__main__":
    run()
