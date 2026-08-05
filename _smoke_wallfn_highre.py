#!/usr/bin/env python3
"""Quick 20-step smoke test for suboff_wallfn_highre_worker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from suboff_wallfn_highre_worker import drag_friction_gradient
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
import math

device = torch.device("sdaa:12")
torch.sdaa.set_device(device)

L = 80.0; nx, ny, nz = 200, 80, 80; u_in = 0.06
Re = 1000; tau = 0.5144; nu = u_in * L / Re; cs_smag = 0.05
config = SuboffConfig(); radius = config.r_over_l * L; D = 2.0 * radius
cx = nx * 0.30; cy = ny * 0.5; cz = nz * 0.5
dpS = 0.5 * u_in**2 * math.pi * D * L

solid, stats = build_suboff_mask(
    hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
    cx=cx, cy=cy, cz=cz, length=L, radius=radius,
    config=config, device=device)
near = get_near_wall_3d(solid)
mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

rho0 = torch.ones((nz, ny, nx), device=device)
ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
im = float(rho0.sum().item())
print(f"Init done. n_solid={int(solid.sum())}, n_near={int(near.sum())}, dpS={dpS:.4f}")

# Test 3 friction formulas on initial field
q_half = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)
fx_a, _, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=None)
fx_b, _, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=q_half)
fx_c, _, _ = drag_friction_gradient(f, mesh, dpS, nu, delta_n=1.0)
print(f"Initial friction: a(2vu)={fx_a:.6f} b(vu/q)={fx_b:.6f} c(vdu/dn)={fx_c:.6f}")
print(f"  |a-b|={abs(fx_a-fx_b):.2e} |a-c|={abs(fx_a-fx_c):.2e}")

# Run 20 steps with standard BB
for step in range(1, 21):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f = bounce_back_cells_3d(f, solid)
    f = stream3d(f)
    f = far_field_bc_3d(f, u_in)
    if step % 200 == 0:
        f = correct_mass3d(f, im)
    fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
    fx_a, _, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=None)
    fx_b, _, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=q_half)
    fx_c, _, _ = drag_friction_gradient(f, mesh, dpS, nu, delta_n=1.0)
    if step % 5 == 0:
        print(f"  step={step} Cd_p={fx_p:.6f} fa={fx_a:.6f} fb={fx_b:.6f} fc={fx_c:.6f}")

# Test wall function (gradient law) for 5 steps
print("\nTesting wall_function_3d (gradient, y_val=0.5)...")
for step in range(1, 6):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f = stream3d(f)
    f, drag_fric, drag_pres = wall_function_3d(
        f, solid, nu, y_val=0.5, wall_law="gradient", near_mask=near)
    f = far_field_bc_3d(f, u_in)
    fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
    cd_f = drag_fric / dpS
    print(f"  wf step={step} Cd_p={fx_p:.6f} Cd_f(wf)={cd_f:.6f}")

print("\nTesting wall_function_3d (log, y_val=1.0)...")
nu2 = u_in * L / 2e6  # Re=2e6 viscosity
for step in range(1, 6):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=0.5000072, C_s=cs_smag)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f = stream3d(f)
    f, drag_fric, drag_pres = wall_function_3d(
        f, solid, nu2, y_val=1.0, wall_law="log", near_mask=near)
    f = far_field_bc_3d(f, u_in)
    fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
    cd_f = drag_fric / dpS
    print(f"  wf-log step={step} Cd_p={fx_p:.6f} Cd_f(wf)={cd_f:.6f}")

print("\nSMOKE TEST PASSED")
