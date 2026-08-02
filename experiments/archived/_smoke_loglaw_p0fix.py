#!/usr/bin/env python3
"""Quick 50-step smoke test for log-law + far_field p_0."""
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

device = torch.device("sdaa:12")
torch.sdaa.set_device(device)

L = 80.0; nx, ny, nz = 200, 80, 80; u_in = 0.06
Re = 100000; tau = 0.500144; nu = u_in * L / Re; cs_smag = 0.05
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
print(f"Init: n_solid={int(solid.sum())}, n_near={int(near.sum())}, dpS={dpS:.4f}")

t0 = time.time()
for step in range(1, 51):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f = stream3d(f)
    f, drag_fric, drag_pres = wall_function_3d(
        f, solid, nu, y_val=1.0, wall_law="log", near_mask=near)
    f = far_field_bc_3d(f, u_in)
    if step % 200 == 0:
        f = correct_mass3d(f, im)
    cd_f = drag_fric / dpS
    # Test all 3 p_0 methods
    fx_near, _, _ = drag_pressure_integration(f, mesh, dpS, p0_method="near_wall", solid=solid)
    fx_far, _, _ = drag_pressure_integration(f, mesh, dpS, p0_method="far_field", solid=solid)
    fx_dom, _, _ = drag_pressure_integration(f, mesh, dpS, p0_method="domain_avg", solid=solid)
    if step % 10 == 0:
        print(f"  step={step} Cd_f={cd_f:.6f} Cd_p(near)={fx_near:.6f} Cd_p(far)={fx_far:.6f} Cd_p(dom)={fx_dom:.6f} ({time.time()-t0:.1f}s)")

print("\nSMOKE TEST PASSED — log law + far_field p_0 works")
