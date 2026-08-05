#!/usr/bin/env python3
"""Quick diagnostic: check force computation on SDAA for VIV cylinder."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch_sdaa  # noqa
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)

dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 8
device = torch.device(f"sdaa:{dev_id}")
torch.sdaa.set_device(dev_id)

nx, ny, nz = 400, 120, 4
D = 48; R = D / 2.0
cx = nx // 4; cy = ny // 2
u_in = 0.1

# Build cylinder mask
yy, xx = torch.meshgrid(
    torch.arange(ny, device=device, dtype=torch.float32),
    torch.arange(nx, device=device, dtype=torch.float32),
    indexing="ij",
)
circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
print(f"solid cells={int(solid.sum().item())}")

near = get_near_wall_3d(solid)
print(f"near-wall cells={int(near.sum().item())}")

mesh = SurfaceMesh.from_cylinder(solid, near, float(cx), float(cy), float(R), axis='z')
print(f"mesh.near sum={int(mesh.near.sum().item())}")
print(f"mesh.nx_n max={float(mesh.nx_n.abs().max().item()):.4f}")
print(f"mesh.ny_n max={float(mesh.ny_n.abs().max().item()):.4f}")

# Init flow: uniform in x
rho0 = torch.ones((nz, ny, nx), device=device)
ux0 = torch.full((nz, ny, nx), u_in, device=device)
ux0[solid] = 0.0
f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)

dpS = 0.5 * u_in ** 2 * D * nz
nu = 0.024

# Check macroscopic fields
rho, ux, uy, uz = macroscopic3d(f)
p = (rho - 1.0) / 3.0
print(f"rho: min={float(rho.min()):.6f} max={float(rho.max()):.6f}")
print(f"p: min={float(p.min()):.6f} max={float(p.max()):.6f}")
print(f"p at near-wall: mean={float((p * near.float()).sum().item() / near.float().sum().clamp(min=1).item()):.6f}")

# Compute force
fx_p, fy_p, fz_p = drag_pressure_integration(
    f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
fx_f, fy_f, fz_f = drag_friction_integration(
    f, mesh, dpS, nu, formula='standard')
print(f"Pressure drag: fx={fx_p:.6f} fy={fy_p:.6f} fz={fz_p:.6f}")
print(f"Friction drag: fx={fx_f:.6f} fy={fy_f:.6f} fz={fz_f:.6f}")
print(f"Total drag: fx={fx_p+fx_f:.6f} fy={fy_p+fy_f:.6f}")
print(f"dpS={dpS:.6f}")
print("DONE")
