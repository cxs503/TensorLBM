#!/usr/bin/env python3
"""Debug script to trace the difference between reference and torch implementation."""

import torch
import sys
sys.path.insert(0, "/root/tp-agent/work/m366/src")

from tensorlbm.d3q19 import OPPOSITE, C, W, equilibrium3d
from tensorlbm.solver3d import stream3d
from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask
from tensorlbm.triton_fused_obstacle import torch_fused_obstacle_xfar_les

device = "cpu"
nx, ny, nz = 16, 16, 16
nu_lb = 0.005

# Create a simple test case with a small obstacle
solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
solid[nz//2, ny//2, nx//2] = True  # Single solid cell at center

rho0 = torch.ones((nz, ny, nx), device=device, dtype=torch.float32)
ux0 = torch.full_like(rho0, 0.05)
ux0[solid] = 0.0
uy0 = torch.zeros_like(rho0)
uz0 = torch.zeros_like(rho0)

f0 = equilibrium3d(rho0, ux0, uy0, uz0)

print("Initial state:")
print(f"  f0 shape: {f0.shape}")
print(f"  Solid cell at: ({nz//2}, {ny//2}, {nx//2})")
print(f"  f0 at solid cell: {f0[:, nz//2, ny//2, nx//2]}")

# === Reference chain ===
print("\n=== Reference chain ===")

# Collision
rho = f0.sum(dim=0)
rho_safe = torch.where(rho > 1e-12, rho, torch.tensor(1e-12, device=device))

cxf = C[:, 0].to(device).float()
cyf = C[:, 1].to(device).float()
czf = C[:, 2].to(device).float()
w = W.to(device).float()

ux = (cxf[:, None, None, None] * f0).sum(dim=0) / rho_safe
uy = (cyf[:, None, None, None] * f0).sum(dim=0) / rho_safe
uz = (czf[:, None, None, None] * f0).sum(dim=0) / rho_safe

usq = ux * ux + uy * uy + uz * uz
cxf_4d = cxf[:, None, None, None]
cyf_4d = cyf[:, None, None, None]
czf_4d = czf[:, None, None, None]
w_4d = w[:, None, None, None]

cu = cxf_4d * ux[None, :, :, :] + cyf_4d * uy[None, :, :, :] + czf_4d * uz[None, :, :, :]
feq = rho_safe[None, :, :, :] * w_4d * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :, :])

tau = 3.0 * nu_lb + 0.5
omega = 1.0 / tau
f_collided = f0 - omega * (f0 - feq)

print(f"After collision:")
print(f"  f_collided at solid cell: {f_collided[:, nz//2, ny//2, nx//2]}")

# Streaming
f_streamed = stream3d(f_collided)
print(f"After streaming:")
print(f"  f_streamed at solid cell: {f_streamed[:, nz//2, ny//2, nx//2]}")

# Bounce-back
f_ref = bounce_back_cells_3d(f_streamed, solid)
print(f"After bounce-back:")
print(f"  f_ref at solid cell: {f_ref[:, nz//2, ny//2, nx//2]}")

# === Torch implementation ===
print("\n=== Torch implementation ===")
f_torch = torch_fused_obstacle_xfar_les(
    f0.clone(),
    nu_lb,
    solid.to(torch.int8),
    Cs=0.0,
    delta=1.0,
    collision="BGK",
)
print(f"f_torch at solid cell: {f_torch[:, nz//2, ny//2, nx//2]}")

# === Compare ===
print("\n=== Comparison ===")
diff = (f_ref - f_torch).abs()
print(f"Max diff at solid cell: {diff[:, nz//2, ny//2, nx//2].max().item():.2e}")
print(f"Overall max diff: {diff.max().item():.2e}")

# Check a few specific directions
print("\nChecking specific directions at solid cell:")
for q in [0, 1, 2, 3, 4]:
    print(f"  q={q}: ref={f_ref[q, nz//2, ny//2, nx//2]:.6f}, torch={f_torch[q, nz//2, ny//2, nx//2]:.6f}, diff={diff[q, nz//2, ny//2, nx//2]:.2e}")
