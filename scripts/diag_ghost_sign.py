#!/usr/bin/env python3
"""Verify build_ghost_plan sample positions against the SHELL_OUTSIDE
neighbour positions (post-fix check).

Reconstructs the plan's actual sample position from (z0,y0,x0,wz,wy,wx):
    continuous = lo + w  (coarse-index coords),  p_world = continuous + 0.5
and compares with the SHELL_OUTSIDE neighbour position
    p_correct = center[i] + c[d_link]*dx.
"""
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.octree_boundary.stepping import build_ghost_plan, build_plane_shell
from tensorlbm.refinement import BoxRegion

DTYPE = torch.float64
device = torch.device("cpu")
shape = (32, 16, 16)          # (nz, ny, nx)
box = BoxRegion(x0=2, x1=shape[2] - 2, y0=2, y1=shape[1] - 2, z0=4, z1=shape[0] - 4)

shell = build_plane_shell(shape, box, device=device)
gp = build_ghost_plan(shell, shape)

coords = shell._l1_coords
centers64 = (coords.to(DTYPE) + 0.5) / 2.0          # (n, 3) x,y,z
links = shell.interface_links                       # (n_link, 2) (i, d)
leaf = links[:, 0]
d_link = links[:, 1]
c = C.to(device)
dx = 0.5

p_correct = centers64[leaf] + c[d_link].to(DTYPE) * dx          # (n,3) x,y,z
p_plan = torch.stack((
    gp.x0.to(DTYPE) + gp.wx + 0.5,
    gp.y0.to(DTYPE) + gp.wy + 0.5,
    gp.z0.to(DTYPE) + gp.wz + 0.5,
), dim=1)                                                     # (n,3) x,y,z

diff = (p_correct - p_plan).abs()
print(f"n_links={links.shape[0]}")
print(f"max |p_correct - p_plan| = {diff.max().item():.6e}")
print(f"mean |p_correct - p_plan| = {diff.mean().item():.6e}")
print(f"aligned links: {int((diff.max(dim=1).values < 1e-9).sum())} / {links.shape[0]}")

# verify a Couette field is now sampled at the correct y
H = float(shape[1])
U = 0.05
def sample_ux(p):
    yi = torch.floor(p[:, 1]).to(torch.int64).clamp(0, shape[1] - 1)
    return U * (yi.to(DTYPE) + 0.5) / H
ux_diff = (sample_ux(p_correct) - sample_ux(p_plan)).abs()
print(f"Couette ux sampling error: max={ux_diff.max().item():.6e} mean={ux_diff.mean().item():.6e}")
