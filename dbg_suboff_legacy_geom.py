#!/usr/bin/env python3
"""Legacy two-level SUBOFF shell geometry (host = coarse grid)."""
import torch

from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

nx, ny, nz = 128, 64, 64
L = 60
cx = 40
config = SuboffConfig()
srad = config.r_over_l * L
bl = max(2.0, round(srad / 2.0))
solid_cpu, _ = build_suboff_mask(
    hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
    cx=cx, cy=ny * 0.5, cz=nz * 0.5,
    length=L, radius=srad, config=config, device="cpu",
)
dev = torch.device("cpu")
octree = build_octree_shell(
    (nz, ny, nx), center=(cx, ny * 0.5, nz * 0.5),
    radius=max(srad, bl * 2),
    bl_thickness_cells=bl, d_max=1,
    lattice="D3Q27", device=dev,
    inside_fn=solid_mask_inside_fn(solid_cpu.bool(), device=dev),
)
print(f"LEGACY: n_leaf={octree.n_leaf} stats={octree.stats}")
bfl = octree.bfl_mask
print(f"LEGACY BFL wall links: {int(bfl.sum())}")
gplan = build_ghost_plan(octree, tuple(octree.meta["shape"]))
print(f"LEGACY ghost plan n_ghost={gplan.n_ghost}")
# ghost donor coarse positions
z0 = gplan.z0; y0 = gplan.y0; x0 = gplan.x0
if gplan.n_ghost:
    cz = z0.float() + 0.5; cy = y0.float() + 0.5; cxx = x0.float() + 0.5
    print("LEGACY ghost donor x range: [%.2f, %.2f]" % (cxx.min().item(), cxx.max().item()))
    print("LEGACY ghost donor y range: [%.2f, %.2f]" % (cy.min().item(), cy.max().item()))
    tail = (cxx > 70.0)
    print(f"LEGACY ghost donors downstream of body tail (x>70): {int(tail.sum())}")
