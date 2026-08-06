#!/usr/bin/env python3
"""Probe: which directions do near-wall (dist<3dx) interface links use? R8 bl=3."""
import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.octree_boundary.geometry import SHELL_OUTSIDE, build_octree_shell
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry,
)
from tensorlbm.static_block_amr import NestedStaticBlockAMR3D, StaticBlockAMRConfig

RATIO, GHOST = 2, 1
dev = torch.device("cpu")
radius, nx, ny, nz = 8.0, 128, 88, 88
shape = (nz, ny, nx)
cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
solid_coarse, _ = build_sphere_geometry(nx, ny, nz, cx, cy, cz, radius, dev)
plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
box1 = plan.box
rho = torch.ones(shape, device=dev)
ux = torch.full_like(rho, 0.06)
zero = torch.zeros_like(rho)
coarse_f = equilibrium3d(rho, ux, zero, zero, device=dev)
s1, fc1, radius1, _l1 = build_fine_block_geometry(
    box1, (cx, cy, cz), radius, RATIO, GHOST, dev)
config1 = StaticBlockAMRConfig(box1, tau_coarse=0.5288, reflux=True,
                               ghost_interpolation="injection")
amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST),
               float(fc1[2] - GHOST))
octree = build_octree_shell(s1, phys_center, radius1,
                            bl_thickness_cells=3.0, d_max=1, transition=1,
                            device=dev)
center = torch.tensor(phys_center, dtype=torch.float64, device=dev)
centers64 = (octree._l1_coords.to(torch.float64) + 0.5) / (
    2.0 ** octree.leaf_level.to(torch.float64))[:, None]
r_vec = centers64 - center
r_norm = r_vec.norm(dim=1).clamp_min(1e-12)
cos_theta = (r_vec[:, 0] / r_norm).clamp(-1.0, 1.0)
dx_leaf = 2.0 ** (-octree.leaf_level.to(torch.float64))
dist_wall_dx = (r_norm - radius1) / dx_leaf

il = octree.interface_links
leaf, dlink = il[:, 0], il[:, 1]
near = dist_wall_dx[leaf] < 3.0
print(f"interface links: total={il.shape[0]} near-wall(<3dx)={int(near.sum())}")
print("near-wall interface links by direction (d, c_d):")
for d in range(1, 19):
    sel = near & (dlink == d)
    if bool(sel.any()):
        leaves = leaf[sel]
        ct = cos_theta[leaves]
        dw = dist_wall_dx[leaves]
        print(f"  d={d:2d} c={tuple(int(v) for v in C[d].tolist())}: "
              f"n={int(sel.sum())} mean_cos_theta={ct.mean().item():+.3f} "
              f"mean_dist_wall={dw.mean().item():+.2f}dx "
              f"min_ct={ct.min().item():+.3f} max_ct={ct.max().item():+.3f}")
# For a few near-wall interface links: what is the neighbor position relative to wall?
print("\nneighbor position of near-wall interface links (leaf dist -> neighbor dist, dx units):")
for d in range(1, 19):
    sel = near & (dlink == d)
    if not bool(sel.any()):
        continue
    leaves = leaf[sel]
    nb_dist = (r_norm[leaves] ** 2
               + (dx_leaf[leaves] * torch.tensor([C[d, 0].item(), C[d, 1].item(), C[d, 2].item()],
                                                 dtype=torch.float64)) .pow(2).sum()
               + 2 * (r_vec[leaves] * (dx_leaf[leaves] * torch.tensor(
                   [C[d, 0].item(), C[d, 1].item(), C[d, 2].item()],
                   dtype=torch.float64))).sum(dim=1)).sqrt()
    nb_dist_dx = (nb_dist - radius1) / dx_leaf[leaves]
    # NB: this approx assumes the link is purely radial; print distribution anyway
    print(f"  d={d:2d}: neighbor dist range [{nb_dist_dx.min().item():+.2f}, "
          f"{nb_dist_dx.max().item():+.2f}] dx (approx radial)")
