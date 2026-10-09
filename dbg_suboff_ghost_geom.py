#!/usr/bin/env python3
"""Diagnose SUBOFF L1 shell geometry: ghost donor positions, coverage."""
import torch

from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.amr_shell_planning import plan_body_shell_box

nx, ny, nz = 128, 64, 64
L = 60
cx = 40
config = SuboffConfig()
srad = config.r_over_l * L
bl = max(2.0, round(srad / 2.0))
print(f"srad={srad:.3f} bl={bl}")

solid_cpu, _ = build_suboff_mask(
    hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
    cx=cx, cy=ny * 0.5, cz=nz * 0.5,
    length=L, radius=srad, config=config, device="cpu",
)
dev = torch.device("cpu")
solid_coarse = solid_cpu.bool().to(dev)
plan = plan_body_shell_box(solid_coarse, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
print(f"L1 box: x[{box.x0},{box.x1}) y[{box.y0},{box.y1}) z[{box.z0},{box.z1})")
print(f"l1_shape (z,y,x) = {l1_shape}")

radius_l1 = max(srad, bl * 2) * 2.0
print(f"radius_l1={radius_l1}")

def _suboff_inside_l1(centers):
    coarse = 0.5 * centers + torch.tensor(
        (box.x0, box.y0, box.z0), dtype=torch.float64, device=centers.device,
    )
    return solid_mask_inside_fn(solid_cpu.bool(), device=dev)(coarse)

octree = build_octree_shell(
    l1_shape, center=(0, 0, 0), radius=radius_l1,
    bl_thickness_cells=bl, d_max=2,
    lattice="D3Q27", device=dev,
    inside_fn=_suboff_inside_l1,
)
n_leaf = octree.n_leaf
print(f"n_leaf={n_leaf} d_max={octree.d_max}")
print(f"stats: {octree.stats}")

# Shell leaf extent in COARSE coords
host = octree.leaf_host_cell  # (n,3) (z,y,x) L1 frame
coarse_host = (host.float() / 2.0) + torch.tensor(
    [box.z0, box.y0, box.x0], dtype=torch.float32)
print("leaf host coarse x range: [%.1f, %.1f]" % (
    coarse_host[:, 2].min().item(), coarse_host[:, 2].max().item()))
print("leaf host coarse y range: [%.1f, %.1f]" % (
    coarse_host[:, 1].min().item(), coarse_host[:, 1].max().item()))
print("leaf host coarse z range: [%.1f, %.1f]" % (
    coarse_host[:, 0].min().item(), coarse_host[:, 0].max().item()))

# levels
lev = octree.leaf_level
print("leaf levels: l1=%d l2=%d" % (
    int((lev == 1).sum()), int((lev == 2).sum())))

# BFL wall links
bfl = octree.bfl_mask
n_wall = int(bfl.sum())
print(f"BFL wall links: {n_wall}")
# wall links by direction
from tensorlbm.d3q27 import C as C27
c = C27
for d in range(1, 27):
    nd = int(bfl[d].sum())
    if nd:
        print(f"  d={d} c={tuple(int(v) for v in c[d])} n={nd}")

# interface links (SHELL_OUTSIDE -> ghost fill)
links = octree.interface_links
print(f"interface_links: {links.shape[0]}")
if links.shape[0]:
    dirs = links[:, 1]
    for d in range(27):
        nd = int((dirs == d).sum())
        if nd:
            print(f"  iface d={d} c={tuple(int(v) for v in c[d])} n={nd}")

# ghost plan donors
gplan = build_ghost_plan(octree, tuple(octree.meta["shape"]))
print(f"ghost plan n_ghost={gplan.n_ghost}")
if gplan.n_ghost:
    # donor sample positions in coarse frame
    leaf = gplan.leaf
    dirn = gplan.direction
    z0 = gplan.z0; y0 = gplan.y0; x0 = gplan.x0
    # sample cell centres in L1 frame: donor cells lo..hi
    # convert to coarse coords
    cz = (z0.float() + 0.5) / 2.0 + box.z0
    cy = (y0.float() + 0.5) / 2.0 + box.y0
    cxx = (x0.float() + 0.5) / 2.0 + box.x0
    print("ghost donor lo-cell coarse x range: [%.2f, %.2f]" % (
        cxx.min().item(), cxx.max().item()))
    print("ghost donor lo-cell coarse y range: [%.2f, %.2f]" % (
        cy.min().item(), cy.max().item()))
    # how many ghost donors lie OUTSIDE the L1 box (coarse frame)?
    # box interior: x in [box.x0, box.x1), y in [box.y0, box.y1), z in ...
    ox = (cxx < box.x0) | (cxx >= box.x1)
    oy = (cy < box.y0) | (cy >= box.y1)
    oz = (cz < box.z0) | (cz >= box.z1)
    outside = (ox | oy | oz).sum().item()
    print(f"ghost donors outside L1 box: {outside} / {gplan.n_ghost}")
    # near wake: donors with coarse x > body tail (70)
    tail = (cxx > 70.0)
    print(f"ghost donors downstream of body tail (x>70): {int(tail.sum())}")
    # solid fallback usage
    print("ghost dirs sample:")
    for d in range(27):
        nd = int((dirn == d).sum())
        if nd:
            print(f"  gdir d={d} c={tuple(int(v) for v in c[d])} n={nd}")

# SHELL_OUTSIDE neighbours of boundary leaves: which links feed the wall leaves?
nt = octree.neighbor_table
opp = octree._opp
bfl_leaves = torch.nonzero(bfl.any(dim=0), as_tuple=False).squeeze(1)
print(f"n bfl leaves: {bfl_leaves.shape[0]}")
# for boundary leaves, count how many incoming links come from ghost cells
from tensorlbm.octree_boundary.geometry import SHELL_OUTSIDE
src = nt[opp]  # (Q,n) upstream donor for each direction
ghost_in = (src[:, bfl_leaves] == SHELL_OUTSIDE)
print(f"boundary-leaf incoming links from SHELL_OUTSIDE ghost: {int(ghost_in.sum())} / {bfl_leaves.shape[0]*27}")
# fraction of the leaf's incoming population that is ghost
per_leaf = ghost_in.sum(dim=0).float()
print("ghost incoming per boundary leaf: mean=%.2f max=%d min=%d" % (
    per_leaf.mean().item(), int(per_leaf.max()), int(per_leaf.min())))
