#!/usr/bin/env python3
"""Static geometry probe for the R8 restriction/reflux imbalance.

Checks:
  1. covered & _solid intersection (candidate: restriction extracts solid momentum)
  2. every covered cell hosts exactly 8 leaves (full8 / volume weight)
  3. correction stencil coverage: every covered-boundary link has an exterior
     correction cell; count exterior cells adjacent to covered
  4. fine interface links (SHELL_OUTSIDE leaf links) vs covered boundary pairing
  5. restriction neq rescale factors (leaf tau chain) at R6 vs R8
  6. reflux cap headroom: requested vs applied inventory per direction
"""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import C
from tensorlbm.octree_boundary.geometry import (
    SHELL_OUTSIDE, build_octree_shell,
)
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.sphere_amr_common import build_fine_block_geometry, build_sphere_geometry

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=128)
p.add_argument("--ny", type=int, default=88)
p.add_argument("--nz", type=int, default=88)
p.add_argument("--radius", type=float, default=8.0)
p.add_argument("--d-max", type=int, default=1)
p.add_argument("--bl-thickness", type=float, default=3.0)
p.add_argument("--reynolds", type=float, default=100.0)
p.add_argument("--lattice-speed", type=float, default=0.06)
args = p.parse_args()

device = torch.device("cpu")
shape = (args.nz, args.ny, args.nx)
cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0
RATIO, GHOST = 2, 1
nu = args.lattice_speed * (2.0 * args.radius) / args.reynolds
tau_coarse = 0.5 + 3.0 * nu

solid_coarse, _ = build_sphere_geometry(args.nx, args.ny, args.nz, cx, cy, cz, args.radius, device)
plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
box1 = plan.box
s1, fc1, radius1, _l1 = build_fine_block_geometry(box1, (cx, cy, cz), args.radius, RATIO, GHOST, device)
nz1, ny1, nx1 = s1
phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
octree = build_octree_shell(
    s1, phys_center, radius_l1 := radius1,
    bl_thickness_cells=args.bl_thickness, d_max=args.d_max,
    transition=1, device=device)
print(f"L1 shape={s1} radius_l1={radius1} phys_center={phys_center}")

covered = octree._shell_mask
solid = octree._solid
assert covered is not None and solid is not None
n_cov = int(covered.sum())
n_sol = int(solid.sum())
inter = int((covered & solid).sum())
print(f"[1] covered={n_cov} solid={n_sol} covered&solid={inter}  -> candidate2 "
      f"{'CONFIRMED (covered contains solid cells!)' if inter else 'clean (disjoint)'}")

# 2. full8
host = octree.leaf_host_cell
cells, cell_id = torch.unique(host, dim=0, return_inverse=True)
counts = torch.bincount(cell_id, minlength=cells.shape[0])
bad = int((counts != 8).sum())
vol = octree.leaf_volume()
vol_sum = torch.zeros(cells.shape[0], dtype=torch.float64).scatter_add_(0, cell_id, vol)
badvol = int((vol_sum != 1.0).sum())
print(f"[2] n_leaf={octree.n_leaf} n_cells={cells.shape[0]} "
      f"cells_without_8_leaves={bad} cells_with_volume!=1={badvol}")

# 3. correction stencil: exterior cells adjacent to covered (all Q dirs, no solid)
c = C.to(device)
adj = torch.zeros_like(covered)
for d in range(1, 19):
    cx_, cy_, cz_ = (int(v) for v in c[d].tolist())
    dest_covered = torch.roll(covered, shifts=(-cz_, -cy_, -cx_), dims=(0, 1, 2))
    dest_solid = torch.roll(solid, shifts=(-cz_, -cy_, -cx_), dims=(0, 1, 2))
    # outgoing: covered cell whose dest is exterior fluid
    adj |= covered & ~dest_covered & ~dest_solid
# exterior cells that receive from covered
ext_recv = torch.zeros_like(covered)
for d in range(1, 19):
    cx_, cy_, cz_ = (int(v) for v in c[d].tolist())
    dest_covered = torch.roll(covered, shifts=(-cz_, -cy_, -cx_), dims=(0, 1, 2))
    ext_recv |= (~covered) & (~solid) & dest_covered
print(f"[3] covered-boundary cells (outgoing into fluid) = {int(adj.sum())}, "
      f"exterior receiving cells = {int(ext_recv.sum())}")

# 4. fine interface links
links = octree.interface_links
print(f"[4] n_interface_links(fine)={links.shape[0]} n_ghost={int((octree.neighbor_table == SHELL_OUTSIDE).sum())}")
# ghost targets' host L1 cells: are they all exterior (not covered)?
gp = build_ghost_plan(octree, s1)
if gp.n_ghost:
    leaf_lev = octree.leaf_level[gp.leaf]
    dx = 2.0 ** (-leaf_lev.to(torch.float64))
    coords = torch.cat((octree._l1_coords, octree._l2_coords), dim=0)
    centers64 = (coords.to(torch.float64) + 0.5) / (2.0 ** octree.leaf_level.to(torch.float64))[:, None]
    p_xyz = centers64[gp.leaf] + c[gp.direction].to(torch.float64) * dx[:, None]
    host_g = torch.floor(p_xyz)[:, [2, 1, 0]].to(torch.int64)
    host_g[:, 0] = host_g[:, 0].clamp(0, nz1 - 1)
    host_g[:, 1] = host_g[:, 1].clamp(0, ny1 - 1)
    host_g[:, 2] = host_g[:, 2].clamp(0, nx1 - 1)
    cov_g = covered[host_g[:, 0], host_g[:, 1], host_g[:, 2]]
    sol_g = solid[host_g[:, 0], host_g[:, 1], host_g[:, 2]]
    print(f"    ghost hosts: covered={int(cov_g.sum())} solid={int(sol_g.sum())} "
          f"exterior={int((~cov_g & ~sol_g).sum())} of {gp.n_ghost}")
    # direction histogram of ghost hosts in covered cells
    for d in range(1, 19):
        sel = gp.direction == d
        if bool(sel.any()) and bool(cov_g[sel].any()):
            print(f"    dir {d:2d} c=({c[d,0].item()},{c[d,1].item()},{c[d,2].item()}): "
                  f"ghost hosts covered={int(cov_g[sel].sum())}")

# 5. restriction rescale chain
from tensorlbm.static_block_amr import convective_refined_tau
t1 = convective_refined_tau(tau_coarse)
taus = [tau_coarse, t1]
print(f"[5] tau_coarse={tau_coarse:.6f} tau_leaf={t1:.6f} "
      f"restrict_scale=taus[0]/(0.5*tau_leaf)={tau_coarse/(0.5*t1):.6f}")

# 6. link pairing: fine link count vs covered-boundary link count per direction
print("[6] per-direction covered boundary links (fluid only) vs fine interface links:")
from collections import Counter
fine_dir = Counter()
if links.shape[0]:
    fine_dir = Counter(int(d) for d in links[:, 1].tolist())
for d in range(1, 19):
    cx_, cy_, cz_ = (int(v) for v in c[d].tolist())
    dest_covered = torch.roll(covered, shifts=(-cz_, -cy_, -cx_), dims=(0, 1, 2))
    dest_solid = torch.roll(solid, shifts=(-cz_, -cy_, -cx_), dims=(0, 1, 2))
    out = int((covered & ~dest_covered & ~dest_solid).sum())
    inc = int((~covered & ~solid & dest_covered).sum())
    print(f"    d={d:2d} c=({cx_},{cy_},{cz_}): coarse_out={out} coarse_in={inc} fine={fine_dir.get(d,0)}")
