#!/usr/bin/env python3
"""Diagnose octree P3 sphere: freeze masks, shell coverage, BFL counts."""
import argparse
import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.geometry import build_octree_shell, SOLID, SHELL_OUTSIDE
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.sphere_amr_common import build_fine_block_geometry, build_sphere_geometry
from tensorlbm.static_block_amr import StaticBlockAMRConfig, NestedStaticBlockAMR3D

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--nx", type=int, default=96)
p.add_argument("--ny", type=int, default=64)
p.add_argument("--nz", type=int, default=64)
p.add_argument("--radius", type=float, default=6.0)
p.add_argument("--d-max", type=int, default=1)
p.add_argument("--bl-thickness", type=float, default=3.0)
p.add_argument("--shell-margin", type=int, default=6)
p.add_argument("--wall-margin", type=int, default=8)
p.add_argument("--wake-cells", type=int, default=32)
args = p.parse_args()

device = torch.device(args.device)
shape = (args.nz, args.ny, args.nx)
cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0
RATIO, GHOST = 2, 1

solid_coarse, solid_coarse_q = build_sphere_geometry(
    args.nx, args.ny, args.nz, cx, cy, cz, args.radius, device)
plan = plan_body_shell_box(solid_coarse, args.shell_margin, args.wake_cells, pad=args.wall_margin)
box1 = plan.box
print(f"L0 coarse solid cells: {int(solid_coarse.sum().item())}")
print(f"L1 box coarse: x=[{box1.x0},{box1.x1}) y=[{box1.y0},{box1.y1}) z=[{box1.z0},{box1.z1})")

rho = torch.ones(shape, device=device)
ux = torch.full_like(rho, 0.06)
zero = torch.zeros_like(rho)
coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

s1, fc1, radius1, _l1 = build_fine_block_geometry(
    box1, (cx, cy, cz), args.radius, RATIO, GHOST, device)
nz1, ny1, nx1 = s1
phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
print(f"L1 shape {s1}  phys_center {phys_center}  radius_l1 {radius1}")

octree = build_octree_shell(
    s1, phys_center, radius1,
    bl_thickness_cells=args.bl_thickness, d_max=args.d_max,
    transition=1, device=device)
print(f"n_leaf={octree.n_leaf}  (l1={octree.n_leaf_level(1)}, l2={octree.n_leaf_level(2)})")
print(f"shell cells={octree.stats['n_shell_cells']}  iface links={octree.stats['n_interface_links']}")

solid = octree._solid
print(f"L1 solid (octree._solid): {int(solid.sum().item())} cells")
shell_mask = octree._shell_mask
print(f"L1 shell_mask: {int(shell_mask.sum().item())} cells")
# overlap checks
print(f"solid & shell overlap: {int((solid & shell_mask).sum().item())}")

nt = octree.neighbor_table
n_solid = int((nt == SOLID).sum().item())
n_out = int((nt == SHELL_OUTSIDE).sum().item())
n_leaf_fluid = int((nt >= 0).sum().item())
print(f"neighbor_table: fluid={n_leaf_fluid} SOLID={n_solid} SHELL_OUTSIDE={n_out} "
      f"total={(nt >= -3).sum().item()} of {nt.numel()}")

bfl = octree.bfl_mask
print(f"BFL mask links: {int(bfl.sum().item())}  per-leaf with BFL: "
      f"{int(bfl.any(dim=0).sum().item())} of {octree.n_leaf}")

# q field sanity: q in (0,1]
q = octree.q_field[bfl]
print(f"BFL q range: [{float(q.min().item()):.4f}, {float(q.max().item()):.4f}]")

# hole check: are there L1 cells inside the sphere whose centre is fluid per the
# L1 mask but which host no leaf and are not solid? (gaps in coverage)
host = octree.leaf_host_cell
host_set = set(map(tuple, host.tolist()))
interior = torch.nonzero(solid, as_tuple=False).tolist()
in_interior_no_leaf = [tuple(c) for c in interior if tuple(c) not in host_set]
print(f"L1 solid cells not hosting any leaf: {len(in_interior_no_leaf)} / {len(interior)}")
# cells that are NOT solid, NOT shell, i.e. plain L1 fluid
plain = (~solid) & (~shell_mask)
print(f"L1 plain fluid cells (neither solid nor shell): {int(plain.sum().item())}")

# check that the L1 solid mask fully encloses the sphere: sample distance field
from tensorlbm.octree_boundary.geometry import sphere_distance_field
dist = sphere_distance_field(s1, phys_center, radius1, device)
dmin_solid = float(dist[solid].max().item())
dmax_fluid = float(dist[~solid].min().item())
print(f"solid cells max dist-to-surface: {dmin_solid:.4f} (should be < 0 => fully inside)")
print(f"fluid cells min dist-to-surface: {dmax_fluid:.4f} (should be > 0)")

# coarse sphere check
print(f"L0: sphere_mask solid={int(solid_coarse.sum().item())}  "
      f"freeze cells in coarse_f shape {tuple(solid_coarse_q.shape)}")
