#!/usr/bin/env python3
"""Static shell-band structure probe R6 vs R8 (no time stepping).

Quantifies the "shell band coverage" hypothesis:
  1. radial leaf rings between wall and shell outer boundary vs cos_theta
  2. BFL upstream donor composition per ring (leaf / ghost / solid / fanout)
  3. BFL link distance-from-wall distribution
  4. interface (ghost-fed) link distance-from-wall distribution
  5. for the innermost BFL ring: is the upstream donor the next ring or a ghost?
"""
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d
from tensorlbm.octree_boundary.geometry import (
    FANOUT, SHELL_OUTSIDE, SOLID, build_octree_shell,
)
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry,
)
from tensorlbm.static_block_amr import NestedStaticBlockAMR3D, StaticBlockAMRConfig

RATIO, GHOST = 2, 1


def diag(radius, nx, ny, nz, bl):
    dev = torch.device("cpu")
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
                                bl_thickness_cells=bl, d_max=1, transition=1,
                                device=dev)
    center = torch.tensor(phys_center, dtype=torch.float64, device=dev)
    centers64 = (octree._l1_coords.to(torch.float64) + 0.5) / (
        2.0 ** octree.leaf_level.to(torch.float64))[:, None]
    r_vec = centers64 - center
    r_norm = r_vec.norm(dim=1).clamp_min(1e-12)
    cos_theta = (r_vec[:, 0] / r_norm).clamp(-1.0, 1.0)
    dist_wall = (r_norm - radius1) * 2.0 ** octree.leaf_level.to(torch.float64)
    # leaf-lattice distance from the wall (in units of the leaf's own dx)
    dx_leaf = 2.0 ** (-octree.leaf_level.to(torch.float64))
    dist_wall_dx = (r_norm - radius1) / dx_leaf

    print(f"=== R{int(radius)} bl={bl} (R_l1={radius1:.1f}) ===")
    print(f"  n_leaf={octree.n_leaf} n_bfl={int(octree.bfl_mask.sum())} "
          f"n_interface={octree.interface_links.shape[0]}")
    # -- 1. radial rings: hist of dist_wall_dx over ALL leaves and over
    #    interface leaves, split by cos_theta region
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 30, 1e9]
    def hist(vals):
        h = [int(((vals >= bins[i]) & (vals < bins[i + 1])).sum()) for i in range(len(bins) - 1)]
        return h
    print("  all-leaves dist_wall/dx histogram:", hist(dist_wall_dx))
    # interface leaves
    il = octree.interface_links[:, 0]
    iface_dist = dist_wall_dx[il]
    print("  interface-leaf dist_wall/dx hist:", hist(iface_dist),
          f"(max={iface_dist.max().item():.1f})")
    # BFL leaves
    bfl_leaf = (octree.bfl_mask.sum(dim=0) > 0)
    bfl_dist = dist_wall_dx[bfl_leaf]
    print("  BFL-leaf     dist_wall/dx hist:", hist(bfl_dist))
    # per cos_theta band: outermost interface-leaf distance (coverage depth)
    for lo, hi in ((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)):
        sel = (cos_theta >= lo) & (cos_theta < hi)
        d_if = dist_wall_dx[il][cos_theta[il] >= lo]
        d_if = dist_wall_dx[il][(cos_theta[il] >= lo) & (cos_theta[il] < hi)]
        print(f"  cos_theta [{lo:+.1f},{hi:+.1f}): leaves={int(sel.sum())} "
              f"iface_outer_dist_max={d_if.max().item() if d_if.numel() else float('nan'):.1f} "
              f"iface_outer_dist_min={d_if.min().item() if d_if.numel() else float('nan'):.1f}")
    # -- 2. BFL upstream donor composition by dist ring
    nt = octree.neighbor_table
    opp = octree._opp
    n_leaf = octree.n_leaf
    up_type = torch.zeros((n_leaf, 5), dtype=torch.int64)  # leaf/ghost/solid/fanout/other
    for d in range(1, 19):
        m = octree.bfl_mask[d]
        if not bool(m.any()):
            continue
        up = nt[int(opp[d].item())]
        for i in torch.nonzero(m, as_tuple=False).squeeze(1):
            u = int(up[i].item())
            if u >= 0:
                up_type[i, 0] += 1
            elif u == SHELL_OUTSIDE:
                up_type[i, 1] += 1
            elif u == SOLID:
                up_type[i, 2] += 1
            elif u == FANOUT:
                up_type[i, 3] += 1
            else:
                up_type[i, 4] += 1
    bfl_i = torch.nonzero(octree.bfl_mask.sum(dim=0) > 0, as_tuple=False).squeeze(1)
    # per ring (dist_wall_dx rounded): upstream donor type counts
    ring = dist_wall_dx[bfl_i].round().to(torch.int64).clamp(0, 5)
    for r in sorted(set(ring.tolist())):
        sel = ring == r
        s = up_type[bfl_i[sel]].sum(dim=0)
        print(f"  BFL ring dist={r}dx: n_leaves={int(sel.sum())} "
              f"up_leaf={int(s[0])} up_ghost={int(s[1])} up_solid={int(s[2])} "
              f"up_fanout={int(s[3])}")
    # -- 3. q-field stats per ring
    qf = octree.q_field
    print("  q<0.5 (lin) fraction per BFL leaf ring:",
          {int(r): round(float((qf[1:, bfl_i[ring == r]] < 0.5).float().mean().item()), 3)
           for r in sorted(set(ring.tolist()))})
    print()


for bl in (3.0, 4.0, 5.0):
    diag(6.0, 96, 64, 64, bl)
    diag(8.0, 128, 88, 88, bl)
