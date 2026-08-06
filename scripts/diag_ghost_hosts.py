#!/usr/bin/env python3
"""Ghost plan host-cell distribution with the FIXED sample positions (c_vec[d_link]).

For R6 and R8: how many interface-link ghost slots land in covered / solid /
exterior L1 cells, per direction, plus the distance-from-surface of the
corresponding leaves.  Also: BFL links whose upstream donor is a ghost, and
the wall distance of those.
"""
import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d
from tensorlbm.octree_boundary.geometry import (
    SHELL_OUTSIDE, SOLID, build_octree_shell,
)
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry,
)
from tensorlbm.static_block_amr import (
    NestedStaticBlockAMR3D, StaticBlockAMRConfig,
)

RATIO, GHOST = 2, 1


def diag(radius, nx, ny, nz):
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
    nz1, ny1, nx1 = s1
    config1 = StaticBlockAMRConfig(box1, tau_coarse=0.5288, reflux=True,
                                   ghost_interpolation="injection")
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    octree = build_octree_shell(s1, phys_center, radius1, bl_thickness_cells=3.0,
                                d_max=1, transition=1, device=dev)
    gp = build_ghost_plan(octree, s1)
    n_link = gp.n_ghost
    solid = octree._solid
    covered = octree._shell_mask
    c_vec = C.to(dev)
    leaf = gp.leaf
    d_link = octree.interface_links[:, 1]
    level_i = octree.leaf_level[leaf]
    dx = 2.0 ** (-level_i.to(torch.float64))
    coords = octree._l1_coords
    centers64 = (coords.to(torch.float64) + 0.5) / (
        2.0 ** octree.leaf_level.to(torch.float64))[:, None]
    p_xyz = centers64[leaf] + c_vec[d_link].to(torch.float64) * dx[:, None]
    p = p_xyz[:, [2, 1, 0]]
    cell_p = torch.stack((
        p[:, 0].floor().to(torch.int64).clamp(0, nz1 - 1),
        p[:, 1].floor().to(torch.int64).clamp(0, ny1 - 1),
        p[:, 2].floor().to(torch.int64).clamp(0, nx1 - 1),
    ), dim=1)
    is_solid = solid[cell_p[:, 0], cell_p[:, 1], cell_p[:, 2]]
    is_cov = covered[cell_p[:, 0], cell_p[:, 1], cell_p[:, 2]]
    n_sol = int(is_solid.sum().item())
    n_cov = int((is_cov & ~is_solid).sum().item())
    n_ext = n_link - n_sol - n_cov
    print(f"=== R{int(radius)} ===")
    print(f"  interface links={n_link} ghost hosts: solid={n_sol} "
          f"({100*n_sol/n_link:.1f}%) covered={n_cov} ({100*n_cov/n_link:.1f}%) "
          f"exterior={n_ext} ({100*n_ext/n_link:.1f}%)")

    # leaf distance from surface for fallback links vs all links
    cen = torch.tensor(phys_center, dtype=torch.float64)
    d_leaf = torch.sqrt(((centers64 - cen) ** 2).sum(dim=1))
    r = float(radius1)
    surf_dist = d_leaf[leaf] - r
    print(f"  all links: leaf surf-dist mean={surf_dist.mean():.3f} "
          f"min={surf_dist.min():.3f} max={surf_dist.max():.3f}")
    print(f"  fallback links: leaf surf-dist mean={surf_dist[is_solid].mean():.3f} "
          f"min={surf_dist[is_solid].min():.3f} max={surf_dist[is_solid].max():.3f}")
    # ghost position distance from surface
    gdist = torch.sqrt(((p_xyz - cen) ** 2).sum(dim=1)) - r
    print(f"  ghost pos surf-dist: all mean={gdist.mean():.3f} "
          f"min={gdist.min():.3f} max={gdist.max():.3f}")
    print(f"    fallback ghost pos: mean={gdist[is_solid].mean():.3f} "
          f"min={gdist[is_solid].min():.3f} max={gdist[is_solid].max():.3f}")
    print(f"    covered-host ghost pos: mean={gdist[is_cov & ~is_solid].mean():.3f}")
    print(f"    exterior-host ghost pos: mean={gdist[~is_cov & ~is_solid].mean():.3f}")

    # per-direction fallback fraction
    from collections import Counter
    fb_dir = Counter()
    tot_dir = Counter()
    for d in range(1, 19):
        sel = d_link == d
        tot_dir[d] = int(sel.sum().item())
        fb_dir[d] = int((sel & is_solid).sum().item())
    rows = [f"d{d}({c_vec[d,0].item():+.0f},{c_vec[d,1].item():+.0f},{c_vec[d,2].item():+.0f}): "
            f"{fb_dir[d]}/{tot_dir[d]}" for d in range(1, 19) if tot_dir[d]]
    print("  per-dir fallback:", " ".join(rows))

    # BFL links: upstream donor class + how many BFL leaves are near ghost links
    m = octree.bfl_mask
    nt = octree.neighbor_table
    opp = OPPOSITE.to(dev)
    n_bfl = int(m.sum().item())
    up_ghost = 0
    up_solid = 0
    up_leaf = 0
    for d in range(1, 19):
        idx = torch.nonzero(m[d], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        up = nt[int(opp[d].item()), idx]
        up_leaf += int((up >= 0).sum().item())
        up_ghost += int((up == SHELL_OUTSIDE).sum().item())
        up_solid += int((up == SOLID).sum().item())
    print(f"  BFL links={n_bfl} upstream: leaf={up_leaf} ghost={up_ghost} solid={up_solid}")
    # BFL leaves with any interface link (these couple wall <-> exterior directly)
    bfl_leaf = m.any(dim=0)
    iface_leaf = torch.zeros(octree.n_leaf, dtype=torch.bool)
    iface_leaf[leaf] = True
    both = int((bfl_leaf & iface_leaf).sum().item())
    print(f"  leaves with BFL links={int(bfl_leaf.sum().item())} "
          f"with interface links={int(iface_leaf.sum().item())} both={both}")
    print()


for (r, nx, ny, nz) in ((6.0, 96, 64, 64), (8.0, 128, 88, 88)):
    diag(r, nx, ny, nz)
