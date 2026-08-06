#!/usr/bin/env python3
"""Geometry comparison R6 vs R8: L1 block, shell band, BFL links, covered cells."""
import torch
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.sphere_amr_common import build_fine_block_geometry, build_sphere_geometry
from tensorlbm.static_block_amr import NestedStaticBlockAMR3D, StaticBlockAMRConfig

RATIO, GHOST = 2, 1

for (nx, ny, nz, radius) in ((96, 64, 64, 6.0), (128, 88, 88, 8.0)):
    device = torch.device("cpu")
    shape = (nz, ny, nx)
    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    solid_coarse, _ = build_sphere_geometry(nx, ny, nz, cx, cy, cz, radius, device)
    plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
    box1 = plan.box
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, 0.06)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)
    s1, fc1, radius1, _l1 = build_fine_block_geometry(
        box1, (cx, cy, cz), radius, RATIO, GHOST, device)
    nz1, ny1, nx1 = s1
    config1 = StaticBlockAMRConfig(box1, tau_coarse=0.5288, reflux=True,
                                   ghost_interpolation="injection")
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    octree = build_octree_shell(s1, phys_center, radius1, bl_thickness_cells=3.0,
                                d_max=1, transition=1, device=device)
    cov = octree._shell_mask
    solid = octree._solid
    # distance from covered region to L1 block edge
    nz_, ny_, nx_ = cov.shape
    cov_idx = cov.nonzero()
    lo = cov_idx.min(dim=0).values
    hi = cov_idx.max(dim=0).values
    dist_edges = min(
        int(lo[0]), int(lo[1]), int(lo[2]),
        int(nz_ - 1 - hi[0]), int(ny_ - 1 - hi[1]), int(nx_ - 1 - hi[2]),
    )
    bfl = octree.bfl_mask
    print(f"R{int(radius)}: coarse={shape} L1={s1} box1=({box1.x0},{box1.x1},{box1.y0},{box1.y1},{box1.z0},{box1.z1})")
    print(f"   phys_center={tuple(round(v,3) for v in phys_center)} radius_l1={radius1:.3f}")
    print(f"   covered={int(cov.sum())} solid_l1={int(solid.sum())} n_leaf={octree.n_leaf} "
          f"n_bfl_links={int(bfl.sum())} n_interface_links={octree.interface_links.shape[0]}")
    print(f"   covered bbox lo={tuple(int(v) for v in lo)} hi={tuple(int(v) for v in hi)} "
          f"dist_to_edge={dist_edges}")
    # how many covered cells are interior (all 8 leaves) vs partial
    host = octree.leaf_host_cell
    cells, counts = torch.unique(host, dim=0, return_counts=True)
    print(f"   host cells={cells.shape[0]} leaf-per-cell min={int(counts.min())} max={int(counts.max())} "
          f"full8={(counts==8).sum().item()}/{cells.shape[0]}")
    # per-leaf level histogram
    lv = octree.leaf_level
    print(f"   leaf levels: l1={int((lv==1).sum())} l2={int((lv==2).sum())}")
    # solid cells adjacent to covered (inner boundary)
    from tensorlbm.octree_boundary.stepping import build_shell_coarse_links
    links_full = build_shell_coarse_links(cov, None, q=19)
    links_solid_excl = build_shell_coarse_links(cov, solid, q=19)
    print(f"   coarse crossing links: full={int(links_full.outgoing_origins.sum()+links_full.incoming_origins.sum())} "
          f"solid-excl={int(links_solid_excl.outgoing_origins.sum()+links_solid_excl.incoming_origins.sum())}")
    print()
