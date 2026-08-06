#!/usr/bin/env python3
"""q-field / BFL-link geometry comparison R6 vs R8 (no simulation)."""
import torch
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import C, equilibrium3d
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
    config1 = StaticBlockAMRConfig(box1, tau_coarse=0.5288, reflux=True,
                                   ghost_interpolation="injection")
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    octree = build_octree_shell(s1, phys_center, radius1, bl_thickness_cells=3.0,
                                d_max=1, transition=1, device=device)
    q = octree.q_field.float()
    m = octree.bfl_mask
    qq = q[m]
    n = m.sum().item()
    print(f"R{int(radius)}: n_bfl={int(n)} "
          f"q<0.5: {int((qq<0.5).sum())} ({100*(qq<0.5).float().mean():.1f}%) "
          f"q>=0.5: {int((qq>=0.5).sum())} "
          f"q_mean={qq.mean():.3f} q<0.25: {int((qq<0.25).sum())} q>0.75: {int((qq>0.75).sum())}")
    # per-direction counts with c_x != 0
    per_dir = []
    for d in range(1, 19):
        cnt = int(m[d].sum().item())
        if cnt:
            per_dir.append((int(C[d,0].item()), int(C[d,1].item()), int(C[d,2].item()), cnt))
    print(f"   dir counts: {per_dir}")
    # links per leaf histogram
    per_leaf = m.sum(dim=0)
    print(f"   per-leaf links: max={int(per_leaf.max())} mean={per_leaf.float().mean():.2f} "
          f"leaves_with_links={int((per_leaf>0).sum())}/{octree.n_leaf}")
    # distance of BFL leaves from sphere surface
    from tensorlbm.octree_boundary.geometry import sphere_distance_field
    dist = sphere_distance_field(s1, phys_center, radius1, device)
    leaf_centers = octree.leaf_center
    # world coords are (x,y,z); dist field is indexed (z,y,x)
    bfl_leaves = (per_leaf > 0)
    lc = leaf_centers[bfl_leaves]
    zz = torch.floor(lc[:, 2]).clamp(0, s1[0]-1).long()
    yy = torch.floor(lc[:, 1]).clamp(0, s1[1]-1).long()
    xx = torch.floor(lc[:, 0]).clamp(0, s1[2]-1).long()
    d_leaf = dist[zz, yy, xx]
    print(f"   bfl leaf dist-to-surface: mean={d_leaf.float().mean():.3f} "
          f"min={d_leaf.min():.3f} max={d_leaf.max():.3f}")
    print()
