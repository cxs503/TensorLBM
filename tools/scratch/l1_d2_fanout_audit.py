"""Part 2: fanout / BFL / ghost Python-loop scale for d1 vs d2 (CPU)."""
import sys
sys.path.insert(0, "/root/TensorLBM_feat2/src")
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, sphere_distance_field
from tensorlbm.amr_shell_planning import plan_body_shell_box

nx, ny, nz = 96, 64, 64
radius = 6.0
center = (nx * 0.5, ny * 0.5, nz * 0.5)
bl = max(2.0, round(radius / 2.0))
dev = torch.device("cpu")
solid_coarse = sphere_distance_field((nz, ny, nx), center, radius, dev) <= 0.0
plan = plan_body_shell_box(solid_coarse, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
center_l1 = (center[0] * 2.0 - box.x0 * 2, center[1] * 2.0 - box.y0 * 2,
             center[2] * 2.0 - box.z0 * 2)

for d_max in (1, 2):
    octree = build_octree_shell(
        l1_shape, center=center_l1, radius=radius * 2.0,
        bl_thickness_cells=bl, d_max=d_max, lattice="D3Q27", device=dev,
    )
    print(f"\n==== d_max={d_max} n_leaf={octree.n_leaf} ====")
    nt = octree.neighbor_table
    print(f"neighbor_table (Q,n_leaf)={tuple(nt.shape)}")
    for name, val in (("SHELL_OUTSIDE", -1), ("SOLID", -2), ("FANOUT", -3), ("DOMAIN_OUT", -4)):
        print(f"  {name}: {int((nt == val).sum())}")
    print(f"  valid leaf enums: {int((nt >= 0).sum())}")
    print(f"  cross_level_donor>=0: {int((octree.cross_level_donor >= 0).sum())}")
    n_fo = len(octree.interface_fanout)
    tot_fo = sum(len(g) for g in octree.interface_fanout.values())
    print(f"  interface_fanout groups={n_fo} total_members={tot_fo}")
    fo_pos = getattr(octree, "fanout_pos", None)
    print(f"  fanout_pos cached: {None if fo_pos is None else tuple(fo_pos.shape)}")
    print(f"  bfl_mask sum: {int(octree.bfl_mask.sum())}")
    print(f"  interface_links rows: {octree.interface_links.shape[0]}")
    print(f"  stats: { {k: octree.stats.get(k) for k in ('n_cross_level_donor','n_fanout_groups')} }")
    # shell/covered geometry
    print(f"  shell cells (L1): {int(octree._shell_mask.sum())}  solid cells: {int(octree._solid.sum())}")
    # per-rank ghost row counts (contiguous, ws=16)
    from tensorlbm.octree_boundary.stepping import build_ghost_plan
    from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
    gp = build_ghost_plan(octree, tuple(octree.meta["shape"]))
    from tensorlbm.octree_boundary.distributed_stepping import split_leaf_bounds
    ws = 16
    for r in (0, 15):
        lo, hi = split_leaf_bounds(octree.n_leaf, ws)[r]
        gpl, _ = _slice_ghost_plan(gp, lo, hi, hi - lo)
        print(f"  contiguous ws=16 rank{r}: leaves={hi-lo} local_ghost={gpl.n_ghost}")
    # ghost_vals tensor per substep
    print(f"  ghost_vals (Q,n_ghost) = {27*gp.n_ghost*4/1e6:.2f}MB")
    print(f"  full_pc (Q,n_leaf) = {27*octree.n_leaf*4/1e6:.2f}MB")
