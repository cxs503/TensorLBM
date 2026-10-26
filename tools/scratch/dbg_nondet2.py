import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, morton_encode_batch
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.octree_boundary.topology import _classify_targets
from tensorlbm.d3q27 import C

center=(80.0,48.0,48.0); radius=10.0
for trial in range(10):
    o = build_octree_shell((96, 96, 160), center=center, radius=radius,
        bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
    n1 = o.n_leaf_level(1)
    l1 = o._l1_coords
    lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
    aligned = bool((l1 == lc2).all())
    # also verify _l1_coords is morton-sorted
    lm = morton_encode_batch(torch.full((n1,),1), l1, o._k)
    sorted_ok = bool((lm.diff() >= 0).all())
    print(f"trial {trial}: aligned={aligned} l1_morton_sorted={sorted_ok} "
          f"l1[86404]={l1[86404].tolist()} nt[5,86404]={int(o.neighbor_table[5,86404])}")
