import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, morton_encode_batch
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

center=(80.0,48.0,48.0); radius=10.0
o = build_octree_shell((96, 96, 160), center=center, radius=radius,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
n1 = o.n_leaf_level(1)
l1 = o._l1_coords
lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
print("l1 == lc2 all:", bool((l1 == lc2).all()))
print("l1[86404]  ", l1[86404].tolist())
print("lc2[86404] ", lc2[86404].tolist())
print("leaf_center[86404]", o.leaf_center[86404].tolist())
# l1 itself sorted by morton?
k = o._k
lm = morton_encode_batch(torch.full((n1,),1), l1, k)
print("l1 morton sorted:", bool((lm.diff() >= 0).all()))
# is there any duplicate row in l1?
print("unique l1 rows:", torch.unique(l1, dim=0).shape[0], "of", n1)
# leaf_morton consistent with leaf_center?
lm2 = morton_encode_batch(torch.full((n1,),1), lc2, k)
print("morton(lc2) == leaf_morton:", bool((lm2 == o.leaf_morton[:n1]).all()))
print("leaf_morton[86404]", int(o.leaf_morton[86404]))
print("lm[86404]", int(lm[86404]), " lm2[86404]", int(lm2[86404]))
