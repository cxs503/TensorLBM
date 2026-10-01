import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, morton_encode_batch
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

center=(80.0,48.0,48.0); radius=10.0
o = build_octree_shell((96, 96, 160), center=center, radius=radius,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
n1 = o.n_leaf_level(1)
l1 = o._l1_coords
k = o._k
lm = morton_encode_batch(torch.full((n1,),1), l1, k)
print("morton(_l1_coords) == leaf_morton[:n1]:", bool((lm == o.leaf_morton[:n1]).all()))
# leaf_morton sorted?
print("leaf_morton[:n1] sorted:", bool((o.leaf_morton[:n1].diff() >= 0).all()))
# derive coords from leaf_center and check its morton
lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
lm2 = morton_encode_batch(torch.full((n1,),1), lc2, k)
print("morton(floor(leaf_center*2-0.5)) == leaf_morton[:n1]:", bool((lm2 == o.leaf_morton[:n1]).all()))
print("_l1_coords == floor(leaf_center*2-0.5):", bool((l1 == lc2).all()))
# shell cells: how many, and are they (z,y,x)?
sc = torch.nonzero(o._shell_mask, as_tuple=False)
print("shell cells count", sc.shape[0], "range z", int(sc[:,0].min()), int(sc[:,0].max()),
      "y", int(sc[:,1].min()), int(sc[:,1].max()), "x", int(sc[:,2].min()), int(sc[:,2].max()))
# level-1 leaves derived from shell cells: 2*cell+offset
n = sc.shape[0]
child = torch.arange(8, dtype=torch.int64)
bx, by, bz = child & 1, (child >> 1) & 1, (child >> 2) & 1
cells = sc[:, [2,1,0]].repeat_interleave(8, dim=0)
offs = torch.stack([bx, by, bz], dim=1).repeat(n, 1)
exp = 2*cells + offs   # (8N, 3) (x,y,z)
# is _l1_coords a permutation of exp?
print("exp shape", exp.shape, "n1", n1)
s_exp = exp[torch.argsort(morton_encode_batch(torch.full((exp.shape[0],),1), exp, k), stable=True)]
print("_l1_coords == sorted(exp):", bool((l1 == s_exp).all()))
s_lc = lc2[torch.argsort(lm2, stable=True)]
print("_l1_coords == sorted(leaf_center-derived):", bool((l1 == s_lc).all()))
print("leaf_morton[:5]", o.leaf_morton[:5].tolist())
print("lm[:5]", lm[:5].tolist())
