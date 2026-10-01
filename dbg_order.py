import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, DOMAIN_OUT
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

center=(80.0,48.0,48.0); radius=10.0
o = build_octree_shell((96, 96, 160), center=center, radius=radius,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
n1 = o.n_leaf_level(1)
l1 = o._l1_coords
print("l1 shape", l1.shape, "n1", n1, "n_leaf", o.n_leaf)
lc = (o.leaf_center[86404].to(torch.float64)*2 - 0.5).long()
print("leaf 86404 center", o.leaf_center[86404].tolist(), "-> lc", lc.tolist())
print("l1[86404]   ", l1[86404].tolist())
print("l1[87420]   ", l1[87420].tolist())
print("leaf 87420 center", o.leaf_center[87420].tolist())
# is leaf_center consistent with l1 row order?
c2 = (l1.to(torch.float64) + 0.5) / 2.0
diff = (c2 - o.leaf_center[:n1].to(torch.float64)).abs().max()
print("max |leaf_center - l1_center| over first n1:", float(diff))
# find rows where leaf_center != l1-derived center
bad = (c2 - o.leaf_center[:n1].to(torch.float64)).abs().max(dim=1).values > 1e-6
print("mismatched rows:", int(bad.sum()), "e.g.", torch.nonzero(bad, as_tuple=False)[:5].flatten().tolist())
nt = o.neighbor_table
print("nt[5,86404] =", int(nt[5,86404]), " nt[5,87420] =", int(nt[5,87420]))
print("nt[5,86404] src target:", (l1[86404] + torch.tensor([0,0,1])).tolist(),
      " bound:", [160<<1, 96<<1, 96<<1])
