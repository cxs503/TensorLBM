import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, DOMAIN_OUT, SHELL_OUTSIDE, SOLID
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.octree_boundary.topology import _classify_targets
from tensorlbm.d3q27 import C
from tensorlbm.octree_boundary.geometry import morton_encode_batch

center=(80.0,48.0,48.0); radius=10.0
o = build_octree_shell((96, 96, 160), center=center, radius=radius,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
nt = o.neighbor_table
n1 = o.n_leaf_level(1); n2 = o.n_leaf_level(2)
print("n1", n1, "n2", n2)
l1 = o._l1_coords
k = o._k
l1_morton = morton_encode_batch(torch.full((n1,),1), l1, k)
l1_sorted, l1_order = torch.sort(l1_morton)
def hit(sorted_m, q):
    p = torch.searchsorted(sorted_m, q)
    pm = p.clamp(max=sorted_m.shape[0]-1)
    return (p < sorted_m.shape[0]) & (sorted_m[pm] == q), p
d = 5
cd = C[d]
tgt = l1 + cd
q1 = morton_encode_batch(torch.full((n1,),1), tgt, k)
h1, p1 = hit(l1_sorted, q1)
cls = _classify_targets(tgt, 1, (96,96,160), center, radius, sphere_inside_fn(center, radius))
lc = (o.leaf_center[86404].to(torch.float64)*2 - 0.5).long()
row = int((l1 == lc).all(dim=1).nonzero()[0])
print(f"d={d} leaf86404 row={row} hit1={bool(h1[row])} cls={int(cls[row])} nt={int(nt[d,86404])}")
print("   target:", tgt[row].tolist(), "bound<<1:", [160<<1, 96<<1, 96<<1])
dd = (nt[d,:n1] == DOMAIN_OUT)
print(f"   nt[d] DOMAIN_OUT={int(dd.sum())}, of which cls says SHELL_OUTSIDE={int((cls[dd]==SHELL_OUTSIDE).sum())} "
      f"DOMAIN_OUT={int((cls[dd]==DOMAIN_OUT).sum())} SOLID={int((cls[dd]==SOLID).sum())} hit1={int(h1[dd].sum())}")
# and the nt value where cls==DOMAIN_OUT: are those hit1?
print("   cls==DOMAIN_OUT rows:", int((cls==DOMAIN_OUT).sum()), " hit1 of them:", int((h1 & (cls==DOMAIN_OUT)).sum()),
      " nt==DOMAIN_OUT of them:", int((nt[d,:n1][cls==DOMAIN_OUT]==DOMAIN_OUT).sum()))
