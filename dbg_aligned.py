import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, DOMAIN_OUT
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
# same import set as the anomalous runs
from tensorlbm.octree_boundary.topology import _classify_targets
from tensorlbm.d3q27 import C

center=(80.0,48.0,48.0); radius=10.0
o = build_octree_shell((96, 96, 160), center=center, radius=radius,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
nt = o.neighbor_table
n_dom = int((nt == DOMAIN_OUT).sum())
dd, ii = torch.nonzero(nt == DOMAIN_OUT, as_tuple=True)
print("DOMAIN_OUT count:", n_dom, "unique leaves:", len(torch.unique(ii)))
if len(ii):
    print("first leaves:", torch.unique(ii)[:8].tolist())
    print("dirs:", torch.unique(dd).tolist())
# leaf z-centers of DOMAIN_OUT leaves
print("DOMAIN_OUT leaf z-centers:", float(o.leaf_center[ii,2].min()), float(o.leaf_center[ii,2].max()))
# check _l1_coords alignment again
n1 = o.n_leaf_level(1)
l1 = o._l1_coords
lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
print("aligned:", bool((l1 == lc2).all()))
