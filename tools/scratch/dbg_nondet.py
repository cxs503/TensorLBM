import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

center=(80.0,48.0,48.0); radius=10.0
for trial in range(5):
    o = build_octree_shell((96, 96, 160), center=center, radius=radius,
        bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
    n1 = o.n_leaf_level(1)
    l1 = o._l1_coords
    lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
    print(f"trial {trial}: l1[86404]={l1[86404].tolist()} leaf_center[86404]={o.leaf_center[86404].tolist()} "
          f"aligned={bool((l1 == lc2).all())} nt[5,86404]={int(o.neighbor_table[5,86404])}")
