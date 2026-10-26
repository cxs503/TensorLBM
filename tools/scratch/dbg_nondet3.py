import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

for trial in range(5):
    # R6 first, then R10, in the same process (like verify_domain_out.py)
    o6 = build_octree_shell((96, 64, 64), center=(48.0, 32.0, 32.0), radius=6.0,
        bl_thickness_cells=3.0, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=sphere_inside_fn((48.0,32.0,32.0), 6.0))
    o = build_octree_shell((96, 96, 160), center=(80.0, 48.0, 48.0), radius=10.0,
        bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=sphere_inside_fn((80.0,48.0,48.0), 10.0))
    n1 = o.n_leaf_level(1)
    l1 = o._l1_coords
    lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
    aligned = bool((l1 == lc2).all())
    print(f"trial {trial}: R6 n_leaf={o6.n_leaf} R10 aligned={aligned} "
          f"l1[86404]={l1[86404].tolist()} nt[5,86404]={int(o.neighbor_table[5,86404])}")
