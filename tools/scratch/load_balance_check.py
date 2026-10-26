"""检查 d_max=2 时每 rank 的 L1/L2 叶子分布 (负载平衡)。"""
import sys
sys.path.insert(0, 'src')
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.octree_boundary.distributed_stepping import split_leaf_bounds
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

nx, ny, nz = 192, 128, 128
L = 80.0
config = SuboffConfig()
radius = config.r_over_l * L
solid, _ = build_suboff_mask(hull_type='bare_hull', nx=nx, ny=ny, nz=nz,
    cx=60, cy=ny*0.5, cz=nz*0.5, length=L, radius=radius, config=config, device='cpu')
solid = solid.bool()
octree = build_octree_shell((nz,ny,nx), center=(60,64,64), radius=max(radius, 4.0),
    bl_thickness_cells=2.0, d_max=2, lattice='D3Q27', device=torch.device('cpu'),
    inside_fn=solid_mask_inside_fn(solid))

n_leaf = octree.n_leaf
level = octree.leaf_level  # 1 or 2
bounds = split_leaf_bounds(n_leaf, 16)
print(f"n_leaf={n_leaf}, L1={int((level==1).sum())}, L2={int((level==2).sum())}")
print(f"{'rank':>4} {'range':>12} {'n':>5} {'L1':>5} {'L2':>5} {'L2%':>6}")
for r, (lo, hi) in enumerate(bounds):
    seg = level[lo:hi]
    n1 = int((seg == 1).sum())
    n2 = int((seg == 2).sum())
    print(f"{r:>4} {f'[{lo},{hi})':>12} {hi-lo:>5} {n1:>5} {n2:>5} {100.0*n2/max(hi-lo,1):>5.1f}%")
