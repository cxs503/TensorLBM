"""验证交错分片 (round-robin) 是否平衡 L1/L2 分布。"""
import sys
sys.path.insert(0, 'src')
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
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
level = octree.leaf_level

def contiguous_bounds(n, nshards):
    base, extra = divmod(n, nshards)
    b, s = [], 0
    for i in range(nshards):
        sz = base + (1 if i < extra else 0)
        b.append((s, s+sz)); s += sz
    return b

def interleaved_bounds(n, nshards):
    # round-robin: each rank gets every nshards-th leaf
    b = []
    for r in range(nshards):
        idx = list(range(r, n, nshards))
        b.append((idx, idx))  # not contiguous — store indices
    return b

print("=== 连续分片 (当前) ===")
for r, (lo, hi) in enumerate(contiguous_bounds(n_leaf, 16)):
    seg = level[lo:hi]
    print(f"rank{r:>2}: L1={int((seg==1).sum()):>5} L2={int((seg==2).sum()):>5}")

print("=== 交错分片 (round-robin) ===")
for r in range(16):
    idx = list(range(r, n_leaf, 16))
    seg = level[torch.tensor(idx)]
    print(f"rank{r:>2}: L1={int((seg==1).sum()):>5} L2={int((seg==2).sum()):>5}")
