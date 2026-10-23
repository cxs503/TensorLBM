"""共性模块验证: geometry_adapters + build_octree_shell 泛化 (球体 + SUBOFF)。"""
import sys
sys.path.insert(0, 'src')
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import (
    sphere_inside_fn, solid_mask_inside_fn, solid_mask_shell_fn,
)

dev = torch.device('cpu')

# ---- 1. 球体 (默认 inside_fn, 回归) ----
shape = (16, 16, 16)
oct_sphere = build_octree_shell(
    shape, center=(8.0, 8.0, 8.0), radius=4.0,
    bl_thickness_cells=2.0, d_max=2, lattice='D3Q27', device=dev,
)
print(f'球体: n_leaf={oct_sphere.stats["n_leaf"]} fanout={len(oct_sphere.interface_fanout)}')

# ---- 2. 球体 (显式 sphere_inside_fn, 应与默认一致) ----
oct_sphere2 = build_octree_shell(
    shape, center=(8.0, 8.0, 8.0), radius=4.0,
    bl_thickness_cells=2.0, d_max=2, lattice='D3Q27', device=dev,
    inside_fn=sphere_inside_fn((8.0, 8.0, 8.0), 4.0),
)
print(f'球体(显式fn): n_leaf={oct_sphere2.stats["n_leaf"]} 一致={oct_sphere.stats["n_leaf"] == oct_sphere2.stats["n_leaf"]}')

# ---- 3. SUBOFF (solid mask 适配器) ----
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
L = 80.0
config = SuboffConfig()
radius = config.r_over_l * L
solid, _ = build_suboff_mask(
    hull_type='bare_hull', nx=96, ny=64, nz=64,
    cx=30, cy=32, cz=32, length=L, radius=radius,
    config=config, device='cpu')
solid = solid.bool()
print(f'SUBOFF solid: {tuple(solid.shape)}, 实体格={int(solid.sum())}')

# shell 带 (流体但靠近实体)
shell_fn = solid_mask_shell_fn(solid, bl_thickness=2.0)
inside_fn = solid_mask_inside_fn(solid)
# 壳层中心应在实体中心附近
oct_suboff = build_octree_shell(
    solid.shape, center=(30.0, 32.0, 32.0), radius=10.0,
    bl_thickness_cells=2.0, d_max=2, lattice='D3Q27', device=dev,
    inside_fn=inside_fn,
)
print(f'SUBOFF壳层: n_leaf={oct_suboff.stats["n_leaf"]} fanout={len(oct_suboff.interface_fanout)}')
print(f'SUBOFF壳层: n_shell_cells={oct_suboff.stats.get("n_shell_cells")}')
