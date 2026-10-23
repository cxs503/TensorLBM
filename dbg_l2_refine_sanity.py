#!/usr/bin/env python3
"""Extra sanity checks after the _level2_leaves refine fix."""
import torch

from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask


def build_suboff(d_max):
    nx, ny, nz = 96, 48, 48
    config = SuboffConfig()
    L = 40.0
    srad = config.r_over_l * L
    solid_cpu, _ = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=nx * 0.5, cy=ny * 0.5, cz=nz * 0.5,
        length=L, radius=srad, config=config, device="cpu",
    )
    bl = max(2.0, round(srad / 2.0))
    plan = plan_body_shell_box(
        solid_cpu.bool(), shell_margin=6, wake_cells=24, pad=8,
    )
    box = plan.box
    l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2,
                (box.x1 - box.x0) * 2)

    def _suboff_inside_l1(centers):
        coarse = 0.5 * centers + torch.tensor(
            (box.x0, box.y0, box.z0), dtype=torch.float64,
            device=centers.device,
        )
        return solid_mask_inside_fn(solid_cpu.bool(), device=torch.device("cpu"))(coarse)

    octree = build_octree_shell(
        l1_shape, center=(0.0, 0.0, 0.0),
        radius=max(srad, bl * 2) * 2.0,
        bl_thickness_cells=bl, d_max=d_max, lattice="D3Q27",
        device=torch.device("cpu"),
        inside_fn=_suboff_inside_l1,
    )
    return octree, _suboff_inside_l1


# ---- 1) d_max=1 regression: no L2 leaves, same L1 count as d_max=2 path's L1 set ----
g1 = build_suboff(1)[0]
print(f"[suboff d_max=1] n_leaf_l1={g1.stats['n_leaf_l1']} "
      f"n_leaf_l2={g1.stats['n_leaf_l2']} n_leaf={g1.n_leaf}")

# ---- 2) d_max=2: all kept leaves must be fluid (centre outside body) ----
g2, inside_fn = build_suboff(2)
inside = inside_fn(g2.leaf_center.to(torch.float64))
n_solid_leak = int(inside.sum().item())
print(f"[suboff d_max=2] n_leaf_l1={g2.stats['n_leaf_l1']} "
      f"n_leaf_l2={g2.stats['n_leaf_l2']} n_leaf={g2.n_leaf} "
      f"| solid leaves kept: {n_solid_leak}")

# level-2 leaves must all be fluid
l2_mask = g2.leaf_level == 2
n_l2 = int(l2_mask.sum().item())
n_l2_solid = int((l2_mask & inside).sum().item())
print(f"[suboff d_max=2] level-2 leaves: {n_l2}, of which solid (leak): {n_l2_solid}")

# some L1 leaves must have been refined (i.e. L2 exists) — already shown by n_leaf_l2

# ---- 3) topology checks ----
print("[suboff d_max=2] checks:", dict(g2.checks))
assert n_solid_leak == 0, "fluid leaves leak into the solid body!"
assert n_l2_solid == 0, "level-2 leaves leak into the solid body!"
assert n_l2 > 0, "no level-2 leaves produced!"
print("ALL ASSERTIONS PASSED")
