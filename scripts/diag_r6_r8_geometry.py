#!/usr/bin/env python3
"""Compare octree shell geometry stats: R6 vs R8 (CPU, no stepping)."""
import math
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell

def diag(radius, nx, ny, nz, bl=3.0, d_max=1, device="cpu"):
    dev = torch.device(device)
    shape = (nz, ny, nx)
    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    # L1 sphere: radius * 2 (ratio 2), centre at (cx*2, cy*2, cz*2)
    octree = build_octree_shell(
        shape, (cx, cy, cz), radius,
        bl_thickness_cells=bl, d_max=d_max, transition=1, device=dev)
    st = octree.stats
    solid = octree._solid
    shell = octree._shell_mask
    # count leaves whose CENTRE is inside the analytic sphere (kept, no BFL wall)
    centers = octree.leaf_center.double()
    c = torch.tensor([cx, cy, cz], dtype=torch.float64)
    d2 = ((centers - c) ** 2).sum(dim=1)
    n_center_inside = int((d2 <= radius ** 2).sum().item())
    # count leaves whose centre is inside but has NO bfl mask at all
    bfl_sum = octree.bfl_mask.sum(dim=0)
    n_ci_no_bfl = int((((d2 <= radius ** 2)) & (bfl_sum == 0)).sum().item())
    # q distribution of bfl links
    q = octree.q_field[octree.bfl_mask].double()
    q_lo = int((q < 0.5).sum().item()) if q.numel() else 0
    q_hi = int((q >= 0.5).sum().item()) if q.numel() else 0
    q_very_lo = int((q < 0.05).sum().item()) if q.numel() else 0
    q_very_hi = int((q > 0.95).sum().item()) if q.numel() else 0
    # shell-cell centre distances relative to analytic shell
    dist = torch.sqrt(((torch.nonzero(shell).double() + 0.5 - c) ** 2).sum(dim=1))
    print(f"--- R={radius} nx={nx} shape={shape} ---")
    print(f"  n_leaf={st['n_leaf']} l1={st['n_leaf_l1']} l2={st['n_leaf_l2']} "
          f"shell_cells={st['n_shell_cells']}")
    print(f"  leaf_volume={st['leaf_volume']:.4f} analytic={st['analytic_shell_volume']:.4f} "
          f"vol_err={st['volume_error']*100:.4f}%")
    print(f"  interface_links={st['n_interface_links']} cross_level={st['n_cross_level_donor']}")
    print(f"  n_center_inside={n_center_inside} n_ci_no_bfl={n_ci_no_bfl}")
    print(f"  bfl_links={int(octree.bfl_mask.sum().item())} q<0.5={q_lo} q>=0.5={q_hi} "
          f"q<0.05={q_very_lo} q>0.95={q_very_hi}")
    print(f"  shell cell dist min={dist.min().item():.3f} max={dist.max().item():.3f}")
    print(f"  solid cells={int(solid.sum().item())} shell={int(shell.sum().item())}")
    # solid cell distance max (are there solid cells beyond the sphere surface?)
    sd = torch.sqrt(((torch.nonzero(solid).double() + 0.5 - c) ** 2).sum(dim=1))
    print(f"  solid cell dist max={sd.max().item():.3f} (R={radius})")
    return octree

if __name__ == "__main__":
    torch.manual_seed(0)
    # R6: nx=96 ny=64 nz=64 radius 6 ; L1 = radius*2 = 12
    o6 = diag(12.0, 96, 64, 64)
    # R8: nx=128 ny=88 nz=88 radius 8 ; L1 = 16
    o8 = diag(16.0, 128, 88, 88)
