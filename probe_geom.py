#!/usr/bin/env python3
"""Probe which geometry config reproduces the production n_leaf (R6=34264, R10=122992)."""
import sys, torch
sys.path.insert(0, "/root/TensorLBM_feat2/src")
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

def build(shape, center, radius, bl, d_max):
    o = build_octree_shell(shape, center=center, radius=radius,
        bl_thickness_cells=bl, d_max=d_max, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
    return o

# R6 candidates: script defaults nx=96,ny=64,nz=64,r=6,bl=3
cands = [
    ("R6 d1 (64,64,96) bl3", (64, 64, 96), (48.0, 32.0, 32.0), 6.0, 3.0, 1),
    ("R6 d2 (64,64,96) bl3", (64, 64, 96), (48.0, 32.0, 32.0), 6.0, 3.0, 2),
    ("R6 d2 (64,64,96) bl2", (64, 64, 96), (48.0, 32.0, 32.0), 6.0, 2.0, 2),
    ("R6 d1 (96,64,64) bl3", (96, 64, 64), (48.0, 32.0, 32.0), 6.0, 3.0, 1),
    ("R6 d2 (96,64,64) bl3", (96, 64, 64), (48.0, 32.0, 32.0), 6.0, 3.0, 2),
    ("R10 d1 (96,96,160) bl5", (96, 96, 160), (80.0, 48.0, 48.0), 10.0, 5.0, 1),
    ("R10 d2 (96,96,160) bl5", (96, 96, 160), (80.0, 48.0, 48.0), 10.0, 5.0, 2),
    ("R10 d2 (96,96,160) bl4", (96, 96, 160), (80.0, 48.0, 48.0), 10.0, 4.0, 2),
]
for tag, shape, center, r, bl, dm in cands:
    try:
        o = build(shape, center, r, bl, dm)
        nt = o.neighbor_table
        from tensorlbm.octree_boundary.geometry import SHELL_OUTSIDE, SOLID, DOMAIN_OUT, FANOUT
        print(f"{tag}: n_leaf={o.n_leaf}  sentinels: shell_out={int((nt==SHELL_OUTSIDE).sum())} "
              f"solid={int((nt==SOLID).sum())} dom_out={int((nt==DOMAIN_OUT).sum())} fanout={int((nt==FANOUT).sum())} "
              f"lvl1={o.n_leaf_level(1)} lvl2={o.n_leaf_level(2)}")
    except Exception as e:
        print(f"{tag}: FAIL {e}")
