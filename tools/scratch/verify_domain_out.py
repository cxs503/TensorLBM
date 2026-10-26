#!/usr/bin/env python3
"""Verify the DOMAIN_OUT hypothesis: R10 (bug-rotated) shell touches the z boundary."""
import sys, torch
sys.path.insert(0, "/root/TensorLBM_feat2/src")
from tensorlbm.octree_boundary.geometry import build_octree_shell, DOMAIN_OUT, SHELL_OUTSIDE, SOLID, FANOUT
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn

def build(shape, center, radius, bl, tag):
    o = build_octree_shell(shape, center=center, radius=radius,
        bl_thickness_cells=bl, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
    return o

R6 = build((96, 64, 64), (48.0, 32.0, 32.0), 6.0, 3.0, "R6")
R10 = build((96, 96, 160), (80.0, 48.0, 48.0), 10.0, 5.0, "R10")

for tag, o in (("R6", R6), ("R10", R10)):
    nt = o.neighbor_table
    print(f"\n=== {tag} neighbor_table sentinels (Q={o.Q}, n_leaf={o.n_leaf}) ===")
    for name, val in (("SHELL_OUTSIDE", SHELL_OUTSIDE), ("SOLID", SOLID),
                      ("DOMAIN_OUT", DOMAIN_OUT), ("FANOUT", FANOUT)):
        print(f"  {name}: {int((nt == val).sum())}")
    if int((nt == DOMAIN_OUT).sum()):
        dd, ii = torch.nonzero(nt == DOMAIN_OUT, as_tuple=True)
        print(f"  DOMAIN_OUT leaves: {torch.unique(ii)[:10].tolist()} (unique {len(torch.unique(ii))})")
        gi = int(torch.unique(ii)[0])
        print(f"  e.g. leaf {gi}: center={o.leaf_center[gi].tolist()} host={o.leaf_host_cell[gi].tolist()} "
              f"dirs={torch.unique(dd).tolist()}")
    # how many DOMAIN_OUT neighbours do rank1's leaves have (interleave / contig)?
    n_leaf = o.n_leaf
    for mode, lidx in (("contig-r1", torch.arange(n_leaf // 2, n_leaf, dtype=torch.int64)),
                       ("ilv-r1", torch.arange(1, n_leaf, 2, dtype=torch.int64))):
        src = nt[:, lidx]
        print(f"  [{mode}] DOMAIN_OUT in rank1 leaves: {int((src == DOMAIN_OUT).sum())} "
              f"leaves: {len(torch.unique(torch.nonzero(src == DOMAIN_OUT, as_tuple=True)[1]))}")
    # checks results
    print(f"  checks: { {k: v for k, v in o.checks.items()} }")

# stream_gather_distributed handling of DOMAIN_OUT: look for the branch
import inspect
from tensorlbm.octree_boundary import distributed_stepping as ds
src_code = inspect.getsource(ds.stream_gather_distributed)
print("\nDOMAIN_OUT handled in stream_gather_distributed:", "DOMAIN_OUT" in src_code)
print("DOMAIN_OUT handled in stepping.stream_gather:", "DOMAIN_OUT" in inspect.getsource(
    __import__('tensorlbm.octree_boundary.stepping', fromlist=['stream_gather']).stream_gather))
