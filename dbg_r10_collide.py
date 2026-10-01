#!/usr/bin/env python3
"""Check leaves 67808..67868: level, host, and substep-1 collide inputs."""
import sys, torch
sys.path.insert(0, "/root/TensorLBM_feat2/src")

from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import build_ghost_plan, _tau_chain
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.cumulant import collide_cumulant_d3q27

torch.set_printoptions(linewidth=220)

octree = build_octree_shell(
    (96, 96, 160), center=(80.0, 48.0, 48.0), radius=10.0,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"),
)
print("n_leaf:", octree.n_leaf, "d_max:", octree.d_max,
      "n_l1:", octree.n_leaf_level(1), "n_l2:", octree.n_leaf_level(2))
print("level_start:", octree.level_start.tolist())

for gi in [67808, 67810, 67812, 67814, 67824, 67863, 67864, 67868, 14775, 30153, 55123]:
    print(f"leaf {gi}: level={int(octree.leaf_level[gi])} "
          f"host={octree.leaf_host_cell[gi].tolist()} "
          f"center={octree.leaf_center[gi].tolist()} "
          f"morton={int(octree.leaf_morton[gi])}")

# count leaves per host cell near 67808
host = octree.leaf_host_cell
target = torch.tensor([48, 44, 69])
same = (host == target).all(dim=1)
print("leaves with host [48,44,69]:", int(same.sum()))
if int(same.sum()):
    idxs = torch.nonzero(same, as_tuple=False).squeeze(1)
    print("  enums:", idxs.tolist(), "levels:", octree.leaf_level[idxs].tolist())

# colliding eq samples: does collide produce NaN for these?
u_in = 0.06
eq_global = equilibrium27(
    torch.ones(96, 96, 160), torch.full((96, 96, 160), u_in),
    torch.zeros(96, 96, 160), torch.zeros(96, 96, 160),
)
tau_coarse = 0.5 + 3.0 * (u_in * 10.0 / 100.0)
taus = _tau_chain(tau_coarse, 1)
print("taus:", taus)
f_leaf = eq_global[:, host[:, 0], host[:, 1], host[:, 2]].clone()
f4 = f_leaf.view(27, 1, 1, -1)
pc = collide_cumulant_d3q27(f4, taus[1], C_s=0.0).view_as(f_leaf)
for gi in [67808, 67810, 67812, 67814, 67824, 67863, 67864, 67868, 14775, 30153, 55123]:
    col = f_leaf[:, gi]
    print(f"leaf {gi}: f_leaf sum={float(col.sum()):.6f} min={float(col.min()):.4f} "
          f"max={float(col.max()):.4f} pc_finite={bool(torch.isfinite(pc[:, gi]).all())}")

# now simulate substep0: collide, stream with ghost plan, then BFL, then collide again
from tensorlbm.octree_boundary.stepping import ShellGhostPlan, _fill_ghost_impl
from tensorlbm.octree_boundary.distributed_stepping import (
    split_leaf_bounds, stream_gather_distributed, _slice_ghost_plan_by_indices,
    interleaved_leaf_indices,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights

plan = build_ghost_plan(octree, (96, 96, 160))
shell_cells = torch.nonzero(octree._shell_mask, as_tuple=False)
coarse_sparse = eq_global.clone()
coarse_sparse[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]] = \
    eq_global[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]]

for rank in range(2):
    lo0, hi0 = split_leaf_bounds(octree.n_leaf, 2)[rank]
    lidx = torch.arange(lo0, hi0, dtype=torch.int64)
    n_local = int(lidx.shape[0])
    p_local, grows = _slice_ghost_plan(plan, lo0, hi0, n_local)
    p = p_local
    gplan_fill = ShellGhostPlan(
        p.n_ghost, lidx[p.leaf.cpu()], p.direction, p.z0, p.y0, p.x0,
        p.z1, p.y1, p.x1, p.wx, p.wy, p.wz, p.volume, p.slot,
    )
    f_leaf = eq_global[:, host[lidx, 0], host[lidx, 1], host[lidx, 2]].clone()
    for s in range(2):
        alpha = s / 2
        parent_t = coarse_sparse
        f4 = f_leaf.view(27, 1, 1, -1)
        pc = collide_cumulant_d3q27(f4, taus[1], C_s=0.0).view_as(f_leaf)
        full_pc = torch.zeros(27, octree.n_leaf)
        full_pc[:, lidx] = pc
        ghost_vals = _fill_ghost_impl(octree.leaf_level, gplan_fill, parent_t, taus)
        out = stream_gather_distributed(octree, full_pc, ghost_vals, p_local.slot,
                                        f_leaf, lidx)
        nnan = int((~torch.isfinite(out)).sum())
        print(f"[rank{rank} substep{s}] pc_finite={bool(torch.isfinite(pc).all())} "
              f"stream_nan={nnan}")
        # BFL needs (Q, n_leaf) global shapes
        out_g = torch.zeros(27, octree.n_leaf)
        out_g[:, lidx] = out
        pc_g = torch.zeros(27, octree.n_leaf)
        pc_g[:, lidx] = pc
        leaf_weights_g = leaf_force_weights(octree)
        out2_g, force = bfl_apply_gather(
            octree, out_g, pc_g, ghost_plan=gplan_fill, ghost_vals=ghost_vals,
            force_weights=leaf_weights_g, return_force=True, q_min=None,
        )
        out2 = out2_g[:, lidx]
        print(f"   bfl_out_finite={bool(torch.isfinite(out2).all())} "
              f"force={force.tolist()}")
        if not bool(torch.isfinite(out2).all()):
            bd, bi = torch.nonzero(~torch.isfinite(out2), as_tuple=True)
            print("   bfl NaN local leaves:", torch.unique(bi).tolist()[:20],
                  "global:", lidx[torch.unique(bi)[:20]].tolist())
        f_leaf = out2
