import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, DOMAIN_OUT, SHELL_OUTSIDE
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.octree_boundary.stepping import build_ghost_plan, _fill_ghost_impl, ShellGhostPlan
from tensorlbm.octree_boundary.distributed_stepping import (
    _slice_ghost_plan_by_indices, split_leaf_bounds, interleaved_leaf_indices,
    stream_gather_distributed,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
from tensorlbm.d3q27 import equilibrium27, OPPOSITE
from tensorlbm.octree_boundary.stepping import _tau_chain
from tensorlbm.cumulant import collide_cumulant_d3q27

center=(80.0,48.0,48.0); radius=10.0
o = build_octree_shell((96, 96, 160), center=center, radius=radius,
    bl_thickness_cells=5.0, d_max=1, lattice="D3Q27",
    device=torch.device("cpu"), inside_fn=sphere_inside_fn(center, radius))
nt = o.neighbor_table
n1 = o.n_leaf_level(1)
l1 = o._l1_coords
lc2 = (o.leaf_center[:n1].to(torch.float64)*2 - 0.5).long()
print("aligned:", bool((l1 == lc2).all()), "l1[86404]:", l1[86404].tolist(),
      "nt[5,86404]:", int(nt[5,86404]))
dom = (nt == DOMAIN_OUT).sum().item()
print("DOMAIN_OUT total:", dom)
dd, ii = torch.nonzero(nt == DOMAIN_OUT, as_tuple=True)
print("first DOMAIN_OUT leaf:", int(ii.min()) if len(ii) else None)

# ---- NaN mechanism demo: pre-fill out with NaN (simulating SDAA torch.empty) ----
plan = build_ghost_plan(o, (96, 96, 160))
nz, ny, nx = 96, 96, 160
Q = o.Q
u_in = 0.06
eq_global = equilibrium27(torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
                          torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx))
taus = _tau_chain(0.5 + 3.0*(u_in*10.0/100.0), o.d_max)
host = o.leaf_host_cell
for mode in ("contig", "interleave"):
    for rank in range(2):
        n_leaf = o.n_leaf
        if mode == "contig":
            lo0, hi0 = split_leaf_bounds(n_leaf, 2)[rank]
            lidx = torch.arange(lo0, hi0, dtype=torch.int64)
            p_local, _ = _slice_ghost_plan(plan, lo0, hi0, int(lidx.shape[0]))
        else:
            lidx = interleaved_leaf_indices(n_leaf, 2, rank)
            p_local, _ = _slice_ghost_plan_by_indices(plan, lidx, int(lidx.shape[0]))
        n_local = int(lidx.shape[0])
        f_leaf = eq_global[:, host[lidx,0], host[lidx,1], host[lidx,2]].clone()
        post = collide_cumulant_d3q27(f_leaf.view(Q,1,1,-1), taus[1], C_s=0.0).view_as(f_leaf)
        full_pc = torch.zeros(Q, n_leaf)
        full_pc[:, lidx] = post
        gplan_fill = ShellGhostPlan(p_local.n_ghost, lidx[p_local.leaf.cpu()], p_local.direction,
            p_local.z0, p_local.y0, p_local.x0, p_local.z1, p_local.y1, p_local.x1,
            p_local.wx, p_local.wy, p_local.wz, p_local.volume, p_local.slot)
        gv = _fill_ghost_impl(o.leaf_level, gplan_fill,
                              eq_global.clone(), taus)  # parent_t = equilibrium
        # simulate SDAA torch.empty containing NaN
        out_nan = torch.full((Q, n_local), float("nan"))
        out = stream_gather_distributed(o, full_pc, gv, p_local.slot, f_leaf, lidx)
        # count DOMAIN_OUT positions per rank
        src = nt[OPPOSITE.to("cpu")][:, lidx]  # nt[opp[d]][i] as in stream
        n_dom = int((src == DOMAIN_OUT).sum())
        # count unhandled (would remain NaN if out prefilled with NaN): emulate by masking
        unhandled = (src == DOMAIN_OUT)
        # also: what about positions where src==REMOTE? none in full table
        print(f"[{mode}] rank{rank}: DOMAIN_OUT srcs={n_dom}  (of Q*n_local={Q*n_local})")
