#!/usr/bin/env python3
"""Reproduce R10 NaN on CPU with the EXACT analytic geometry (no inside_fn)."""
import sys, torch
sys.path.insert(0, "/root/TensorLBM_feat2/src")

from tensorlbm.octree_boundary.geometry import build_octree_shell, SHELL_OUTSIDE, SOLID, FANOUT, DOMAIN_OUT
from tensorlbm.octree_boundary.stepping import build_ghost_plan, _fill_ghost_impl, ShellGhostPlan, _tau_chain
from tensorlbm.octree_boundary.distributed_stepping import (
    _slice_ghost_plan_by_indices, split_leaf_bounds, interleaved_leaf_indices,
    stream_gather_distributed,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
from tensorlbm.d3q27 import equilibrium27, OPPOSITE

torch.set_printoptions(linewidth=220)

def build(shape, center, radius, bl, tag):
    octree = build_octree_shell(
        shape, center=center, radius=radius,
        bl_thickness_cells=bl, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"),
    )
    plan = build_ghost_plan(octree, tuple(octree.meta["shape"]))
    return octree, plan

def check_all(octree, plan, tag, world_size=2):
    nz, ny, nx = octree.meta["shape"]
    Q = octree.Q
    n_leaf = octree.n_leaf
    dev = octree.leaf_host_cell.device
    opp = octree._opp.to(dev)
    nt = octree.neighbor_table
    bfl = octree.bfl_mask
    print(f"\n===== {tag}: shape=({nz},{ny},{nx}) n_leaf={n_leaf} "
          f"n_link={plan.n_ghost} n_shell={int(octree._shell_mask.sum())} "
          f"bfl_links={int(bfl.sum())} =====")

    # sentinel coverage in global neighbor table
    for nm, v in (("SHELL_OUTSIDE", SHELL_OUTSIDE), ("SOLID", SOLID),
                  ("FANOUT", FANOUT), ("DOMAIN_OUT", DOMAIN_OUT)):
        print(f"  nt=={nm}: {int((nt == v).sum())}")

    # ---------- H1a: GLOBAL slot completeness ----------
    src = nt[opp]
    need_stream = (src == SHELL_OUTSIDE)
    slot = plan.slot
    miss_stream = need_stream & (slot < 0)
    print(f"[H1-global] stream-ghost needed={int(need_stream.sum())} "
          f"missing-slot={int(miss_stream.sum())}")
    if int(miss_stream.sum()):
        rows = torch.nonzero(miss_stream, as_tuple=False)
        print("     missing (d,i):", rows[:20].tolist())
    # any (d,i) where src is NOT covered by valid/ghost/solid/fanout?
    covered = (src >= 0) | (src == SHELL_OUTSIDE) | (src == SOLID) | (src == FANOUT)
    print(f"[cov] src NOT covered by stream branches: {int((~covered).sum())}")

    # ---------- H1b: SLICED slot completeness ----------
    for mode in ("contig", "interleave"):
        for rank in range(world_size):
            if mode == "contig":
                lo0, hi0 = split_leaf_bounds(n_leaf, world_size)[rank]
                lidx = torch.arange(lo0, hi0, dtype=torch.int64)
                p_local, grows = _slice_ghost_plan(plan, lo0, hi0, int(lidx.shape[0]))
            else:
                lidx = interleaved_leaf_indices(n_leaf, world_size, rank)
                p_local, grows = _slice_ghost_plan_by_indices(plan, lidx, int(lidx.shape[0]))
            n_local = int(lidx.shape[0])
            src_local = nt[opp][:, lidx]
            need = src_local == SHELL_OUTSIDE
            miss = need & (p_local.slot < 0)
            # ghost_vals[d, slot] indexing check: any slot >= n_ghost or < 0?
            sl = p_local.slot
            bad = (sl < -1) | (sl >= p_local.n_ghost)
            print(f"[H1-{mode}] rank{rank} n_local={n_local} n_ghost={p_local.n_ghost} "
                  f"stream-need={int(need.sum())} missing={int(miss.sum())} "
                  f"slot-bad-range={int(bad.sum())}")
            if int(miss.sum()):
                rows = torch.nonzero(miss, as_tuple=False)
                print("     missing (d, local_i):", rows[:20].tolist())
    return

def simulate_step1(octree, plan, tag, world_size=2, interleave=False):
    """Reproduce BOTH substeps (s=0 and s=1) like the real stepper."""
    nz, ny, nx = octree.meta["shape"]
    Q = octree.Q
    n_leaf = octree.n_leaf
    dev = octree.leaf_host_cell.device
    u_in = 0.06
    tau_coarse = 0.5 + 3.0 * (u_in * 10.0 / 100.0)
    taus = _tau_chain(tau_coarse, octree.d_max)
    tau_shell = taus[1]
    host = octree.leaf_host_cell

    eq_global = equilibrium27(
        torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
        torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx),
    )
    if interleave:
        lidx_all = [interleaved_leaf_indices(n_leaf, world_size, r) for r in range(world_size)]
    else:
        lidx_all = [torch.arange(*split_leaf_bounds(n_leaf, world_size)[r], dtype=torch.int64)
                    for r in range(world_size)]

    from tensorlbm.cumulant import collide_cumulant_d3q27
    def advance_shell(f, tau, level, substep):
        f4 = f.view(Q, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    shell_cells = torch.nonzero(octree._shell_mask, as_tuple=False)
    n_shell = shell_cells.shape[0]
    full_sc = eq_global[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]].clone()
    coarse_sparse = equilibrium27(
        torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
        torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx),
    )
    coarse_sparse[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]] = full_sc
    l1_old = coarse_sparse
    l1_f = coarse_sparse

    print(f"[sim {tag} {interleave and 'interleave' or 'contig'}] n_shell={n_shell} tau_shell={tau_shell:.6f}")
    n_substeps = 1 << octree.d_max
    for r in range(world_size):
        lidx = lidx_all[r]
        n_local = int(lidx.shape[0])
        f_leaf = eq_global[:, host[lidx, 0], host[lidx, 1], host[lidx, 2]].clone()
        if interleave:
            p_local, grows = _slice_ghost_plan_by_indices(plan, lidx, n_local)
        else:
            lo0, hi0 = split_leaf_bounds(n_leaf, world_size)[r]
            p_local, grows = _slice_ghost_plan(plan, lo0, hi0, n_local)
        p = p_local
        gplan_fill = ShellGhostPlan(
            p.n_ghost, lidx[p.leaf.cpu()], p.direction, p.z0, p.y0, p.x0,
            p.z1, p.y1, p.x1, p.wx, p.wy, p.wz, p.volume, p.slot,
        )
        full_pc = torch.zeros(Q, n_leaf)
        for s in range(n_substeps):
            alpha = s / n_substeps
            parent_t = torch.lerp(l1_old, l1_f, alpha)
            populations, post_collision = advance_shell(f_leaf, tau_shell, 1, s), f_leaf
            full_pc.zero_()
            full_pc[:, lidx] = post_collision
            ghost_vals = _fill_ghost_impl(octree.leaf_level, gplan_fill, parent_t, taus)
            gv_finite = bool(torch.isfinite(ghost_vals).all())
            out = stream_gather_distributed(octree, full_pc, ghost_vals, p_local.slot,
                                            f_leaf, lidx)
            nan_elems = int((~torch.isfinite(out)).sum())
            print(f"[sim {tag}] rank{r} substep{s} ghost_finite={gv_finite} "
                  f"stream NaN elems={nan_elems} / {Q * n_local}")
            if nan_elems:
                bad = (~torch.isfinite(out))
                dd, ii = torch.nonzero(bad, as_tuple=True)
                uni = torch.unique(ii)
                print(f"     NaN leaves: {len(uni)} unique, global leaf {lidx[uni[:8]].tolist()}")
                print(f"     NaN dirs: {torch.unique(dd).tolist()}")
                li = int(uni[0])
                gi = int(lidx[li])
                src = octree.neighbor_table[octree._opp][:, gi]
                print(f"     leaf {gi}: level={int(octree.leaf_level[gi])} "
                      f"host={host[gi].tolist()} center={octree.leaf_center[gi].tolist()}")
                for d in torch.unique(dd).tolist():
                    print(f"       d={d}: src={int(src[d])} slot={int(p_local.slot[d, li])} "
                          f"out={float(out[d, li]):.4f}")
            f_leaf = out
    return

if __name__ == "__main__":
    R6 = build((96, 64, 64), (48.0, 32.0, 32.0), 6.0, 3.0, "R6")
    check_all(*R6, "R6 r=6 bl=3")
    R10 = build((96, 96, 160), (80.0, 48.0, 48.0), 10.0, 5.0, "R10")
    check_all(*R10, "R10 r=10 bl=5")
    print("\n================ STEP-1 FULL SIM (both substeps) ================")
    simulate_step1(*R6, "R6", interleave=False)
    simulate_step1(*R10, "R10", interleave=False)
    simulate_step1(*R10, "R10", interleave=True)
