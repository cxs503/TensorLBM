#!/usr/bin/env python3
"""Full 2-step CPU sim of the integration loop WITH the buggy full_f reconstruction
(commit 6e8f616): verify when NaN first appears and its count."""
import sys, torch
sys.path.insert(0, "/root/TensorLBM_feat2/src")
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan, _fill_ghost_impl, ShellGhostPlan, _tau_chain,
)
from tensorlbm.octree_boundary.distributed_stepping import (
    _slice_ghost_plan_by_indices, split_leaf_bounds, interleaved_leaf_indices,
    stream_gather_distributed, restrict_shell_to_block,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.cumulant import collide_cumulant_d3q27

torch.set_printoptions(linewidth=220)

def build(shape, center, radius, bl):
    return build_octree_shell(
        shape, center=center, radius=radius,
        bl_thickness_cells=bl, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"),
    )

def main():
    octree = build((96, 96, 160), (80.0, 48.0, 48.0), 10.0, 5.0)  # R10
    nz, ny, nx = octree.meta["shape"]
    Q = octree.Q
    n_leaf = octree.n_leaf
    u_in = 0.06
    tau_coarse = 0.5 + 3.0 * (u_in * 10.0 / 100.0)
    taus = _tau_chain(tau_coarse, octree.d_max)
    tau_shell = taus[1]
    host = octree.leaf_host_cell
    world_size = 2
    rank = 0
    lidx = torch.arange(*split_leaf_bounds(n_leaf, world_size)[rank], dtype=torch.int64)
    n_local = int(lidx.shape[0])
    plan = build_ghost_plan(octree, tuple(octree.meta["shape"]))
    lo0, hi0 = split_leaf_bounds(n_leaf, world_size)[rank]
    p_local, grows = _slice_ghost_plan(plan, lo0, hi0, n_local)
    p = p_local
    gplan_fill = ShellGhostPlan(
        p.n_ghost, lidx[p.leaf.cpu()], p.direction, p.z0, p.y0, p.x0,
        p.z1, p.y1, p.x1, p.wx, p.wy, p.wz, p.volume, p.slot,
    )

    def advance_shell(f, tau, level, substep):
        f4 = f.view(Q, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    # --- coarse field init: uniform inflow equilibrium on a single rank's view
    eq_global = equilibrium27(
        torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
        torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx),
    )
    coarse_f = eq_global.clone()          # coarse field (per-rank full copy here)
    octree.f_leaf = eq_global[:, host[lidx, 0], host[lidx, 1], host[lidx, 2]].clone()

    shell_mask = octree._shell_mask
    shell_cells = torch.nonzero(shell_mask, as_tuple=False)

    def build_coarse_sparse(coarse):
        n_shell = shell_cells.shape[0]
        full_sc = coarse[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]].clone()
        cs = equilibrium27(
            torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
            torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx),
        )
        cs[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]] = full_sc
        return cs

    for step in range(1, 3):
        # coarse evolve (one collide+stream, simplified; uniform field stays eq)
        post = collide_cumulant_d3q27(coarse_f.view(Q, 1, 1, -1), tau_coarse, C_s=0.0).view_as(coarse_f)
        coarse_f = post
        coarse_sparse = build_coarse_sparse(coarse_f)
        l1_old = l1_f = coarse_sparse
        n_substeps = 1 << octree.d_max
        for s in range(n_substeps):
            alpha = s / n_substeps
            parent_t = torch.lerp(l1_old, l1_f, alpha)
            populations, post_collision = advance_shell(octree.f_leaf, tau_shell, 1, s), octree.f_leaf
            octree.f_leaf = populations
            global_pc = torch.zeros(Q, n_leaf)
            global_pc[:, lidx] = post_collision
            full_pc = global_pc.clone()   # correct gather (sum of disjoint shards)
            ghost_vals = _fill_ghost_impl(octree.leaf_level, gplan_fill, parent_t, taus)
            gv_finite = bool(torch.isfinite(ghost_vals).all())
            out = stream_gather_distributed(octree, full_pc, ghost_vals, p_local.slot, octree.f_leaf, lidx)
            nan_elems = int((~torch.isfinite(out)).sum())
            print(f"step{step} substep{s}: ghost_finite={gv_finite} stream NaN={nan_elems}/{Q*n_local}")
            if nan_elems:
                bad = ~torch.isfinite(out)
                dd, ii = torch.nonzero(bad, as_tuple=True)
                print("   NaN dirs:", torch.unique(dd).tolist())
                print("   NaN leaves:", len(torch.unique(ii)), "e.g.", lidx[torch.unique(ii)[:6]].tolist())
                # what did the ghost sample for a bad leaf
                li = int(torch.unique(ii)[0])
                gi = int(lidx[li])
                src = octree.neighbor_table[octree._opp][:, gi]
                for d in torch.unique(dd).tolist():
                    print(f"   leaf {gi} d={d}: src={int(src[d])} slot={int(p_local.slot[d, li])} ghost_val={float(ghost_vals[d, int(p_local.slot[d, li])]) if p_local.slot[d, li] >= 0 else 'na'}")
            octree.f_leaf = out
        # ---- buggy reconstruction (line 423-427 + 491) ----
        full_f = torch.zeros(Q, n_leaf)
        full_f[:, lidx] = octree.f_leaf
        # BUG: full_f = torch.zeros(...) again discards the scatter
        full_f = torch.zeros(Q, n_leaf)
        full_f2 = torch.zeros(Q, n_leaf)   # what the correct code would produce
        full_f2[:, lidx] = octree.f_leaf
        restricted, cells = restrict_shell_to_block(octree, full_f, taus)
        restricted_ok, _ = restrict_shell_to_block(octree, full_f2, taus)
        print(f"step{step} end: restricted(all-zero bug) finite={bool(torch.isfinite(restricted).all())} "
              f"n_cells={cells.shape[0]}; correct-restriction finite={bool(torch.isfinite(restricted_ok).all())}")
        # write back into coarse field (integration script line 407)
        mine = (cells[:, 2] >= 0) & (cells[:, 2] < nx)
        if bool(mine.any()):
            coarse_f[:, cells[mine, 0], cells[mine, 1], cells[mine, 2]] = restricted[:, mine]
        octree.f_leaf = full_f[:, lidx].contiguous()
        nz_leaf = int((octree.f_leaf == 0).sum())
        print(f"step{step} end: f_leaf zeroed by bug: {nz_leaf}/{octree.f_leaf.numel()}")

if __name__ == "__main__":
    main()
