#!/usr/bin/env python3
"""CPU static verification of the R10 (vs R6) ghost plan / host cell / donor coords.

Reproduces exactly the octree_integrated_validate.py geometry on CPU and checks:
  H1: ghost plan slicing (_slice_ghost_plan / _slice_ghost_plan_by_indices) produces
      no invalid slot (slot<0 where the stream/BFL needs one)
  H2: leaf_host_cell out-of-bounds vs (nz,ny,nx)
  H3: ghost donor coords (z0/y0/x0/z1/y1/x1) out-of-bounds
  H4: f_leaf eq_global sampling bounds
Plus a full first-step substep-0 stream simulation to reproduce the NaN.
"""
import sys, torch
sys.path.insert(0, "/root/TensorLBM_feat2/src")

from tensorlbm.octree_boundary.geometry import build_octree_shell, SHELL_OUTSIDE, SOLID
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.octree_boundary.stepping import build_ghost_plan, _fill_ghost_impl, ShellGhostPlan
from tensorlbm.octree_boundary.distributed_stepping import (
    _slice_ghost_plan_by_indices, split_leaf_bounds, interleaved_leaf_indices,
    stream_gather_distributed,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
from tensorlbm.d3q27 import equilibrium27, OPPOSITE

torch.set_printoptions(linewidth=200)

def build(shape, center, radius, bl, tag):
    octree = build_octree_shell(
        shape, center=center, radius=radius,
        bl_thickness_cells=bl, d_max=1, lattice="D3Q27",
        device=torch.device("cpu"),
        inside_fn=sphere_inside_fn(center, radius),
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

    # ---------- H2: leaf_host_cell ----------
    host = octree.leaf_host_cell
    print(f"[H2] host z [{int(host[:,0].min())},{int(host[:,0].max())}] vs nz={nz} "
          f"y [{int(host[:,1].min())},{int(host[:,1].max())}] vs ny={ny} "
          f"x [{int(host[:,2].min())},{int(host[:,2].max())}] vs nx={nx}")
    # pre-clamp floor of leaf centres
    centers64 = octree.leaf_center.to(torch.float64)  # (n,3) x,y,z world
    raw = torch.floor(centers64)[:, [2, 1, 0]].to(torch.int64)  # (z,y,x)
    oob = (raw[:, 0] < 0) | (raw[:, 0] > nz - 1) | (raw[:, 1] < 0) | (raw[:, 1] > ny - 1) \
        | (raw[:, 2] < 0) | (raw[:, 2] > nx - 1)
    clamped = (raw != host).any(dim=1)
    print(f"[H2] pre-clamp OOB leaves: {int(oob.sum())}  clamped-by-builder: {int(clamped.sum())}")
    if int(oob.sum()):
        bad = torch.nonzero(oob, as_tuple=False).squeeze(1)
        print("     example raw hosts:", raw[bad[:10]].tolist(), "clamped to", host[bad[:10]].tolist())

    # ---------- H3: ghost donor coords ----------
    if plan.n_ghost:
        for nm, lo_, hi_ in (("z", plan.z0, plan.z1), ("y", plan.y0, plan.y1), ("x", plan.x0, plan.x1)):
            print(f"[H3] donor {nm}: lo [{int(lo_.min())},{int(lo_.max())}] hi [{int(hi_.min())},{int(hi_.max())}]", end="  ")
        print()
        zmax, ymax, xmax = nz - 1, ny - 1, nx - 1
        bad = ((plan.z0 < 0) | (plan.z0 > zmax) | (plan.z1 < 0) | (plan.z1 > zmax)
               | (plan.y0 < 0) | (plan.y0 > ymax) | (plan.y1 < 0) | (plan.y1 > ymax)
               | (plan.x0 < 0) | (plan.x0 > xmax) | (plan.x1 < 0) | (plan.x1 > xmax))
        print(f"[H3] donor coord out-of-bounds rows: {int(bad.sum())} / {plan.n_ghost}")
        w = torch.stack((plan.wx, plan.wy, plan.wz), dim=1)
        print(f"[H3] weights range [{float(w.min())},{float(w.max())}]")

    # ---------- H1a: GLOBAL slot completeness ----------
    # stream needs: for every (d_out, i) with nt[opp[d_out], i] == SHELL_OUTSIDE -> slot[d_out, i] >= 0
    src = nt[opp]  # (Q, n)
    need_stream = (src == SHELL_OUTSIDE)
    slot = plan.slot
    miss_stream = need_stream & (slot < 0)
    print(f"[H1-global] stream-ghost needed={int(need_stream.sum())} "
          f"missing-slot={int(miss_stream.sum())}")
    # BFL needs: slot[d, i] >= 0 where nt[opp[d], i] == SHELL_OUTSIDE (BFL upstream ghost)
    miss_bfl = bfl & need_stream & (slot < 0)
    print(f"[H1-global] bfl-mask links needing ghost={int((bfl & need_stream).sum())} "
          f"missing-slot={int(miss_bfl.sum())}")
    if int(miss_stream.sum()):
        rows = torch.nonzero(miss_stream, as_tuple=False)
        print("     missing (d,i):", rows[:20].tolist())

    # duplicate (direction, leaf) entries in the plan (slot overwrite check)
    key = torch.stack((plan.direction, plan.leaf), dim=1)
    uniq = torch.unique(key, dim=0).shape[0]
    print(f"[H1-global] plan rows={plan.n_ghost} unique(direction,leaf)={uniq} "
          f"duplicates={plan.n_ghost - uniq}")

    # ---------- H1b: SLICED slot completeness (both shardings, both ranks) ----------
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
            # stream needs for this rank's leaves
            src_local = nt[opp][:, lidx]                       # (Q, n_local)
            need = src_local == SHELL_OUTSIDE
            miss = need & (p_local.slot < 0)
            # bfl upstream ghost needs
            bfl_need = bfl[:, lidx] & need
            miss_bfl = bfl_need & (p_local.slot < 0)
            # sanity: sliced slot max < n_ghost_local
            ok_range = int((p_local.slot >= p_local.n_ghost).sum()) if p_local.n_ghost else 0
            # sanity: p_local.leaf within [0, n_local)
            leaf_ok = bool((p_local.leaf >= 0).all() and (p_local.leaf < n_local).all())
            print(f"[H1-{mode}] rank{rank} n_local={n_local} n_ghost={p_local.n_ghost} "
                  f"stream-need={int(need.sum())} missing={int(miss.sum())} "
                  f"bfl-ghost-need={int(bfl_need.sum())} bfl-missing={int(miss_bfl.sum())} "
                  f"slot>=n_ghost:{ok_range} leaf_ok:{leaf_ok}")
            if int(miss.sum()):
                rows = torch.nonzero(miss, as_tuple=False)
                print("     missing (d, local_i):", rows[:20].tolist())
    return

def simulate_step1(octree, plan, tag, world_size=2, interleave=False):
    """Reproduce octree_integrated_validate.py step-1 substep-0 on CPU."""
    nz, ny, nx = octree.meta["shape"]
    Q = octree.Q
    n_leaf = octree.n_leaf
    dev = octree.leaf_host_cell.device
    u_in = 0.06
    tau_coarse = 0.5 + 3.0 * (u_in * 10.0 / 100.0)
    from tensorlbm.octree_boundary.stepping import _tau_chain
    taus = _tau_chain(tau_coarse, octree.d_max)
    tau_shell = taus[1]
    host = octree.leaf_host_cell

    # --- f_leaf init exactly as the example ---
    eq_global = equilibrium27(
        torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
        torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx),
    )
    if interleave:
        lidx_all = [interleaved_leaf_indices(n_leaf, world_size, r) for r in range(world_size)]
    else:
        lidx_all = [torch.arange(*split_leaf_bounds(n_leaf, world_size)[r], dtype=torch.int64)
                    for r in range(world_size)]
    # OOB check for the eq_global sampling
    for r in range(world_size):
        lidx = lidx_all[r]
        hz, hy, hx = host[lidx, 0], host[lidx, 1], host[lidx, 2]
        oob = (hz < 0) | (hz > nz - 1) | (hy < 0) | (hy > ny - 1) | (hx < 0) | (hx > nx - 1)
        print(f"[H4-{tag} {interleave and 'ilv' or 'ctg'}] rank{r} f_leaf sample OOB: {int(oob.sum())}")

    from tensorlbm.cumulant import collide_cumulant_d3q27
    def advance_shell(f, tau, level, substep):
        f4 = f.view(Q, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    # coarse field: build the coarse_sparse exactly as the example (shell cells filled,
    # rest equilibrium). n_shell cells:
    shell_cells = torch.nonzero(octree._shell_mask, as_tuple=False)
    n_shell = shell_cells.shape[0]
    full_sc = torch.zeros(Q, n_shell)
    # use the exact equilibrium field at shell cells (uniform flow -> same as eq anyway)
    full_sc[:, :] = eq_global[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]]
    coarse_sparse = equilibrium27(
        torch.ones(nz, ny, nx), torch.full((nz, ny, nx), u_in),
        torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx),
    )
    coarse_sparse[:, shell_cells[:, 0], shell_cells[:, 1], shell_cells[:, 2]] = full_sc
    l1_old = coarse_sparse
    l1_f = coarse_sparse
    alpha = 0.0
    parent_t = torch.lerp(l1_old, l1_f, alpha)

    print(f"[sim {tag} {interleave and 'interleave' or 'contig'}] n_shell={n_shell} "
          f"tau_shell={tau_shell:.6f}")
    for r in range(world_size):
        lidx = lidx_all[r]
        n_local = int(lidx.shape[0])
        f_leaf = eq_global[:, host[lidx, 0], host[lidx, 1], host[lidx, 2]].clone()
        # collide
        populations, post_collision = advance_shell(f_leaf, tau_shell, 1, 0), f_leaf
        # slice plan
        if interleave:
            p_local, grows = _slice_ghost_plan_by_indices(plan, lidx, n_local)
        else:
            lo0, hi0 = split_leaf_bounds(n_leaf, world_size)[r]
            p_local, grows = _slice_ghost_plan(plan, lo0, hi0, n_local)
        # rebuild gplan_fill with global enums (as the stepper does)
        p = p_local
        gplan_fill = ShellGhostPlan(
            p.n_ghost, lidx[p.leaf.cpu()], p.direction, p.z0, p.y0, p.x0,
            p.z1, p.y1, p.x1, p.wx, p.wy, p.wz, p.volume, p.slot,
        )
        ghost_vals = _fill_ghost_impl(octree.leaf_level, gplan_fill, parent_t, taus)
        gv_finite = bool(torch.isfinite(ghost_vals).all())
        # full post-collision (single-process sim of the all_gather sum)
        full_pc = torch.zeros(Q, n_leaf)
        full_pc[:, lidx] = post_collision
        out = stream_gather_distributed(octree, full_pc, ghost_vals, p_local.slot,
                                        f_leaf, lidx)
        nan_elems = int((~torch.isfinite(out)).sum())
        print(f"[sim {tag}] rank{r} n_local={n_local} ghost_finite={gv_finite} "
              f"stream NaN elems={nan_elems} / {Q * n_local}")
        if nan_elems:
            bad = (~torch.isfinite(out))
            dd, ii = torch.nonzero(bad, as_tuple=True)
            uni = torch.unique(ii)
            print(f"     NaN leaves: {len(uni)} unique, e.g. global leaf "
                  f"{lidx[uni[:8]].tolist()} (local {uni[:8].tolist()})")
            # which directions
            print(f"     NaN dirs: {torch.unique(dd).tolist()}")
            # inspect one leaf
            li = int(uni[0])
            gi = int(lidx[li])
            src = octree.neighbor_table[octree._opp][:, gi]
            print(f"     leaf {gi}: level={int(octree.leaf_level[gi])} "
                  f"host={host[gi].tolist()} center={octree.leaf_center[gi].tolist()}")
            for d in torch.unique(dd).tolist():
                print(f"       d={d}: src={int(src[d])} slot={int(p_local.slot[d, li])} "
                      f"post[opp]={float(post_collision[int(OPPOSITE[d]), li]):.4f} "
                      f"out={float(out[d, li]):.4f}")
        # BFL check with ghost slots (replicate bfl_apply_gather's slot lookup)
        bfl = octree.bfl_mask[:, lidx]
        opp = octree._opp
        nt_l = octree.neighbor_table[opp][:, lidx]
        up_ghost = (nt_l == SHELL_OUTSIDE) & bfl
        if bool(up_ghost.any()):
            dd2, ii2 = torch.nonzero(up_ghost, as_tuple=True)
            slots = p_local.slot[dd2, ii2]
            print(f"     BFL-upstream-ghost: {len(dd2)} links, min slot={int(slots.min())} "
                  f"(<0: {int((slots < 0).sum())})")
    return

if __name__ == "__main__":
    R6 = build((96, 64, 64), (48.0, 32.0, 32.0), 6.0, 3.0, "R6")
    check_all(*R6, "R6 r=6 bl=3")
    R10 = build((96, 96, 160), (80.0, 48.0, 48.0), 10.0, 5.0, "R10")
    check_all(*R10, "R10 r=10 bl=5")
    print("\n================ STEP-1 SUBSTEP-0 SIMULATION ================")
    simulate_step1(*R6, "R6", interleave=False)
    simulate_step1(*R6, "R6", interleave=True)
    simulate_step1(*R10, "R10", interleave=False)
    simulate_step1(*R10, "R10", interleave=True)
