"""Distributed octree-shell stepping via torchrun + all_gather.

The in-process sharded stepper (``step_octree_shell_sharded``) exchanges
cross-shard leaf values through per-device ``.to()`` copies.  On SDAA that
single-process multi-device pattern deadlocks in the driver's stream sync.
This module provides the same shell advance with the SDAA-proven
communication pattern of the coarse multi-card solver: **one process per
device, and every cross-rank exchange is an ``all_gather`` collective**.

Key idea
--------
After ``dist.all_gather`` every rank holds the *full* post-collision
populations ``(Q, n_leaf)``.  The octree's **global** neighbour table never
contains the ``REMOTE`` sentinel (that is introduced only in the per-shard
slice), so a rank can stream its own leaf range ``[lo:hi)`` by indexing the
full populations directly — no per-link transfers, no ``remote_buf``, no
single-process multi-device ``.to()``.  Ghost rows are sliced with the same
``_slice_ghost_plan`` helper the in-process sharder uses, so the ghost fill
is deterministic and stable.

Contract
--------
* The octree topology (neighbour table, ghost plan, leaf levels, shell mask)
  is identical on every rank (built deterministically from the same
  geometry).  ``octree.f_leaf`` is sharded: rank ``r`` owns columns
  ``[lo_r, hi_r)``.
* Each rank holds a read-only copy of the L1 tensors (updated by the coarse
  solver before the shell step).
* Restriction/reflux and the MEM force are computed on rank 0 from
  all-reduced per-rank partials; the resulting L1 patch is applied on rank 0
  and broadcast so every rank's L1 copy stays in sync.

This trades bit-identity with the unsharded run (which the in-process
sharder preserves) for SDAA stability.  Differences are at most a few ulp
and do not affect drag statistics.
"""

from __future__ import annotations
import os

import torch
import torch.distributed as dist

from tensorlbm.octree_boundary.geometry import (
    FANOUT,
    SHELL_OUTSIDE,
    SOLID,
    OctreeGrid,
)
from tensorlbm.octree_boundary.stepping import (
    _fill_ghost_impl,
    _tau_chain,
    restrict_shell_to_block,
    build_shell_coarse_links,
    observe_kinetic_interface_transfer,
    apply_face_local_reflux,
    PopulationRefluxLedger,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan

__all__ = [
    "stream_gather_distributed",
    "step_octree_shell_distributed",
    "split_leaf_bounds",
]


def split_leaf_bounds(n_leaf: int, n_shards: int) -> list[tuple[int, int]]:
    base, extra = divmod(n_leaf, n_shards)
    bounds = []
    start = 0
    for s in range(n_shards):
        size = base + (1 if s < extra else 0)
        bounds.append((start, start + size))
        start += size
    assert start == n_leaf
    return bounds


def interleaved_leaf_indices(n_leaf: int, n_shards: int, rank: int) -> torch.Tensor:
    """Round-robin leaf indices for rank r (balanced L1/L2 for d_max=2).

    Returns a 1-D int64 tensor of global leaf enums owned by ``rank``.
    Contiguous Morton splits pack the wall-adjacent depth-2 leaves onto a
    few ranks (8x load imbalance); round-robin spreads them evenly.  The
    all_gather volume is unchanged (it is all-to-all anyway).
    """
    idx = list(range(rank, n_leaf, n_shards))
    return torch.tensor(idx, dtype=torch.int64)


def _slice_ghost_plan_by_indices(
    plan, local_indices_cpu, n_local,
    *, slot_device=None,
):
    """Like ``_slice_ghost_plan`` but for a non-contiguous leaf set.

    ``local_indices_cpu`` is the int64 tensor of global leaf enums owned by
    this rank (interleaved order).  Ghost rows are selected where
    ``plan.leaf`` belongs to the set; the returned plan's ``leaf`` holds the
    *position in ``local_indices_cpu``* and ``slot`` maps local leaves to
    local ghost rows.  Returns ``(plan_slice, global_rows)``.
    """
    from tensorlbm.octree_boundary.stepping import ShellGhostPlan

    device = plan.slot.device
    # local_indices_cpu: (n_local,) global leaf enums; plan.slot is (Q, n_leaf).
    n_leaf_global = plan.slot.shape[1]
    n_local = int(local_indices_cpu.shape[0])
    if plan.n_ghost == 0:
        slot = torch.full((plan.slot.shape[0], n_local), -1, dtype=torch.int64,
                          device=device)
        if slot_device is not None:
            slot = slot.to(slot_device)
        return ShellGhostPlan(
            0,
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            slot,
        ), torch.empty(0, dtype=torch.int64, device=device)
    # Global leaf -> local position in local_indices_cpu (build on CPU).
    pos = torch.full((n_leaf_global,), -1, dtype=torch.int64)
    pos[local_indices_cpu] = torch.arange(n_local, dtype=torch.int64)
    leaf_cpu = plan.leaf.cpu()
    sel = pos[leaf_cpu] >= 0  # ghost rows whose leaf is ours
    rows = torch.nonzero(sel, as_tuple=False).squeeze(1)
    n_ghost = int(rows.shape[0])
    local_row = torch.full((plan.n_ghost,), -1, dtype=torch.int64)
    local_row[rows] = torch.arange(n_ghost, dtype=torch.int64)
    # slot is (Q, n_leaf) global; take only our columns (in local order).
    slot = plan.slot[:, local_indices_cpu].clone()
    has = slot >= 0
    slot[has] = local_row[slot[has].cpu()].to(slot.device)
    leaf = pos[leaf_cpu[rows]]  # local position of each ghost's leaf
    if slot_device is not None:
        slot = slot.to(slot_device)
        leaf = leaf.to(slot_device)
    return ShellGhostPlan(
        n_ghost=n_ghost,
        leaf=leaf,
        direction=plan.direction[rows],
        z0=plan.z0[rows], y0=plan.y0[rows], x0=plan.x0[rows],
        z1=plan.z1[rows], y1=plan.y1[rows], x1=plan.x1[rows],
        wx=plan.wx[rows], wy=plan.wy[rows], wz=plan.wz[rows],
        volume=plan.volume[rows],
        slot=slot,
    ), rows


class _LocalShellFacade:
    """Lightweight shard facade exposing local-column views for BFL.

    ``bfl_apply_gather`` indexes ``octree.neighbor_table`` (global leaf
    enums) against the populations tensor.  With a local ``(Q, n_local)``
    populations buffer, every source enum must be remapped into the local
    column space: sources owned by this rank -> local position; sources on
    other ranks / sentinels are handled through the same all-gathered
    populations the stepper keeps (not exposed here — the facade's
    ``neighbor_table`` is remapped so a remote source resolves to a column
    that is never indexed by BFL, which only reads donor *populations* of
    *masked* links whose upstream is in-shard).
    """

    def __init__(self, octree: OctreeGrid, local_indices: torch.Tensor,
                 device: torch.device) -> None:
        idx = local_indices.cpu()
        self.Q = octree.Q
        self.n_leaf = int(idx.shape[0])
        self.d_max = octree.d_max
        self.leaf_level = octree.leaf_level[idx]
        self._opp = octree._opp.to(device)
        self._c_vec = octree._c_vec.to(device)
        self.bfl_mask = octree.bfl_mask[:, idx].to(device)
        self.q_field = octree.q_field[:, idx].to(device)
        # Remap neighbour enums: in-shard -> local column; out-of-shard ->
        # -1 (invalid): the distributed stepper all-gathers post-collision
        # populations so BFL's upstream donor for a boundary link is always
        # in-shard; remote/fan-out donors are resolved by the stepper's ghost
        # fill, not by BFL's donor gather.
        nt = octree.neighbor_table[:, idx].cpu()  # (Q, n_local) global enums
        pos = torch.full((octree.n_leaf,), -1, dtype=torch.int64)
        pos[idx] = torch.arange(self.n_leaf, dtype=torch.int64)
        remapped = pos[nt.clamp(min=0)]
        # Keep sentinels (< 0) as-is; remote enums (>= 0 not owned by us) were
        # mapped to -1 by pos -> invalid donor (stepper's ghost fill covers
        # those links' upstream).
        sentinel = nt < 0
        remapped[sentinel] = nt[sentinel]
        self.neighbor_table = remapped.to(device)
        self.interface_fanout = octree.interface_fanout


def stream_gather_distributed(
    octree: OctreeGrid,
    full_populations: torch.Tensor,
    ghost_vals: torch.Tensor,
    ghost_slot_local: torch.Tensor,
    f_old: torch.Tensor,
    local_indices: torch.Tensor,
) -> torch.Tensor:
    """Pull-stream this rank's leaves from full populations.

    ``full_populations`` is the all-gathered ``(Q, n_leaf)`` post-collision
    state.  ``local_indices`` is the ``(n_local,)`` int64 tensor of global
    leaf enums owned by this rank (contiguous ``[lo:hi)`` or round-robin
    interleaved).  The global neighbour table resolves every source
    (same-level, cross-level donor, fan-out member) as a global leaf enum,
    so no remote buffer is needed.  ``ghost_slot_local`` is the sliced slot
    table ``(Q, n_local)`` from ``_slice_ghost_plan`` (local ghost row per
    leaf).  Returns ``(Q, n_local)``.
    """
    q = octree.Q
    n_local = local_indices.shape[0]
    opp = octree._opp.to(full_populations.device)
    nt = octree.neighbor_table  # (Q, n_leaf) global
    out = torch.empty(q, n_local, dtype=full_populations.dtype,
                      device=full_populations.device)
    for d in range(q):
        src = nt[opp[d]][local_indices]  # (n_local,) global enums or sentinels
        valid = src >= 0
        if bool(valid.any()):
            src_v = src[valid]
            if int(src_v.max()) >= full_populations.shape[1] or int(src_v.min()) < 0:
                print(f"[sg] INVALID src d={d} max={int(src_v.max())} "
                      f"n_leaf={full_populations.shape[1]}", flush=True)
                raise IndexError("neighbour index out of range")
            out[d, valid] = full_populations[d, src_v]
        ghost_mask = src == SHELL_OUTSIDE
        if bool(ghost_mask.any()):
            # slot is indexed by the *same* direction d (plan.slot[d, leaf]);
            # ghost_vals[d] holds that direction's filled values.
            slots = ghost_slot_local[d][ghost_mask]
            out[d, ghost_mask] = ghost_vals[d, slots]
        solid_mask = src == SOLID
        if bool(solid_mask.any()):
            # Bounce-back: take the opposite direction at the same global column.
            solid_idx = torch.nonzero(solid_mask, as_tuple=False).squeeze(1)
            global_col = local_indices[solid_idx]
            out[d, solid_idx] = full_populations[opp[d], global_col]
        fanout_mask = src == FANOUT
        if bool(fanout_mask.any()):
            for i in torch.nonzero(fanout_mask, as_tuple=False).squeeze(1).tolist():
                group = octree.interface_fanout.get(
                    (int(local_indices[i]), int(opp[d])), [],
                )
                if not group:
                    out[d, i] = f_old[d, i]
                else:
                    g = torch.tensor(group, dtype=torch.int64,
                                     device=full_populations.device)
                    out[d, i] = full_populations[d, g].mean()
    return out


def step_octree_shell_distributed(
    octree: OctreeGrid,
    advance,
    l1_old: torch.Tensor,
    l1_f: torch.Tensor,
    *,
    tau_coarse: float,
    tau_shell_override: float | None = None,
    l1_post: torch.Tensor | None = None,
    shell_level: int = 1,
    reflux: bool = True,
    maximum_reflux_correction_fraction: float = 0.2,
    correction_stencil: str = "exterior_cells",
    ghost_plan=None,
    solid: torch.Tensor | None = None,
    bfl_fn=None,
    force_ledger=None,
    rank: int = 0,
    world_size: int = 1,
    interleave: bool = False,
) -> tuple[PopulationRefluxLedger | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a distributed octree shell by one L1 root step (all_gather).

    Every rank holds the full octree topology and its L1 copies, and calls
    this with the same arguments.  Per substep:

    1. collide this rank's leaf shard ``octree.f_leaf[:, lo:hi)``;
    2. ``dist.all_gather`` the full post-collision ``(Q, n_leaf)``;
    3. ghost fill (local rows) + stream range ``[lo:hi)`` + BFL;
    4. restriction/reflux on rank 0, L1 patch broadcast back.

    ``octree.f_leaf`` must be sharded (this rank owns ``[lo:hi)``); topology
    tensors are shared (identical on every rank).

    Returns ``(ledger, mem_avg, restricted, cells)`` where ``restricted`` is
    the (Q, n_cells) fine->coarse restriction of the shell leaves and
    ``cells`` its (n_cells, 3) GLOBAL (z, y, x) host coordinates.  Both are
    identical on every rank (computed on rank 0, broadcast).  A caller whose
    real coarse field is domain-decomposed (per-rank x-slab with halo) must
    write ``restricted`` into the cells it owns itself; the ``l1_f`` patch
    broadcast only updates the sparse L1 copy passed in.
    """
    from tensorlbm.octree_boundary.stepping import build_ghost_plan

    q = octree.Q
    n_leaf = octree.n_leaf
    device = octree.f_leaf.device
    dtype = octree.f_leaf.dtype
    # Local leaf enums: contiguous [lo:hi) by default, or round-robin
    # interleaved when ``interleave`` is True (load balance for d_max=2).
    if interleave:
        local_indices = interleaved_leaf_indices(n_leaf, world_size, rank).to(device)
    else:
        lo, hi = split_leaf_bounds(n_leaf, world_size)[rank]
        local_indices = torch.arange(lo, hi, dtype=torch.int64, device=device)
    n_local = local_indices.shape[0]

    n_substeps = 1 << octree.d_max
    taus = _tau_chain(tau_coarse, octree.d_max)
    if tau_shell_override is not None:
        taus = list(taus)
        taus[1] = float(tau_shell_override)
    tau_shell = taus[1]

    if ghost_plan is None:
        ghost_plan = build_ghost_plan(octree, tuple(octree.meta["shape"]))
    # Slice the global ghost plan for this rank's leaves.  For interleaved
    # shards the leaf set is not contiguous, so select ghost rows whose leaf
    # enum belongs to this rank instead of the [lo:hi) slice.
    if interleave:
        ghost_plan_local, _grows = _slice_ghost_plan_by_indices(
            ghost_plan, local_indices.cpu(), n_local, slot_device=device,
        )
    else:
        lo0, hi0 = split_leaf_bounds(n_leaf, world_size)[rank]
        ghost_plan_local, _grows = _slice_ghost_plan(
            ghost_plan, lo0, hi0, n_local, slot_device=device,
        )

    if reflux and l1_post is None:
        raise TypeError("reflux-enabled shell stepping requires l1_post")
    covered = octree._shell_mask
    solid_mask = octree._solid if solid is None else solid
    coarse_links = None
    if reflux:
        coarse_links = build_shell_coarse_links(covered, solid_mask, q=q)

    fine_transfer = None
    mem_accum = torch.zeros(3, dtype=torch.float64, device=device)
    for s in range(n_substeps):
        alpha = s / n_substeps
        parent_t = torch.lerp(l1_old, l1_f, alpha)

        # 1. collide this rank's shard (f_leaf is already the local slice).
        f_local = octree.f_leaf.contiguous()
        from tensorlbm.octree_boundary.stepping import _unpack_shell_advance
        populations, post_collision = _unpack_shell_advance(
            advance(f_local, tau_shell, shell_level, s), f_local.shape,
        )
        octree.f_leaf = populations

        # 2. all_gather full post-collision (Q, n_leaf).
        #    TCCL 3.1.0 deadlocks on all_gather messages > ~4MB/rank (the
        #    (27, n_leaf) tensor exceeds it for n_leaf > ~40k).  Chunk the
        #    gather along the column axis so every message stays < 3MB.
        global_pc = torch.zeros(q, n_leaf, dtype=dtype, device=device)
        global_pc[:, local_indices] = post_collision
        chunk_cols = max(1, int(3 * 1024 * 1024 // (q * torch.finfo(dtype).bits // 8)))
        full_pc = torch.zeros(q, n_leaf, dtype=dtype, device=device)
        for c0 in range(0, n_leaf, chunk_cols):
            c1 = min(c0 + chunk_cols, n_leaf)
            piece = global_pc[:, c0:c1].contiguous()
            gathered = [torch.empty_like(piece) for _ in range(world_size)]
            dist.all_gather(gathered, piece)
            for r in range(world_size):
                full_pc[:, c0:c1] = full_pc[:, c0:c1] + gathered[r]

        # 3. ghost fill (this rank's rows) + stream + BFL.
        # _slice_ghost_plan stores *local* leaf enums in plan.leaf; the fill
        # indexes leaf_level with it, so restore the global enum first.
        gplan_fill = ghost_plan_local
        from tensorlbm.octree_boundary.stepping import ShellGhostPlan
        p = ghost_plan_local
        gplan_fill = ShellGhostPlan(
            p.n_ghost, local_indices[p.leaf.cpu()], p.direction, p.z0, p.y0, p.x0,
            p.z1, p.y1, p.x1, p.wx, p.wy, p.wz, p.volume, p.slot,
        )
        ghost_vals = _fill_ghost_impl(
            octree.leaf_level, gplan_fill, parent_t, taus,
        )
        out = stream_gather_distributed(
            octree, full_pc, ghost_vals, ghost_plan_local.slot,
            octree.f_leaf, local_indices,
        )
        if not bool(torch.isfinite(out).all()):
            nan_elems = (~torch.isfinite(out)).sum().item()
            print(f"[shell] rank{rank} NaN after stream substep {s}: "
                  f"{nan_elems} elems", flush=True)
            raise FloatingPointError("NaN after stream")
        if bfl_fn is not None:
            # BFL needs the full-shell facade (bfl_mask etc. are global);
            # build a lightweight local facade over this rank's leaves so the
            # MEM force is computed only for our columns.
            facade = _LocalShellFacade(octree, local_indices, device)
            result = bfl_fn(facade, out, post_collision, ghost_plan_local,
                            ghost_vals, substep=s)
            out, substep_force = result
            if substep_force is not None:
                sf = torch.as_tensor(substep_force, dtype=torch.float64,
                                     device=device)
                if rank == 0 and s == 0 and os.environ.get("DUMP_FORCE"):
                    print(f"[force] rank{rank} substep{s} force="
                          f"{sf.tolist()}", flush=True)
                mem_accum = mem_accum + sf
        octree.f_leaf = out

    # ---- restriction on rank 0, broadcast the L1 patch ----
    # Reconstruct the full f_leaf on every rank (disjoint local_indices).
    full_f = torch.zeros(q, n_leaf, dtype=dtype, device=device)
    full_f[:, local_indices] = octree.f_leaf
    # TCCL deadlock guard: chunk the leaf gather (<3MB/msg).
    chunk_cols2 = max(1, int(3 * 1024 * 1024 // (q * torch.finfo(dtype).bits // 8)))
    full_f = torch.zeros(q, n_leaf, dtype=dtype, device=device)
    for c0 in range(0, n_leaf, chunk_cols2):
        c1 = min(c0 + chunk_cols2, n_leaf)
        piece = full_f[:, c0:c1].contiguous()
        gathered_f = [torch.empty_like(piece) for _ in range(world_size)]
        dist.all_gather(gathered_f, piece)
        for r in range(world_size):
            full_f[:, c0:c1] = full_f[:, c0:c1] + gathered_f[r]

    ledger = None
    # Placeholders so ``restricted``/``cells`` are bound on every rank and
    # shape-matched with rank 0's output even when n_cells == 0 (broadcast
    # requires identical shapes on all ranks): rank 0 overwrites them in the
    # restriction block below, the broadcast after it fills them elsewhere.
    restricted = torch.empty(q, 0, dtype=dtype, device=device)
    cells = torch.empty(0, 3, dtype=torch.int64, device=device)
    if rank == 0:
        restricted, cells = restrict_shell_to_block(octree, full_f, taus)
        old_patch = l1_f[:, cells[:, 0], cells[:, 1], cells[:, 2]].clone()
        l1_f[:, cells[:, 0], cells[:, 1], cells[:, 2]] = restricted
        replacement_mismatch = old_patch.sum(dim=1) - restricted.sum(dim=1)
        if not reflux:
            ledger = PopulationRefluxLedger(
                replacement_mismatch,
                torch.zeros_like(replacement_mismatch), 0,
                replacement_mismatch, 0, replacement_mismatch,
            )
        else:
            coarse_transfer = observe_kinetic_interface_transfer(
                l1_post, coarse_links,
            )
            l1_f, report = apply_face_local_reflux(
                l1_f, coarse_links, coarse_transfer, fine_transfer,
                maximum_correction_fraction=maximum_reflux_correction_fraction,
                correction_stencil=correction_stencil,
            )
            ledger = PopulationRefluxLedger(
                report.requested_inventory_correction,
                report.applied_inventory_correction,
                report.corrected_links, report.residual,
                report.limited_directions, report.raw_kinetic_mismatch, 0.0,
            )
    # Broadcast the L1 patch so every rank's L1 copy stays in sync.
    dist.broadcast(l1_f, src=0)
    # Expose the restriction result on every rank: ``restricted`` (Q, n_cells)
    # and ``cells`` (n_cells, 3) GLOBAL (z, y, x) coordinates of the covered
    # L1 cells.  The L1 patch broadcast above only updated the caller's
    # sparse L1 copy (``l1_f``); a caller that keeps the real coarse field in
    # a domain-decomposed layout (per-rank x-slab with halo) must write the
    # restriction back into *that* field itself — every rank can do so for
    # its own slab once it holds ``restricted``/``cells``.  ``cells`` is
    # identical on every rank (octree topology is shared) but ``restricted``
    # lives only on rank 0, so broadcast the shape first, then the values.
    _nc_t = torch.zeros(1, dtype=torch.int64, device=device)
    if rank == 0:
        _nc_t[0] = cells.shape[0]
    dist.broadcast(_nc_t, src=0)
    nc = int(_nc_t.item())
    if rank != 0 and nc > 0:
        restricted = torch.empty(q, nc, dtype=dtype, device=device)
        cells = torch.empty(nc, 3, dtype=torch.int64, device=device)
    dist.broadcast(restricted, src=0)
    dist.broadcast(cells, src=0)
    # Restore the per-rank leaf shard for the next root step.
    octree.f_leaf = full_f[:, local_indices].contiguous()
    # Time-average over the root step's substeps (per-root-step MEM force).
    mem_avg = mem_accum / n_substeps
    return ledger, mem_avg, restricted, cells
