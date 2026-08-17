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
    build_ghost_plan_coarse_parent,
    ensure_fanout_tables,
    restrict_shell_to_block,
    build_shell_coarse_links,
    observe_kinetic_interface_transfer,
    apply_face_local_reflux,
    PopulationRefluxLedger,
)
from tensorlbm.kinetic_flux_register import KineticInterfaceTransfer
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan

__all__ = [
    "stream_gather_distributed",
    "step_octree_shell_distributed",
    "split_leaf_bounds",
]

_DBG_ROOT_STEP = 0  # debug-only root-step counter (DBG_NAN instrumentation)


def _dbg_root_step() -> int:
    return _DBG_ROOT_STEP


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
        lev=plan.lev[rows] if plan.lev is not None else None,
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
        # Set per substep by the stepper: (Q, n_local) float64 fanout member
        # means (computed from the all-gathered post-collision state).  BFL's
        # fanout donor branch reads this instead of the dict.
        self.fanout_mean: torch.Tensor | None = None


def _build_local_fanout_cache(
    octree: OctreeGrid,
    local_indices: torch.Tensor,
    device: torch.device,
) -> dict:
    """Static per-rank fanout positions (topology is fixed per root step).

    Returns a dict of tensors on ``device``:

    * ``n``: number of FANOUT cells owned by this rank whose group is live;
    * ``d`` / ``i``: ``(n,)`` stream directions / LOCAL leaf columns with a
      FANOUT source (``src_all[d, i] == FANOUT``);
    * ``pad`` / ``vp``: ``(n, max_len)`` global member enums / valid mask —
      the member means are ``mean(full_populations[d, members])``;
    * ``fb_n`` / ``fb_d`` / ``fb_i``: defensive fallback cells (FANOUT but no
      registered group) whose stream output falls back to ``f_old``.
    """
    empty = {"n": 0, "d": None, "i": None, "pad": None, "vp": None,
             "fb_n": 0, "fb_d": None, "fb_i": None}
    rowidx, pad_live = ensure_fanout_tables(octree)
    rowidx = rowidx.detach().cpu()                     # lookups stay on CPU
    pad_live = pad_live.to(device)
    opp = octree._opp.cpu()
    # Locate this rank's FANOUT cells on the CPU only.  SDAA's ``nonzero``
    # kernel faults with SDAA_ERROR_MISALIGNED_ADDRESS when the dynamic
    # output is large (observed on ``src_local == FANOUT`` for d_max=2), so
    # the mask -> indices conversion must never run on the device.  The
    # cache is rebuilt once per root step, so the CPU scan is negligible.
    rows = torch.nonzero(octree.neighbor_table.cpu() == FANOUT, as_tuple=False)
    if rows.shape[0] == 0:
        return empty
    q_f = rows[:, 0]                                   # neighbour-table direction
    g_f = rows[:, 1]                                   # global leaf enum
    # Global enum -> local column map for this rank's shard.
    idx_cpu = local_indices.detach().cpu()
    pos_of = torch.full((octree.n_leaf,), -1, dtype=torch.int64)
    pos_of[idx_cpu] = torch.arange(idx_cpu.shape[0], dtype=torch.int64)
    i_f = pos_of[g_f]                                  # -1 when not owned here
    own = i_f >= 0
    if not bool(own.any()):
        return empty
    d_f = opp[q_f[own]]                                # pull direction
    i_f = i_f[own]
    g_f = g_f[own]
    ridx = rowidx[q_f[own], g_f]                       # live row / -1 (CPU)
    has = ridx >= 0
    cache = dict(empty)
    cache["n"] = int(has.sum())
    if cache["n"]:
        cache["d"] = d_f[has].to(device)
        cache["i"] = i_f[has].to(device)
        # CPU gather (large index tensors fault on SDAA), then move to device.
        cache["pad"] = pad_live[ridx[has]].to(device)
        cache["vp"] = cache["pad"] >= 0
    fb = ~has
    cache["fb_n"] = int(fb.sum())
    if cache["fb_n"]:
        cache["fb_d"] = d_f[fb].to(device)
        cache["fb_i"] = i_f[fb].to(device)
    return cache


def stream_gather_distributed(
    octree: OctreeGrid,
    full_populations: torch.Tensor,
    ghost_vals: torch.Tensor,
    ghost_slot_local: torch.Tensor,
    f_old: torch.Tensor,
    local_indices: torch.Tensor,
    fan_cache: dict | None = None,
    fan_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pull-stream this rank's leaves from full populations.

    ``full_populations`` is the all-gathered ``(Q, n_leaf)`` post-collision
    state.  ``local_indices`` is the ``(n_local,)`` int64 tensor of global
    leaf enums owned by this rank (contiguous ``[lo:hi)`` or round-robin
    interleaved).  The global neighbour table resolves every source
    (same-level, cross-level donor, fan-out member) as a global leaf enum,
    so no remote buffer is needed.  ``ghost_slot_local`` is the sliced slot
    table ``(Q, n_local)`` from ``_slice_ghost_plan`` (local ghost row per
    leaf).  ``fan_cache`` is the static per-rank fanout table from
    ``_build_local_fanout_cache`` (built once per root step).  Returns
    ``(Q, n_local)``.

    Fully batched: one ``torch.gather`` for all leaf sources, one ``nonzero``
    per sentinel branch (ghost / solid / fanout) — zero per-direction Python
    loops and zero per-direction device syncs (each ``bool(...)`` in the old
    per-direction loop forced a device sync, ~108 per substep).
    """
    q = octree.Q
    n_local = local_indices.shape[0]
    n_leaf = full_populations.shape[1]
    opp = octree._opp.to(full_populations.device)
    nt = octree.neighbor_table.to(full_populations.device)
    out = torch.empty(q, n_local, dtype=full_populations.dtype,
                      device=full_populations.device)
    # ---- one batched pass: every source, every direction ------------------
    src_all = nt[opp][:, local_indices]          # (Q, n_local) enums/sentinels
    valid = src_all >= 0
    if bool(valid.any()):
        if int(src_all.max()) >= n_leaf:
            raise IndexError("neighbour index out of range")
        gathered = torch.gather(full_populations, 1, src_all.clamp(min=0))
        out = torch.where(valid, gathered, torch.zeros_like(gathered))
    else:
        out.zero_()
    # ghost 分支 (SHELL_OUTSIDE): 一次 nonzero 收集所有 (d, i), 批量取 slot
    ghost_all = src_all == SHELL_OUTSIDE
    if bool(ghost_all.any()):
        rows = torch.nonzero(ghost_all, as_tuple=False)    # (n_g, 2) (d, i)
        d_g, i_g = rows[:, 0], rows[:, 1]
        slots = ghost_slot_local[d_g, i_g]
        out[d_g, i_g] = ghost_vals[d_g, slots]
    # solid 分支 (SOLID): bounce-back, 同全局列取 opp[d] 方向
    solid_all = src_all == SOLID
    if bool(solid_all.any()):
        rows = torch.nonzero(solid_all, as_tuple=False)    # (n_s, 2) (d, i)
        d_s, i_s = rows[:, 0], rows[:, 1]
        out[d_s, i_s] = full_populations[opp[d_s], local_indices[i_s]]
    # fanout 分支: 预缓存位置一次批量 gather + 段均值
    if fan_cache is not None and fan_cache["n"] > 0:
        d_f, i_f = fan_cache["d"], fan_cache["i"]
        if fan_mean is not None:
            # float64 mean computed once per substep by the stepper
            # (shared with BFL's upstream donor resolution).
            out[d_f, i_f] = fan_mean[d_f, i_f].to(full_populations.dtype)
        else:
            pad, vp = fan_cache["pad"], fan_cache["vp"]
            vals = full_populations[d_f.unsqueeze(1), pad.clamp(min=0)]
            means = (vals * vp).sum(dim=1) / vp.sum(dim=1).clamp_min(1)
            out[d_f, i_f] = means.to(full_populations.dtype)
    if fan_cache is not None and fan_cache["fb_n"] > 0:
        # 防御: FANOUT 但无注册组 -> 保留旧值 (与旧循环一致)
        out[fan_cache["fb_d"], fan_cache["fb_i"]] = (
            f_old[fan_cache["fb_d"], fan_cache["fb_i"]]
        )
    return out


def step_octree_shell_distributed(
    octree: OctreeGrid,
    advance,
    l1_old: torch.Tensor,
    l1_f: torch.Tensor,
    *,
    tau_coarse: float,
    tau_shell_override: float | None = None,
    l1_post: torch.Tensor | list[torch.Tensor] | None = None,
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
    ghost_parent_old: torch.Tensor | None = None,
    ghost_parent_new: torch.Tensor | None = None,
    ghost_parent_offset: tuple[int, int, int] | None = None,
    ghost_parent_tau: float | None = None,
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

    Ghost supply source (SUBOFF L1 P0 fix)
    --------------------------------------
    By default the ghost fill samples the time-lerped ``l1_old``/``l1_f``
    fields through the L1-frame ghost plan (``parent_t``).  When
    ``ghost_parent_old``/``ghost_parent_new`` are given (the evolved coarse
    window fields at the root-step start/end) together with
    ``ghost_parent_offset`` (the L1 physical origin in window coordinates,
    ``(box.z0 - win.z0, box.y0 - win.y0, box.x0 - win.x0)``) and
    ``ghost_parent_tau`` (the real coarse relaxation time), the ghost fill
    instead samples the time-lerped *coarse* field through a coarse-frame
    ghost plan (:func:`build_ghost_plan_coarse_parent`) — the genuine
    evolved coarse supply of the legacy two-level path, skipping the 2:1
    injection of the L1 block field.  The shell host remains the L1 grid:
    collide/stream/BFL/restriction/reflux are unchanged, only the ghost
    values handed to the stream's SHELL_OUTSIDE branch (and to BFL's
    donor resolution / the reflux incoming observation) change source.

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
    # ---- P0 fix: optional coarse-parent ghost supply (see docstring) ----
    use_coarse_parent = ghost_parent_old is not None
    if use_coarse_parent:
        if ghost_parent_new is None or ghost_parent_offset is None \
                or ghost_parent_tau is None:
            raise TypeError(
                "step_octree_shell_distributed: ghost_parent_old/new/offset/"
                "tau must all be provided together",
            )
        _wz, _wy, _wx = ghost_parent_old.shape[1:]
        ghost_plan_coarse = build_ghost_plan_coarse_parent(
            octree, (_wz, _wy, _wx), ghost_parent_offset,
        )
        # The ghost rescale chain: the sampled parent is the REAL coarse
        # field (relaxation ghost_parent_tau), so the chain must start at
        # tau_coarse — [tau_c, tau_l1, tau_shell] for d_max=2 — matching
        # the legacy two-level path's [tau_c, tau_shell] convention
        # (level-l ghost tau_f = chain[lev], base = chain[0]).
        # Chain length: the L1-hosted octree's levels are relative to the
        # L1 physical grid (already 2x finer than coarse), so a depth-l
        # leaf sits at coarse level l+1; the deepest leaf (l = d_max)
        # indexes chain[d_max+1], hence d_max+1 refinements.
        taus_ghost = _tau_chain(ghost_parent_tau, octree.d_max + 1)
    else:
        ghost_plan_coarse = None
        taus_ghost = None

    def _slice_plan(plan):
        """Per-rank slice of a global ghost plan (contiguous or interleaved)."""
        if interleave:
            return _slice_ghost_plan_by_indices(
                plan, local_indices.cpu(), n_local, slot_device=device,
            )[0]
        lo0, hi0 = split_leaf_bounds(n_leaf, world_size)[rank]
        return _slice_ghost_plan(
            plan, lo0, hi0, n_local, slot_device=device,
        )[0]

    def _restore_global(plan_slice):
        """Map sliced-plan local leaf enums back to global enums."""
        p = plan_slice
        return ShellGhostPlan(
            p.n_ghost, local_indices[p.leaf.cpu()], p.direction,
            p.z0, p.y0, p.x0, p.z1, p.y1, p.x1,
            p.wx, p.wy, p.wz, p.volume, p.slot,
            lev=p.lev if p.lev is not None else None,
        )

    # Static per-rank fanout positions (topology is fixed): the rank's
    # FANOUT cells and their member tables.  Built once per root step, reused
    # by every substep's stream + BFL donor resolution.
    fan_cache = _build_local_fanout_cache(octree, local_indices, device)
    # Slice the global ghost plan for this rank's leaves.  For interleaved
    # shards the leaf set is not contiguous, so select ghost rows whose leaf
    # enum belongs to this rank instead of the [lo:hi) slice.
    ghost_plan_local = _slice_plan(ghost_plan)

    if reflux and l1_post is None:
        raise TypeError("reflux-enabled shell stepping requires l1_post")
    covered = octree._shell_mask
    solid_mask = octree._solid if solid is None else solid
    coarse_links = None
    observation_links = None
    if reflux:
        coarse_links = build_shell_coarse_links(covered, solid_mask, q=q)
        # Mass-conservation observation must count the FULL covered boundary
        # (including covered<->solid links whose fine-side counterparts are
        # the ghost-filled inner-wall interface links).  The solid-excluded
        # ``coarse_links`` remain the correction stencil so the reflux never
        # writes into solid cells.  This pairing closes the joint mass
        # identity ``dM = -residual.sum()`` exactly (same as the unsharded
        # ``step_octree_shell``).
        observation_links = build_shell_coarse_links(
            coarse_links.inside, None, q=q,
        )

    fine_transfer = None
    mem_accum = torch.zeros(3, dtype=torch.float64, device=device)
    dbg_mass_log = []  # (root step tag, substep, stage, value)
    for s in range(n_substeps):
        alpha = s / n_substeps
        parent_t = torch.lerp(l1_old, l1_f, alpha)

        # 1. collide this rank's shard (f_leaf is already the local slice).
        f_local = octree.f_leaf.contiguous()
        f_in_save = f_local.clone()
        from tensorlbm.octree_boundary.stepping import _unpack_shell_advance
        populations, post_collision = _unpack_shell_advance(
            advance(f_local, tau_shell, shell_level, s), f_local.shape,
        )
        octree.f_leaf = populations
        if os.environ.get("DBG_NAN"):
            dbg_mass_log.append((s, "collide_in", float(f_in_save.sum().item())))
            dbg_mass_log.append((s, "post_collide", float(post_collision.sum().item())))
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
        from tensorlbm.octree_boundary.stepping import ShellGhostPlan
        p = ghost_plan_local
        gplan_fill = ShellGhostPlan(
            p.n_ghost, local_indices[p.leaf.cpu()], p.direction, p.z0, p.y0, p.x0,
            p.z1, p.y1, p.x1, p.wx, p.wy, p.wz, p.volume, p.slot,
            lev=p.lev if p.lev is not None else None,
        )
        if use_coarse_parent:
            # P0 fix: supply the shell ghosts from the genuine evolved
            # coarse field (time-lerped cw_old/cw_new, coarse-frame
            # trilinear stencil) instead of the 2:1-injected L1 block
            # field.  Same row order / slot table as the L1-frame plan
            # (identical leaf/direction rows), so stream, BFL and the
            # reflux incoming observation stay consistent.
            assert ghost_parent_old is not None and ghost_parent_new is not None
            assert ghost_parent_tau is not None and taus_ghost is not None
            gplan_fill_coarse = _restore_global(_slice_plan(ghost_plan_coarse))
            ghost_vals = _fill_ghost_impl(
                octree.leaf_level, gplan_fill_coarse,
                torch.lerp(ghost_parent_old, ghost_parent_new, alpha),
                taus_ghost,
            )
        else:
            ghost_vals = _fill_ghost_impl(
                octree.leaf_level, gplan_fill, parent_t, taus,
            )
        if os.environ.get("DBG_NAN"):
            dbg_mass_log.append((s, "ghost_vals_sum", float(ghost_vals.sum().item())))
            n_gh = ghost_plan_local.n_ghost
            dbg_mass_log.append((s, "ghost_rows", float(n_gh)))
            dbg_mass_log.append((s, "ghost_leaf_count",
                                 float(torch.unique(ghost_plan_local.leaf).shape[0])))
        if os.environ.get("DBG_NAN") and not bool(torch.isfinite(ghost_vals).all()):
            nnan = (~torch.isfinite(ghost_vals)).sum().item()
            dd, rr = torch.nonzero(~torch.isfinite(ghost_vals), as_tuple=True)
            print(f"[dbg] rank{rank} substep{s} ghost_vals NaN: {nnan} elems "
                  f"rows={torch.unique(rr).tolist()[:10]}", flush=True)
        # Fanout member means over the all-gathered post-collision state —
        # one batched gather per substep (no per-group Python loop).  The
        # same (d, leaf) cells serve the stream fanout branch and BFL's
        # upstream donor resolution (identical direction convention), so the
        # table is computed once and shared.
        fan_mean_t = None
        if fan_cache["n"] > 0:
            _vals = full_pc[fan_cache["d"].unsqueeze(1),
                            fan_cache["pad"].clamp(min=0)]
            _means = (
                _vals.to(torch.float64) * fan_cache["vp"]
            ).sum(dim=1) / fan_cache["vp"].sum(dim=1).clamp_min(1)
            fan_mean_t = torch.zeros(q, n_local, dtype=torch.float64,
                                     device=device)
            fan_mean_t[fan_cache["d"], fan_cache["i"]] = _means
        out = stream_gather_distributed(
            octree, full_pc, ghost_vals, ghost_plan_local.slot,
            octree.f_leaf, local_indices,
            fan_cache=fan_cache, fan_mean=fan_mean_t,
        )
        if os.environ.get("DBG_NAN"):
            dbg_mass_log.append((s, "post_stream", float(out.sum().item())))
            # mass pulled from ghosts into this rank's leaves
            src_all = octree.neighbor_table[octree._opp][:, local_indices]
            gh = src_all == SHELL_OUTSIDE
            ghost_pulled = float(out[gh].sum().item()) if bool(gh.any()) else 0.0
            dbg_mass_log.append((s, "ghost_pulled_in", ghost_pulled))
        if os.environ.get("DBG_NAN"):
            for gc in [14775, 30153, 45060, 55123, 55127, 67864, 67868,
                       77931, 83239, 92838, 92841, 108216, 111655]:
                m = (local_indices == gc).nonzero(as_tuple=False)
                if m.numel():
                    loc = int(m[0])
                    ov = out[:, loc]
                    fv = f_in_save[:, loc]
                    print(f"[dbg] rank{rank} substep{s} global_col {gc}: "
                          f"f_in sum={float(fv.sum()):.6g} "
                          f"stream_out sum={float(ov.sum()):.6g} "
                          f"out_finite={bool(torch.isfinite(ov).all())}",
                          flush=True)
        if not bool(torch.isfinite(out).all()):
            nan_elems = (~torch.isfinite(out)).sum().item()
            print(f"[shell] rank{rank} NaN after stream substep {s}: "
                  f"{nan_elems} elems", flush=True)
            if os.environ.get("DBG_NAN"):
                bad = ~torch.isfinite(out)
                dd, ii = torch.nonzero(bad, as_tuple=True)
                uni = torch.unique(ii)
                print(f"[dbg] NaN leaves {len(uni)} unique global "
                      f"{local_indices[uni[:10]].tolist()} dirs "
                      f"{torch.unique(dd).tolist()}", flush=True)
                opp_t = octree._opp.to(full_pc.device)
                for li in uni[:5].tolist():
                    gi = int(local_indices[li])
                    src = octree.neighbor_table[opp_t][:, gi]
                    print(f"[dbg] leaf {gi} host="
                          f"{octree.leaf_host_cell[gi].tolist()} "
                          f"q={octree.q_field[:, gi].tolist()}", flush=True)
                    for d in torch.unique(dd).tolist():
                        print(f"[dbg]   d={d} src={int(src[d])} "
                              f"slot={int(ghost_plan_local.slot[d, li])} "
                              f"fullpc={float(full_pc[d, gi])} "
                              f"out={float(out[d, li])}", flush=True)
                # where do NaN leaves get their values? check each branch
                print(f"[dbg] post_collision finite="
                      f"{bool(torch.isfinite(post_collision).all())} "
                      f"full_pc finite={bool(torch.isfinite(full_pc).all())}",
                      flush=True)
                if not bool(torch.isfinite(full_pc).all()):
                    bd, bi = torch.nonzero(~torch.isfinite(full_pc), as_tuple=True)
                    cols = torch.unique(bi).tolist()[:20]
                    print(f"[dbg] full_pc NaN cols={cols} "
                          f"dirs={torch.unique(bd).tolist()}", flush=True)
                    # collide input at those columns (local col = global - lo)
                    # NOTE: for interleave, local col != global col; map back.
                    for gc in cols:
                        loc = int((local_indices == gc).nonzero(as_tuple=False)[0])
                        fin = f_in_save[:, loc]
                        print(f"[dbg]   col {gc} collide-input finite="
                              f"{bool(torch.isfinite(fin).all())} "
                              f"sum={float(fin.sum()):.6g} "
                              f"min={float(fin.min()):.6g} "
                              f"max={float(fin.max()):.6g} "
                              f"pc={post_collision[:, loc].tolist()[:6]}",
                              flush=True)
            raise FloatingPointError("NaN after stream")
        if bfl_fn is not None:
            # BFL needs the full-shell facade (bfl_mask etc. are global);
            # build a lightweight local facade over this rank's leaves so the
            # MEM force is computed only for our columns.
            facade = _LocalShellFacade(octree, local_indices, device)
            if fan_mean_t is not None:
                # Precomputed fanout donor means (float64, from full_pc):
                # bfl_apply_gather's fanout branch reads these instead of the
                # per-group dict loop (which was both a CPU hotspot AND a
                # correctness bug — it keyed the global registry with LOCAL
                # columns, always missing and falling back to fp_d).
                facade.fanout_mean = fan_mean_t
            result = bfl_fn(facade, out, post_collision, ghost_plan_local,
                            ghost_vals, substep=s)
            out, substep_force = result
            if os.environ.get("DBG_NAN"):
                dbg_mass_log.append((s, "post_bfl", float(out.sum().item())))
            if os.environ.get("DBG_NAN") and not bool(torch.isfinite(out).all()):
                nnan = (~torch.isfinite(out)).sum().item()
                dd, ii = torch.nonzero(~torch.isfinite(out), as_tuple=True)
                print(f"[dbg] rank{rank} substep{s} BFL-out NaN: {nnan} elems "
                      f"dirs={torch.unique(dd).tolist()[:10]} "
                      f"local_leaves={torch.unique(ii).tolist()[:10]} "
                      f"q_range=[{float(octree.q_field.min())},"
                      f"{float(octree.q_field.max())}] "
                      f"bfl_links={int(octree.bfl_mask.sum())}", flush=True)
            if substep_force is not None:
                sf = torch.as_tensor(substep_force, dtype=torch.float64,
                                     device=device)
                if rank == 0 and s == 0 and os.environ.get("DUMP_FORCE"):
                    print(f"[force] rank{rank} substep{s} force="
                          f"{sf.tolist()}", flush=True)
                mem_accum = mem_accum + sf
        # 4. reflux observation (distributed): observe the fine-side
        #    interface transfer from the all-gathered full_pc (outgoing)
        #    and the local ghost_vals (incoming, all-reduced).  This is
        #    the distributed counterpart of ``observe_shell_interface_transfer``
        #    in the unsharded stepper — each staircase link is counted once,
        #    per-leaf volume scaling, accumulated over substeps.
        if reflux:
            _obs_out = torch.zeros(q, dtype=dtype, device=device)
            _obs_in = torch.zeros(q, dtype=dtype, device=device)
            _if_links = octree.interface_links
            _leaf_vol = octree.leaf_volume()
            # Batched per-direction observation (old code: 26-iteration
            # Python loop with ~52 device-sync ``bool(...)`` per substep).
            # One scatter_add per side; direction 0 is never a link (rest
            # direction self-references) but is zeroed for exact equivalence
            # with the old ``range(1, q)`` loop.
            if _if_links.shape[0]:
                _d_l = _if_links[:, 1]
                _li = _if_links[:, 0]
                _obs_out.scatter_add_(
                    0, _d_l,
                    (full_pc[_d_l, _li] * _leaf_vol[_li].to(dtype)),
                )
            if ghost_plan_local.n_ghost:
                _gdir = ghost_plan_local.direction
                _grow = torch.arange(ghost_plan_local.n_ghost, device=device)
                _obs_in.scatter_add_(
                    0, _gdir,
                    (ghost_vals[_gdir, _grow]
                     * ghost_plan_local.volume.to(dtype)),
                )
            _obs_out[0] = 0
            _obs_in[0] = 0
            # outgoing is identical on every rank (full_pc is global);
            # incoming is a per-rank partial — all-reduce to assemble the
            # global incoming sum (one all_reduce per substep).
            dist.all_reduce(_obs_in, op=dist.ReduceOp.SUM)
            _observed = KineticInterfaceTransfer(_obs_out, _obs_in)
            fine_transfer = (
                _observed if fine_transfer is None
                else fine_transfer + _observed
            )
        octree.f_leaf = out

    # ---- restriction on rank 0, broadcast the L1 patch ----
    # Reconstruct the full f_leaf on every rank (disjoint local_indices).
    # NOTE: do NOT scatter the shard into full_f before the gather-sum below.
    # The chunked all_gather sum trick reconstructs each chunk as
    # ``piece_r + sum_r gathered[r]``; if the accumulator already holds the
    # local shard, every rank's OWN columns are counted twice in its copy of
    # full_f and the ``full_f[:, local_indices]`` restore doubles the whole
    # shell's mass every root step (1 -> 2 -> 4 ... -> NaN by step ~4).
    # Scatter into a scratch buffer and accumulate into a zeroed full_f
    # instead (the same pattern the substep loop uses for full_pc).
    scatter_f = torch.zeros(q, n_leaf, dtype=dtype, device=device)
    scatter_f[:, local_indices] = octree.f_leaf
    full_f = torch.zeros(q, n_leaf, dtype=dtype, device=device)
    # TCCL deadlock guard: chunk the leaf gather (<3MB/msg).
    chunk_cols2 = max(1, int(3 * 1024 * 1024 // (q * torch.finfo(dtype).bits // 8)))
    for c0 in range(0, n_leaf, chunk_cols2):
        c1 = min(c0 + chunk_cols2, n_leaf)
        piece = scatter_f[:, c0:c1].contiguous()
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
            if fine_transfer is None:
                raise RuntimeError(
                    "distributed shell stepping omitted the fine "
                    "interface transfer (reflux=True but no substep "
                    "observation ran)",
                )
            if observation_links is None or l1_post is None:
                raise RuntimeError(
                    "reflux bookkeeping lost l1_post or observation links",
                )
            if isinstance(l1_post, (tuple, list)):
                if len(l1_post) == 0:
                    raise ValueError("l1_post sequence must not be empty")
                coarse_transfer = observe_kinetic_interface_transfer(
                    l1_post[0], observation_links,
                )
                for _post in l1_post[1:]:
                    coarse_transfer = coarse_transfer + (
                        observe_kinetic_interface_transfer(
                            _post, observation_links,
                        )
                    )
            else:
                coarse_transfer = observe_kinetic_interface_transfer(
                    l1_post, observation_links,
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
                1.0, 0.0, 1.0,
                report.maximum_applied_correction_fraction,
            )
    # Broadcast the L1 patch so every rank's L1 copy stays in sync.
    # TCCL 3.1.0 deadlocks on broadcast messages > ~4MB (same limit as
    # all_gather).  l1_f is the full (Q, nz, ny, nx) global coarse field;
    # for large grids (e.g. R12 192x128x128 -> 340MB) a single broadcast
    # hangs.  Flatten and chunk along the element axis so every message
    # stays < 3MB, matching the all_gather chunking strategy above.
    _l1_flat = l1_f.contiguous().view(-1)
    _l1_n = _l1_flat.shape[0]
    _l1_chunk = max(1, int(3 * 1024 * 1024 // l1_f.element_size()))
    for _c0 in range(0, _l1_n, _l1_chunk):
        _c1 = min(_c0 + _l1_chunk, _l1_n)
        _piece = _l1_flat[_c0:_c1].clone()
        dist.broadcast(_piece, src=0)
        _l1_flat[_c0:_c1] = _piece
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
    # Chunk the restricted broadcast (<3MB/msg) for TCCL safety —
    # restricted is (Q, nc); for large shells nc can approach the 4MB
    # limit (R12: 2.84MB).  Chunk along the cell axis.
    _r_chunk = max(1, int(3 * 1024 * 1024 // (q * restricted.element_size())))
    for _rc0 in range(0, nc, _r_chunk):
        _rc1 = min(_rc0 + _r_chunk, nc)
        _r_piece = restricted[:, _rc0:_rc1].contiguous()
        dist.broadcast(_r_piece, src=0)
        restricted[:, _rc0:_rc1] = _r_piece
    dist.broadcast(cells, src=0)
    # Restore the per-rank leaf shard for the next root step.
    octree.f_leaf = full_f[:, local_indices].contiguous()
    if os.environ.get("DBG_NAN"):
        # Per-root-step global mass summary: all-reduce local sums so rank 0
        # prints the shell-wide totals.
        global _DBG_ROOT_STEP
        _DBG_ROOT_STEP += 1

        def _reduce(v):
            t = torch.tensor(float(v), dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return float(t.item())
        if dbg_mass_log:
            for s, stage, val in dbg_mass_log:
                dbg_mass_log_sum = _reduce(val)
                if rank == 0:
                    print(f"[dbg] RS{_DBG_ROOT_STEP} s{s} {stage} global_sum="
                          f"{dbg_mass_log_sum:.9g}", flush=True)
        shard_sum = float(octree.f_leaf.sum().item())
        shell_sum = _reduce(shard_sum)
        restr_sum = _reduce(float(restricted.sum().item()) if rank == 0 else 0.0)
        if rank == 0:
            print(f"[dbg] RS{_DBG_ROOT_STEP} shell_total={shell_sum:.9g} "
                  f"restricted_total={restr_sum:.9g}", flush=True)
    # Time-average over the root step's substeps (per-root-step MEM force).
    mem_avg = mem_accum / n_substeps
    return ledger, mem_avg, restricted, cells
