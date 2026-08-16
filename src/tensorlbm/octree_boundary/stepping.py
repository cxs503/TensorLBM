"""P2 time stepping for the octree boundary shell of the hybrid AMR architecture.

Implements the shell side of the P2 acceptance contract (design doc
``docs/octree-boundary-design.md`` §3.3 / §4):

* **substep scheduling** — the shell advances ``2 ** d_max`` substeps per L1
  root step (d_max=1 → 2 substeps, d_max=2 → 4 substeps; all leaves advance in
  lockstep in P2; the recursive depth-1/depth-2 split with internal
  ``TemporalInterp`` is the documented P3 item);
* **ghost fill** — boundary leaves interpolate their incoming directions from
  the L1 parent field with a two-time-point ``lerp`` (the coarse layer's two
  half-time states), trilinear space interpolation and
  ``rescale_nonequilibrium`` (per-leaf level scaling);
* **streaming** — pull semantics through the ``neighbor_table`` gather/scatter
  (no ``torch.roll``; leaves are not a regular lattice).  Cross-level links use
  the P2 approximation: a fine leaf pulls directly from its coarse donor leaf,
  a coarse leaf pointing into a refined region pulls the mean of the fan-out
  leaves; solid neighbours keep their pre-step populations (the BFL wall
  reconstruction is P3);
* **restriction + reflux** — volume-weighted leaf restriction back into the
  covered L1 cells plus a kinetic-flux reflux ledger observed on the shell
  interface links (each staircase link counted once, per-leaf volume scaling,
  accumulated over substeps) and applied face-local on the L1 side.

``step_octree_shell`` is the single entry point.  Its ``advance`` callback has
the same contract as :data:`tensorlbm.static_block_amr.Advance3D` (collision
operator agnostic); for the shell it must return the post-collision/pre-stream
state (streaming is performed by the stepper).  The function mutates ``l1_f``
(covered cells ← restriction, exterior cells ← reflux correction) and
``octree.f_leaf`` and returns a :class:`PopulationRefluxLedger` with the same
schema as ``StaticBlockAMR3D.step``.

Mass-conservation identity: with a mass-conserving collision, the combined
L1-exterior + shell volume-integrated mass changes by exactly
``-ledger.residual.sum()`` per root step, so a reflux residual < 1e-10 bounds
the joint-system mass drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import time as _time
import warnings

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.kinetic_flux_register import (
    KineticInterfaceLinks,
    KineticInterfaceTransfer,
    apply_face_local_reflux,
    observe_kinetic_interface_transfer,
)
from tensorlbm.octree_boundary.geometry import (
    DOMAIN_OUT,
    FANOUT,
    REMOTE,
    SHELL_OUTSIDE,
    SOLID,
    OctreeGrid,
    _axis_bits,
    morton_encode_batch,
)
from tensorlbm.octree_boundary.topology import run_topology_checks
from tensorlbm.refinement import BoxRegion
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    Advance3D,
    PopulationRefluxLedger,
    convective_refined_tau,
)


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------


def _tau_chain(tau_coarse: float, d_max: int) -> list[float]:
    """Convective relaxation chain ``[tau_c, tau_1, ..., tau_dmax]``."""
    taus = [float(tau_coarse)]
    for _ in range(d_max):
        taus.append(convective_refined_tau(taus[-1]))
    return taus


def _rescale_nonequilibrium_per_cell(
    f: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """``f + (scale-1) * neq`` per cell — the vectorised analogue of
    ``rescale_nonequilibrium`` with a per-cell scale (needed because shell
    leaves of depth 1 and 2 live at different ratios to the L1 level).

    ``f`` has shape ``(Q, n)`` and ``scale`` shape ``(n,)``.  The equilibrium
    decomposition uses the lattice's macroscopic/equilibrium pair and the
    same operation order as ``rescale_nonequilibrium``, so for a constant
    scale the D3Q19 result is bit-identical to it.
    """
    q, n = f.shape
    if q == 19:
        rho, ux, uy, uz = macroscopic3d(f.view(q, n, 1, 1))
        feq = equilibrium3d(rho, ux, uy, uz).view(q, n)
    elif q == 27:
        from tensorlbm.d3q27 import equilibrium27, macroscopic27

        rho, ux, uy, uz = macroscopic27(f.view(q, n, 1, 1))
        feq = equilibrium27(rho, ux, uy, uz).view(q, n)
    else:
        raise ValueError(f"unsupported lattice Q={q} (D3Q19 or D3Q27)")
    return feq + scale.to(f.dtype).unsqueeze(0) * (f - feq)


def _unpack_shell_advance(
    result: torch.Tensor | AMRAdvanceResult,
    expected_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(populations, post_collision)`` from an ``Advance3D`` result.

    For the shell the callback is expected to return the post-collision /
    pre-stream state (the stepper performs the neighbour-table streaming), so
    a collide-only callback may return ``AMRAdvanceResult(post, post)``.
    """
    if isinstance(result, AMRAdvanceResult):
        if result.populations.shape != expected_shape:
            raise ValueError("advance changed the shell population shape")
        if result.post_collision.shape != expected_shape:
            raise ValueError("advance changed the shell population shape")
        return result.populations, result.post_collision
    if isinstance(result, torch.Tensor):
        if result.shape != expected_shape:
            raise ValueError("advance changed the shell population shape")
        return result, result
    raise TypeError(
        "advance must return a tensor or AMRAdvanceResult, "
        f"got {type(result).__name__}",
    )


# ---------------------------------------------------------------------------
# Ghost fill plan (leaf boundary directions -> L1 trilinear donor stencil)
# ---------------------------------------------------------------------------


@dataclass
class ShellGhostPlan:
    """Precomputed donor stencil for the shell's virtual ghost cells.

    For every shell interface link ``(leaf i, direction d)`` (neighbour leaves
    the shell) the virtual ghost cell adjacent to ``i`` along ``d`` supplies
    the incoming population of direction ``opp[d]`` at leaf ``i``.  The sample
    position is the ghost-cell centre ``x_i + c_d * dx_i`` (an exact dyadic
    rational in the L1-cell world frame), converted to coarse-index
    continuous coordinates and trilinearly decomposed over the eight L1 donor
    cells — the same convention as ``StaticBlockAMR3D._build_ghost_sampling_plan``.
    """

    n_ghost: int
    leaf: torch.Tensor            # (n_ghost,) leaf enum
    direction: torch.Tensor       # (n_ghost,) filled direction at the leaf
    z0: torch.Tensor              # (n_ghost,) donor lower cells (z, y, x)
    y0: torch.Tensor
    x0: torch.Tensor
    z1: torch.Tensor              # (n_ghost,) donor upper cells
    y1: torch.Tensor
    x1: torch.Tensor
    wz: torch.Tensor              # (n_ghost,) trilinear weights
    wy: torch.Tensor
    wx: torch.Tensor
    volume: torch.Tensor          # (n_ghost,) fine-cell volume (2^-3l)
    slot: torch.Tensor            # (Q, n_leaf) int64, -1 = no ghost


def build_ghost_plan(
    octree: OctreeGrid, l1_shape: tuple[int, int, int],
    *,
    solid_fallback: bool = True,
) -> ShellGhostPlan:
    """Build the ghost fill donor stencil for an octree shell.

    Leaf centres are recomputed exactly in float64 from the Morton lattice
    (``(coords + 0.5) / 2^level``), so the sample positions are exact dyadic
    rationals and the trilinear weights are exact — matching the block AMR's
    ghost sampling bit for bit on the plane-degenerate topology.

    ``solid_fallback``: when a ghost sample position falls inside an L1-solid
    cell (the surface-straddling ring), replace the stencil with the leaf's
    own covered host cell (``True``, the production behaviour) or keep the
    trilinear sample over the frozen-solid cell (``False``, diagnostic mode
    for the R8 drag-deficit investigation).
    """
    nz, ny, nx = l1_shape
    device = octree.leaf_morton.device
    q = octree.Q
    n_leaf = octree.n_leaf
    links = octree.interface_links                      # (n_link, 2) (i, d)
    n_link = int(links.shape[0])
    slot = torch.full((q, n_leaf), -1, dtype=torch.int64, device=device)
    if n_link == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return ShellGhostPlan(
            0, empty, empty, empty, empty, empty, empty, empty, empty,
            empty, empty, empty, empty, slot,
        )
    leaf = links[:, 0]
    d_link = links[:, 1]
    opp = octree._opp.to(device)
    c_vec = octree._c_vec.to(device)
    direction = opp[d_link]                             # filled direction
    level_i = octree.leaf_level[leaf]
    dx = 2.0 ** (-level_i.to(torch.float64))
    if octree._l2_coords is not None and octree._l2_coords.numel() > 0:
        coords = torch.cat((octree._l1_coords, octree._l2_coords), dim=0)
    else:
        coords = octree._l1_coords
    centers64 = (
        coords.to(torch.float64) + 0.5
    ) / (2.0 ** octree.leaf_level.to(torch.float64))[:, None]   # (n, 3) x,y,z
    # The ghost cell sits at the SHELL_OUTSIDE neighbour position, i.e. the
    # cell adjacent to leaf i along the *link* direction d_link
    # (``x_i + c_vec[d_link] * dx``), and supplies the incoming population of
    # direction ``opp[d_link]`` that streams from there into the leaf.  The
    # position must use ``c_vec[d_link]`` (NOT ``c_vec[direction]`` — that is
    # the mirrored, shell-interior point whose sampling decouples the shell
    # boundary leaves from the exterior L1 flow; see the stream_gather /
    # bfl_apply_gather donor conventions).
    p_xyz = centers64[leaf] + c_vec[d_link].to(torch.float64) * dx[:, None]
    p = p_xyz[:, [2, 1, 0]]                             # (z, y, x) world
    # The restricted shell stores volume averages in the L1 covered cells.
    # Those parent values are sampled as cell-centred quantities at the
    # shell/L1 interface, hence the local world-to-index map retains the
    # half-cell offset even though the L1 wall mask itself is node-centred.
    # Keeping this interface convention avoids changing the established
    # subcycling transfer stencil while the parent wall geometry is aligned.
    continuous = p - 0.5                                # coarse-index coords
    lo = torch.floor(continuous).to(torch.int64)
    hi = lo + 1
    bounds = torch.tensor(
        [nz, ny, nx], dtype=torch.int64, device=device,
    )
    lo = lo.clamp(torch.zeros_like(lo), bounds - 2)
    hi = hi.clamp(torch.ones_like(hi), bounds - 1)
    w = (continuous - lo.to(continuous.dtype)).clamp(0.0, 1.0)
    if octree._solid is not None and bool(
        (octree._solid[p[:, 0].floor().to(torch.int64).clamp(0, nz - 1),
                       p[:, 1].floor().to(torch.int64).clamp(0, ny - 1),
                       p[:, 2].floor().to(torch.int64).clamp(0, nx - 1)]).any()
    ) and solid_fallback:
        # Solid-host fallback: a ghost position can be fluid at leaf
        # resolution yet fall inside an L1-solid cell (the surface-straddling
        # ring — L1 cells whose centre is inside the sphere but whose outer
        # part is outside).  The L1 field there is the frozen-collision core,
        # not a fluid state; sample the leaf's own (covered) host cell
        # instead, which holds the restricted leaf state — the same local
        # band fluid the old mirror sampling happened to hit for these links.
        cell_p = torch.stack((
            p[:, 0].floor().to(torch.int64).clamp(0, nz - 1),
            p[:, 1].floor().to(torch.int64).clamp(0, ny - 1),
            p[:, 2].floor().to(torch.int64).clamp(0, nx - 1),
        ), dim=1)
        solid_host = octree._solid[
            cell_p[:, 0], cell_p[:, 1], cell_p[:, 2],
        ]
        if bool(solid_host.any()):
            lo = lo.clone()
            hi = hi.clone()
            w = w.clone()
            host = octree.leaf_host_cell[leaf]          # (n, 3) (z, y, x)
            lo[solid_host] = host[solid_host]
            hi[solid_host] = host[solid_host]
            w[solid_host] = 0.0
    volume = 2.0 ** (-3.0 * level_i.to(torch.float64))
    slot[direction, leaf] = torch.arange(
        n_link, dtype=torch.int64, device=device,
    )
    return ShellGhostPlan(
        n_ghost=n_link,
        leaf=leaf,
        direction=direction,
        z0=lo[:, 0], y0=lo[:, 1], x0=lo[:, 2],
        z1=hi[:, 0], y1=hi[:, 1], x1=hi[:, 2],
        wz=w[:, 0], wy=w[:, 1], wx=w[:, 2],
        volume=volume,
        slot=slot,
    )


def fill_ghost(
    octree: OctreeGrid,
    plan: ShellGhostPlan,
    parent_t: torch.Tensor,
    taus: list[float],
) -> torch.Tensor:
    """Time-lerped, trilinearly interpolated, rescaled L1 ghost populations.

    ``parent_t`` is the L1 state at the current substep's time point (the
    caller forms ``lerp(l1_old, l1_f, alpha)``).  Returns ``(Q, n_ghost)``.
    """
    return _fill_ghost_impl(octree.leaf_level, plan, parent_t, taus)


def _fill_ghost_impl(
    leaf_level: torch.Tensor,
    plan: ShellGhostPlan,
    parent_t: torch.Tensor,
    taus: list[float],
) -> torch.Tensor:
    """Shared ghost-fill body; ``leaf_level`` is the (global or shard-local)
    per-leaf level array indexed by ``plan.leaf``."""
    q = parent_t.shape[0]
    if plan.n_ghost == 0:
        return torch.empty(
            (q, 0), dtype=parent_t.dtype, device=parent_t.device,
        )
    wdtype = parent_t.dtype
    wx = plan.wx.unsqueeze(0).to(dtype=wdtype)
    wy = plan.wy.unsqueeze(0).to(dtype=wdtype)
    wz = plan.wz.unsqueeze(0).to(dtype=wdtype)
    v00 = torch.lerp(
        parent_t[:, plan.z0, plan.y0, plan.x0],
        parent_t[:, plan.z0, plan.y0, plan.x1], wx,
    )
    v01 = torch.lerp(
        parent_t[:, plan.z0, plan.y1, plan.x0],
        parent_t[:, plan.z0, plan.y1, plan.x1], wx,
    )
    v10 = torch.lerp(
        parent_t[:, plan.z1, plan.y0, plan.x0],
        parent_t[:, plan.z1, plan.y0, plan.x1], wx,
    )
    v11 = torch.lerp(
        parent_t[:, plan.z1, plan.y1, plan.x0],
        parent_t[:, plan.z1, plan.y1, plan.x1], wx,
    )
    sampled = torch.lerp(
        torch.lerp(v00, v01, wy), torch.lerp(v10, v11, wy), wz,
    )
    lev = leaf_level[plan.leaf]
    tau_f = torch.tensor(
        taus, dtype=torch.float64, device=sampled.device,
    )[lev.to(device=sampled.device)]
    scale = tau_f / (
        (2.0 ** lev.to(torch.float64)) * taus[0]
    )
    # The ghost cell is a virtual leaf-lattice neighbour, and stream_gather
    # pulls *post-collision* populations from real leaf neighbours; the ghost
    # must therefore supply its post-collision state too.  The rescaled
    # sampled value is the ghost's pre-collision state, so relax its neq with
    # the leaf's own relaxation time before injection.  Without this the
    # interface over-injects the coarse neq by tau_f/(tau_f - 1) (~3.5x at
    # tau_f = 0.7) and a spurious stress layer forms at the AMR interface
    # (driven-Couette ux error ~1.2e-3 — see scripts/octree_analytic_check.py).
    # Pure-equilibrium fields (neq = 0) are unaffected, so the P2 plane-shell
    # equivalence to StaticBlockAMR3D is preserved bit for bit.
    scale = scale * (1.0 - 1.0 / tau_f)
    return _rescale_nonequilibrium_per_cell(sampled, scale)


# ---------------------------------------------------------------------------
# Streaming through the neighbour table
# ---------------------------------------------------------------------------


def ensure_fanout_tables(octree) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrected live fanout tables, built once and cached on ``octree``.

    The registry ``interface_fanout`` is keyed ``(leaf, direction q)`` with
    ``neighbor_table[q, leaf] == FANOUT`` — *q is the neighbour-table
    direction*.  The topology pre-cache ``fanout_pos``/``fanout_pad`` stores
    rows in the same convention but includes dead keys (registered during the
    fine-leaf pass whose reverse neighbour never became a FANOUT entry), so it
    is filtered here to the live rows.

    Returns ``(rowidx, pad)``:

    * ``rowidx``: ``(Q, n_leaf)`` int64 — row index into ``pad`` for every
      ``(q, i)`` with ``neighbor_table[q, i] == FANOUT``, else -1.
    * ``pad``: ``(n_live, max_len)`` int64 — member leaf enums (global column
      space), padded with -1.

    **Pull direction**: streaming / BFL read the member populations along
    ``opp[q]`` (``src_all[d, i] = neighbor_table[opp[d], i]``), so callers
    must index the member values with ``opp[q]`` — never ``q`` directly (the
    original batched prototype indexed with ``q``, a direction-flip bug that
    only manifests at d_max=2 where fanout groups exist).
    """
    rowidx = getattr(octree, "_fanout_rowidx", None)
    if rowidx is not None:
        return rowidx, octree._fanout_pad_live
    nt = octree.neighbor_table
    fo_pos = octree.fanout_pos
    fo_pad = octree.fanout_pad
    # Do all indexing on the CPU: SDAA's advanced-index gather faults with
    # SDAA_ERROR_MISALIGNED_ADDRESS for large dynamic index tensors (seen at
    # d_max=2 with ~30k fanout rows).  The tables are static topology, built
    # once, so the CPU cost is negligible.
    dev = nt.device
    nt_c = nt.detach().cpu()
    fo_pos_c = fo_pos.detach().cpu()
    fo_pad_c = fo_pad.detach().cpu()
    rowidx = torch.full((octree.Q, nt.shape[1]), -1, dtype=torch.int64)
    if fo_pos_c.shape[0] > 0 and fo_pad_c.shape[0] > 0:
        # Defensive bounds guard: an out-of-range (q, leaf) row would make
        # the advanced index ``nt[fo_pos[:, 0], fo_pos[:, 1]]`` (and the
        # scatter into ``rowidx`` below) run out of bounds — on SDAA that
        # surfaces as SDAA_ERROR_MISALIGNED_ADDRESS.  Such rows cannot be
        # FANOUT links of this table; drop them and warn loudly (they
        # indicate a topology inconsistency worth investigating).
        n_leaf = nt.shape[1]
        qv, lv = fo_pos_c[:, 0], fo_pos_c[:, 1]
        ok = (qv >= 0) & (qv < octree.Q) & (lv >= 0) & (lv < n_leaf)
        if not bool(ok.all()):
            warnings.warn(
                "ensure_fanout_tables: dropping "
                f"{int((~ok).sum())} fanout_pos row(s) out of range "
                f"(Q={octree.Q}, n_leaf={n_leaf})",
                RuntimeWarning,
            )
            fo_pos_c = fo_pos_c[ok]
            fo_pad_c = fo_pad_c[ok]
        live = nt_c[fo_pos_c[:, 0], fo_pos_c[:, 1]] == FANOUT
        pos = fo_pos_c[live]
        pad = fo_pad_c[live]
        rowidx[pos[:, 0], pos[:, 1]] = torch.arange(
            pos.shape[0], dtype=torch.int64,
        )
    else:
        # Fallback: build from the registry directly (octrees built before
        # the pre-cache existed).  Live keys are exactly the FANOUT entries.
        rows = torch.nonzero(nt_c == FANOUT, as_tuple=False)   # (n_live, 2)
        if rows.shape[0]:
            members = [
                octree.interface_fanout[(int(r[1]), int(r[0]))]
                for r in rows.tolist()
            ]
            max_len = max(len(m) for m in members)
            pad = torch.full((rows.shape[0], max_len), -1, dtype=torch.int64)
            for r, m in enumerate(members):
                pad[r, :len(m)] = torch.tensor(m, dtype=torch.int64)
            rowidx[rows[:, 0], rows[:, 1]] = torch.arange(
                rows.shape[0], dtype=torch.int64,
            )
        else:
            pad = torch.empty((0, 0), dtype=torch.int64)
    octree._fanout_rowidx = rowidx
    octree._fanout_pad_live = pad
    return rowidx, pad


def _fanout_segment_mean(
    values: torch.Tensor,
    d_idx: torch.Tensor,
    pad: torch.Tensor,
) -> torch.Tensor:
    """Masked mean over the padded fanout member table.

    ``values[d_idx[:, None], pad.clamp(min=0)]`` gathered in the population
    direction ``d_idx``, averaged over the valid (``pad >= 0``) members.
    Returns ``(n_rows,)`` in ``values.dtype``.
    """
    valid = pad >= 0
    vals = values[d_idx.unsqueeze(1), pad.clamp(min=0)]
    return (vals * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def stream_gather(
    octree: OctreeGrid,
    plan: ShellGhostPlan,
    populations: torch.Tensor,
    f_old: torch.Tensor,
    ghost_vals: torch.Tensor,
) -> torch.Tensor:
    """Pull-stream the shell leaves through ``neighbor_table``.

    ``populations`` is the post-collision / pre-stream state, ``f_old`` the
    pre-substep state (kept for the defensive empty fan-out fallback),
    ``ghost_vals`` the ``(Q, n_ghost)`` filled boundary values.  Sentinel
    handling:

    * ``>= 0`` — leaf enum (same level or cross-level donor, P2 direct pull);
    * ``SHELL_OUTSIDE`` — virtual ghost cell, take the filled value;
    * ``FANOUT`` — mean over the registered fan-out fine leaves;
    * ``SOLID`` — bounce-back fallback ``post[opp[d], i]`` (the P2 pre-BFL
      wall).  This keeps the shell's mass bookkeeping exact: every
      post-collision population is either pulled by its downstream leaf,
      reflected by the wall, or leaves through an interface link, so
      ``dM_shell = -outgoing + incoming`` holds to machine precision;
    * ``DOMAIN_OUT`` — topology violation, raise.
    """
    q, n = populations.shape
    out = torch.empty_like(populations)
    opp = octree._opp.to(populations.device)
    nt = octree.neighbor_table
    # 批量版: 所有方向一次 torch.gather (SDAA 上 27 方向循环 = 135 次小 kernel 极慢)
    src_all = nt[opp]  # (q, n)
    valid_all = src_all >= 0
    idx_safe = src_all.clamp(min=0)
    gathered_all = torch.gather(populations, 1, idx_safe)  # (q, n)
    out = torch.where(valid_all, gathered_all, out)
    # ghost 分支 (SHELL_OUTSIDE): 批量 (nonzero 一次收集所有 ghost 位置)
    ghost_all = src_all == SHELL_OUTSIDE
    if bool(ghost_all.any()):
        rows = torch.nonzero(ghost_all, as_tuple=False)  # (n_ghost, 2): (d, i)
        d_idx = rows[:, 0]
        i_idx = rows[:, 1]
        slots = plan.slot[d_idx, i_idx]  # 高级索引批量取 slot
        vals = ghost_vals[d_idx, slots]
        out[d_idx, i_idx] = vals
    # solid 分支 (SOLID): 批量
    solid_all = src_all == SOLID
    if bool(solid_all.any()):
        rows_s = torch.nonzero(solid_all, as_tuple=False)  # (n_solid, 2)
        d_s = rows_s[:, 0]
        i_s = rows_s[:, 1]
        out[d_s, i_s] = populations[opp[d_s], i_s]
    # fanout 分支: 纯张量批量 (修正后的预缓存 — 行方向 q 是 neighbour-table
    # 方向, 拉取/写出方向是 opp[q]; 旧版直接用 fanout_pos[:,0]=q 索引
    # populations 是方向翻转 bug, 且 fo_mask 过滤后 90% 的 fanout 单元仍走
    # Python 回退循环)
    fanout_all = src_all == FANOUT
    if bool(fanout_all.any()):
        rows_f = torch.nonzero(fanout_all, as_tuple=False)   # (n_fan, 2) (d, i)
        d_f, i_f = rows_f[:, 0], rows_f[:, 1]
        rowidx, pad_live = ensure_fanout_tables(octree)
        rowidx = rowidx.to(populations.device)
        pad_live = pad_live.to(populations.device)
        ridx = rowidx[opp[d_f], i_f]                         # live 行号 / -1
        has = ridx >= 0
        if bool(has.any()):
            means = _fanout_segment_mean(
                populations, d_f[has], pad_live[ridx[has]],
            )
            out[d_f[has], i_f[has]] = means
        fb = ~has
        if bool(fb.any()):
            # 防御: FANOUT 但无注册组 -> 保留旧值 (与旧循环一致)
            out[d_f[fb], i_f[fb]] = f_old[d_f[fb], i_f[fb]]
    # domain 分支: 检查
    domain_all = src_all == DOMAIN_OUT
    if bool(domain_all.any()):
        raise RuntimeError(
            "DOMAIN_OUT neighbour in shell streaming — the shell must be "
            "fully embedded in the L1 block",
        )
    return out


# ---------------------------------------------------------------------------
# Shell-side kinetic interface transfer observation
# ---------------------------------------------------------------------------


def observe_shell_interface_transfer(
    octree: OctreeGrid,
    plan: ShellGhostPlan,
    post_collision: torch.Tensor,
    ghost_vals: torch.Tensor,
) -> KineticInterfaceTransfer:
    """Observe the shell-side pre-stream flux on the interface links.

    Outgoing: ``sum(post[d, i] * vol_i)`` over interface links ``(i, d)``.
    Incoming: ``sum(ghost[d, i] * vol_i)`` over ghost targets ``(i, d)`` (the
    virtual ghost cells' incoming populations).  Each staircase link is
    counted exactly once, including edge/corner links.
    """
    q = octree.Q
    dtype = post_collision.dtype
    device = post_collision.device
    outgoing = torch.zeros(q, dtype=dtype, device=device)
    incoming = torch.zeros_like(outgoing)
    links = octree.interface_links
    vol = octree.leaf_volume()
    for d in range(1, q):
        sel = links[:, 1] == d
        if bool(sel.any()):
            li = links[sel, 0]
            outgoing[d] = (
                post_collision[d, li] * vol[li].to(dtype)
            ).sum()
        gsel = plan.direction == d
        if bool(gsel.any()):
            incoming[d] = (
                ghost_vals[d, gsel] * plan.volume[gsel].to(dtype)
            ).sum()
    return KineticInterfaceTransfer(outgoing, incoming)


# ---------------------------------------------------------------------------
# Restriction (fine leaves -> covered L1 cells)
# ---------------------------------------------------------------------------


def _segmented_sum(segments: torch.Tensor, values: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Deterministic segmented reduction: sum ``values[..., i]`` grouped by ``segments[i]``.

    ``segments``: ``(n,)`` int64 with values in ``[0, n_segments)``.
    ``values``: ``(..., n)`` tensor.
    Returns ``(..., n_segments)``.

    Unlike ``scatter_add_`` (whose CUDA kernel uses a run-dependent atomic
    accumulation order and therefore drifts by ~1 ulp between otherwise
    identical runs), this sums each group in a fixed, device-independent
    order (stable-sorted by segment id, i.e. ascending leaf enum within each
    group) via a prefix sum.  This makes the shell restriction reproducible
    bit for bit, which is what the sharded-vs-unsharded equivalence contract
    requires (the restriction is the only state-affecting reduction that was
    still order-unspecified).
    """
    n = segments.numel()
    device = segments.device
    out_shape = (*values.shape[:-1], n_segments)
    if n == 0:
        return torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    order = torch.argsort(segments, stable=True)
    seg_sorted = segments[order]
    vals_sorted = values[..., order]
    csum = torch.cumsum(vals_sorted, dim=-1)
    # start position of every present group (first occurrence in sorted order)
    start_mask = torch.zeros(n, dtype=torch.bool, device=device)
    start_mask[0] = True
    if n > 1:
        start_mask[1:] = seg_sorted[1:] != seg_sorted[:-1]
    start_pos = torch.nonzero(start_mask, as_tuple=False).squeeze(1)  # (k,)
    seg_ids = seg_sorted[start_pos]                                   # (k,)
    # end position of group starting at start_pos[k] is start_pos[k+1]-1
    end_pos = torch.cat(
        (
            start_pos[1:] - 1,
            torch.tensor([n - 1], dtype=torch.int64, device=device),
        ),
    )
    csum_end = csum[..., end_pos]
    prefix = torch.zeros_like(csum_end)
    prefix[..., 1:] = csum_end[..., :-1]
    sums = csum_end - prefix
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    out[..., seg_ids] = sums
    return out


def restrict_shell_to_block(
    octree: OctreeGrid,
    f_leaf: torch.Tensor,
    taus: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict the shell leaves into their host L1 cells.

    For each covered L1 cell the populations are the volume-weighted mean over
    its leaves (exact 2x2x2 mean when every leaf is depth 1), followed by the
    fine->coarse non-equilibrium rescale with the cell's finest leaf level.
    Returns ``(restricted (Q, n_cells), cells (n_cells, 3) (z, y, x))``.
    """
    q, n = f_leaf.shape
    device = f_leaf.device
    host = octree.leaf_host_cell
    vol = octree.leaf_volume().to(device)
    cells, cell_id = torch.unique(host, dim=0, return_inverse=True)
    n_cells = cells.shape[0]
    volume_sum = _segmented_sum(cell_id, vol, n_cells)
    weight = (vol / volume_sum[cell_id]).to(f_leaf.dtype)
    f_mean = _segmented_sum(
        cell_id, weight.unsqueeze(0) * f_leaf, n_cells,
    )
    level_max = torch.zeros(
        n_cells, dtype=torch.float64, device=device,
    ).scatter_reduce_(
        0, cell_id, octree.leaf_level.to(torch.float64),
        reduce="amax", include_self=False,
    )
    tau_f = torch.tensor(
        taus, dtype=torch.float64, device=device,
    )[level_max.to(torch.int64)]
    scale = taus[0] / ((2.0 ** (-level_max)) * tau_f)
    return _rescale_nonequilibrium_per_cell(f_mean, scale), cells


# ---------------------------------------------------------------------------
# L1-side coarse links (staircase interface, solid-excluded)
# ---------------------------------------------------------------------------


def build_shell_coarse_links(
    covered: torch.Tensor,
    solid: torch.Tensor | None,
    *,
    q: int,
) -> KineticInterfaceLinks:
    """Coarse-side crossing links of the shell-covered L1 region.

    Identical to ``build_kinetic_interface_links`` for a solid-free covered
    mask.  With a ``solid`` mask the links whose destination (outgoing) or
    origin (incoming) is a solid L1 cell are dropped, which yields the
    **correction stencil** for ``apply_face_local_reflux``: body-wall links
    with a bounce-back fine-side counterpart carry no ghost-filled transfer
    of their own and must never be corrected (the BFL wall is P3).

    The mass-conservation observation, by contrast, must count the *full*
    covered boundary: the L1 advance streams the covered region as ordinary
    fluid, so kinetic mass crosses the covered<->solid links too, and those
    links' fine-side counterparts (ghost-filled inner-wall interface links)
    are already in the fine transfer.  ``step_octree_shell`` therefore
    observes the coarse transfer on ``build_shell_coarse_links(covered,
    None, ...)`` while correcting on the solid-excluded stencil — the pair
    closes the joint mass identity ``dM = -residual.sum()`` exactly.
    """
    if covered.ndim != 3 or covered.dtype is not torch.bool:
        raise ValueError("covered must be a 3-D boolean tensor")
    if solid is None:
        solid = torch.zeros_like(covered)
    if solid.shape != covered.shape or solid.dtype is not torch.bool:
        raise ValueError("solid must be a boolean tensor with the covered shape")
    if q == 19:
        from tensorlbm.d3q19 import C
    elif q == 27:
        from tensorlbm.d3q27 import C
    else:
        raise ValueError(f"unsupported lattice Q={q} (D3Q19 or D3Q27)")
    c = C.to(covered.device)
    outgoing = torch.zeros(
        (q, *covered.shape), dtype=torch.bool, device=covered.device,
    )
    incoming = torch.zeros_like(outgoing)
    for d in range(1, q):
        cx, cy, cz = (int(v) for v in c[d].tolist())
        dest_covered = torch.roll(
            covered, shifts=(-cz, -cy, -cx), dims=(0, 1, 2),
        )
        dest_solid = torch.roll(
            solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2),
        )
        outgoing[d] = covered & ~dest_covered & ~dest_solid
        incoming[d] = ~covered & ~solid & dest_covered
    return KineticInterfaceLinks(covered, outgoing, incoming)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def step_octree_shell(
    octree: OctreeGrid,
    advance: Advance3D,
    l1_old: torch.Tensor,
    l1_f: torch.Tensor,
    *,
    tau_coarse: float,
    tau_fine: float | None = None,
    tau_shell_override: float | None = None,
    l1_post: torch.Tensor | list[torch.Tensor] | None = None,
    shell_level: int = 1,
    reflux: bool = True,
    maximum_reflux_correction_fraction: float = 0.2,
    correction_stencil: str = "exterior_cells",
    ghost_plan: ShellGhostPlan | None = None,
    coarse_links: KineticInterfaceLinks | None = None,
    solid: torch.Tensor | None = None,
    bfl_fn=None,
    force_ledger=None,
) -> PopulationRefluxLedger:
    """Advance the octree shell by one L1 root step.

    Args:
        octree: the :class:`OctreeGrid` shell (``f_leaf`` is mutated in place).
        advance: ``Advance3D``-compatible callback.  For the shell it must
            return the post-collision/pre-stream state (``AMRAdvanceResult``
            with both fields equal for a collide-only callback); streaming is
            performed here through the neighbour table.
        l1_old: L1 populations at the root-step start (time-lerp anchor).
        l1_f: L1 populations after the block's own advance; mutated in place —
            covered cells receive the restriction, exterior cells the reflux
            correction.
        tau_coarse: L1 relaxation time (convective chain is derived from it).
        tau_fine: shell relaxation time; must equal
            ``convective_refined_tau(tau_coarse)`` when given.
        tau_shell_override: diagnostic knob — force the shell's relaxation
            time to this value (instead of the convective chain's fine value)
            while keeping the chain's coarse value for the ghost-fill and
            restriction rescales.  ``None`` = production convective chain.
        l1_post: L1 post-collision state (required when ``reflux=True``) used
            for the coarse-side transfer observation.
        shell_level: hierarchy level index passed to ``advance`` for the shell.
        reflux: enable the kinetic-flux reflux ledger.
        maximum_reflux_correction_fraction, correction_stencil: passed
            through to ``apply_face_local_reflux``.
        ghost_plan: cached :class:`ShellGhostPlan` (built on demand).
        coarse_links: cached coarse-side links (built on demand from the
            octree's shell mask and solid mask).
        solid: L1 solid mask override (defaults to ``octree._solid``).
        bfl_fn: optional ``bfl_fn(octree, out, post_collision, ghost_plan,
            ghost_vals, *, substep) -> (f_out, force | None)`` applied
            after streaming in every substep (P3 gather-based Bouzidi BFL,
            see :mod:`tensorlbm.octree_boundary.bfl`).  ``force`` is the
            per-substep MEM impulse in leaf lattice units (already per-leaf
            weighted by the caller).
        force_ledger: optional object with ``add_substep_force(force)``
            (e.g. :class:`tensorlbm.octree_boundary.force.ShellForceLedger`)
            that accumulates the substep forces.  ``None`` disables force
            bookkeeping even when ``bfl_fn`` returns forces.

    Returns:
        A :class:`PopulationRefluxLedger` with the same schema as
        ``StaticBlockAMR3D.step``.  The combined L1-exterior + shell mass
        drift over the step equals ``-ledger.residual.sum()``.
    """
    if not isinstance(octree, OctreeGrid):
        raise TypeError("octree must be an OctreeGrid")
    if l1_old.shape != l1_f.shape:
        raise ValueError("l1_old and l1_f must share shape")
    l1_shape = tuple(l1_f.shape[1:])
    if l1_shape != octree.meta["shape"]:
        raise ValueError("L1 populations do not match the octree's block shape")
    if correction_stencil not in ("exterior_cells", "crossing_links"):
        raise ValueError(
            "correction_stencil must be exterior_cells or crossing_links",
        )
    n_substeps = 1 << octree.d_max
    taus = _tau_chain(tau_coarse, octree.d_max)
    if tau_fine is not None and abs(tau_fine - taus[1]) > 1.0e-12:
        raise ValueError("dynamic tau_fine must preserve convective scaling")
    if tau_shell_override is not None:
        taus = list(taus)
        taus[1] = float(tau_shell_override)
    tau_shell = taus[1]
    if reflux and l1_post is None:
        raise TypeError(
            "reflux-enabled shell stepping requires l1_post (the L1 "
            "post-collision/pre-stream state)",
        )
    if ghost_plan is None:
        ghost_plan = build_ghost_plan(octree, l1_shape)
    if reflux and coarse_links is None:
        covered = octree._shell_mask
        if covered is None:
            raise RuntimeError(
                "octree carries no shell mask; pass coarse_links explicitly",
            )
        solid_mask = octree._solid if solid is None else solid
        coarse_links = build_shell_coarse_links(
            covered, solid_mask, q=octree.Q,
        )
    if reflux:
        if coarse_links is None:
            raise RuntimeError(
                "reflux bookkeeping lost the coarse interface links",
            )
        # The L1 advance streams the covered ("frozen") region as ordinary
        # fluid, so kinetic mass crosses *every* link of the covered-region
        # boundary — including links into solid L1 cells, whose fine-side
        # counterparts are the ghost-filled inner-wall interface links.  The
        # coarse transfer must therefore be observed on the full boundary,
        # while the solid-excluded ``coarse_links`` remain the correction
        # stencil so the reflux never writes into solid cells.  With this
        # pairing the joint mass identity ``dM = -residual.sum()`` closes
        # exactly (the fine side and the coarse side then count the same
        # flux set).
        observation_links = build_shell_coarse_links(
            coarse_links.inside, None, q=octree.Q,
        )
    else:
        observation_links = None

    fine_transfer: KineticInterfaceTransfer | None = None
    for s in range(n_substeps):
        alpha = s / n_substeps
        parent_t = torch.lerp(l1_old, l1_f, alpha)
        ghost_vals = fill_ghost(octree, ghost_plan, parent_t, taus)
        populations, post_collision = _unpack_shell_advance(
            advance(octree.f_leaf, tau_shell, shell_level, s),
            octree.f_leaf.shape,
        )
        out = stream_gather(
            octree, ghost_plan, populations, octree.f_leaf, ghost_vals,
        )
        if bfl_fn is not None:
            result = bfl_fn(
                octree, out, post_collision, ghost_plan, ghost_vals,
                substep=s,
            )
            out, substep_force = result
            if force_ledger is not None and substep_force is not None:
                force_ledger.add_substep_force(substep_force)
        if reflux:
            observed = observe_shell_interface_transfer(
                octree, ghost_plan, post_collision, ghost_vals,
            )
            fine_transfer = (
                observed if fine_transfer is None else fine_transfer + observed
            )
        octree.f_leaf = out

    restricted, cells = restrict_shell_to_block(octree, octree.f_leaf, taus)
    old_patch = l1_f[:, cells[:, 0], cells[:, 1], cells[:, 2]].clone()
    l1_f[:, cells[:, 0], cells[:, 1], cells[:, 2]] = restricted
    replacement_mismatch = old_patch.sum(dim=1) - restricted.sum(dim=1)
    if not reflux:
        return PopulationRefluxLedger(
            replacement_mismatch,
            torch.zeros_like(replacement_mismatch),
            0,
            replacement_mismatch,
            0,
            replacement_mismatch,
        )
    if fine_transfer is None:
        raise RuntimeError("shell stepping omitted the fine interface transfer")
    if l1_post is None or coarse_links is None or observation_links is None:
        raise RuntimeError(
            "reflux bookkeeping lost l1_post, coarse_links or observation_links",
        )
    if isinstance(l1_post, (tuple, list)):
        # The L1 block of the hybrid hierarchy advances `ratio` substeps per
        # root step; the coarse-side transfer must count every one of them to
        # pair with the shell's `2^d` accumulated fine substeps.
        if len(l1_post) == 0:
            raise ValueError("l1_post sequence must not be empty")
        coarse_transfer = observe_kinetic_interface_transfer(
            l1_post[0], observation_links,
        )
        for post in l1_post[1:]:
            coarse_transfer = coarse_transfer + observe_kinetic_interface_transfer(
                post, observation_links,
            )
    else:
        coarse_transfer = observe_kinetic_interface_transfer(
            l1_post, observation_links,
        )
    l1_f, report = apply_face_local_reflux(
        l1_f,
        coarse_links,
        coarse_transfer,
        fine_transfer,
        maximum_correction_fraction=maximum_reflux_correction_fraction,
        correction_stencil=correction_stencil,
    )
    octree.meta["last_fine_transfer"] = fine_transfer
    return PopulationRefluxLedger(
        report.requested_inventory_correction,
        report.applied_inventory_correction,
        report.corrected_links,
        report.residual,
        report.limited_directions,
        report.raw_kinetic_mismatch,
        0.0,
        1.0,
        0.0,
        1.0,
        report.maximum_applied_correction_fraction,
    )


# ---------------------------------------------------------------------------
# Multi-device shell stepping (fine_devices sharding)
# ---------------------------------------------------------------------------


_DEBUG_STEP = {"n": 0}


def _debug_nan(tag: str, tensors: dict, *, substep: int) -> None:
    """Env-gated (OCTREE_DEBUG_NAN=1) finiteness report per pipeline stage."""
    import os

    if not os.environ.get("OCTREE_DEBUG_NAN"):
        return
    step = _DEBUG_STEP["n"]
    for name, t in tensors.items():
        if t is None or t.numel() == 0:
            continue
        if not bool(torch.isfinite(t).all()):
            bad = ~torch.isfinite(t)
            nbad = int(bad.sum().item())
            print(
                f"[dbg] step={step} substep={substep} stage={tag} {name} "
                f"non-finite: {nbad}/{t.numel()} "
                f"min={float(t.min().item()):.4e} max={float(t.max().item()):.4e}",
                flush=True,
            )
        else:
            print(
                f"[dbg] step={step} substep={substep} stage={tag} {name} finite "
                f"min={float(t.min().item()):.4e} max={float(t.max().item()):.4e}",
                flush=True,
            )


def _stream_gather_shard(shard, populations, f_old, ghost_vals) -> torch.Tensor:
    """Pull-stream one leaf shard through its REMOTE-rewritten neighbour table.

    Semantics identical to :func:`stream_gather` per output element: in-shard
    sources gather from ``populations``, cross-shard sources resolve through
    the pre-filled ``shard.remote_buf``, ghost cells take ``ghost_vals``,
    solid neighbours bounce back, fan-out groups average their (globally
    ordered) group buffer — the gather itself is element-wise equivalent to the
    unsharded path; the collision callback may still introduce roundoff-level
    differences when it reduces over different shard sizes.
    """
    q, n = populations.shape
    out = torch.empty_like(populations)
    opp = shard.opp
    nt = shard.neighbor_table
    remote_buf = shard.remote_buf
    remote_pos = shard.remote_pos
    fan_off = shard.fan_off
    fan_len = shard.fan_len
    ghost_slot = shard.ghost_slot
    for d in range(q):
        src = nt[opp[d]]
        valid = src >= 0
        if bool(valid.any()):
            out[d, valid] = populations[d, src[valid]]
        remote_mask = src == REMOTE
        if bool(remote_mask.any()):
            slots = remote_pos[opp[d], remote_mask]
            out[d, remote_mask] = remote_buf[slots]
        ghost_mask = src == SHELL_OUTSIDE
        if bool(ghost_mask.any()):
            slots = ghost_slot[d, ghost_mask]
            out[d, ghost_mask] = ghost_vals[d, slots]
        solid_mask = src == SOLID
        if bool(solid_mask.any()):
            out[d, solid_mask] = populations[opp[d], solid_mask]
        fanout_mask = src == FANOUT
        if bool(fanout_mask.any()):
            for i in torch.nonzero(
                fanout_mask, as_tuple=False,
            ).squeeze(1).tolist():
                off = int(fan_off[opp[d], i].item())
                ln = int(fan_len[opp[d], i].item())
                if ln > 0:
                    out[d, i] = remote_buf[off:off + ln].mean()
                else:
                    out[d, i] = f_old[d, i]
        domain_mask = src == DOMAIN_OUT
        if bool(domain_mask.any()):
            raise RuntimeError(
                "DOMAIN_OUT neighbour in shell streaming — the shell must be "
                "fully embedded in the L1 block",
            )
    return out


def _assemble_shell_transfer(
    shards,
    *,
    q: int,
    dtype: torch.dtype,
    root_device: torch.device,
) -> KineticInterfaceTransfer:
    """Combine the per-shard interface-transfer observations on the root device.

    Every link's value is scattered into a per-direction buffer at its global
    row rank (the position it holds in the unsharded link/plan row order), so
    the per-direction reductions reproduce the unsharded sums bit for bit.
    """
    outgoing = torch.zeros(q, dtype=dtype, device=root_device)
    incoming = torch.zeros(q, dtype=dtype, device=root_device)
    for d in range(1, q):
        n_out = sum(
            int((shard.link_dir == d).sum().item()) for shard in shards
        )
        if n_out:
            buf = torch.zeros(n_out, dtype=dtype, device=root_device)
            for shard in shards:
                sel = shard.link_dir == d
                if not bool(sel.any()):
                    continue
                # link_dir/link_leaf live on different devices (root vs shard);
                # move the mask to the shard device before indexing
                li = shard.link_leaf[sel.to(shard.device)]
                vals = (
                    shard.post_collision[d, li]
                    * shard.leaf_volume[li].to(dtype)
                )
                buf[shard.out_rank[sel]] = vals.to(root_device)
            outgoing[d] = buf.sum()
        n_in = sum(
            int((shard.ghost_plan.direction == d).sum().item())
            for shard in shards
        )
        if n_in:
            buf = torch.zeros(n_in, dtype=dtype, device=root_device)
            for shard in shards:
                gsel = shard.ghost_plan.direction == d
                if not bool(gsel.any()):
                    continue
                vol = shard.ghost_plan.volume[gsel].to(dtype).to(shard.device)
                gsel_dev = gsel.to(shard.device)
                vals = shard.ghost_vals[d, gsel_dev] * vol
                buf[shard.in_rank[gsel]] = vals.to(root_device)
            incoming[d] = buf.sum()
    return KineticInterfaceTransfer(outgoing, incoming)


def _assemble_bfl_force(
    shards,
    records,
    *,
    q: int,
    root_device: torch.device,
) -> torch.Tensor:
    """Global-order assembly of the per-substep MEM force.

    ``records`` is a per-shard list of ``(d, idx_local, link)`` triples
    emitted by the ``link_sink`` hook of :func:`bfl_apply_gather`.  Each
    direction's per-link contributions are scattered into a global buffer at
    their rank among the direction's boundary links (enum order) and summed
    in the same order and dtype as the unsharded accumulation.
    """
    force = torch.zeros(3, dtype=torch.float64, device=root_device)
    for d in range(1, q):
        per_shard = [
            (s, link)
            for s, recs in enumerate(records)
            for (dd, _idx, link) in recs
            if dd == d
        ]
        if not per_shard:
            continue
        n_d = sum(int(link.shape[0]) for _s, link in per_shard)
        buf = torch.zeros(n_d, 3, dtype=torch.float64, device=root_device)
        for s, link in per_shard:
            rank = shards[s].bfl_links[d][1]
            buf[rank] = link.to(root_device)
        force += buf.sum(dim=0)
    return force


def _merge_shard_ghost_plans(shards) -> ShellGhostPlan:
    """Reassemble the global ghost plan from the per-shard slices.

    Each shard's ``ghost_plan`` holds the rows of the global plan whose
    leaves it owns, and ``ghost_rows`` records each row's position in the
    global (unsharded) row order.  Scattering the slices back at those
    positions reproduces the global plan row for row (the global row order is
    direction-major — ``interface_links`` come from ``torch.nonzero`` — so a
    plain per-shard concatenation would NOT recover it), letting the sharded
    stepper run one ``_fill_ghost_impl`` over the full ghost batch.

    The global-order call keeps the ghost interpolation itself reproducible;
    it avoids an additional drift from reducing smaller, differently ordered
    batches.  The user collision callback is still allowed to differ by a few
    ulps across shard sizes.
    """
    if not shards:
        raise ValueError("_merge_shard_ghost_plans requires at least one shard")
    n_ghost = sum(int(s.ghost_plan.n_ghost) for s in shards)
    dev = shards[0].ghost_plan.z0.device
    q = int(shards[0].ghost_plan.slot.shape[0])
    n_leaf = int(shards[-1].hi)
    slot = torch.full((q, n_leaf), -1, dtype=torch.int64, device=dev)
    if n_ghost == 0:
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        empty64 = torch.empty(0, dtype=torch.float64, device=dev)
        return ShellGhostPlan(
            0, empty, empty, empty, empty, empty, empty, empty, empty,
            empty64, empty64, empty64, empty64, slot,
        )

    def _scatter(name: str, dtype: torch.dtype, *, local_leaf: bool = False):
        buf = torch.empty(n_ghost, dtype=dtype, device=dev)
        for s in shards:
            rows = s.ghost_rows
            if rows.numel() == 0:
                continue
            vals = getattr(s.ghost_plan, name).to(device=dev)
            if local_leaf:
                vals = vals + s.lo        # shard-local leaf enum -> global
            buf[rows] = vals
        return buf

    leaf = _scatter("leaf", torch.int64, local_leaf=True)
    direction = _scatter("direction", torch.int64)
    slot[direction, leaf] = torch.arange(n_ghost, dtype=torch.int64, device=dev)
    return ShellGhostPlan(
        n_ghost=n_ghost,
        leaf=leaf,
        direction=direction,
        z0=_scatter("z0", torch.int64), y0=_scatter("y0", torch.int64),
        x0=_scatter("x0", torch.int64), z1=_scatter("z1", torch.int64),
        y1=_scatter("y1", torch.int64), x1=_scatter("x1", torch.int64),
        wz=_scatter("wz", torch.float64), wy=_scatter("wy", torch.float64),
        wx=_scatter("wx", torch.float64), volume=_scatter("volume", torch.float64),
        slot=slot,
    )


def _prepare_shard_les_context(octree: OctreeGrid, shards: list) -> None:
    """Exchange remote leaf velocities and assemble sparse LES neighbours.

    Streaming exchanges one population per direction, which is insufficient
    for a velocity-gradient closure.  This routine exchanges the three local
    velocity components of each deduplicated remote leaf, then materialises a
    ``(3,Q,n_local)`` neighbour field.  Direct cross-shard links and coarse to
    fine fan-out links therefore use the same gradients as the unsharded
    octree; physical sentinels remain invalid and contribute one-sided/zero
    gradients as appropriate.
    """
    if not shards:
        return
    Q = int(octree.Q)
    vel_local: list[torch.Tensor] = []
    if Q == 27:
        from tensorlbm.d3q27 import macroscopic27
        macro = macroscopic27
    elif Q == 19:
        macro = macroscopic3d
    else:
        raise ValueError(f"unsupported octree lattice Q={Q}")
    for shard in shards:
        rho, ux, uy, uz = macro(shard.f_leaf.view(Q, 1, 1, -1))
        del rho
        vel_local.append(torch.stack((ux.reshape(-1), uy.reshape(-1), uz.reshape(-1))))
    for s, shard in enumerate(shards):
        if shard.les_remote_velocity is not None and shard.les_remote_velocity.numel():
            for t, src_idx, dst_idx in shard.les_requests:
                shard.les_remote_velocity[:, dst_idx] = vel_local[t][:, src_idx].to(
                    device=shard.device,
                )
        n = int(shard.n_leaf)
        nv = torch.zeros((3, Q, n), dtype=shard.f_leaf.dtype, device=shard.device)
        nd = torch.zeros((Q, n), dtype=shard.f_leaf.dtype, device=shard.device)
        nt = shard.neighbor_table
        centers = octree.leaf_center[shard.lo:shard.hi].to(shard.device)
        local_vel = vel_local[s]
        for d in range(Q):
            src = nt[d]
            local = src >= 0
            if bool(local.any()):
                idx = src[local]
                nv[:, d, local] = local_vel[:, idx]
                delta = centers[local] - centers[idx]
                nd[d, local] = delta.abs().amax(dim=1).to(nd.dtype)
            remote = src == REMOTE
            if bool(remote.any()) and shard.les_remote_pos.numel():
                slots = shard.les_remote_pos[d, remote]
                nv[:, d, remote] = shard.les_remote_velocity[:, slots]
                global_i = torch.nonzero(remote, as_tuple=False).squeeze(1)
                src_cent = shard.les_remote_centers[slots]
                nd[d, global_i] = (
                    centers[global_i] - src_cent
                ).abs().amax(dim=1).to(nd.dtype)
            fan = src == FANOUT
            if bool(fan.any()):
                for i in torch.nonzero(fan, as_tuple=False).squeeze(1).tolist():
                    encoded = shard.les_fan_members.get((d, int(i)), [])
                    if not encoded:
                        continue
                    vals = []
                    for code in encoded:
                        if code >= 0:
                            vals.append(local_vel[:, code])
                        else:
                            slot = -code - 1
                            vals.append(shard.les_remote_velocity[:, slot])
                    if vals:
                        nv[:, d, i] = torch.stack(vals, dim=1).mean(dim=1)
                        nd[d, i] = torch.as_tensor(
                            shard.les_fan_distance.get((d, int(i)), 0.0),
                            dtype=nd.dtype, device=nd.device,
                        )
        shard.les_neighbor_velocity = nv
        shard.les_neighbor_distance = nd


def step_octree_shell_sharded(
    octree: OctreeGrid,
    shards: list,
    advance: Advance3D,
    l1_old: torch.Tensor,
    l1_f: torch.Tensor,
    *,
    tau_coarse: float,
    tau_fine: float | None = None,
    tau_shell_override: float | None = None,
    l1_post: torch.Tensor | list[torch.Tensor] | None = None,
    shell_level: int = 1,
    reflux: bool = True,
    maximum_reflux_correction_fraction: float = 0.2,
    correction_stencil: str = "exterior_cells",
    ghost_plan: ShellGhostPlan | None = None,
    coarse_links: KineticInterfaceLinks | None = None,
    solid: torch.Tensor | None = None,
    bfl_fn=None,
    force_ledger=None,
) -> PopulationRefluxLedger:
    """Advance a device-sharded octree shell by one L1 root step.

    Same contract as :func:`step_octree_shell`, with the leaf set split into
    the :class:`~tensorlbm.octree_boundary.sharding.OctreeLeafShard` shards of
    ``shards`` (one per fine device; the octree and the L0/L1 hierarchy stay
    on the root device).  Per substep the ``advance`` callback runs once per
    shard on the shard's own device (the dominant shell collision is the
    parallelised workload), cross-shard neighbour/fan-out values are exchanged
    through the precomputed request plans, and the ghost fill is computed as a
    single full-batch call on the root device — the per-shard ghost plans are
    reassembled into the global (unsharded) row order first — then the
    per-shard values are moved to the fine devices.  Every state-affecting
    reduction (restriction, reflux interface transfers, the MEM force) is
    assembled on the root device in global order; the final state is expected
    to agree with the unsharded path to roundoff, not bit-for-bit for arbitrary
    collision callbacks.

    ``ghost_plan`` is accepted for API symmetry with :func:`step_octree_shell`
    but ignored: the per-shard ghost plans are baked into the shards at build
    time (:func:`~tensorlbm.octree_boundary.sharding.shard_octree_shell`).

    ``bfl_fn`` receives the per-shard facade as ``octree`` and must return
    ``(f_out, force | None)``; the force is re-assembled in global order from
    the per-link contributions (``bfl_apply_gather``'s ``link_sink`` hook) so
    the returned per-shard force is ignored.
    """
    if not isinstance(octree, OctreeGrid):
        raise TypeError("octree must be an OctreeGrid")
    if not shards:
        raise ValueError("step_octree_shell_sharded requires at least one shard")
    if l1_old.shape != l1_f.shape:
        raise ValueError("l1_old and l1_f must share shape")
    l1_shape = tuple(l1_f.shape[1:])
    if l1_shape != octree.meta["shape"]:
        raise ValueError("L1 populations do not match the octree's block shape")
    if correction_stencil not in ("exterior_cells", "crossing_links"):
        raise ValueError(
            "correction_stencil must be exterior_cells or crossing_links",
        )
    # New callbacks may request the active shard explicitly.  Keep the
    # historical four-argument callback ABI intact for existing applications,
    # while avoiding device-key collisions when multiple shards share a device
    # (CPU regression mode or MIG partitions).
    import inspect
    try:
        _advance_params = inspect.signature(advance).parameters
        advance_accepts_shard = (
            "shard" in _advance_params
            or any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in _advance_params.values())
        )
    except (TypeError, ValueError):
        advance_accepts_shard = False
    advance_uses_sparse_les = bool(
        getattr(advance, "uses_sparse_les", advance_accepts_shard)
    )
    n_substeps = 1 << octree.d_max
    taus = _tau_chain(tau_coarse, octree.d_max)
    if tau_fine is not None and abs(tau_fine - taus[1]) > 1.0e-12:
        raise ValueError("dynamic tau_fine must preserve convective scaling")
    if tau_shell_override is not None:
        taus = list(taus)
        taus[1] = float(tau_shell_override)
    tau_shell = taus[1]
    if reflux and l1_post is None:
        raise TypeError(
            "reflux-enabled shell stepping requires l1_post (the L1 "
            "post-collision/pre-stream state)",
        )
    if reflux and coarse_links is None:
        covered = octree._shell_mask
        if covered is None:
            raise RuntimeError(
                "octree carries no shell mask; pass coarse_links explicitly",
            )
        solid_mask = octree._solid if solid is None else solid
        coarse_links = build_shell_coarse_links(
            covered, solid_mask, q=octree.Q,
        )
    observation_links = None
    if reflux:
        if coarse_links is None:
            raise RuntimeError(
                "reflux bookkeeping lost the coarse interface links",
            )
        observation_links = build_shell_coarse_links(
            coarse_links.inside, None, q=octree.Q,
        )

    root_device = octree.f_leaf.device
    q = octree.Q
    dtype = octree.f_leaf.dtype
    # One global ghost plan reassembled from the per-shard slices in the
    # unsharded row order, so the per-substep ghost fill runs as a single
    # full-batch call.  This avoids an additional 1-ulp drift from reducing
    # smaller, differently ordered AMR-interface batches.
    merged_plan = _merge_shard_ghost_plans(shards)
    _DEBUG_STEP["n"] += 1
    fine_transfer: KineticInterfaceTransfer | None = None
    for s in range(n_substeps):
        alpha = s / n_substeps
        parent_t = torch.lerp(l1_old, l1_f, alpha)
        _debug_nan("ghost-source", {"parent_t": parent_t}, substep=s)
        # LES needs all neighbour velocities, including leaves that live on a
        # different shard.  Exchange this context before invoking the user
        # collision callback; the callback can consume the assembled fields
        # through ``shard.les_neighbor_velocity`` and
        # ``shard.les_neighbor_distance``.
        if advance_uses_sparse_les and len(shards) > 1:
            _prepare_shard_les_context(octree, shards)
        # 1. collision on the fine devices (the parallelised workload)
        for shard in shards:
            if advance_accepts_shard:
                result = advance(
                    shard.f_leaf, tau_shell, shell_level, s, shard=shard,
                )
            else:
                result = advance(shard.f_leaf, tau_shell, shell_level, s)
            shard.populations, shard.post_collision = _unpack_shell_advance(
                result,
                shard.f_leaf.shape,
            )
            _debug_nan("after-collision", {"populations": shard.populations}, substep=s)
        # 2. ghost fill on the root device: ONE global-order call over the
        # merged plan (batch order == unsharded), then scatter per shard
        gv_global = _fill_ghost_impl(
            octree.leaf_level, merged_plan, parent_t, taus,
        )
        _debug_nan("after-ghostfill", {"ghost_vals": gv_global}, substep=s)
        for shard in shards:
            shard.ghost_vals = gv_global[:, shard.ghost_rows].to(
                device=shard.device,
            )
        # 3. cross-shard post-collision value exchange
        for shard in shards:
            for t, d_idx, j_idx, out_pos in shard.requests:
                vals = shards[t].populations[d_idx, j_idx]
                shard.remote_buf[out_pos] = vals.to(device=shard.device)
        # 4. streaming per shard
        for shard in shards:
            shard.out = _stream_gather_shard(
                shard, shard.populations, shard.f_leaf, shard.ghost_vals,
            )
            _debug_nan("after-stream", {"out": shard.out}, substep=s)
        # 5. BFL per shard (force contributions captured for global assembly)
        force_records: list[list] = [[] for _ in shards]
        if bfl_fn is not None:
            for idx_s, shard in enumerate(shards):
                shard.remote_values = shard.remote_buf
                shard._link_sink = (
                    lambda d, idx, link, _s=idx_s: force_records[_s].append(
                        (d, idx, link),
                    )
                )
                result = bfl_fn(
                    shard, shard.out, shard.post_collision,
                    shard.ghost_plan, shard.ghost_vals, substep=s,
                )
                shard.out, _substep_force = result
                _debug_nan("after-bfl", {"out": shard.out}, substep=s)
                shard._link_sink = None
        # 6. reflux observation (assembled in global order on the root)
        if reflux:
            observed = _assemble_shell_transfer(
                shards, q=q, dtype=dtype, root_device=root_device,
            )
            fine_transfer = (
                observed if fine_transfer is None else fine_transfer + observed
            )
        # 7. force bookkeeping (one global force per substep, like unsharded)
        if force_ledger is not None and bfl_fn is not None:
            force_root = _assemble_bfl_force(
                shards, force_records, q=q, root_device=root_device,
            )
            force_ledger.add_substep_force(force_root)
        # 8. commit
        for shard in shards:
            shard.f_leaf = shard.out
            shard.populations = None
            shard.post_collision = None
            shard.ghost_vals = None
            shard.out = None
            shard.remote_values = None

    # ---- restriction + reflux on the root device (identical to unsharded) --
    from tensorlbm.octree_boundary.sharding import refresh_octree_f_leaf

    refresh_octree_f_leaf(octree, shards)
    restricted, cells = restrict_shell_to_block(octree, octree.f_leaf, taus)
    old_patch = l1_f[:, cells[:, 0], cells[:, 1], cells[:, 2]].clone()
    l1_f[:, cells[:, 0], cells[:, 1], cells[:, 2]] = restricted
    replacement_mismatch = old_patch.sum(dim=1) - restricted.sum(dim=1)
    if not reflux:
        return PopulationRefluxLedger(
            replacement_mismatch,
            torch.zeros_like(replacement_mismatch),
            0,
            replacement_mismatch,
            0,
            replacement_mismatch,
        )
    if fine_transfer is None:
        raise RuntimeError("shell stepping omitted the fine interface transfer")
    if l1_post is None or coarse_links is None or observation_links is None:
        raise RuntimeError(
            "reflux bookkeeping lost l1_post, coarse_links or observation_links",
        )
    if isinstance(l1_post, (tuple, list)):
        if len(l1_post) == 0:
            raise ValueError("l1_post sequence must not be empty")
        coarse_transfer = observe_kinetic_interface_transfer(
            l1_post[0], observation_links,
        )
        for post in l1_post[1:]:
            coarse_transfer = coarse_transfer + observe_kinetic_interface_transfer(
                post, observation_links,
            )
    else:
        coarse_transfer = observe_kinetic_interface_transfer(
            l1_post, observation_links,
        )
    l1_f, report = apply_face_local_reflux(
        l1_f,
        coarse_links,
        coarse_transfer,
        fine_transfer,
        maximum_correction_fraction=maximum_reflux_correction_fraction,
        correction_stencil=correction_stencil,
    )
    octree.meta["last_fine_transfer"] = fine_transfer
    return PopulationRefluxLedger(
        report.requested_inventory_correction,
        report.applied_inventory_correction,
        report.corrected_links,
        report.residual,
        report.limited_directions,
        report.raw_kinetic_mismatch,
        0.0,
        1.0,
        0.0,
        1.0,
        report.maximum_applied_correction_fraction,
    )


# ---------------------------------------------------------------------------
# Plane-degenerate shell (P2 verification topology)
# ---------------------------------------------------------------------------


def build_plane_shell(
    shape: tuple[int, int, int],
    box: BoxRegion,
    *,
    lattice: str = "D3Q19",
    device: torch.device | str = "cpu",
) -> OctreeGrid:
    """Degenerate plane shell: a flat box of depth-1 leaves.

    The shell degenerates to a rectangular box covering exactly the L1 cells
    of ``box`` — the plane special case of the P2 acceptance contract
    (design doc §4.3).  All leaves are depth 1 (``dx = 1/2``, ratio 2 to the
    L1 block), so the shell is the exact octree analogue of a
    ``StaticBlockAMR3D`` fine block and must reproduce its transfers and
    reflux ledger link by link.  ``meta["leaf_fine_flat"]`` maps each leaf to
    the flat index of the corresponding fine-block physical cell.
    """
    if lattice != "D3Q19":
        raise NotImplementedError("plane shell supports D3Q19 only")
    nz, ny, nx = shape
    if not (
        0 < box.x0 < box.x1 < nx - 1
        and 0 < box.y0 < box.y1 < ny - 1
        and 0 < box.z0 < box.z1 < nz - 1
    ):
        raise ValueError("plane shell box must be strictly interior")
    q = 19
    k = _axis_bits(shape)
    dev = torch.device(device)
    cells = torch.cartesian_prod(
        torch.arange(box.z0, box.z1, device=dev),
        torch.arange(box.y0, box.y1, device=dev),
        torch.arange(box.x0, box.x1, device=dev),
    )                                                       # (n, 3) (z,y,x)
    n_cells = cells.shape[0]
    child = torch.arange(8, device=dev)
    bx, by, bz = child & 1, (child >> 1) & 1, (child >> 2) & 1
    coords = torch.stack(
        (
            (2 * cells[:, 2]).repeat_interleave(8) + bx.repeat(n_cells),
            (2 * cells[:, 1]).repeat_interleave(8) + by.repeat(n_cells),
            (2 * cells[:, 0]).repeat_interleave(8) + bz.repeat(n_cells),
        ),
        dim=1,
    )                                                       # (8n, 3) x,y,z
    morton = morton_encode_batch(
        torch.ones(coords.shape[0], dtype=torch.int64, device=dev), coords, k,
    )
    order = torch.argsort(morton, stable=True)
    coords_sorted = coords[order]
    morton_sorted = morton[order]
    n_leaf = int(coords_sorted.shape[0])
    centers = (coords_sorted.to(torch.float64) + 0.5) / 2.0
    host_cell = (coords_sorted // 2)[:, [2, 1, 0]].to(torch.int64)

    nt = torch.full((q, n_leaf), SHELL_OUTSIDE, dtype=torch.int64, device=dev)
    nt[0] = torch.arange(n_leaf, dtype=torch.int64, device=dev)
    for d in range(1, q):
        target = coords_sorted + C[d].to(dev)
        q_target = morton_encode_batch(
            torch.ones(target.shape[0], dtype=torch.int64, device=dev),
            target, k,
        )
        pos = torch.searchsorted(morton_sorted, q_target)
        pos_m = pos.clamp(max=n_leaf - 1)
        hit = (pos < n_leaf) & (morton_sorted[pos_m] == q_target)
        # leaf enums are the sorted-Morton positions, so the neighbour enum
        # of a found target is pos_m itself.
        nt[d] = torch.where(
            hit,
            pos_m,
            torch.full_like(hit, SHELL_OUTSIDE, dtype=torch.int64),
        )
    links = torch.nonzero(nt == SHELL_OUTSIDE, as_tuple=False)
    if links.shape[0]:
        links = links[:, [1, 0]].contiguous()

    nz_f, ny_f, nx_f = (
        2 * (box.z1 - box.z0),
        2 * (box.y1 - box.y0),
        2 * (box.x1 - box.x0),
    )
    fine_flat = (
        (coords_sorted[:, 2] - 2 * box.z0) * ny_f
        + (coords_sorted[:, 1] - 2 * box.y0)
    ) * nx_f + (coords_sorted[:, 0] - 2 * box.x0)

    covered = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    covered[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1] = True

    grid = OctreeGrid(
        n_leaf=n_leaf,
        d_max=1,
        Q=q,
        level_start=torch.tensor([0, n_leaf, n_leaf], dtype=torch.int64, device=dev),
        leaf_morton=morton_sorted,
        leaf_level=torch.ones(n_leaf, dtype=torch.int64, device=dev),
        leaf_center=centers.to(torch.float32),
        leaf_box=torch.stack(
            [centers - 0.25, centers + 0.25], dim=1,
        ).to(torch.float32),
        neighbor_table=nt,
        q_field=torch.full((q, n_leaf), 0.5, dtype=torch.float32, device=dev),
        bfl_mask=torch.zeros((q, n_leaf), dtype=torch.bool, device=dev),
        interface_links=links.to(torch.int64),
        interface_fanout={},
        cross_level_donor=torch.full((q, n_leaf), -1, dtype=torch.int64, device=dev),
        leaf_host_cell=host_cell,
        f_leaf=torch.zeros((q, n_leaf), dtype=torch.float32, device=dev),
        morton_to_index={
            int(m): i for i, m in enumerate(morton_sorted.tolist())
        },
        meta={
            "shape": tuple(shape),
            "box": (box.z0, box.z1, box.y0, box.y1, box.x0, box.x1),
            "d_max": 1,
            "lattice": lattice,
            "axis_bits": int(k),
            "plane": True,
            "fine_shape": (nz_f, ny_f, nx_f),
            "leaf_fine_flat": fine_flat,
        },
    )
    grid._l1_coords = coords_sorted.contiguous()
    grid._l2_coords = torch.empty((0, 3), dtype=torch.int64, device=dev)
    grid._k = k
    grid._c_vec = C.to(dev)
    grid._opp = OPPOSITE.to(dev)
    grid._solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    grid._shell_mask = covered
    grid._delta_mask = 0.0
    grid.stats = {
        "n_leaf": n_leaf,
        "n_interface_links": int(links.shape[0]),
    }
    grid.checks = run_topology_checks(grid)
    return grid


__all__ = [
    "ShellGhostPlan",
    "build_ghost_plan",
    "build_plane_shell",
    "build_shell_coarse_links",
    "fill_ghost",
    "observe_shell_interface_transfer",
    "restrict_shell_to_block",
    "step_octree_shell",
    "stream_gather",
]
