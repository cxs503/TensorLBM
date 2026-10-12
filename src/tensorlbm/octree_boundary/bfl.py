"""P3 body-fitted Bouzidi BFL on the octree shell leaves (gather-based).

Implements the P3 acceptance contract of ``docs/octree-boundary-design.md``
§3.4 / §4: the same linear (``q < 0.5``) / quadratic (``q >= 0.5``)
interpolated bounce-back as ``tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19``
and ``tensorlbm.bfl_common.bfl_bounce_back_common``, with the **single**
documented change: the upstream query ``f_d(x - c_d)`` is a per-direction
donor-table gather instead of ``torch.roll`` (the shell leaves are not a
regular lattice).

Donor convention
----------------
For a boundary link ``(leaf i, direction d)`` (``bfl_mask[d, i]`` True, the
neighbour point ``x_i + c_d * dx_i`` lies inside the body), the unknown
incoming population is ``f[opp[d], i]`` and the upstream outgoing population
``f_prev[d](x_i - c_d * dx_i)`` is gathered from the leaf at the position
``x_i + c_opp[d] * dx_i`` — exactly ``neighbor_table[opp[d], i]``.  The
table therefore needs no separate tensor: ``upstream_donor_table(octree)``
exposes ``neighbor_table[OPPOSITE]``.  Cross-level entries (depth-1 donor
leaves), virtual ghost cells (``SHELL_OUTSIDE``, filled from the ghost plan)
and refined fan-out groups are handled like ``stream_gather``.

Interpolation-source axis
-------------------------
``bfl_apply_gather(..., bfl_source=...)`` / env ``OCTREE_BFL_INTERP``
selects the reconstruction source (see :func:`bfl_apply_gather`):
``"f_opp_post"`` (default) keeps the historical form, which — because
:func:`tensorlbm.octree_boundary.stepping.stream_gather` cell-locally
bounces SOLID links — reads the local post-collision ``fp_d`` back from the
streamed slot and degenerates the linear branch to a q-blind ``fp_d``;
``"legacy"`` restores the documented donor form (purely post-collision
``fp_d`` / ``fp_up`` / ``fp_opp``), the single-grid
``bfl_common``/``bfl_d3q19`` contract the design doc §3.4 pins for this
module.

Moving wall
-----------
The ramp startup of ``bfl_sphere_advance`` is preserved: during the first
``ramp_steps`` the wall moves with the local fluid velocity
(``(1 - activation) * u``), smoothly transitioning to a stationary no-slip
wall.  The population correction is the standard Bouzidi moving-wall term
(``cs^2 = 1/3``: linear branch ``-6 w rho (c_d . u_w)``, quadratic branch
``-(3/q) w rho (c_d . u_w)``).

Force
-----
With ``return_force=True`` the laboratory-frame link momentum exchange
``sum_links c_d (f_prev[d] + f_bc)`` is returned (float64, ``(3,)``) — the
same impulse that ``bouzidi_bounce_back_d3q19`` reports and that the
single-grid sphere benchmark validates against the control-volume force.
Per-leaf substep weights ``2^-(d_max - d_leaf)`` (``leaf_force_weights``)
are applied when supplied; the accumulation into a per-root-step force is
the responsibility of :mod:`tensorlbm.octree_boundary.force`.  The
per-substep weight carries the cadence frame
``host_steps_per_shell_call`` (default ``1.0`` = one shell call per
host-grid lattice step); see :func:`leaf_force_weights` for the hook
cadence correction.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from tensorlbm.octree_boundary.geometry import (
    DOMAIN_OUT,
    FANOUT,
    REMOTE,
    SHELL_OUTSIDE,
    SOLID,
)

if TYPE_CHECKING:
    from tensorlbm.octree_boundary.geometry import OctreeGrid
    from tensorlbm.octree_boundary.stepping import ShellGhostPlan


def _ramp_activation(step: int, steps: int) -> float:
    """Raised-cosine startup ramp (identical to ``ramp_activation``)."""
    if steps <= 0 or step >= steps:
        return 1.0
    import math

    return 0.5 * (1.0 - math.cos(math.pi * step / steps))


def upstream_donor_table(octree: OctreeGrid) -> torch.Tensor:
    """``(Q, n_leaf)`` int64 donor table of the BFL upstream points.

    ``upstream_donor[d, i]`` is the leaf enum at ``x_i - c_d * dx_i``, i.e.
    the neighbour of leaf ``i`` along ``opp[d]`` — a plain (re)indexing of
    ``neighbor_table``.  Values keep the sentinels
    ``SHELL_OUTSIDE / SOLID / FANOUT / DOMAIN_OUT``.
    """
    return octree.neighbor_table[octree._opp]


def leaf_force_spatial_weights(
    octree: OctreeGrid,
    *,
    reference_level: int = 1,
) -> torch.Tensor:
    """Per-leaf **convective spatial** factor ``(dx_leaf / dx_ref)^2``.

    The link impulse of a leaf is expressed in that leaf's own lattice
    units; to add impulses from leaves at *different* depths they must be
    mapped to a common lattice with the convective scaling
    ``F ~ (dx^4 / dt^2) F_lu`` (``rho0 = 1``, ``dt_leaf = dx_leaf``), i.e.
    ``(dx_leaf/dx_ref)^2 = 2^-2(level - reference_level)`` (bfl design
    doc §3.5).  ``reference_level`` defaults to 1 — the coarsest shell level
    — so the returned force is expressed in a *fixed* lattice and is
    refinement-invariant across ``d_max`` (a depth-1 reference leaves the
    depth-1 leaves at factor 1 and down-weights the finer, smaller-area
    leaves by ``2^-2`` per level).

    This factor is the piece :func:`leaf_force_weights` used to miss:
    without it a mixed-depth shell (``d_max=2`` mixes level-1 and level-2
    leaves) sums impulses from ``dx=0.5`` and ``dx=0.25`` lattices with
    identical weights, over-counting the fine-level contribution by
    ``(dx_1/dx_2)^2 = 4`` — the ``d_max=1 -> 2`` force ratio ``~3.7``
    instead of ``~1`` (scripts/validate_shell_bfl_force_analytic_cpu.py).
    """
    level = octree.leaf_level.to(torch.float64)
    return 2.0 ** (-2.0 * (level - float(reference_level)))


def leaf_force_weights(
    octree: OctreeGrid,
    *,
    include_spatial: bool = False,
    reference_level: int = 1,
    host_steps_per_shell_call: float = 1.0,
) -> torch.Tensor:
    """Per-leaf substep weight ``2^-(d_max - d_leaf) * host_steps_per_shell_call``.

    The shell advances every leaf ``2**d_max`` lockstep substeps per shell
    call; a depth-``d`` leaf should only contribute ``2**d`` of them to the
    force per **host lattice step** (its own convective time step is
    ``2^-d`` host steps).  The weight corrects the lockstep over-sampling:
    ``sum_substeps w = 2**d`` per host step of accumulated samples.

    **Cadence frame (E3 lockstep adjudication, 2026-10-10).**  Each weighted
    sample represents ``rate * dt_host``: accumulating over ``C`` shell calls
    therefore yields the impulse over ``C * dt_host`` of host time, which is
    only the impulse over the caller's force interval ``T`` when the host
    advances exactly one lattice step per shell call (``C * dt_host == T``,
    the legacy two-level design of ``examples/octree_{sphere,integrated_
    distributed,multigpu}_validate.py`` and the wrapper cadence of the E3
    arms A_S/A_S_tau/B/B2 — measured ``force_ratio_corr_over_lib == 1.0``).
    The hybrid ``pre_restrict_hook`` cadence (``StaticBlockAMR3D.step``,
    Phase 0.5) fires :func:`tensorlbm.octree_boundary.stepping.step_octree_shell`
    once per outer-root step while the host advances ``config.ratio``
    (typically 2) lattice steps in between, so the stock weight under-counts
    every leaf by exactly ``ratio`` — E3 arms A/A_tau (``d_max=2``, hook)
    measured ``force_ratio_corr_over_lib = 2.0`` at **both** leaf depths
    (``phase05/e3_lockstep_ab2.json``); the depth split is cadence-driven,
    not ``d_max``-driven.  Pass ``host_steps_per_shell_call =
    host_lattice_steps_per_root_step / shell_calls_per_root_step`` (hook:
    ``2.0``) to make the accumulated ledger the impulse over the caller's
    interval; this reproduces the formal runner's manual
    ``leaf_force_weights(octree, include_spatial=True) * 2.0`` cadence-P
    convention (F-4/F-5/E3 A1 record) exactly (bit-for-bit: the factor is a
    power of two).

    Args:
        octree: the shell grid.
        include_spatial: fold in the per-leaf convective spatial factor
            :func:`leaf_force_spatial_weights` (``(dx_leaf/dx_ref)^2``).
            **Required for a correct force on a mixed-depth shell**: the
            raw time weight alone assumes every leaf's impulse is already in
            a common lattice, which only holds for a single-level shell.
            Defaults to ``False`` for backward compatibility with the
            historical (time-only) convention; the shell force path should
            pass ``True``.
        reference_level: leaf level the spatial factor is referenced to
            (see :func:`leaf_force_spatial_weights`); only meaningful with
            ``include_spatial=True``.
        host_steps_per_shell_call: host-grid lattice steps advanced per
            shell call (the cadence frame).  ``1.0`` — the default — is the
            legacy "one shell call per host-grid lattice step" frame and is
            bit-identical to the historical weight.  Under the
            ``pre_restrict_hook`` cadence (one call per outer-root step,
            ``ratio`` host steps per root step) pass the ratio (e.g.
            ``2.0``); any positive value is accepted (fractional cadences
            ``C > R`` give ``< 1``).

    The coordinate conversion that maps the resulting leaf force to the L1
    root lattice is :func:`tensorlbm.octree_boundary.force.convert_leaf_force_to_l1`
    — with ``include_spatial=True`` it must be called with the *reference*
    level's ``dx_leaf`` (a single value), never with the individual leaf
    depths (see its docstring).
    """
    levels = octree.leaf_level.to(torch.float64)
    kappa = float(host_steps_per_shell_call)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError(
            f"host_steps_per_shell_call must be finite and positive, got {host_steps_per_shell_call!r}",
        )
    weights = 2.0 ** (-(octree.d_max - levels))
    if kappa != 1.0:
        weights = weights * kappa
    if include_spatial:
        weights = weights * leaf_force_spatial_weights(
            octree,
            reference_level=reference_level,
        )
    return weights


def leaf_macroscopic(
    octree: OctreeGrid,
    f_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-leaf ``(rho, ux, uy, uz)`` from the pre-stream populations."""
    if octree.Q == 27:
        from tensorlbm.d3q27 import macroscopic27

        rho, ux, uy, uz = macroscopic27(f_prev.view(octree.Q, 1, 1, -1))
    else:
        from tensorlbm.d3q19 import macroscopic3d

        rho, ux, uy, uz = macroscopic3d(f_prev.view(octree.Q, 1, 1, -1))
    return rho.view(-1), ux.view(-1), uy.view(-1), uz.view(-1)


def bfl_ramp_wall_velocity(
    octree: OctreeGrid,
    f_prev: torch.Tensor,
    step: int,
    ramp_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-leaf wall velocity + density for the startup ramp.

    Returns ``(rho, ux, uy, uz)`` with the velocity fields scaled by
    ``(1 - activation)`` — the moving-wall ramp of ``bfl_sphere_advance``.
    After ``ramp_steps`` the wall velocity is exactly zero.
    """
    rho, ux, uy, uz = leaf_macroscopic(octree, f_prev)
    activation = _ramp_activation(step, ramp_steps)
    factor = 1.0 - activation
    return rho, factor * ux, factor * uy, factor * uz


def bfl_apply_gather(
    octree: OctreeGrid,
    f: torch.Tensor,
    f_prev: torch.Tensor,
    *,
    ghost_plan: ShellGhostPlan | None = None,
    ghost_vals: torch.Tensor | None = None,
    wall_velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    wall_density: torch.Tensor | None = None,
    force_weights: torch.Tensor | None = None,
    return_force: bool = False,
    q_min: float | None = None,
    link_sink=None,
    bfl_source: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gather-based Bouzidi BFL on the octree shell leaves.

    Args:
        octree: the shell (``bfl_mask``, ``q_field``, ``neighbor_table``).
            For a sharded shell (:class:`~tensorlbm.octree_boundary.sharding.OctreeLeafShard`)
            the facade must additionally expose ``remote_values`` (the filled
            per-substep cross-shard value buffer) and ``remote_pos``/``fan_off``/
            ``fan_len`` ((Q, n) int64 position tables); cross-shard upstream
            donors then resolve through the ``REMOTE`` neighbour sentinel.
        f: post-stream populations ``(Q, n_leaf)`` (mutated copy returned).
        f_prev: pre-stream (post-collision) populations ``(Q, n_leaf)``.
        ghost_plan / ghost_vals: needed only when an upstream point of a
            boundary link falls on a virtual ghost cell (``SHELL_OUTSIDE``);
            the ghost cell at ``x_i - c_d * dx_i`` is ``plan.slot[d, i]``.
        wall_velocity: per-leaf wall velocity ``(ux, uy, uz)`` — the ramp
            startup of ``bfl_sphere_advance``; ``None`` = stationary wall.
        wall_density: per-leaf wall density (required with wall_velocity).
        force_weights: per-leaf substep weights (``leaf_force_weights``);
            applied to the returned force only.
        return_force: also return the ``(3,)`` float64 link momentum
            exchange of this substep (leaf lattice units).
        q_min: clamp tiny BFL q values (high-Re safeguard).
        link_sink: optional ``link_sink(d, idx, link)`` hook receiving every
            per-link force contribution ``link`` (``(n_links_d, 3)`` float64,
            already weight-scaled) together with the local leaf indices
            ``idx``, for global-order assembly by the sharded stepper.  When
            given, the per-direction in-place ``force`` accumulation is
            skipped (the returned force is then all zeros and must not be
            used).
        bfl_source: interpolation-source axis, one of ``"f_opp_post"``
            (default; the historical/defective form, see below) or
            ``"legacy"`` (the documented donor form).  ``None`` defers to the
            ``OCTREE_BFL_INTERP`` environment variable, itself defaulting to
            ``"f_opp_post"`` — the calibration-harness A/B gate.  Fail-closed
            on any other value.

    Returns:
        ``f_out`` or ``(f_out, force)``.  ``f_out[d]`` is only changed on
        the unknown boundary directions ``opp[d]`` at masked leaves.
    """
    if f.shape != f_prev.shape:
        raise ValueError("f and f_prev must share shape")
    if f.shape[0] != octree.Q or f.shape[1] != octree.n_leaf:
        raise ValueError("populations must be (Q, n_leaf) shell tensors")
    if wall_velocity is not None and wall_density is None:
        raise ValueError("wall_density is required with wall_velocity")
    if wall_velocity is not None and len(wall_velocity) != 3:
        raise ValueError("wall_velocity must be (ux, uy, uz)")

    Q = octree.Q
    device = f.device
    dtype = f.dtype
    opp = octree._opp.to(device)
    c_vec = octree._c_vec.to(device)
    if Q == 27:
        from tensorlbm.d3q27 import W as _W
    else:
        from tensorlbm.d3q19 import W as _W19

        _W = _W19

    mask = octree.bfl_mask.to(device).clone()
    # The legacy implementation only processed directions 1..Q-1 (the rest
    # direction d=0 was never touched); keep that contract exactly.
    mask[0] = False
    q_field = octree.q_field
    nt = octree.neighbor_table
    # sharded-shell facade hooks (absent on a plain OctreeGrid)
    remote_values = getattr(octree, "remote_values", None)
    remote_pos = getattr(octree, "remote_pos", None)
    fan_off = getattr(octree, "fan_off", None)
    fan_len = getattr(octree, "fan_len", None)
    if link_sink is None:
        # the sharded stepper installs a per-substep sink on the facade
        link_sink = getattr(octree, "_link_sink", None)
    f_out = f.clone()
    force = torch.zeros(3, dtype=torch.float64, device=device)

    # ---- batched upstream donor table: all directions in one pass ---------
    # ``src[d, i] = neighbor_table[opp[d], i]`` — the donor of leaf i along
    # direction d (identical table to the streamer's ``src_all``).  Only
    # MASKED cells are examined (the legacy per-direction loop only ever
    # looked at ``idx = nonzero(mask[d])``): this also keeps the facade's
    # out-of-shard donor remap (-1, same value as SHELL_OUTSIDE) from being
    # misread as a ghost cell on unmasked links.
    src = nt[opp].to(device)  # (Q, n)
    fp_d = f_prev.to(torch.float64)
    fp_opp = f_prev[opp].to(torch.float64)
    fp_up = torch.zeros_like(fp_d)
    cells = torch.nonzero(mask, as_tuple=False)  # (n_m, 2) (d, i)
    if cells.shape[0]:
        d_c, i_c = cells[:, 0], cells[:, 1]
        src_m = src[d_c, i_c]  # (n_m,)
        valid = src_m >= 0
        if bool(valid.any()):
            if int(src_m.max()) >= f_prev.shape[1]:
                raise IndexError("BFL upstream donor index out of range")
            fp_up[d_c[valid], i_c[valid]] = f_prev[d_c[valid], src_m[valid].clamp(min=0)].to(
                torch.float64
            )
        if remote_values is not None and remote_pos is not None:
            remote = src_m == REMOTE
            if bool(remote.any()):
                d_r = d_c[remote]
                i_r = i_c[remote]
                # ``remote_pos[q, i]`` is keyed by the stream-link direction q,
                # while BFL's upstream donor for wall direction d is
                # ``neighbor_table[opp[d], i]``.  Use that same donor direction
                # for the remote slot; using d silently addresses a different
                # (often absent) cross-shard request.
                slots = remote_pos[opp[d_r], i_r]
                if bool((slots < 0).any()):
                    raise RuntimeError(
                        "sharded BFL upstream point is REMOTE but has no "
                        "remote slot (shard plan inconsistency)",
                    )
                fp_up[d_r, i_r] = remote_values[slots].to(torch.float64)
        ghost = src_m == SHELL_OUTSIDE
        if bool(ghost.any()):
            if ghost_plan is None or ghost_vals is None:
                raise RuntimeError(
                    "BFL upstream point is a ghost cell but no ghost values were supplied",
                )
            d_g = d_c[ghost]
            i_g = i_c[ghost]
            slots = ghost_plan.slot[d_g, i_g]
            if bool((slots < 0).any()):
                bad = torch.nonzero(slots < 0, as_tuple=False).squeeze(1)
                print(
                    f"[bfl] ghost slot missing: n_bad={len(bad)} "
                    f"ghost_plan.slot shape={tuple(ghost_plan.slot.shape)}",
                    flush=True,
                )
                raise RuntimeError(
                    "BFL upstream ghost cell has no ghost slot (shell band too thin)",
                )
            fp_up[d_g, i_g] = ghost_vals[d_g, slots].to(torch.float64)
        fanout = src_m == FANOUT
        if bool(fanout.any()):
            d_f = d_c[fanout]
            i_f = i_c[fanout]
            fm = getattr(octree, "fanout_mean", None)
            if fm is not None:
                # Distributed facade: float64 member means precomputed by the
                # stepper from the all-gathered post-collision state (also
                # fixes the P1 bug where the global-keyed dict was queried
                # with LOCAL columns and always missed -> fp_d fallback).
                fp_up[d_f, i_f] = fm[d_f, i_f]
            elif remote_values is not None and fan_off is not None and fan_len is not None:
                # Sharded facade: members live in the remote value buffer at
                # fan_off/fan_len (variable-length segments).
                offs = fan_off[d_f, i_f]
                lens = fan_len[d_f, i_f]
                has = lens > 0
                if bool(has.any()):
                    max_ln = int(lens.max())
                    col = torch.arange(max_ln, device=device)
                    pad_idx = offs[has].unsqueeze(1) + col.unsqueeze(0)
                    pad_ok = col.unsqueeze(0) < lens[has].unsqueeze(1)
                    rv = remote_values[pad_idx.clamp(max=max(remote_values.shape[0] - 1, 0))]
                    fp_up[d_f[has], i_f[has]] = (rv * pad_ok).to(torch.float64).sum(
                        dim=1
                    ) / pad_ok.sum(dim=1).clamp_min(1).to(torch.float64)
                fb = ~has
                if bool(fb.any()):
                    fp_up[d_f[fb], i_f[fb]] = fp_d[d_f[fb], i_f[fb]]
            else:
                # Plain octree (global columns): corrected live fanout tables.
                from tensorlbm.octree_boundary.stepping import (
                    _fanout_segment_mean,
                    ensure_fanout_tables,
                )

                rowidx, pad_live = ensure_fanout_tables(octree)
                # the cached fanout tables live on the CPU (the SDAA-safe
                # build path); move them to the populations device before
                # the advanced index — indexing a CPU tensor with CUDA
                # indices raises (seen on the E4 SUBOFF shell, d_max=2)
                rowidx = rowidx.to(device)
                pad_live = pad_live.to(device)
                ridx = rowidx[opp[d_f], i_f]  # live row / -1
                has = ridx >= 0
                if bool(has.any()):
                    fp_up[d_f[has], i_f[has]] = _fanout_segment_mean(
                        f_prev,
                        d_f[has],
                        pad_live[ridx[has]],
                    ).to(torch.float64)
                fb = ~has
                if bool(fb.any()):
                    # defensive: keep the local outgoing population
                    fp_up[d_f[fb], i_f[fb]] = fp_d[d_f[fb], i_f[fb]]
        solid_up = src_m == SOLID
        if bool(solid_up.any()):
            # upstream inside the body: degenerate, keep the local outgoing
            # population (defensive; geometry excludes this case)
            fp_up[d_c[solid_up], i_c[solid_up]] = fp_d[d_c[solid_up], i_c[solid_up]]
        if bool((src_m == DOMAIN_OUT).any()):
            raise RuntimeError("BFL upstream leaves the L1 block")

    # ---- Bouzidi reconstruction (full-tensor, one pass) ------------------
    qq = q_field.to(torch.float64)
    if q_min is not None:
        qq = torch.clamp(qq, min=float(q_min))
    # Interpolation-source axis (suboff_highre_20261010 diagnosis /
    # libfix_20261011): stream_gather's SOLID branch performs a cell-local
    # full-way bounce, so at every BFL link the streamed unknown slot
    # f[opp[d]] is bitwise the local post-collision outgoing fp_d (machine
    # probe: frac_bitwise_equal=1.0) and the historical "f_opp_post" form
    # degenerates to f_bc = fp_d — q-blind and wall-information-free in the
    # linear branch (the wall is pinned to the link midpoint).  "legacy" is
    # the documented contract (docs/octree-boundary-design.md §3.4 =
    # bfl_common.bfl_bounce_back_common = bfl_d3q19.bouzidi_bounce_back_
    # d3q19): the reconstruction is expressed purely on post-collision
    # populations — local outgoing fp_d, local opposite fp_opp, and the TRUE
    # upstream donor fp_up = f_prev[d](x_i - c_d) gathered explicitly by the
    # machinery above — never on the streamed slot.  At q = 0.5 both forms
    # reduce to fp_d (the half-way bounce-back fixed point, bit-identical to
    # the single-grid references); for every q != 0.5 only the legacy form
    # honours q and the donor.
    _source = (
        bfl_source if bfl_source is not None else os.environ.get("OCTREE_BFL_INTERP", "f_opp_post")
    )
    if _source not in ("f_opp_post", "legacy"):
        raise ValueError(
            f"bfl_source/OCTREE_BFL_INTERP must be 'f_opp_post' or 'legacy', got {_source!r}",
        )
    lin = qq < 0.5
    safe_q = torch.where(lin, torch.ones_like(qq), qq)
    if _source == "legacy":
        f_bc_lin = 2.0 * qq * fp_d + (1.0 - 2.0 * qq) * fp_up
        f_bc_quad = fp_d / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    else:
        # f_opp_post = post-stream opposite-direction population (streamed FROM
        # the solid side).  The legacy formula used fp_d (pre-stream incident)
        # which reconstructs bounce-back from the wrong source and drains shell
        # momentum every substep (octree sphere Cd=3.0 vs 1.09).
        f_opp_post = f[opp].to(torch.float64)
        f_bc_lin = 2.0 * qq * f_opp_post + (1.0 - 2.0 * qq) * fp_d
        f_bc_quad = f_opp_post / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    f_bc = torch.where(lin, f_bc_lin, f_bc_quad)
    if wall_velocity is not None:
        uwx, uwy, uwz = wall_velocity
        dev_w = f_bc.device
        if uwx.device != dev_w:
            uwx = uwx.to(dev_w)
            uwy = uwy.to(dev_w)
            uwz = uwz.to(dev_w)
        u_stack = torch.stack([uwx, uwy, uwz], dim=0)  # (3, n)
        c_dot_uw = c_vec.to(torch.float64) @ u_stack.to(torch.float64)
        rho_w = wall_density.to(torch.float64)
        if rho_w.device != dev_w:
            rho_w = rho_w.to(dev_w)
        moving_base = _W.to(torch.float64).unsqueeze(1).to(dev_w) * rho_w.unsqueeze(0) * c_dot_uw
        f_bc = torch.where(
            lin,
            f_bc - 6.0 * moving_base,
            f_bc - (3.0 / safe_q) * moving_base,
        )

    # Masked write: only the unknown boundary directions opp[d] at masked
    # leaves (identical positions to the per-direction loop).
    f_out[opp] = torch.where(mask, f_bc.to(dtype), f_out[opp])

    if return_force:
        # Laboratory-frame momentum exchange c_d*(f_d + f_bc), the impulse
        # that closes the fixed-frame control-volume balance (identical
        # convention to bouzidi_bounce_back_d3q19).
        exchange = fp_d + f_bc
        link = (exchange.unsqueeze(2) * c_vec.to(torch.float64).unsqueeze(1)) * mask.unsqueeze(2)
        if force_weights is not None:
            link = link * force_weights.to(torch.float64).unsqueeze(0).unsqueeze(2)
        if link_sink is not None:
            # per-direction sink hook (global-order assembly by the sharded
            # stepper); the returned force stays zero there.
            for d in range(1, Q):
                idx_d = torch.nonzero(mask[d], as_tuple=False).squeeze(1)
                if idx_d.numel() == 0:
                    continue
                link_sink(d, idx_d, link[d, idx_d])
        else:
            # per-direction reduction first (matches the legacy accumulation
            # order: sum over links of each direction, then over directions)
            force = link.sum(dim=1).sum(dim=0)

    if return_force:
        return f_out, force
    return f_out


__all__ = [
    "bfl_apply_gather",
    "bfl_ramp_wall_velocity",
    "leaf_force_spatial_weights",
    "leaf_force_weights",
    "leaf_macroscopic",
    "upstream_donor_table",
]
