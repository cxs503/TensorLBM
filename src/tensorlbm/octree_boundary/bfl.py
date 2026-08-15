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
the responsibility of :mod:`tensorlbm.octree_boundary.force`.
"""
from __future__ import annotations

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


def leaf_force_weights(octree: OctreeGrid) -> torch.Tensor:
    """Per-leaf substep weight ``2^-(d_max - d_leaf)`` (design doc §3.5).

    The shell advances every leaf ``2**d_max`` lockstep substeps per root
    step; a depth-``d`` leaf should only contribute ``2**d`` of them to the
    force (its own convective time step is ``2^-d`` root units).  The weight
    corrects the lockstep over-sampling: ``sum_substeps w = 2**d``.
    """
    return 2.0 ** (
        -(octree.d_max - octree.leaf_level.to(torch.float64))
    )


def leaf_macroscopic(
    octree: OctreeGrid, f_prev: torch.Tensor,
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
    opp = octree._opp.to(device)
    c_vec = octree._c_vec.to(device)
    if Q == 27:
        from tensorlbm.d3q27 import W as _W
    else:
        from tensorlbm.d3q19 import W as _W19

        _W = _W19

    mask = octree.bfl_mask
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

    for d in range(1, Q):
        m = mask[d]
        if not bool(m.any()):
            continue
        od = int(opp[d].item())
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        qq = q_field[d, idx].to(torch.float64)
        if q_min is not None:
            # High-Re safeguard: clamp tiny q to avoid the 1/(2q) divergence in
            # the quadratic branch. q_min=0 disables (keeps validated low-Re
            # results bit-identical).
            qq = torch.clamp(qq, min=float(q_min))
        fp_d = f_prev[d, idx].to(torch.float64)
        fp_opp = f_prev[od, idx].to(torch.float64)

        # ---- upstream donor gather: f_prev[d](x_i - c_d * dx_i) ----------
        up = nt[od, idx]
        fp_up = torch.zeros_like(fp_d)
        valid = up >= 0
        if bool(valid.any()):
            fp_up[valid] = f_prev[d, up[valid]].to(torch.float64)
        if remote_values is not None and remote_pos is not None:
            remote = up == REMOTE
            if bool(remote.any()):
                slots = remote_pos[od, idx[remote]]
                if bool((slots < 0).any()):
                    raise RuntimeError(
                        "sharded BFL upstream point is REMOTE but has no "
                        "remote slot (shard plan inconsistency)",
                    )
                fp_up[remote] = remote_values[slots].to(torch.float64)
        ghost = up == SHELL_OUTSIDE
        if bool(ghost.any()):
            if ghost_plan is None or ghost_vals is None:
                raise RuntimeError(
                    "BFL upstream point is a ghost cell but no ghost values "
                    "were supplied",
                )
            slots = ghost_plan.slot[d, idx[ghost]]
            if bool((slots < 0).any()):
                raise RuntimeError(
                    "BFL upstream ghost cell has no ghost slot "
                    "(shell band too thin)",
                )
            fp_up[ghost] = ghost_vals[d, slots].to(torch.float64)
        fanout = up == FANOUT
        if bool(fanout.any()):
            for pos in torch.nonzero(
                fanout, as_tuple=False,
            ).squeeze(1).tolist():
                leaf_i = int(idx[pos].item())
                if remote_values is not None and fan_off is not None and fan_len is not None:
                    off = int(fan_off[od, leaf_i].item())
                    ln = int(fan_len[od, leaf_i].item())
                    if ln > 0:
                        fp_up[pos] = remote_values[off:off + ln].to(
                            torch.float64,
                        ).mean()
                        continue
                group = octree.interface_fanout.get((leaf_i, int(od)), [])
                if group:
                    fp_up[pos] = f_prev[
                        d, torch.tensor(group, device=device)
                    ].to(torch.float64).mean()
                else:
                    fp_up[pos] = fp_d[pos]
        solid_up = up == SOLID
        if bool(solid_up.any()):
            # upstream inside the body: degenerate, keep the local outgoing
            # population (defensive; geometry excludes this case)
            fp_up[solid_up] = fp_d[solid_up]
        if bool((up == DOMAIN_OUT).any()):
            raise RuntimeError("BFL upstream leaves the L1 block")

        # ---- Bouzidi reconstruction --------------------------------------
        # Aligned with the verified reference bfl_d3q19.bouzidi_bounce_back_d3q19:
        #   f_opp = f[opp_d]               (post-stream value streamed FROM the
        #                                    solid side; in the pull scheme
        #                                    f[opp_d](x_f) = f_post[opp_d](x_s))
        #   f_bc_lin  = 2q*f_opp + (1-2q)*fp_d
        #   f_bc_quad = f_opp/(2q) + (2q-1)/(2q)*fp_opp
        # The legacy octree formula used fp_d (pre-stream incident) in place
        # of f_opp, which reconstructs the bounce-back from the wrong source
        # population and drains shell momentum every substep (octree sphere
        # Cd=3.0 vs 1.09; see octree VR force audit).
        f_opp_post = f[od, idx].to(torch.float64)
        lin = qq < 0.5
        f_bc_lin = 2.0 * qq * f_opp_post + (1.0 - 2.0 * qq) * fp_d
        safe_q = torch.where(lin, torch.ones_like(qq), qq)
        f_bc_quad = (
            f_opp_post / (2.0 * safe_q)
            + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
        )
        f_bc = torch.where(lin, f_bc_lin, f_bc_quad)
        if wall_velocity is not None:
            uwx, uwy, uwz = wall_velocity
            c_d = c_vec[d]
            c_dot_uw = (
                float(c_d[0].item()) * uwx[idx]
                + float(c_d[1].item()) * uwy[idx]
                + float(c_d[2].item()) * uwz[idx]
            )
            rho_w = wall_density[idx].to(torch.float64)
            moving_base = float(_W[d].item()) * rho_w * c_dot_uw
            f_bc = torch.where(
                lin,
                f_bc - 6.0 * moving_base,
                f_bc - (3.0 / safe_q) * moving_base,
            )

        if return_force:
            # Laboratory-frame momentum exchange c_d*(f_d + f_bc), the
            # impulse that closes the fixed-frame control-volume balance
            # (identical convention to bouzidi_bounce_back_d3q19).
            exchange = fp_d + f_bc
            link = exchange.unsqueeze(1) * c_vec[d].to(torch.float64)
            if force_weights is not None:
                link = link * force_weights[idx].unsqueeze(1)
            if link_sink is not None:
                link_sink(d, idx, link)
            else:
                force += link.sum(dim=0)

        f_out[od, idx] = f_bc.to(f.dtype)

    if return_force:
        return f_out, force
    return f_out


__all__ = [
    "bfl_apply_gather",
    "bfl_ramp_wall_velocity",
    "leaf_force_weights",
    "leaf_macroscopic",
    "upstream_donor_table",
]
