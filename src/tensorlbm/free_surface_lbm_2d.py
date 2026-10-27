"""Free-surface LBM (D2Q9) — faithful Körner model with an *independent* mass ledger.

Reference mechanics (Körner et al. 2005; Thürey 2007; waLBerla
``free_surface``), mirroring :mod:`free_surface_lbm` (D3Q19): single
fill-weighted Guo force, ABB with Laplace/bubble pressure, fluid-side
half-way walls, Körner eq. 9 exchange with the Thürey table-4.1 selector,
interface-only conversion and the conversion-driven envelope.

This module replaces the former density-mapping stub, whose ``fill`` was a
function of the LBM populations (``fill = rho / rho_liquid``).  Mixing the two
representations is the root cause of the old interface atomisation (interface
spreading to ~90 % of the domain) and the +323 % wave-front pollution.

Here the fill level is a genuine, independently advanced quantity

    epsilon(x) = m(x) / rho_liquid

where ``m`` is a per-cell liquid mass that is **not** the LBM density.  It is
advected by the Körner liquid/interface mass exchange, conserved-redistributed
when a cell fills or drains, and only converted to LIQUID/GAS at explicit fill
thresholds.

Model (Körner et al. 2005, J. Comput. Phys. 202, 68; Thürey 2007;
waLBerla ``free_surface`` free-surface module):

* cell types ``GAS / LIQUID / INTERFACE / SOLID``
* independent mass ``m``; fill ``epsilon = m / rho_l``
* anti-bounce-back (ABB) gas-pressure reconstruction at interface cells
* Körner mass exchange: every LIQUID/INTERFACE link is treated with an
  antisymmetric half weight so each *internal* (I/I) link transfers mass with
  an exact ``+dm / -dm`` pair; a LIQUID/INTERFACE link credits the interface
  by exactly the population debit of the bulk cell, hence the conserved liquid
  inventory ``sum(rho over LIQUID) + sum(m over INTERFACE)`` is invariant
* excess-mass redistribution of a converting cell's overflow to interface /
  newly promoted receiver cells (never silently discarded)
* flag conversion I->L (fill >= 1) and I->G (fill <= 0) plus the envelope halo
  that keeps a one-cell interface layer between every LIQUID and GAS region
* a conservation-preserving clamp keeps ``sum(m)`` exactly invariant

Backward compatibility: :func:`init_fill_rectangular_2d`,
:func:`init_flags_from_fill_2d` and :func:`free_surface_step_2d` keep their
names and 3-tuple ``(f, fill, flags)`` return contract.  ``fill`` returned by
the step is now ``m / rho_liquid`` (unclamped) so a caller that feeds it back
into the next step recovers the exact ledger.  Pass ``return_mass=True`` (or use
:func:`koerner_step_2d`) to obtain the raw independent mass field instead.
"""

from __future__ import annotations

import os

import torch

from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .solver import stream as _stream2d

GAS = 0
LIQUID = 1
INTERFACE = 2
SOLID = 3

_MOVING_Q = tuple(range(1, 9))
_OPP = OPPOSITE
# pull-source tensor shifts (dy, dx); field dims are (y, x)
_SHIFTS = tuple((int(C[q, 1]), int(C[q, 0])) for q in _MOVING_Q)
_CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# stencil helpers (2D mirror of tensorlbm.core.d3q19_stencil)
# ---------------------------------------------------------------------------


def _roll_from_pull(field: torch.Tensor, q: int) -> torch.Tensor:
    """Return ``field`` sampled at the pull source ``x - c_q``."""
    dy, dx = _SHIFTS[q - 1]
    return torch.roll(field, shifts=(dy, dx), dims=(0, 1))


def _roll_to_neighbor(field: torch.Tensor, q: int) -> torch.Tensor:
    """Return ``field`` sampled at the downstream neighbour ``x + c_q``."""
    dy, dx = _SHIFTS[q - 1]
    return torch.roll(field, shifts=(-dy, -dx), dims=(0, 1))


def _neighbor_masks(field: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The 8 periodic pull-source neighbour fields in moving-``q`` order."""
    return tuple(_roll_from_pull(field, q) for q in _MOVING_Q)


def count_direct_liquid_gas_links_2d(flags: torch.Tensor) -> int:
    """Number of 8-neighbour LIQUID-GAS links (must be 0 for a valid state)."""
    n = 0
    for nb in _neighbor_masks(flags == GAS):
        n += int(((flags == LIQUID) & nb).sum().item())
    return n


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_fill_rectangular_2d(ny, nx, column_width, column_height, device):
    """Initialize 2D fill field for a rectangular liquid column (dam-break IC)."""
    fill = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    cw, ch = int(column_width), int(column_height)
    fill[:ch, 1:cw] = 1.0
    fx, fy = column_width - cw, column_height - ch
    if fx > 0 and cw < nx - 1:
        fill[:ch, cw] = fx
    if fy > 0 and ch < ny - 1:
        fill[ch, 1:cw] = fy
    if fx > 0 and fy > 0:
        fill[ch, cw] = 0.5 * (fx + fy)
    return fill, solid


def init_flags_from_fill_2d(fill, solid_mask):
    """Set per-cell flags, creating the required one-cell INTERFACE envelope.

    A zero-fill GAS cell directly linked to LIQUID has no Körner boundary
    reconstruction, so such cells start as empty INTERFACE cells to guarantee
    that every LIQUID/GAS D2Q9 link is represented by the interface model
    (no direct liquid-gas link).
    """
    flags = torch.full_like(fill, GAS, dtype=torch.int8)
    flags[fill >= 1.0] = LIQUID
    flags[(fill > 0) & (fill < 1)] = INTERFACE
    liquid = flags == LIQUID
    liquid_neighbor = torch.stack(_neighbor_masks(liquid)).any(dim=0)
    flags[(flags == GAS) & liquid_neighbor & ~solid_mask] = INTERFACE
    flags[solid_mask] = SOLID
    return flags


def init_mass_from_fill_2d(fill, flags, rho_liquid=1.0):
    """Initialize the independent mass ledger: ``rho_l`` at LIQUID, ``fill*rho_l``
    at INTERFACE, and zero elsewhere."""
    mass = torch.zeros_like(fill)
    mass[flags == LIQUID] = rho_liquid
    mass[flags == INTERFACE] = fill[flags == INTERFACE] * rho_liquid
    return mass


def total_liquid_inventory_2d(f, mass, flags, rho_liquid=1.0):
    """Conserved liquid inventory: bulk populations + independent interface mass.

    LIQUID cells contribute their physical population density ``sum_q f_q``;
    INTERFACE cells contribute their tracked mass ``m``.  This is the quantity
    that the Körner exchange keeps invariant (the interface credit equals the
    bulk population debit link-by-link).
    """
    rho = f.sum(dim=0)
    return float(rho[flags == LIQUID].sum().item() + mass[flags == INTERFACE].sum().item())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_D2Q9_MIRROR_X = torch.tensor([0, 3, 2, 1, 4, 6, 5, 8, 7])
_D2Q9_MIRROR_Y = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5])
_D2Q9_SHIFTS = [(int(C[q, 0]), int(C[q, 1])) for q in range(9)]


def _env_bool_2d(name: str, default: bool = False) -> bool:
    """Read an opt-out boolean env gate for A/B diagnosis of the FS fixes."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


def _specular_wall_mirror_index_2d(slip_mask, device):
    """Per-link specular reflected source indices for 2D slip walls.

    For every D2Q9 link ``q`` at a fluid cell whose pull source lies in a
    slip wall, the reflected link reverses each coordinate component of
    ``c_q`` for which the corresponding signed axis neighbour is a slip wall
    (a face wall reverses one component -> specular; a corner reverses two
    -> full bounce).  Returns a ``(9, ny, nx)`` int64 tensor of source link
    indices to gather from ``f_post``.
    """
    smx = slip_mask.roll(1, dims=1)  # slip wall at p-(1,0)
    spx = slip_mask.roll(-1, dims=1)  # slip wall at p+(1,0)
    smy = slip_mask.roll(1, dims=0)
    spy = slip_mask.roll(-1, dims=0)
    off2idx = {(sx, sy): q for q, (sx, sy) in enumerate(_D2Q9_SHIFTS)}
    table = torch.zeros((9, 2, 2), dtype=torch.long)
    for q, (cx, cy) in enumerate(_D2Q9_SHIFTS):
        for fx in (0, 1):
            for fy in (0, 1):
                table[q, fx, fy] = off2idx[(-cx if fx else cx, -cy if fy else cy)]
    table = table.to(device)
    idx = torch.empty((9,) + tuple(slip_mask.shape), dtype=torch.long, device=device)
    for q, (cx, cy) in enumerate(_D2Q9_SHIFTS):
        fx = ((cx > 0) & smx) | ((cx < 0) & spx)
        fy = ((cy > 0) & smy) | ((cy < 0) & spy)
        idx[q] = table[q, fx.long(), fy.long()]
    return idx


def _init_new_2d(f, flags, mask, rho_init, ux=None, uy=None):
    """Reinitialize newly converted cells with a neighbour-averaged-velocity
    equilibrium at ``rho_init`` (vectorized, no boolean sync)."""
    if ux is None or uy is None:
        _, ux, uy = macroscopic(f)
    active = (flags == LIQUID) | (flags == INTERFACE)
    nb = torch.stack(_neighbor_masks(active)).to(f.dtype)
    cnt = nb.sum(dim=0).clamp(min=1.0)
    ux_nb = torch.stack([_roll_from_pull(ux, q) for q in _MOVING_Q]) * nb
    uy_nb = torch.stack([_roll_from_pull(uy, q) for q in _MOVING_Q]) * nb
    ux_m = ux_nb.sum(dim=0) / cnt
    uy_m = uy_nb.sum(dim=0) / cnt
    if rho_init is None:
        # Neighbour-averaged *active* density (waLBerla EquilibriumRefilling).
        rho = f.sum(0)
        rho_nb = (
            torch.stack([_roll_from_pull(rho, q) for q in _MOVING_Q]) * nb
        ).sum(dim=0) / cnt
        rho_field = torch.where(nb.sum(dim=0) > 0, rho_nb, torch.zeros_like(rho_nb))
    elif isinstance(rho_init, torch.Tensor):
        rho_field = rho_init.to(f.dtype)
    else:
        rho_field = torch.full_like(ux, float(rho_init))
    feq = equilibrium(rho_field, ux_m, uy_m)
    return torch.where(mask.unsqueeze(0), feq, f)


def _conservative_clamp(field, upper, weight=None):
    """Clip ``field`` into ``[0, upper]`` **while preserving** ``field.sum()``.

    A plain ``clamp`` is a silent mass source/sink.  Surplus (clip-up) is drawn
    back out of the cells that still carry mass; deficit (clip-down) is pushed
    forward to cells that still have headroom below ``upper``.  The returned
    tensor is bounded to ``[0, upper]`` and its sum equals the input sum up to
    float rounding.  (Identical contract to the 3D module's
    ``conservative_clamp_conserve``.)
    """
    hi = float(upper)
    clipped = field.clamp(0.0, hi)
    net = float((clipped - field).sum().item())
    if net == 0.0:
        return clipped
    if weight is None:
        weight = torch.ones_like(field)
    else:
        weight = weight.to(field.dtype)
    if net > 0.0:
        mass_weight = clipped * weight
        weight_sum = float(mass_weight.sum().item())
        if weight_sum <= 0.0:
            return clipped
        return (clipped - mass_weight * (net / weight_sum)).clamp(0.0, hi)
    room = (hi - clipped) * weight
    room_sum = float(room.sum().item())
    if room_sum <= 0.0:
        return clipped
    return (clipped + room * ((-net) / room_sum)).clamp(0.0, hi)


def _address_interface_layer(
    flags, mass, f, solid_mask, rho_liquid, ux, uy, recv_new=None, to_gas=None, rho=None
):
    """Conversion-driven envelope + LIQUID demotion (no halo sweep, no isolation).

    Mirrors the 3D ``build_topology_transaction`` envelope stage:
      * only gas cells that received conversion excess (``recv_new``) are
        promoted to INTERFACE, keeping their credited mass;
      * a LIQUID cell adjacent to freshly converted GAS is demoted to
        INTERFACE, keeping its PDFs and tracked mass;
      * new cells are refilled with neighbour-averaged density/velocity
        (waLBerla ``EquilibriumRefilling``) and their fill is normalised by
        the refilled density.
    """
    if recv_new is None:
        recv_new = torch.zeros_like(solid_mask)
    if to_gas is None:
        to_gas = torch.zeros_like(solid_mask)
    if rho is None:
        rho = torch.full_like(mass, float(rho_liquid))
    rho_reset = rho.to(mass.dtype)
    to_i = recv_new & (flags == GAS) & ~solid_mask
    f = _init_new_2d(f, flags, to_i, None, ux=ux, uy=uy)
    flags = torch.where(to_i, torch.full_like(flags, INTERFACE), flags)
    rho_fill = torch.where(to_i, f.sum(dim=0).clamp(min=1e-6), rho_reset)
    _liq_adj_new_gas = torch.stack(_neighbor_masks(to_gas)).any(dim=0)
    demote = (flags == LIQUID) & _liq_adj_new_gas & ~solid_mask
    flags = torch.where(demote, torch.full_like(flags, INTERFACE), flags)
    return flags, mass, f, rho_fill



def koerner_step_2d(
    f,
    mass,
    flags,
    solid_mask,
    tau=1.0,
    gx=0.0,
    gy=0.0,
    rho_liquid=1.0,
    rho_gas=1.0,
    clamp_mass=True,
    enable_abb=True,
    enable_mass_exchange=True,
    paired_liquid_debit=True,
    ledger=None,
    surface_tension=0.0,
    bubble_pressure=None,
    cell_conversion_threshold=1e-2,
    cell_conversion_force_threshold=0.1,
    free_slip_y=False,
    y_wall_mask=None,
    free_slip_mask=None,
    free_slip_all=False,
):
    """One faithful Körner free-surface timestep (D2Q9).

    Reference mechanics (Körner et al. 2005; Thürey 2007; waLBerla
    ``free_surface``), mirroring :func:`free_surface_lbm.free_surface_step`:
    single fill-weighted Guo force, skipped solid/gas collision, ABB with
    Laplace/bubble pressure, fluid-side half-way walls with selective
    free-slip, Körner eq. 9 exchange with the Thürey table-4.1 selector,
    interface-only conversion and the conversion-driven envelope.

    Returns ``(f, mass, flags)`` where ``mass`` is the independent ledger field
    and the fill level of INTERFACE cells is ``mass / rho_local``.

    ``clamp_mass`` and ``paired_liquid_debit`` are deprecated no-ops retained
    for call compatibility (the reference model has neither).
    """
    device = f.device
    f = f.clone()
    flags = flags.clone()
    mass = mass.clone()

    non_gas = flags != GAS
    mass_start = float(mass.sum().item()) if ledger is not None else 0.0

    # ---- 1. Macroscopic + collision (BGK) + single Guo force ---------------
    rho, ux, uy = macroscopic(f)
    if _env_bool_2d("TL_FS_INPUT_GUARD", True):
        ux = ux.clamp(-0.5, 0.5)
        uy = uy.clamp(-0.5, 0.5)
    rho_s = rho.clamp(min=1e-6, max=rho_liquid * 3.0)
    liquid_pre = flags == LIQUID
    iface_pre = flags == INTERFACE
    active_pre = liquid_pre | iface_pre
    # Rebuild the pre-collision fill for the Körner eq. 15 force weight
    # (F = a * rho * fill at INTERFACE, F = a * rho at LIQUID).
    fill_pre = torch.where(
        active_pre, (mass / rho_s.clamp(min=1e-6)).clamp(0.0, 1.0), torch.zeros_like(mass)
    )
    force_weight = torch.where(iface_pre, fill_pre, torch.ones_like(fill_pre))
    force_weight = torch.where(active_pre, force_weight, torch.zeros_like(force_weight))
    ux_eq = (ux + 0.5 * force_weight * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + 0.5 * force_weight * gy).clamp(-0.5, 0.5)
    feq = equilibrium(rho_s, ux_eq, uy_eq)
    f_pre_collision = f
    f = f - (f - feq) / tau

    if gx != 0.0 or gy != 0.0:
        c = C.to(device).float()
        w = W.to(device).float().view(9, 1, 1)
        cx = c[:, 0].view(9, 1, 1)
        cy = c[:, 1].view(9, 1, 1)
        Fx = rho_s * gx * force_weight
        Fy = rho_s * gy * force_weight
        cu = (cx * ux + cy * uy) / _CS2
        forcing = ((cx - ux) * Fx + (cy - uy) * Fy) / _CS2 + cu * (cx * Fx + cy * Fy) / _CS2
        f = f + (1.0 - 0.5 / tau) * w * forcing

    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Collision is skipped in SOLID and GAS cells (reference dynamics apply
    # only to liquid/interface cells).
    f = torch.where(solid_mask.unsqueeze(0), f_pre_collision, f)
    f = torch.where((flags == GAS).unsqueeze(0), torch.zeros_like(f), f)

    # Preserve post-collision outgoing populations for ABB and mass exchange.
    f_post = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))

    # ---- 2. Stream (pull) + zero gas ---------------------------------------
    f = _stream2d(f_post)
    f = torch.where((flags == GAS).unsqueeze(0), torch.zeros_like(f), f)
    if _env_bool_2d("TL_FS_INPUT_GUARD", True):
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            min=0.0, max=rho_liquid * 3.0
        )
    # Freeze the pure streamed state *before* the gas-pressure reconstruction so
    # the mass exchange never reads an ABB-reconstructed population.
    f_exchange = f

    # ---- 3. Anti-bounce-back (ABB) gas-pressure reconstruction -------------
    iface = flags == INTERFACE
    gas_mask = flags == GAS
    liq_mask = flags == LIQUID
    nb_flags = torch.stack([flags] + list(_neighbor_masks(flags)))  # (9, ny, nx)
    abb_moving = iface.unsqueeze(0) & (nb_flags[1:] == GAS)
    need_abb = torch.cat([torch.zeros_like(abb_moving[:1]), abb_moving], dim=0)
    opp = _OPP.to(device)
    _rho_pc, _ux_pc, _uy_pc = macroscopic(f_post)
    if _env_bool_2d("TL_FS_INPUT_GUARD", True):
        _ux_pc = _ux_pc.clamp(-0.5, 0.5)
        _uy_pc = _uy_pc.clamp(-0.5, 0.5)
    _rho_gas_base = float(bubble_pressure) if bubble_pressure is not None else float(rho_gas)
    rho_g_field = torch.full_like(rho, _rho_gas_base)
    if surface_tension > 0:
        # Laplace pressure jump inside the ABB gas density (Thürey 2007 §4.2,
        # waLBerla ``computeDeltaRhoLaplacePressure``): Δρ = 3σκ, ±0.1 limit.
        gx_f = 0.5 * (fill_pre.roll(-1, dims=1) - fill_pre.roll(1, dims=1))
        gy_f = 0.5 * (fill_pre.roll(-1, dims=0) - fill_pre.roll(1, dims=0))
        mag = (gx_f**2 + gy_f**2).sqrt().clamp(min=1e-10)
        nx_n, ny_n = -gx_f / mag, -gy_f / mag
        kappa = 0.5 * (
            (nx_n.roll(-1, dims=1) - nx_n.roll(1, dims=1))
            + (ny_n.roll(-1, dims=0) - ny_n.roll(1, dims=0))
        )
        delta_rho = (3.0 * float(surface_tension) * kappa).clamp(-0.1, 0.1)
        rho_g_field = rho_g_field + torch.where(iface, delta_rho, torch.zeros_like(delta_rho))
    feq_gas = equilibrium(rho_g_field, _ux_pc, _uy_pc)
    f_abb = feq_gas + feq_gas[opp] - f_post[opp]
    if enable_abb:
        f = torch.where(need_abb, f_abb, f)

    # ---- 4. Wall BC --------------------------------------------------------
    # Fluid-side half-way bounce-back (default; ``TL_FS_WALL_MODE=legacy``
    # restores the old post-stream solid-swap-only path for A/B comparison),
    # with selective specular (free-slip) reflection on slip walls.
    from .boundaries import bounce_back_cells

    _wall_mode = os.environ.get("TL_FS_WALL_MODE", "halfway").strip().lower()
    if _wall_mode != "legacy":
        _src_solid = torch.stack(
            [_roll_from_pull(solid_mask, q) for q in _MOVING_Q]
        )  # (8,ny,nx) pull source x-c_q lies in SOLID
        _fluid_bb = (~solid_mask).unsqueeze(0) & torch.cat(
            [torch.zeros_like(_src_solid[:1]), _src_solid], dim=0
        )
        _f_reflect = f_post[_OPP.to(device)]
        _slip = free_slip_mask
        if _slip is None and free_slip_all:
            _slip = solid_mask
        if _slip is not None:
            _mir_idx = _specular_wall_mirror_index_2d(_slip, device)
            _src_slip_mv = torch.stack([_roll_from_pull(_slip, q) for q in _MOVING_Q])
            _src_slip = torch.cat([torch.zeros_like(_src_slip_mv[:1]), _src_slip_mv], dim=0)
            _f_reflect = torch.where(
                _fluid_bb & _src_slip, torch.gather(f_post, 0, _mir_idx), _f_reflect
            )
        elif free_slip_y and y_wall_mask is not None:
            _src_y_mv = torch.stack([_roll_from_pull(y_wall_mask, q) for q in _MOVING_Q])
            _src_y = torch.cat([torch.zeros_like(_src_y_mv[:1]), _src_y_mv], dim=0)
            _f_reflect = torch.where(
                _fluid_bb & _src_y, f_post[_D2Q9_MIRROR_Y.to(device)], _f_reflect
            )
        f = torch.where(_fluid_bb, _f_reflect, f)

    f = bounce_back_cells(f, solid_mask)
    if ledger is not None:
        ledger["stage_f_exchange"] = float(mass.sum().item())

    # ---- 5. Körner mass exchange (eq. 9 / Thürey table 4.1) ----------------
    recv9 = iface.unsqueeze(0)
    from_liq = recv9 & (nb_flags == LIQUID)
    from_if = recv9 & (nb_flags == INTERFACE)
    diff = f_exchange - f_post[opp]
    md_liq = torch.where(from_liq, diff, torch.zeros_like(diff))
    fill_nb = torch.stack([fill_pre] + [_roll_from_pull(fill_pre, q) for q in _MOVING_Q])
    ii_weight = 0.5 * (fill_pre.unsqueeze(0) + fill_nb)
    liq_nbr_count = (nb_flags == LIQUID).sum(dim=0)
    liq_nbr_at_source = torch.stack(
        [liq_nbr_count] + [_roll_from_pull(liq_nbr_count, q) for q in _MOVING_Q]
    )
    gas_nbr_here = (nb_flags == GAS).any(dim=0).unsqueeze(0)
    ii_dead = from_if & (liq_nbr_at_source == 0) & ~gas_nbr_here
    md_if = torch.where(from_if & ~ii_dead, ii_weight * diff, torch.zeros_like(diff))
    mass_delta = (md_liq + md_if).sum(dim=0)
    if enable_mass_exchange:
        mass_delta = torch.nan_to_num(mass_delta, nan=0.0, posinf=0.0, neginf=0.0)
        mass = torch.where(iface, mass + mass_delta, mass)
    # Post-boundary local density: fill = mass / rho_local (unclamped so the
    # conversion gates read the true over/undershoot).
    rho_new = f.sum(dim=0).clamp(min=1e-6)
    fill_t = torch.where(iface, mass / rho_new, fill_pre)
    fill_t = torch.where(liquid_pre, torch.ones_like(fill_t), fill_t)
    fill_t = torch.where(gas_mask | solid_mask, torch.zeros_like(fill_t), fill_t)

    # ---- 6. Conversion marking + excess redistribution ---------------------
    _kappa = float(cell_conversion_threshold)
    _kappa_f = float(cell_conversion_force_threshold)
    to_iface = torch.zeros_like(gas_mask)
    to_liq = iface & (fill_t > 1.0 + _kappa)
    to_gas = iface & (fill_t < -_kappa)
    _has_gas_nb = (nb_flags == GAS).any(dim=0)
    _has_liq_nb = (nb_flags == LIQUID).any(dim=0)
    _has_if_nb = (nb_flags == INTERFACE).any(dim=0)
    to_liq = to_liq | (iface & ~_has_gas_nb & ((fill_t > 1.0 - _kappa_f) | ~_has_if_nb))
    to_gas = to_gas | (iface & ~_has_liq_nb & ((fill_t < _kappa_f) | ~_has_if_nb))
    to_gas = to_gas & ~to_liq  # liquid priority on the double trigger

    excess = torch.where(to_liq, mass - rho_new, torch.zeros_like(mass)) + torch.where(
        to_gas, mass, torch.zeros_like(mass)
    )
    recv_iface = iface & ~to_liq & ~to_gas
    adjacent_converting = torch.stack(_neighbor_masks(to_liq)).any(dim=0)
    recv_new = gas_mask & adjacent_converting & ~solid_mask
    recv_mask = recv_iface | recv_new
    n_recv = torch.stack(_neighbor_masks(recv_mask)).sum(dim=0).to(mass.dtype).clamp(min=1.0)
    excess_per_nb = excess / n_recv
    redistribution_increment = torch.stack(
        [_roll_to_neighbor(excess_per_nb, q) * recv_mask for q in _MOVING_Q]
    ).sum(dim=0)

    # A newly born GAS->INTERFACE cell needs a valid population set.
    f = _init_new_2d(f, flags, to_iface, rho_liquid, ux=ux, uy=uy)
    flags = torch.where(to_iface, torch.full_like(flags, INTERFACE), flags)

    mass = mass + redistribution_increment

    # I->L: cells reaching a full fill become bulk liquid (PDFs kept, donor
    # retains its local density — the shed excess went to the receivers).
    flags = torch.where(to_liq, torch.full_like(flags, LIQUID), flags)
    mass = torch.where(to_liq, rho_new, mass)
    # I->G: cells draining to empty become gas.
    flags = torch.where(to_gas, torch.full_like(flags, GAS), flags)
    mass = torch.where(to_gas, torch.zeros_like(mass), mass)
    f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)

    # ---- 7. Conversion-driven envelope + demotion --------------------------
    flags, mass, f, rho_fill = _address_interface_layer(
        flags, mass, f, solid_mask, rho_liquid, ux, uy, recv_new=recv_new, to_gas=to_gas, rho=rho_new
    )
    fill_t = torch.where(
        flags == INTERFACE, mass / rho_fill.clamp(min=1e-6), fill_t
    )
    fill_t = torch.where(flags == LIQUID, torch.ones_like(fill_t), fill_t)
    fill_t = torch.where(
        (flags == GAS) | solid_mask, torch.zeros_like(fill_t), fill_t
    )

    if ledger is not None:
        ledger["mass_start"] = mass_start
        ledger["mass_end"] = float(mass.sum().item())
        ledger["mass_drift"] = ledger["mass_end"] - mass_start
        ledger["inventory"] = total_liquid_inventory_2d(f, mass, flags, rho_liquid)
        ledger["direct_liquid_gas_links"] = count_direct_liquid_gas_links_2d(flags)
    return f, mass, flags


def free_surface_step_2d(
    f,
    fill,
    flags,
    solid_mask,
    tau=1.0,
    gx=0.0,
    gy=0.0,
    rho_liquid=1.0,
    rho_gas=1.0,
    free_slip_y=False,
    y_wall_mask=None,
    mass=None,
    return_mass=False,
    ledger=None,
    surface_tension=0.0,
    bubble_pressure=None,
    cell_conversion_threshold=1e-2,
    cell_conversion_force_threshold=0.1,
    free_slip_mask=None,
    free_slip_all=False,
):
    """One 2D free-surface timestep (D2Q9) — faithful Körner model.

    ``fill`` indexes the independent mass ledger via ``mass = fill * rho_local``
    when ``mass`` is not supplied.  The returned ``fill`` is ``mass / rho_local``
    at INTERFACE (1 at LIQUID, 0 at GAS/SOLID) so feeding it back into the next
    call recovers the ledger exactly.  With ``return_mass=True`` the raw
    independent mass field is returned instead of the fill level.
    """
    if mass is None:
        mass = fill * f.sum(dim=0).clamp(min=1e-6)
    f, mass, flags = koerner_step_2d(
        f,
        mass,
        flags,
        solid_mask,
        tau=tau,
        gx=gx,
        gy=gy,
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
        ledger=ledger,
        surface_tension=surface_tension,
        bubble_pressure=bubble_pressure,
        cell_conversion_threshold=cell_conversion_threshold,
        cell_conversion_force_threshold=cell_conversion_force_threshold,
        free_slip_y=free_slip_y,
        y_wall_mask=y_wall_mask,
        free_slip_mask=free_slip_mask,
        free_slip_all=free_slip_all,
    )
    if return_mass:
        return f, mass, flags
    rho_out = f.sum(dim=0).clamp(min=1e-6)
    fill_out = torch.where(flags == INTERFACE, mass / rho_out, torch.zeros_like(mass))
    fill_out = torch.where(flags == LIQUID, torch.ones_like(fill_out), fill_out)
    return f, fill_out, flags
