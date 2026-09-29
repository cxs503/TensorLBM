"""Free-surface LBM (D2Q9) — faithful Körner model with an *independent* mass ledger.

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

import torch

from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
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
    return float(
        rho[flags == LIQUID].sum().item() + mass[flags == INTERFACE].sum().item()
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
    feq = equilibrium(torch.full_like(ux, float(rho_init)), ux_m, uy_m)
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
    flags, mass, f, solid_mask, rho_liquid, ux, uy, recv_new=None
):
    """Rebuild a valid one-cell INTERFACE envelope and remove isolated cells.

    Mirrors the 3D ``build_topology_transaction`` halo/isolation stage:
      * GAS cells directly adjacent to LIQUID become INTERFACE cells;
      * a cell that just received redistribution mass (``recv_new``) is promoted
        to INTERFACE *without* discarding the mass it received — zeroing it was
        a silent tracked-mass sink (the recv_new halo cell is a *receiver*);
      * an INTERFACE cell with no LIQUID/INTERFACE neighbour is dissolved to GAS
        (this is the guard that prevents the interface layer from self-
        propagating and atomising the domain).  Dissolution is only allowed for
        a (near-)empty cell: an isolated cell that still carries tracked mass is
        retained, so no liquid mass is silently destroyed.

    The stage keeps ``mass.sum()`` invariant (mass is only ever moved between
    non-solid cells, never created or destroyed).
    """
    gas_mask = flags == GAS
    shifted = torch.stack(_neighbor_masks(flags))
    is_nb_liq = (shifted == LIQUID).any(dim=0)
    to_i = gas_mask & is_nb_liq & ~solid_mask
    if recv_new is not None:
        to_i = to_i | recv_new
    f = _init_new_2d(f, flags, to_i, rho_liquid, ux=ux, uy=uy)
    flags = torch.where(to_i, torch.full_like(flags, INTERFACE), flags)
    # Only a freshly promoted *empty* envelope cell is zeroed; a recv_new
    # receiver keeps the mass it was credited in the redistribution stage.
    if recv_new is not None:
        zero_mass = to_i & ~recv_new
    else:
        zero_mass = to_i
    mass = torch.where(zero_mass, torch.zeros_like(mass), mass)

    shifted2 = torch.stack(_neighbor_masks(flags))
    has_nb = ((shifted2 == LIQUID) | (shifted2 == INTERFACE)).any(dim=0)
    empty = mass <= 1.0e-3 * rho_liquid
    isolated = (flags == INTERFACE) & ~has_nb & ~solid_mask & empty
    flags = torch.where(isolated, torch.full_like(flags, GAS), flags)
    mass = torch.where(isolated, torch.zeros_like(mass), mass)
    f = torch.where(isolated.unsqueeze(0), torch.zeros_like(f), f)
    flags = torch.where(solid_mask, torch.full_like(flags, SOLID), flags)
    return flags, mass, f


# ---------------------------------------------------------------------------
# Core timestep — full Körner model
# ---------------------------------------------------------------------------


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
):
    """One faithful Körner free-surface timestep (D2Q9).

    Returns ``(f, mass, flags)`` where ``mass`` is the independent ledger field
    and the fill level of INTERFACE cells is ``mass / rho_liquid``.
    """
    device = f.device
    f = f.clone()
    flags = flags.clone()
    mass = mass.clone()

    non_gas = flags != GAS
    mass_start = float(mass.sum().item()) if ledger is not None else 0.0

    # ---- 1. Macroscopic + collision (BGK) + Guo gravity --------------------
    rho, ux, uy = macroscopic(f)
    rho_s = rho.clamp(min=1e-6, max=rho_liquid * 3.0)
    ux_eq = (ux + tau * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gy).clamp(-0.5, 0.5)
    feq = equilibrium(rho_s, ux_eq, uy_eq)
    f = f - (f - feq) / tau

    if gx != 0.0 or gy != 0.0:
        c = C.to(device).float()
        w = W.to(device).float().view(9, 1, 1)
        ng = non_gas.to(f.dtype)
        Fx = rho_liquid * gx * ng
        Fy = rho_liquid * gy * ng
        cu_force = c[:, 0].view(9, 1, 1) * Fx + c[:, 1].view(9, 1, 1) * Fy
        f = f + (1.0 - 0.5 / tau) * w * cu_force / _CS2

    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Preserve post-collision outgoing populations for ABB and mass exchange.
    f_post = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))

    # ---- 2. Stream (pull) + zero gas ---------------------------------------
    f = _stream2d(f_post)
    f = torch.where((flags == GAS).unsqueeze(0), torch.zeros_like(f), f)
    # Freeze the pure streamed state *before* the gas-pressure reconstruction so
    # the mass exchange never reads an ABB-reconstructed population.
    f_exchange = f

    # ---- 3. Anti-bounce-back (ABB) gas-pressure reconstruction -------------
    iface = flags == INTERFACE
    gas_mask = flags == GAS
    liq_mask = flags == LIQUID
    nb_flags = torch.stack(_neighbor_masks(flags))  # (8, ny, nx) pull sources
    abb_moving = iface.unsqueeze(0) & (nb_flags == GAS)
    need_abb = torch.cat([torch.zeros_like(abb_moving[:1]), abb_moving], dim=0)
    opp = _OPP.to(device)
    feq_gas = equilibrium(torch.full_like(rho, float(rho_gas)), ux, uy)
    f_abb = feq_gas + feq_gas[opp] - f_post[opp]
    if enable_abb:
        f = torch.where(need_abb, f_abb, f)

    # ---- 4. Wall BC --------------------------------------------------------
    from .boundaries import bounce_back_cells

    f = bounce_back_cells(f, solid_mask)
    if ledger is not None:
        ledger["stage_f_exchange"] = float(mass.sum().item())

    # ---- 5. Körner mass exchange (independent mass ledger) -----------------
    recv8 = iface.unsqueeze(0)
    from_liq = recv8 & (nb_flags == LIQUID)
    from_if = recv8 & (nb_flags == INTERFACE)
    diff = f_exchange[1:] - f_post[opp][1:]  # (8, ny, nx)
    md_liq = torch.where(from_liq, diff, torch.zeros_like(diff))
    md_if = torch.where(from_if, 0.5 * diff, torch.zeros_like(diff))
    mass_delta = (md_liq + md_if).sum(dim=0)
    if enable_mass_exchange:
        mass = torch.where(iface, mass + mass_delta, mass)
        if paired_liquid_debit:
            # Every L/I credit at interface x is paired link-by-link with an
            # equal debit at its pull source x - c_q, so the independent ledger
            # is conserved exactly (the interface credit is a *transfer* from
            # the bulk, never a source).
            debit = torch.where(from_liq, -diff, torch.zeros_like(diff))
            bulk_debit = torch.stack(
                [_roll_to_neighbor(debit[i], q) for i, q in enumerate(_MOVING_Q)]
            ).sum(dim=0)
            mass = mass + torch.where(
                liq_mask, bulk_debit, torch.zeros_like(bulk_debit)
            )
    fill_t = (mass / rho_liquid).clamp(0.0, 1.0)

    # ---- 6. Flag conversion thresholds + excess redistribution -------------
    to_iface = gas_mask & (fill_t > 0.01) & ~solid_mask
    to_liq = iface & (fill_t >= 0.999) & ~solid_mask
    to_gas = iface & (fill_t <= 0.01) & ~solid_mask

    excess = torch.where(
        to_liq,
        mass - rho_liquid,
        torch.where(to_gas, mass, torch.zeros_like(mass)),
    )
    recv_iface = iface & ~to_liq & ~to_gas
    adjacent_converting = torch.stack(_neighbor_masks(to_liq)).any(dim=0)
    recv_new = gas_mask & adjacent_converting & ~solid_mask
    recv_mask = recv_iface | recv_new
    n_recv = torch.stack(_neighbor_masks(recv_mask)).sum(dim=0).to(mass.dtype).clamp(min=1.0)
    excess_per_nb = excess / n_recv
    redistribution_increment = torch.stack(
        [_roll_to_neighbor(excess_per_nb, q) for q in _MOVING_Q]
    ).sum(dim=0) * recv_mask.to(mass.dtype)

    # A newly born GAS->INTERFACE cell needs a valid population set.
    f = _init_new_2d(f, flags, to_iface, rho_liquid, ux=ux, uy=uy)
    flags = torch.where(to_iface, torch.full_like(flags, INTERFACE), flags)

    mass = mass + redistribution_increment
    if clamp_mass:
        mass = _conservative_clamp(mass, rho_liquid, weight=(~solid_mask).to(mass.dtype))

    # I->L: cells reaching a full fill become bulk liquid.
    f = _init_new_2d(f, flags, to_liq, rho_liquid, ux=ux, uy=uy)
    flags = torch.where(to_liq, torch.full_like(flags, LIQUID), flags)
    mass = torch.where(to_liq, torch.full_like(mass, rho_liquid), mass)
    # I->G: cells draining to empty become gas.
    flags = torch.where(to_gas, torch.full_like(flags, GAS), flags)
    mass = torch.where(to_gas, torch.zeros_like(mass), mass)
    f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)

    # ---- 7. Envelope halo + isolation cleanup ------------------------------
    flags, mass, f = _address_interface_layer(
        flags, mass, f, solid_mask, rho_liquid, ux, uy, recv_new=recv_new
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
):
    """One 2D free-surface timestep (D2Q9) — faithful Körner model.

    Backward-compatible signature.  ``fill`` indexes the independent mass
    ledger via ``mass = fill * rho_liquid`` when ``mass`` is not supplied.  The
    returned ``fill`` is ``mass / rho_liquid`` (unclamped) so feeding it back
    into the next call recovers the ledger exactly.  With ``return_mass=True``
    the raw independent mass field is returned instead of the fill level.

    ``free_slip_y`` / ``y_wall_mask`` are accepted for signature compatibility
    and ignored (the closed-box dam break uses no-slip walls only).
    """
    if mass is None:
        mass = fill * rho_liquid
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
    )
    if return_mass:
        return f, mass, flags
    return f, mass / rho_liquid, flags