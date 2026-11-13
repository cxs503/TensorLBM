"""Repo-side transcription of the TensorLBM mass-exchange selector — anchor
``XVAL-LBM-MASS-EXCHANGE-SELECTOR``.

Transcribed from ``src/tensorlbm/free_surface_lbm_2d.py`` (line numbers are of
the branch this XVAL ran on) so that the differential test compares *rules*,
not whole solvers:

=============================  =============================================
repo location                  rule reproduced here
=============================  =============================================
``:604-605``                   ``_kappa = cell_conversion_threshold`` (1e-2),
                               ``_kappa_f = cell_conversion_force_threshold``
                               (0.1)
``:607-608``                   pure-fill pair: ``to_liq`` if ``fill > 1+kappa``,
                               ``to_gas`` if ``fill < -kappa``
``:610-614``                   neighbour-gated pair: no-GAS-neighbour promotion
                               at ``fill > 1-kappa_f``, no-LIQUID-neighbour
                               demotion at ``fill < kappa_f``, both also firing
                               when the cell has no INTERFACE neighbour;
                               LIQUID wins the double trigger
``:617-626``                   excess redistribution: the donor's overflow is
                               split **uniformly** over every receiver
                               (interface cells that do not convert + gas cells
                               that become interface), ``1/n_recv`` each
=============================  =============================================

Coordinates: ``(ny, nx)`` like the repo tensors, periodic ``np.roll`` like
``torch.roll`` in ``_roll_from_pull`` / ``_roll_to_neighbor``.
"""

from __future__ import annotations

import numpy as np

GAS, LIQUID, INTERFACE, SOLID = 0, 1, 2, 3

# the 8 moving D2Q9 offsets as (dy, dx); the rule set is symmetric so the
# enumeration order only has to be a consistent covering of the ring
OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))

DEFAULT_KAPPA = 1e-2        # :604  cell_conversion_threshold
DEFAULT_KAPPA_F = 0.1       # :605  cell_conversion_force_threshold


def neighbour_masks(mask):
    """The 8 periodic pull-source neighbour fields (``_neighbor_masks``)."""
    return [np.roll(mask, shift=(dy, dx), axis=(0, 1)) for dy, dx in OFFSETS]


def mark(flags, fill, kappa=DEFAULT_KAPPA, kappa_f=DEFAULT_KAPPA_F):
    """``:607-614`` — interface conversion decisions (no side effects)."""
    iface = flags == INTERFACE
    has_gas = np.zeros(flags.shape, dtype=bool)
    has_liq = np.zeros(flags.shape, dtype=bool)
    has_if = np.zeros(flags.shape, dtype=bool)
    for nb in neighbour_masks(flags):
        has_gas |= nb == GAS
        has_liq |= nb == LIQUID
        has_if |= nb == INTERFACE
    fill = np.asarray(fill, dtype=np.float64)
    to_liq = iface & (fill > 1.0 + kappa)
    to_gas = iface & (fill < -kappa)
    to_liq = to_liq | (iface & ~has_gas & ((fill > 1.0 - kappa_f) | ~has_if))
    to_gas = to_gas | (iface & ~has_liq & ((fill < kappa_f) | ~has_if))
    to_gas = to_gas & ~to_liq          # :614 liquid priority
    return to_liq, to_gas, {"has_liq": has_liq, "has_gas": has_gas,
                            "has_if": has_if}


def redistribute(flags, mass, rho_new, to_liq, to_gas):
    """``:617-626`` — uniform ``1/n_recv`` split of the donor's excess.

    ``recv_mask`` = interface cells that do not convert + gas cells adjacent
    to a converting cell (the soon-to-be-born interface cells).  The repo's
    ``n_recv >= 1`` clamp means a donor with no receiver keeps its excess
    (inventory-conserving).
    """
    flags = np.asarray(flags)
    mass = np.asarray(mass, dtype=np.float64)
    # :619 ``excess = where(to_liq, mass - rho_new, 0) + where(to_gas, mass, 0)``
    excess = (np.where(to_liq, mass - np.asarray(rho_new, dtype=np.float64), 0.0)
              + np.where(to_gas, mass, 0.0))
    recv_iface = (flags == INTERFACE) & ~to_liq & ~to_gas
    adjacent_converting = np.zeros(flags.shape, dtype=bool)
    for nb in neighbour_masks(to_liq):
        adjacent_converting |= nb
    recv_new = (flags == GAS) & adjacent_converting
    recv_mask = recv_iface | recv_new
    n_recv = np.zeros(flags.shape, dtype=np.float64)
    for nb in neighbour_masks(recv_mask):
        n_recv += nb
    n_recv = np.clip(n_recv, 1.0, None)
    excess_per_nb = excess / n_recv
    # ``_roll_to_neighbor(excess_per_nb, q) * recv_mask`` then summed over q:
    # each receiver gets one share per converting neighbour it is adjacent to
    increment = np.zeros(flags.shape, dtype=np.float64)
    for nb in neighbour_masks(excess_per_nb):
        increment += nb * recv_mask
    return increment, excess, recv_mask, n_recv