"""Canonical tensor-coordinate traversal for the D3Q27 lattice.

The D3Q27 lattice descriptor stores velocities in ``(x, y, z)`` order, while
solver fields are indexed ``(z, y, x)``.  This module is the sole
domain-neutral bridge between those two conventions for moving D3Q27 links.

It mirrors :mod:`tensorlbm.core.d3q19_stencil` but covers all 26 moving
directions of the D3Q27 stencil (6 face + 12 edge + 8 corner).  Both bind the
same lattice-parameterised implementation from
:mod:`tensorlbm.core._stencil_base`; only the descriptor and the public names
differ.
"""

from __future__ import annotations

import torch

from ..d3q27 import C
from ._stencil_base import build_stencil_ops

D3Q27_MOVING_Q = tuple(range(1, 27))

# Validates the descriptor at import time and precomputes the 26 shifts.
_OPS = build_stencil_ops(C, 27, "D3Q27")
_MOVING_TENSOR_SHIFTS = _OPS.shifts


def moving_tensor_shifts_27() -> tuple[tuple[int, int, int], ...]:
    """Return moving D3Q27 shifts in field order ``(dz, dy, dx)``."""
    return _MOVING_TENSOR_SHIFTS


def tensor_shift_for_q_27(q: int) -> tuple[int, int, int]:
    """Return the pull-source tensor shift for one moving direction."""
    return _OPS.tensor_shift_for_q(q)


def roll_from_pull_source_27(field: torch.Tensor, q: int) -> torch.Tensor:
    """Roll a field from the periodic pull source for moving direction ``q``."""
    return _OPS.roll_from_pull_source(field, q)


def roll_to_neighbor_27(field: torch.Tensor, q: int) -> torch.Tensor:
    """Roll a donor field to its periodic neighbour along moving direction ``q``."""
    return _OPS.roll_to_neighbor(field, q)


def all_moving_neighbor_masks_27(mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the 26 periodic pull-source neighbour masks in q order."""
    return _OPS.all_moving_neighbor_masks(mask)


def assert_no_direct_phase_links_27(
    flags: torch.Tensor,
    source_flag: int,
    target_flag: int,
    error_prefix: str,
) -> None:
    """Reject any moving D3Q27 link from ``source_flag`` to ``target_flag``."""
    _OPS.assert_no_direct_phase_links(flags, source_flag, target_flag, error_prefix)
