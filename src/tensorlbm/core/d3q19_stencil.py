"""Canonical tensor-coordinate traversal for the production D3Q19 lattice.

The production lattice descriptor stores velocities in ``(x, y, z)`` order,
while solver fields are indexed ``(z, y, x)``.  This module is the sole
domain-neutral bridge between those two conventions for moving D3Q19 links.

The traversal itself is lattice-parameterised and lives in
:mod:`tensorlbm.core._stencil_base`, shared with the D3Q27 stencil; this module
binds it to the production D3Q19 descriptor and owns the public names.
"""

from __future__ import annotations

import torch

from ..d3q19 import C
from ._stencil_base import build_stencil_ops

D3Q19_MOVING_Q = tuple(range(1, 19))

# Validates the descriptor at import time and precomputes the 18 shifts.
_OPS = build_stencil_ops(C, 19, "D3Q19")
_MOVING_TENSOR_SHIFTS = _OPS.shifts


def moving_tensor_shifts() -> tuple[tuple[int, int, int], ...]:
    """Return moving D3Q19 shifts in field order ``(dz, dy, dx)``."""
    return _MOVING_TENSOR_SHIFTS


def tensor_shift_for_q(q: int) -> tuple[int, int, int]:
    """Return the pull-source tensor shift for one moving direction."""
    return _OPS.tensor_shift_for_q(q)


def roll_from_pull_source(field: torch.Tensor, q: int) -> torch.Tensor:
    """Roll a field from the periodic pull source for moving direction ``q``."""
    return _OPS.roll_from_pull_source(field, q)


def roll_to_neighbor(field: torch.Tensor, q: int) -> torch.Tensor:
    """Roll a donor field to its periodic neighbour along moving direction ``q``."""
    return _OPS.roll_to_neighbor(field, q)


def all_moving_neighbor_masks(mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the 18 periodic pull-source neighbour masks in q order."""
    return _OPS.all_moving_neighbor_masks(mask)


def assert_no_direct_phase_links(
    flags: torch.Tensor,
    source_flag: int,
    target_flag: int,
    error_prefix: str,
) -> None:
    """Reject any moving D3Q19 link from ``source_flag`` to ``target_flag``."""
    _OPS.assert_no_direct_phase_links(flags, source_flag, target_flag, error_prefix)
