"""Canonical tensor-coordinate traversal for the D3Q27 lattice.

The D3Q27 lattice descriptor stores velocities in ``(x, y, z)`` order,
while solver fields are indexed ``(z, y, x)``.  This module is the sole
domain-neutral bridge between those two conventions for moving D3Q27 links.

It mirrors :mod:`tensorlbm.core.d3q19_stencil` but covers all 26 moving
directions of the D3Q27 stencil (6 face + 12 edge + 8 corner).
"""
from __future__ import annotations

import torch

from ..d3q27 import C

D3Q27_MOVING_Q = tuple(range(1, 27))


def _validate_production_d3q27() -> None:
    """Fail closed if the production descriptor no longer is D3Q27."""
    if C.shape != (27, 3):
        raise ValueError(f"production D3Q27 C must have shape (27, 3), got {tuple(C.shape)}")
    if not torch.equal(C[0], torch.zeros(3, dtype=C.dtype, device=C.device)):
        raise ValueError("production D3Q27 C[0] must be the rest direction")
    if bool((C[1:].abs().sum(dim=1) == 0).any()):
        raise ValueError("production D3Q27 moving directions must be nonzero")


_validate_production_d3q27()
_MOVING_TENSOR_SHIFTS = tuple(
    (int(C[q, 2]), int(C[q, 1]), int(C[q, 0])) for q in D3Q27_MOVING_Q
)


def _require_field_3d(field: torch.Tensor) -> None:
    if not isinstance(field, torch.Tensor):
        raise TypeError(f"D3Q27 tensor field must be a torch.Tensor, got {type(field).__name__}")
    if field.ndim != 3:
        raise ValueError(f"D3Q27 tensor field must have exactly three dimensions (z, y, x), got {field.ndim}")


def _require_moving_q(q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int):
        raise TypeError(f"D3Q27 moving q must be a non-bool int, got {type(q).__name__}")
    if q not in D3Q27_MOVING_Q:
        raise ValueError(f"D3Q27 moving q must be in [1, 26], got {q}")


def moving_tensor_shifts_27() -> tuple[tuple[int, int, int], ...]:
    """Return moving D3Q27 shifts in field order ``(dz, dy, dx)``."""
    return _MOVING_TENSOR_SHIFTS


def tensor_shift_for_q_27(q: int) -> tuple[int, int, int]:
    """Return the pull-source tensor shift for one moving direction."""
    _require_moving_q(q)
    return _MOVING_TENSOR_SHIFTS[q - 1]


def roll_from_pull_source_27(field: torch.Tensor, q: int) -> torch.Tensor:
    """Roll a field from the periodic pull source for moving direction ``q``."""
    _require_field_3d(field)
    return torch.roll(field, shifts=tensor_shift_for_q_27(q), dims=(0, 1, 2))


def roll_to_neighbor_27(field: torch.Tensor, q: int) -> torch.Tensor:
    """Roll a donor field to its periodic neighbour along moving direction ``q``."""
    _require_field_3d(field)
    return torch.roll(field, shifts=tuple(-delta for delta in tensor_shift_for_q_27(q)), dims=(0, 1, 2))


def all_moving_neighbor_masks_27(mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the 26 periodic pull-source neighbour masks in q order."""
    _require_field_3d(mask)
    return tuple(roll_from_pull_source_27(mask, q) for q in D3Q27_MOVING_Q)


def assert_no_direct_phase_links_27(
    flags: torch.Tensor,
    source_flag: int,
    target_flag: int,
    error_prefix: str,
) -> None:
    """Reject any moving D3Q27 link from ``source_flag`` to ``target_flag``."""
    _require_field_3d(flags)
    source = flags == source_flag
    count = sum(
        int((source & target).sum().item())
        for target in all_moving_neighbor_masks_27(flags == target_flag)
    )
    if count:
        raise ValueError(f"{error_prefix}: found {count} direct phase link(s)")
