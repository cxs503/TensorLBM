"""Lattice-parameterised tensor-coordinate traversal shared by the D3Q* stencils.

:mod:`tensorlbm.core.d3q19_stencil` and :mod:`tensorlbm.core.d3q27_stencil`
were line-for-line copies differing only in the lattice size, the descriptor
they import and the ``D3Q19``/``D3Q27`` text in their messages — the second was
introduced as a copy of the first when D3Q27 free surface landed, so every fix
had to be applied twice and the two could silently drift.  The traversal logic
lives here once, parameterised by the lattice descriptor; the two modules stay
as the public entry points and keep their own names and error strings.

The shared concern is the coordinate-order bridge: lattice descriptors store
velocities in ``(x, y, z)`` order while solver fields are indexed ``(z, y, x)``,
so a link's tensor shift is the reversed velocity row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TensorShift = tuple[int, int, int]


def validate_descriptor(c: torch.Tensor, q: int, label: str) -> None:
    """Fail closed if descriptor ``c`` is no longer the expected ``label`` lattice.

    Runs at import time in each stencil module: these helpers index ``C`` by
    position, so a descriptor that changed shape or direction order would
    silently produce wrong neighbours rather than raising.
    """
    if c.shape != (q, 3):
        raise ValueError(f"production {label} C must have shape ({q}, 3), got {tuple(c.shape)}")
    if not torch.equal(c[0], torch.zeros(3, dtype=c.dtype, device=c.device)):
        raise ValueError(f"production {label} C[0] must be the rest direction")
    if bool((c[1:].abs().sum(dim=1) == 0).any()):
        raise ValueError(f"production {label} moving directions must be nonzero")


@dataclass(frozen=True)
class StencilOps:
    """Traversal helpers bound to one lattice.

    Built by :func:`build_stencil_ops`; the stencil modules re-export the
    attributes under their public names so call sites are unchanged.
    """

    label: str
    moving_q: tuple[int, ...]
    shifts: tuple[TensorShift, ...]

    # -- guards ------------------------------------------------------------
    def require_field_3d(self, field: torch.Tensor) -> None:
        if not isinstance(field, torch.Tensor):
            raise TypeError(
                f"{self.label} tensor field must be a torch.Tensor, got {type(field).__name__}"
            )
        if field.ndim != 3:
            raise ValueError(
                f"{self.label} tensor field must have exactly three dimensions (z, y, x), "
                f"got {field.ndim}"
            )

    def require_moving_q(self, q: int) -> None:
        if isinstance(q, bool) or not isinstance(q, int):
            raise TypeError(f"{self.label} moving q must be a non-bool int, got {type(q).__name__}")
        if q not in self.moving_q:
            raise ValueError(f"{self.label} moving q must be in [1, {self.moving_q[-1]}], got {q}")

    # -- traversal ---------------------------------------------------------
    def tensor_shift_for_q(self, q: int) -> TensorShift:
        """Pull-source tensor shift ``(dz, dy, dx)`` for one moving direction."""
        self.require_moving_q(q)
        return self.shifts[q - 1]

    def roll_from_pull_source(self, field: torch.Tensor, q: int) -> torch.Tensor:
        """Roll a field from the periodic pull source for moving direction ``q``."""
        self.require_field_3d(field)
        return torch.roll(field, shifts=self.tensor_shift_for_q(q), dims=(0, 1, 2))

    def roll_to_neighbor(self, field: torch.Tensor, q: int) -> torch.Tensor:
        """Roll a donor field to its periodic neighbour along moving direction ``q``."""
        self.require_field_3d(field)
        return torch.roll(
            field, shifts=tuple(-delta for delta in self.tensor_shift_for_q(q)), dims=(0, 1, 2)
        )

    def all_moving_neighbor_masks(self, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Periodic pull-source neighbour masks for every moving direction, in q order."""
        self.require_field_3d(mask)
        return tuple(self.roll_from_pull_source(mask, q) for q in self.moving_q)

    def assert_no_direct_phase_links(
        self,
        flags: torch.Tensor,
        source_flag: int,
        target_flag: int,
        error_prefix: str,
    ) -> None:
        """Reject any moving link from ``source_flag`` to ``target_flag``."""
        self.require_field_3d(flags)
        source = flags == source_flag
        count = sum(
            int((source & target).sum().item())
            for target in self.all_moving_neighbor_masks(flags == target_flag)
        )
        if count:
            raise ValueError(f"{error_prefix}: found {count} direct phase link(s)")


def build_stencil_ops(c: torch.Tensor, q: int, label: str) -> StencilOps:
    """Validate descriptor ``c`` and precompute the moving tensor shifts for it."""
    validate_descriptor(c, q, label)
    moving_q = tuple(range(1, q))
    # (x, y, z) descriptor row -> (dz, dy, dx) field-order shift.
    shifts = tuple((int(c[i, 2]), int(c[i, 1]), int(c[i, 0])) for i in moving_q)
    return StencilOps(label=label, moving_q=moving_q, shifts=shifts)
