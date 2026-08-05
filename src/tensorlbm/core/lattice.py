"""Backend-neutral descriptions of LBM velocity sets.

These immutable descriptors deliberately use only Python scalar values.  Tensor
backends adapt them into device arrays; this core package does not import a
framework or an existing framework-specific kernel.
"""

from __future__ import annotations

from dataclasses import dataclass


Direction = tuple[int, int, int]


@dataclass(frozen=True)
class LatticeDescriptor:
    """Metadata for a discrete velocity lattice owned by an existing kernel."""

    q: int
    directions: tuple[Direction, ...]
    weights: tuple[float, ...]
    opposite: tuple[int, ...]
    cs2: float


def _opposites(directions: tuple[Direction, ...]) -> tuple[int, ...]:
    """Return the unique index of each velocity's additive inverse."""
    index = {direction: position for position, direction in enumerate(directions)}
    try:
        return tuple(index[(-x, -y, -z)] for x, y, z in directions)
    except KeyError as error:  # pragma: no cover - protects future descriptors.
        raise ValueError("lattice directions must be closed under negation") from error


_D3Q19_DIRECTIONS: tuple[Direction, ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)
D3Q19 = LatticeDescriptor(
    q=19,
    directions=_D3Q19_DIRECTIONS,
    weights=(1.0 / 3.0,) + (1.0 / 18.0,) * 6 + (1.0 / 36.0,) * 12,
    opposite=_opposites(_D3Q19_DIRECTIONS),
    cs2=1.0 / 3.0,
)

_D3Q27_DIRECTIONS: tuple[Direction, ...] = tuple(
    (x, y, z) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
)
D3Q27 = LatticeDescriptor(
    q=27,
    directions=_D3Q27_DIRECTIONS,
    weights=tuple(
        8.0 / 27.0 if direction == (0, 0, 0) else
        2.0 / 27.0 if sum(component * component for component in direction) == 1 else
        1.0 / 54.0 if sum(component * component for component in direction) == 2 else
        1.0 / 216.0
        for direction in _D3Q27_DIRECTIONS
    ),
    opposite=_opposites(_D3Q27_DIRECTIONS),
    cs2=1.0 / 3.0,
)

__all__ = ["D3Q19", "D3Q27", "Direction", "LatticeDescriptor"]
