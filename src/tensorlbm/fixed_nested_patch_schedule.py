"""Pure ownership and time schedule for one fixed 2:1 nested D3Q27 patch.

This module is deliberately metadata only: it allocates no populations and
performs no collision, streaming, transfer, or reflux.  Coordinates are
half-open integer cell extents.  The coarse extent is expressed on the coarse
lattice; the fine extent is expressed on a globally aligned fine lattice, so a
fine extent divided by two is its covered coarse volume.

The fine patch must be strictly interior to the coarse patch.  That restriction
makes every coarse/fine boundary a planar face and gives a complete, disjoint
coarse ownership partition.  Edge and corner coupling is intentionally not
represented: D3Q27 exchange/reflux work is scheduled only for its six planar
faces, which are the responsibilities of the existing planar-interface and
2:1 temporal-reflux primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_RATIO: Final = 2
_AXES: Final = 3


@dataclass(frozen=True, order=True)
class CellExtent3D:
    """A non-empty half-open, integer 3-D cell extent ``[lower, upper)``."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name, point in (("lower", self.lower), ("upper", self.upper)):
            if not isinstance(point, tuple) or len(point) != _AXES:
                raise TypeError(f"{name} must be a three-component integer tuple")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in point):
                raise TypeError(f"{name} must be a three-component integer tuple")
        if any(low >= high for low, high in zip(self.lower, self.upper, strict=True)):
            raise ValueError("cell extent must be non-empty in every dimension")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.upper[0] - self.lower[0],
            self.upper[1] - self.lower[1],
            self.upper[2] - self.lower[2],
        )

    @property
    def volume(self) -> int:
        x, y, z = self.shape
        return x * y * z

    def contains_cell(self, cell: tuple[int, int, int]) -> bool:
        if not isinstance(cell, tuple) or len(cell) != _AXES:
            raise TypeError("cell must be a three-component integer tuple")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in cell):
            raise TypeError("cell must be a three-component integer tuple")
        return all(low <= value < high for value, low, high in zip(cell, self.lower, self.upper, strict=True))

    def contains_extent(self, other: CellExtent3D) -> bool:
        return all(
            own_low <= other_low and other_high <= own_high
            for own_low, own_high, other_low, other_high in zip(
                self.lower, self.upper, other.lower, other.upper, strict=True
            )
        )

    def overlaps(self, other: CellExtent3D) -> bool:
        """Whether two half-open volumes share one or more cells."""
        return all(
            max(first_low, second_low) < min(first_high, second_high)
            for first_low, first_high, second_low, second_high in zip(
                self.lower, self.upper, other.lower, other.upper, strict=True
            )
        )


@dataclass(frozen=True)
class D3Q27PlanarInterfaceFace:
    """One coarse/fine planar boundary, oriented coarse-to-fine.

    ``coarse_face_extent`` and ``fine_face_extent`` are the source-owned layers
    adjacent to the interface in their respective lattice coordinates.  The
    normal is a signed Cartesian unit vector directed from coarse to fine.
    Every listed face needs per-substep exchange and end-of-step reflux.
    """

    axis: int
    side: int
    normal: tuple[int, int, int]
    coarse_face_extent: CellExtent3D
    fine_face_extent: CellExtent3D
    requires_exchange: bool = True
    requires_reflux: bool = True


@dataclass(frozen=True)
class FineSubstepD3Q27:
    """One ordered fine half-step inside a normalized coarse step ``[0, 1]``."""

    index: int
    time_start: float
    time_end: float
    exchange_faces: tuple[D3Q27PlanarInterfaceFace, ...]


@dataclass(frozen=True)
class FixedNestedPatchScheduleD3Q27:
    """Validated ownership and two-substep schedule for a fixed D3Q27 patch.

    The descriptor has exactly a 2:1 spatial and temporal refinement ratio.
    Coarse cells in ``coarse_coverage`` are fine-owned and therefore excluded
    from ``coarse_owned_extents``; all other coarse cells remain coarse-owned.
    """

    coarse_extent: CellExtent3D
    fine_extent: CellExtent3D

    def __post_init__(self) -> None:
        if not isinstance(self.coarse_extent, CellExtent3D) or not isinstance(self.fine_extent, CellExtent3D):
            raise TypeError("coarse_extent and fine_extent must be CellExtent3D instances")
        if any(value % _RATIO for value in (*self.fine_extent.lower, *self.fine_extent.upper)):
            raise ValueError("fine extent bounds must be aligned and even on the 2:1 coarse lattice")
        if any(size % _RATIO for size in self.fine_extent.shape):
            raise ValueError("fine extent dimensions must be even for a 2:1 ratio")
        coverage = self.coarse_coverage
        if not self.coarse_extent.contains_extent(coverage):
            raise ValueError("fine patch coverage must lie inside the coarse extent")
        if any(
            coverage.lower[axis] <= self.coarse_extent.lower[axis]
            or coverage.upper[axis] >= self.coarse_extent.upper[axis]
            for axis in range(_AXES)
        ):
            raise ValueError("fine patch coverage must lie strictly inside the coarse extent")
        owned = self.coarse_owned_extents
        if any(first.overlaps(second) for index, first in enumerate(owned) for second in owned[index + 1 :]):
            raise ValueError("coarse ownership extents must not overlap")
        if sum(extent.volume for extent in owned) + coverage.volume != self.coarse_extent.volume:
            raise ValueError("coarse ownership extents must exactly partition the non-refined coarse cells")

    @property
    def refinement_ratio(self) -> int:
        return _RATIO

    @property
    def coarse_coverage(self) -> CellExtent3D:
        return CellExtent3D(
            (
                self.fine_extent.lower[0] // _RATIO,
                self.fine_extent.lower[1] // _RATIO,
                self.fine_extent.lower[2] // _RATIO,
            ),
            (
                self.fine_extent.upper[0] // _RATIO,
                self.fine_extent.upper[1] // _RATIO,
                self.fine_extent.upper[2] // _RATIO,
            ),
        )

    @property
    def coarse_owned_extents(self) -> tuple[CellExtent3D, ...]:
        """Six non-overlapping slabs that exactly exclude the fine-owned volume."""
        c, r = self.coarse_extent, self.coarse_coverage
        cl, cu, rl, ru = c.lower, c.upper, r.lower, r.upper
        return (
            CellExtent3D((cl[0], cl[1], cl[2]), (rl[0], cu[1], cu[2])),
            CellExtent3D((ru[0], cl[1], cl[2]), (cu[0], cu[1], cu[2])),
            CellExtent3D((rl[0], cl[1], cl[2]), (ru[0], rl[1], cu[2])),
            CellExtent3D((rl[0], ru[1], cl[2]), (ru[0], cu[1], cu[2])),
            CellExtent3D((rl[0], rl[1], cl[2]), (ru[0], ru[1], rl[2])),
            CellExtent3D((rl[0], rl[1], ru[2]), (ru[0], ru[1], cu[2])),
        )

    def owns_coarse_cell(self, cell: tuple[int, int, int]) -> bool:
        return self.coarse_extent.contains_cell(cell) and not self.coarse_coverage.contains_cell(cell)

    def owns_fine_cell(self, cell: tuple[int, int, int]) -> bool:
        return self.fine_extent.contains_cell(cell)

    @property
    def interface_faces(self) -> tuple[D3Q27PlanarInterfaceFace, ...]:
        c, f, r = self.coarse_extent, self.fine_extent, self.coarse_coverage
        del c  # Geometry validation has already established all adjacent layers.
        faces: list[D3Q27PlanarInterfaceFace] = []
        for axis in range(_AXES):
            for side in (-1, 1):
                coarse_lower = list(r.lower)
                coarse_upper = list(r.upper)
                fine_lower = list(f.lower)
                fine_upper = list(f.upper)
                if side < 0:
                    coarse_lower[axis] -= 1
                    coarse_upper[axis] = r.lower[axis]
                    fine_upper[axis] = f.lower[axis] + 1
                else:
                    coarse_lower[axis] = r.upper[axis]
                    coarse_upper[axis] += 1
                    fine_lower[axis] = f.upper[axis] - 1
                normal = (side if axis == 0 else 0, side if axis == 1 else 0, side if axis == 2 else 0)
                faces.append(
                    D3Q27PlanarInterfaceFace(
                        axis=axis,
                        side=side,
                        normal=normal,
                        coarse_face_extent=CellExtent3D(
                            (coarse_lower[0], coarse_lower[1], coarse_lower[2]),
                            (coarse_upper[0], coarse_upper[1], coarse_upper[2]),
                        ),
                        fine_face_extent=CellExtent3D(
                            (fine_lower[0], fine_lower[1], fine_lower[2]),
                            (fine_upper[0], fine_upper[1], fine_upper[2]),
                        ),
                    )
                )
        return tuple(faces)

    @property
    def fine_substeps(self) -> tuple[FineSubstepD3Q27, FineSubstepD3Q27]:
        faces = self.interface_faces
        return (
            FineSubstepD3Q27(0, 0.0, 0.5, faces),
            FineSubstepD3Q27(1, 0.5, 1.0, faces),
        )

    @property
    def reflux_faces(self) -> tuple[D3Q27PlanarInterfaceFace, ...]:
        return self.interface_faces


__all__ = [
    "CellExtent3D",
    "D3Q27PlanarInterfaceFace",
    "FineSubstepD3Q27",
    "FixedNestedPatchScheduleD3Q27",
]
