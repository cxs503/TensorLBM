"""Tests for the pure fixed 2:1 D3Q27 nested-patch schedule descriptor."""
from __future__ import annotations

import pytest

from tensorlbm.fixed_nested_patch_schedule import (
    CellExtent3D,
    FixedNestedPatchScheduleD3Q27,
)


def _small_valid_schedule() -> FixedNestedPatchScheduleD3Q27:
    # Fine coordinates use the globally aligned fine lattice.  [4, 12)^3 maps
    # exactly to the interior coarse coverage [2, 6)^3 of [0, 8)^3.
    return FixedNestedPatchScheduleD3Q27(
        coarse_extent=CellExtent3D((0, 0, 0), (8, 8, 8)),
        fine_extent=CellExtent3D((4, 4, 4), (12, 12, 12)),
    )


def test_small_aligned_patch_has_explicit_ownership_six_faces_and_two_substeps() -> None:
    schedule = _small_valid_schedule()

    assert schedule.refinement_ratio == 2
    assert schedule.coarse_coverage == CellExtent3D((2, 2, 2), (6, 6, 6))
    assert schedule.fine_extent == CellExtent3D((4, 4, 4), (12, 12, 12))
    assert schedule.owns_coarse_cell((1, 1, 1))
    assert not schedule.owns_coarse_cell((2, 2, 2))
    assert schedule.owns_fine_cell((4, 4, 4))
    assert not schedule.owns_fine_cell((3, 4, 4))

    # Six disjoint slabs cover the coarse patch except for the refined volume.
    owned = schedule.coarse_owned_extents
    assert len(owned) == 6
    assert sum(extent.volume for extent in owned) + schedule.coarse_coverage.volume == schedule.coarse_extent.volume
    for index, first in enumerate(owned):
        for second in owned[index + 1 :]:
            assert not first.overlaps(second)

    faces = schedule.interface_faces
    assert [(face.axis, face.side, face.normal) for face in faces] == [
        (0, -1, (-1, 0, 0)),
        (0, 1, (1, 0, 0)),
        (1, -1, (0, -1, 0)),
        (1, 1, (0, 1, 0)),
        (2, -1, (0, 0, -1)),
        (2, 1, (0, 0, 1)),
    ]
    assert all(face.requires_exchange and face.requires_reflux for face in faces)
    assert all(face.coarse_face_extent.volume == 16 for face in faces)
    assert all(face.fine_face_extent.volume == 64 for face in faces)

    assert [(substep.index, substep.time_start, substep.time_end) for substep in schedule.fine_substeps] == [
        (0, 0.0, 0.5),
        (1, 0.5, 1.0),
    ]
    assert all(substep.exchange_faces == faces for substep in schedule.fine_substeps)
    assert schedule.reflux_faces == faces


@pytest.mark.parametrize(
    ("coarse", "fine", "message"),
    [
        (CellExtent3D((0, 0, 0), (8, 8, 8)), CellExtent3D((5, 4, 4), (12, 12, 12)), "aligned"),
        (CellExtent3D((0, 0, 0), (8, 8, 8)), CellExtent3D((4, 4, 4), (11, 12, 12)), "even"),
        (CellExtent3D((0, 0, 0), (8, 8, 8)), CellExtent3D((0, 4, 4), (8, 12, 12)), "strictly inside"),
        (CellExtent3D((0, 0, 0), (8, 8, 8)), CellExtent3D((12, 4, 4), (20, 12, 12)), "inside"),
    ],
)
def test_invalid_patch_geometry_is_rejected(coarse: CellExtent3D, fine: CellExtent3D, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FixedNestedPatchScheduleD3Q27(coarse, fine)


def test_empty_and_non_integer_extents_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CellExtent3D((1, 0, 0), (1, 2, 2))
    with pytest.raises(TypeError, match="integer"):
        CellExtent3D((0, 0, 0), (2, 2, 1.5))  # type: ignore[arg-type]
