"""Canonical D3Q27 tensor-stencil contract.

The independent oracle below reads the production lattice descriptor directly,
but does not use the production stencil helpers under test.

Mirrors test_d3q19_stencil.py for the D3Q27 lattice (26 moving directions:
6 face + 12 edge + 8 corner).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.core.d3q27_stencil import (
    D3Q27_MOVING_Q,
    all_moving_neighbor_masks_27,
    assert_no_direct_phase_links_27,
    moving_tensor_shifts_27,
    roll_from_pull_source_27,
    roll_to_neighbor_27,
    tensor_shift_for_q_27,
)
from tensorlbm.d3q27 import C


def _oracle_shift(q: int) -> tuple[int, int, int]:
    """Independent oracle: D3Q27 C is (x,y,z) order, field is (z,y,x)."""
    return int(C[q, 2]), int(C[q, 1]), int(C[q, 0])


def test_moving_q_and_tensor_shifts_are_complete_production_d3q27_oracle() -> None:
    """D3Q27 has 26 moving directions (q=1..26)."""
    assert D3Q27_MOVING_Q == tuple(range(1, 27))
    expected_shifts = tuple(_oracle_shift(q) for q in D3Q27_MOVING_Q)
    assert moving_tensor_shifts_27() == expected_shifts
    for q in D3Q27_MOVING_Q:
        assert tensor_shift_for_q_27(q) == _oracle_shift(q)


@pytest.mark.parametrize("q", [0, -1, 27, 1.0, "1", True, False])
def test_tensor_shift_rejects_nonmoving_or_invalid_q(q: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tensor_shift_for_q_27(q)  # type: ignore[arg-type]


@pytest.mark.parametrize("q", D3Q27_MOVING_Q)
def test_pull_roll_matches_independent_oracle_including_periodic_seam(q: int) -> None:
    field = torch.arange(5 * 6 * 7, dtype=torch.int64).reshape(5, 6, 7)
    expected = torch.roll(field, shifts=_oracle_shift(q), dims=(0, 1, 2))
    actual = roll_from_pull_source_27(field, q)
    assert torch.equal(actual, expected)
    assert actual.device == field.device


@pytest.mark.parametrize("q", D3Q27_MOVING_Q)
def test_push_roll_matches_independent_oracle_including_periodic_seam(q: int) -> None:
    """roll_to_neighbor_27 pushes along +c_q (opposite sign of pull)."""
    field = torch.arange(5 * 6 * 7, dtype=torch.int64).reshape(5, 6, 7)
    dz, dy, dx = _oracle_shift(q)
    # Push = roll by negative of pull shift
    expected = torch.roll(field, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
    actual = roll_to_neighbor_27(field, q)
    assert torch.equal(actual, expected)
    assert actual.device == field.device


def test_neighbor_masks_and_phase_guard_cover_every_moving_link() -> None:
    flags = torch.zeros((5, 6, 7), dtype=torch.int8)
    flags[0, 0, 0] = 1
    masks = all_moving_neighbor_masks_27(flags == 1)
    assert len(masks) == 26
    assert all(mask.device == flags.device for mask in masks)
    for q, mask in zip(D3Q27_MOVING_Q, masks):
        assert torch.equal(mask, torch.roll(flags == 1, _oracle_shift(q), dims=(0, 1, 2)))

    # Phase guard: place source and target such that a moving link connects them
    violating_q = 7  # any moving direction
    dz, dy, dx = _oracle_shift(violating_q)
    flags[(-dz) % 5, (-dy) % 6, (-dx) % 7] = 2
    with pytest.raises(ValueError, match=r"canonical guard: found 1 direct phase link"):
        assert_no_direct_phase_links_27(flags, 1, 2, "canonical guard")


def test_roll_and_neighbor_masks_require_exactly_three_tensor_dimensions() -> None:
    for field in (torch.zeros((2, 3)), torch.zeros((2, 3, 4, 5))):
        with pytest.raises(ValueError, match="exactly three dimensions"):
            roll_from_pull_source_27(field, 1)
        with pytest.raises(ValueError, match="exactly three dimensions"):
            roll_to_neighbor_27(field, 1)
        with pytest.raises(ValueError, match="exactly three dimensions"):
            all_moving_neighbor_masks_27(field.bool())


def test_stencil_rejects_non_tensor_field() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        roll_from_pull_source_27([[0]], 1)  # type: ignore[arg-type]


def test_d3q27_includes_face_edge_corner_directions() -> None:
    """Verify D3Q27 stencil covers all three direction classes."""
    shifts = moving_tensor_shifts_27()
    # Face directions: exactly one non-zero component (6 total)
    face = [s for s in shifts if sum(1 for c in s if c != 0) == 1]
    # Edge directions: exactly two non-zero components (12 total)
    edge = [s for s in shifts if sum(1 for c in s if c != 0) == 2]
    # Corner directions: all three non-zero components (8 total)
    corner = [s for s in shifts if sum(1 for c in s if c != 0) == 3]
    assert len(face) == 6, f"Expected 6 face directions, got {len(face)}"
    assert len(edge) == 12, f"Expected 12 edge directions, got {len(edge)}"
    assert len(corner) == 8, f"Expected 8 corner directions, got {len(corner)}"


def test_d3q27_opposite_pairs() -> None:
    """Every moving direction q has an opposite direction opp where c_opp = -c_q."""
    for q in D3Q27_MOVING_Q:
        cx, cy, cz = int(C[q, 0]), int(C[q, 1]), int(C[q, 2])
        # Find opposite
        opp_found = False
        for qp in D3Q27_MOVING_Q:
            if int(C[qp, 0]) == -cx and int(C[qp, 1]) == -cy and int(C[qp, 2]) == -cz:
                opp_found = True
                break
        assert opp_found, f"No opposite found for q={q} c=({cx},{cy},{cz})"


def test_pull_push_roundtrip_identity() -> None:
    """Pull then push (or vice versa) along the same direction is identity."""
    field = torch.randn(8, 10, 12, dtype=torch.float32)
    for q in [1, 7, 13, 20, 26]:  # sample face, edge, corner
        pulled = roll_from_pull_source_27(field, q)
        roundtrip = roll_to_neighbor_27(pulled, q)
        assert torch.allclose(roundtrip, field, atol=0), f"Roundtrip failed for q={q}"

        pushed = roll_to_neighbor_27(field, q)
        roundtrip2 = roll_from_pull_source_27(pushed, q)
        assert torch.allclose(roundtrip2, field, atol=0), f"Push-pull roundtrip failed for q={q}"
