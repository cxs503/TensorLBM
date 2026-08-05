"""Canonical D3Q19 tensor-stencil contract.

The independent oracle below reads the production lattice descriptor directly,
but does not use the production stencil helpers under test.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C
from tensorlbm.core.d3q19_stencil import (
    D3Q19_MOVING_Q,
    all_moving_neighbor_masks,
    assert_no_direct_phase_links,
    moving_tensor_shifts,
    roll_from_pull_source,
    tensor_shift_for_q,
)


def _oracle_shift(q: int) -> tuple[int, int, int]:
    return int(C[q, 2]), int(C[q, 1]), int(C[q, 0])


def test_moving_q_and_tensor_shifts_are_complete_production_d3q19_oracle() -> None:
    assert D3Q19_MOVING_Q == tuple(range(1, 19))
    assert moving_tensor_shifts() == tuple(_oracle_shift(q) for q in D3Q19_MOVING_Q)
    for q in D3Q19_MOVING_Q:
        assert tensor_shift_for_q(q) == _oracle_shift(q)


@pytest.mark.parametrize("q", [0, -1, 19, 1.0, "1", True, False])
def test_tensor_shift_rejects_nonmoving_or_invalid_q(q: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tensor_shift_for_q(q)  # type: ignore[arg-type]


@pytest.mark.parametrize("q", D3Q19_MOVING_Q)
def test_pull_roll_matches_independent_oracle_including_periodic_seam(q: int) -> None:
    field = torch.arange(5 * 6 * 7, dtype=torch.int64).reshape(5, 6, 7)
    expected = torch.roll(field, shifts=_oracle_shift(q), dims=(0, 1, 2))
    actual = roll_from_pull_source(field, q)
    assert torch.equal(actual, expected)
    assert actual.device == field.device


def test_neighbor_masks_and_phase_guard_cover_every_moving_link() -> None:
    flags = torch.zeros((5, 6, 7), dtype=torch.int8)
    flags[0, 0, 0] = 1
    masks = all_moving_neighbor_masks(flags == 1)
    assert len(masks) == 18
    assert all(mask.device == flags.device for mask in masks)
    for q, mask in zip(D3Q19_MOVING_Q, masks):
        assert torch.equal(mask, torch.roll(flags == 1, _oracle_shift(q), dims=(0, 1, 2)))

    violating_q = 7
    dz, dy, dx = _oracle_shift(violating_q)
    flags[(-dz) % 5, (-dy) % 6, (-dx) % 7] = 2
    with pytest.raises(ValueError, match=r"canonical guard: found 1 direct phase link"):
        assert_no_direct_phase_links(flags, 1, 2, "canonical guard")


def test_roll_and_neighbor_masks_require_exactly_three_tensor_dimensions() -> None:
    for field in (torch.zeros((2, 3)), torch.zeros((2, 3, 4, 5))):
        with pytest.raises(ValueError, match="exactly three dimensions"):
            roll_from_pull_source(field, 1)
        with pytest.raises(ValueError, match="exactly three dimensions"):
            all_moving_neighbor_masks(field.bool())


def test_stencil_rejects_non_tensor_field() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        roll_from_pull_source([[0]], 1)  # type: ignore[arg-type]
