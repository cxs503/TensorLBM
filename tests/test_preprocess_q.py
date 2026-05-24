"""Tests for polygon q-field generation."""
from __future__ import annotations

import torch

from tensorlbm.preprocess_geo import poly_to_mask_2d, poly_to_mask_and_q_2d


def test_poly_to_mask_and_q_2d_shape() -> None:
    mask, q = poly_to_mask_and_q_2d(
        [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        5,
        5,
        torch.device("cpu"),
    )
    assert mask.shape == (5, 5)
    assert q.shape == (9, 5, 5)


def test_poly_to_mask_and_q_2d_q_range() -> None:
    _, q = poly_to_mask_and_q_2d(
        [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        5,
        5,
        torch.device("cpu"),
    )
    assert torch.all((q >= 0.0) & (q <= 1.0))


def test_poly_to_mask_and_q_2d_consistent_mask() -> None:
    verts = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
    mask0 = poly_to_mask_2d(verts, 5, 5, torch.device("cpu"))
    mask1, _ = poly_to_mask_and_q_2d(verts, 5, 5, torch.device("cpu"))
    assert torch.equal(mask0, mask1)
