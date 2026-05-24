"""Tests for tensorlbm.constants – D2Q9 lattice constants."""

from __future__ import annotations

import torch

from tensorlbm.constants import D2Q9


class TestD2Q9Velocities:
    def test_c_shape(self) -> None:
        assert D2Q9.c.shape == (9, 2)

    def test_c_dtype(self) -> None:
        assert D2Q9.c.dtype == torch.int64

    def test_rest_direction_is_zero(self) -> None:
        """Direction 0 (rest) must have cx=0 and cy=0."""
        assert int(D2Q9.c[0, 0]) == 0
        assert int(D2Q9.c[0, 1]) == 0

    def test_velocities_are_unit_or_zero(self) -> None:
        """Each velocity component must be -1, 0, or +1."""
        for val in D2Q9.c.flatten().tolist():
            assert val in (-1, 0, 1), f"Unexpected velocity component: {val}"

    def test_opposite_directions_sum_to_zero(self) -> None:
        """For each direction i (1–8) there should be a direction j such that c[i] = -c[j]."""
        c = D2Q9.c.tolist()
        for i in range(1, 9):
            neg = [-c[i][0], -c[i][1]]
            assert neg in c, f"No opposite direction found for direction {i}: {c[i]}"


class TestD2Q9Weights:
    def test_w_shape(self) -> None:
        assert D2Q9.w.shape == (9,)

    def test_w_dtype(self) -> None:
        assert D2Q9.w.dtype == torch.float32

    def test_weights_sum_to_one(self) -> None:
        assert abs(float(D2Q9.w.sum().item()) - 1.0) < 1e-6

    def test_rest_weight(self) -> None:
        """Rest direction weight should be 4/9."""
        assert abs(float(D2Q9.w[0].item()) - 4.0 / 9.0) < 1e-6

    def test_axis_weights(self) -> None:
        """Axis-aligned (speed-1) directions should have weight 1/9."""
        for i in range(1, 5):
            assert abs(float(D2Q9.w[i].item()) - 1.0 / 9.0) < 1e-6

    def test_diagonal_weights(self) -> None:
        """Diagonal (speed-√2) directions should have weight 1/36."""
        for i in range(5, 9):
            assert abs(float(D2Q9.w[i].item()) - 1.0 / 36.0) < 1e-6

    def test_weights_non_negative(self) -> None:
        assert (D2Q9.w >= 0.0).all()
