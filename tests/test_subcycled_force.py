from __future__ import annotations

import math

import pytest

from tensorlbm.subcycled_force import UniformSubcycleAverager


def test_recursive_depth_defines_exact_uniform_sample_count() -> None:
    average = UniformSubcycleAverager(refinement_depth=3)

    assert average.expected_samples == 8
    assert average.mean(range(1, 9), observable="drag") == pytest.approx(4.5)
    assert average.provenance(8) == {
        "refinement_depth": 3,
        "expected_samples": 8,
        "observed_samples": 8,
        "uniform_sample_count_met": True,
    }


def test_wrong_sample_count_cannot_change_the_denominator_silently() -> None:
    average = UniformSubcycleAverager(refinement_depth=3)

    with pytest.raises(RuntimeError, match="requires 8.*observed 4"):
        average.mean((1.0, 2.0, 3.0, 4.0), observable="drag")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_force_sample_is_rejected(value: float) -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        UniformSubcycleAverager(1).mean((1.0, value))


@pytest.mark.parametrize("depth", [-1, 1.5, True])
def test_invalid_refinement_depth_is_rejected(depth: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        UniformSubcycleAverager(depth)  # type: ignore[arg-type]
