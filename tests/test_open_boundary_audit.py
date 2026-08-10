from __future__ import annotations

import pytest

from tensorlbm.open_boundary_audit import audit_open_boundary_history


def test_open_boundary_history_uses_grid_normalized_scales() -> None:
    result = audit_open_boundary_history(
        [
            {
                "mass_delta": 2.0,
                "momentum_delta": [3.0, 4.0, 0.0],
                "finite": True,
                "stages": [
                    {
                        "face_sum_mass_closure_error": 0.1,
                        "face_sum_momentum_closure_error": [0.0, 0.2, 0.0],
                    }
                ],
            },
            {
                "mass_delta": -1.0,
                "momentum_delta": [0.0, 0.0, 5.0],
                "finite": True,
                "stages": [],
            },
        ],
        reference_mass=100.0,
        reference_momentum=10.0,
    )

    assert result.samples == 2
    assert result.cumulative_mass_delta == 1.0
    assert result.mean_absolute_mass_delta_fraction == pytest.approx(0.015)
    assert result.maximum_absolute_mass_delta_fraction == pytest.approx(0.02)
    assert result.cumulative_momentum_delta == (3.0, 4.0, 5.0)
    assert result.mean_momentum_delta_fraction == pytest.approx(0.5)
    assert result.maximum_face_sum_mass_closure_fraction == pytest.approx(0.001)
    assert result.maximum_face_sum_momentum_closure_fraction == pytest.approx(0.02)
    assert result.finite


def test_empty_open_boundary_history_is_finite_zero_exposure() -> None:
    result = audit_open_boundary_history(
        [],
        reference_mass=100.0,
        reference_momentum=10.0,
    )

    assert result.samples == 0
    assert result.finite
    assert result.maximum_momentum_delta_fraction == 0.0


def test_open_boundary_history_rejects_invalid_reference_scale() -> None:
    with pytest.raises(ValueError, match="reference_mass"):
        audit_open_boundary_history(
            [],
            reference_mass=0.0,
            reference_momentum=1.0,
        )
