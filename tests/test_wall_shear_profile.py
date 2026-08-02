from __future__ import annotations

import pytest
import torch

from tensorlbm.wall_shear_profile import summarize_axial_wall_shear


def test_axial_wall_shear_profile_preserves_force_and_area() -> None:
    profile = summarize_axial_wall_shear(
        torch.tensor((0.0, 1.0, 8.0, 9.0)),
        torch.tensor((1.0, 2.0, 3.0, -1.0)),
        torch.tensor((2.0, 2.0, 4.0, 4.0)),
        torch.tensor((100.0, 200.0, 300.0, 400.0)),
        torch.tensor((0.1, 0.2, 0.3, 0.4)),
        torch.tensor((0.01, 0.02, 0.03, 0.04)),
        bins=2,
    )

    assert len(profile) == 2
    assert sum(item["signed_shear_x_sum_lu"] for item in profile) == pytest.approx(5.0)
    assert sum(item["area_sum_lu2"] for item in profile) == pytest.approx(12.0)
    assert sum(item["absolute_shear_x_fraction"] for item in profile) == pytest.approx(1.0)
    assert profile[0]["signed_shear_x_fraction"] == pytest.approx(3.0 / 5.0)
    assert profile[1]["median_y_plus"] == pytest.approx(350.0)
    assert profile[1]["median_tangential_speed_lu"] == pytest.approx(0.35)
    assert profile[1]["median_friction_velocity_lu"] == pytest.approx(0.035)


def test_axial_wall_shear_profile_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        summarize_axial_wall_shear(
            torch.ones(2),
            torch.ones(1),
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
        )
