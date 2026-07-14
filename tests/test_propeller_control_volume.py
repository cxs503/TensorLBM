"""Control-volume momentum-budget contracts for actual propeller samples."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.propeller_benchmark import (
    _d3q19_momentum_x,
    _summarize_control_volume_cross_check,
)


def test_d3q19_cv_x_momentum_includes_diagonal_populations() -> None:
    """A diagonal-only D3Q19 population must contribute to x momentum."""
    distributions = torch.zeros((19, 1, 1, 1), dtype=torch.float64)
    distributions[7, 0, 0, 0] = 2.5  # c_7 = (+1, +1, 0), not f[1] - f[2]
    distributions[12, 0, 0, 0] = 0.5  # c_12 = (-1, 0, -1)

    assert _d3q19_momentum_x(distributions).item() == pytest.approx(2.0)


def test_control_volume_budget_reports_every_term_and_sign_convention() -> None:
    report = _summarize_control_volume_cross_check([
        {
            "fluid_momentum_delta_x": 1.0,
            "wall_me_load_x": 0.9,
            "open_face_momentum_flux_x": 0.2,
            "collision_momentum_contribution_x": 0.1,
            "streaming_momentum_contribution_x": 0.0,
            "fixed_channel_wall_momentum_contribution_x": 0.0,
            "moving_mask_reset_momentum_contribution_x": 0.3,
            "wall_momentum_contribution_x": 0.4,
            "budget_residual_x": 0.0,
            "open_faces_available": True,
        },
        {
            "fluid_momentum_delta_x": 1.2,
            "wall_me_load_x": 1.0,
            "open_face_momentum_flux_x": 0.3,
            "collision_momentum_contribution_x": 0.1,
            "streaming_momentum_contribution_x": 0.0,
            "fixed_channel_wall_momentum_contribution_x": 0.0,
            "moving_mask_reset_momentum_contribution_x": 0.2,
            "wall_momentum_contribution_x": 0.6,
            "budget_residual_x": 0.0,
            "open_faces_available": True,
        },
    ])

    assert report["available"] is True
    assert report["status"] == "noncomparable"
    assert report["method"] == "discrete_full_control_volume_momentum_budget"
    assert report["sample_count"] == 2
    assert report["fluid_momentum_delta_x_mean"] == pytest.approx(1.1)
    assert report["open_face_momentum_flux_x_mean"] == pytest.approx(0.25)
    assert report["collision_momentum_contribution_x_mean"] == pytest.approx(0.1)
    assert report["moving_mask_reset_momentum_contribution_x_mean"] == pytest.approx(0.25)
    assert report["wall_momentum_contribution_x_mean"] == pytest.approx(0.5)
    assert report["budget_residual_x_mean"] == pytest.approx(0.0)
    # Positive ME load is force on body, while positive wall contribution is fluid gain.
    assert report["me_vs_cv_wall_nonclosure_x_mean"] == pytest.approx(1.45)
    assert report["me_vs_cv_wall_nonclosure_abs_x_mean"] == pytest.approx(1.45)
    assert report["me_vs_cv_wall_nonclosure_rel"] == pytest.approx(1.45 / 0.95)
    assert "before moving_wall_bounce_back_3d" in report["me_vs_cv_comparison_reason"]
    assert report["sign_convention"]["positive_x"] == "positive streamwise (+x) momentum"


def test_control_volume_budget_is_withheld_when_open_faces_are_unavailable() -> None:
    report = _summarize_control_volume_cross_check([{
        "fluid_momentum_delta_x": 1.0,
        "wall_me_load_x": 0.9,
        "open_faces_available": False,
    }])

    assert report["available"] is False
    assert report["status"] == "withheld"
    assert report["reason"] == "open_face_momentum_flux_unavailable"


def test_control_volume_cross_check_fails_closed_without_samples() -> None:
    assert _summarize_control_volume_cross_check([]) == {
        "available": False,
        "status": "withheld",
        "reason": "no_post_warmup_samples",
    }
