"""Control-volume momentum-budget contracts for actual propeller samples."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.propeller_benchmark import (
    _d3q19_momentum_x,
    _summarize_control_volume_cross_check,
    moving_wall_bounce_back_3d_with_reaction,
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
            "wall_reaction_x": -0.4,
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
            "wall_reaction_x": -0.6,
            "budget_residual_x": 0.0,
            "open_faces_available": True,
        },
    ])

    assert report["available"] is True
    assert report["status"] == "comparable"
    assert report["method"] == "discrete_full_control_volume_momentum_budget"
    assert report["sample_count"] == 2
    assert report["fluid_momentum_delta_x_mean"] == pytest.approx(1.1)
    assert report["open_face_momentum_flux_x_mean"] == pytest.approx(0.25)
    assert report["collision_momentum_contribution_x_mean"] == pytest.approx(0.1)
    assert report["moving_mask_reset_momentum_contribution_x_mean"] == pytest.approx(0.25)
    assert report["wall_momentum_contribution_x_mean"] == pytest.approx(0.5)
    assert report["budget_residual_x_mean"] == pytest.approx(0.0)
    # Same-operator reaction is the negative of the fluid impulse, whereas
    # the legacy static ME remains explicitly non-comparable.
    assert report["same_operator_action_reaction_residual_x_mean"] == pytest.approx(0.0)
    assert report["same_operator_action_reaction_abs_residual_x_max"] == pytest.approx(0.0)
    assert report["same_operator_action_reaction_relative_residual_max"] == pytest.approx(0.0)
    torque_report = report["same_operator_torque_action_reaction"]
    assert torque_report == {
        "status": "withheld",
        "reason": "missing_same_operator_torque_fields",
        "sample_count": 2,
        "missing_fields": ["wall_fluid_torque_impulse_x", "wall_reaction_torque_x"],
        "required_fields": ["wall_fluid_torque_impulse_x", "wall_reaction_torque_x"],
    }
    assert report["me_vs_cv_comparison_status"] == "noncomparable"
    assert report["sign_convention"]["positive_x"] == "positive streamwise (+x) momentum"


def test_control_volume_torque_diagnostic_is_withheld_for_mixed_samples() -> None:
    """Torque is comparable only when every sample carries both operator terms."""
    common = {
        "fluid_momentum_delta_x": 1.0,
        "wall_me_load_x": 0.9,
        "open_face_momentum_flux_x": 0.2,
        "collision_momentum_contribution_x": 0.1,
        "streaming_momentum_contribution_x": 0.0,
        "fixed_channel_wall_momentum_contribution_x": 0.0,
        "moving_mask_reset_momentum_contribution_x": 0.3,
        "wall_momentum_contribution_x": 0.4,
        "wall_reaction_x": -0.4,
        "budget_residual_x": 0.0,
        "open_faces_available": True,
    }
    report = _summarize_control_volume_cross_check([
        common | {
            "wall_fluid_torque_impulse_x": 2.0,
            "wall_reaction_torque_x": -2.0,
        },
        common,
    ])

    assert report["status"] == "comparable"
    assert report["same_operator_torque_action_reaction"] == {
        "status": "withheld",
        "reason": "missing_same_operator_torque_fields",
        "sample_count": 2,
        "missing_fields": ["wall_fluid_torque_impulse_x", "wall_reaction_torque_x"],
        "required_fields": ["wall_fluid_torque_impulse_x", "wall_reaction_torque_x"],
    }


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


def test_moving_wall_operator_reports_its_exact_fluid_impulse_and_body_reaction() -> None:
    f = torch.zeros((19, 2, 2, 2), dtype=torch.float64)
    f[1] = 0.2
    f[2] = 0.1
    mask = torch.zeros((2, 2, 2), dtype=torch.bool)
    mask[0, 1, 1] = True
    zero = torch.zeros_like(mask, dtype=torch.float64)

    after, reaction = moving_wall_bounce_back_3d_with_reaction(
        f, mask, zero, zero, zero, origin=(0.0, 0.0, 0.0),
    )

    assert torch.allclose(reaction.fluid_impulse, torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(reaction.body_reaction, -reaction.fluid_impulse)
    assert reaction.action_reaction_signed_residual_norm == pytest.approx(0.0)
    assert reaction.action_reaction_absolute_residual_norm == pytest.approx(0.0)
    assert reaction.action_reaction_relative_residual == pytest.approx(0.0)
    assert _d3q19_momentum_x(after) - _d3q19_momentum_x(f) == pytest.approx(reaction.fluid_impulse[0].item())
