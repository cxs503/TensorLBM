"""Control-volume cross-check contracts for actual propeller campaign samples."""
from __future__ import annotations

import pytest

from tensorlbm.propeller_benchmark import _summarize_control_volume_cross_check


def test_control_volume_cross_check_reports_distribution_delta_minus_me_load() -> None:
    report = _summarize_control_volume_cross_check([
        {"distribution_momentum_delta_x": 1.0, "fx_me_lu": 0.9},
        {"distribution_momentum_delta_x": 1.2, "fx_me_lu": 1.0},
    ])

    assert report["available"] is True
    assert report["method"] == "global_momentum_delta"
    assert report["sample_count"] == 2
    assert report["distribution_momentum_delta_x_mean"] == pytest.approx(1.1)
    assert report["me_force_x_mean"] == pytest.approx(0.95)
    # Momentum change of the fluid is opposite to the ME load on the wall.
    assert report["residual_x_mean"] == pytest.approx(2.05)


def test_control_volume_cross_check_fails_closed_without_samples() -> None:
    assert _summarize_control_volume_cross_check([]) == {
        "available": False,
        "reason": "no_post_warmup_samples",
    }
