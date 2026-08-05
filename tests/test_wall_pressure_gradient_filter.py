from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.wall_pressure_gradient_filter import WallPressureGradientFilter


def _vector(x: float) -> torch.Tensor:
    return torch.tensor(((x, 0.0, 0.0),), dtype=torch.float64)


def test_filter_exact_decay_is_diagnostic_cadence_invariant() -> None:
    one_step = WallPressureGradientFilter(time_constant_steps=10.0)
    two_step = WallPressureGradientFilter(time_constant_steps=10.0)
    one_step.update(_vector(0.0))
    two_step.update(_vector(0.0))
    first, _, _ = one_step.update(_vector(1.0), delta_steps=1.0)
    first, _, _ = one_step.update(_vector(1.0), delta_steps=1.0)
    second, _, diagnostics = two_step.update(_vector(1.0), delta_steps=2.0)
    torch.testing.assert_close(first, second)
    assert diagnostics.relaxation_fraction == pytest.approx(
        1.0 - math.exp(-0.2),
    )


def test_invalid_evidence_clears_state_and_reinitializes_without_stale_mean() -> None:
    wall_filter = WallPressureGradientFilter(time_constant_steps=5.0)
    wall_filter.update(_vector(3.0))
    output, valid, diagnostics = wall_filter.update(
        _vector(100.0),
        valid=torch.tensor((False,)),
    )
    assert not bool(valid.item())
    assert torch.isnan(output).all()
    assert diagnostics.cleared_nodes == 1
    output, valid, diagnostics = wall_filter.update(_vector(-2.0))
    assert bool(valid.item())
    assert output[0, 0].item() == -2.0
    assert diagnostics.newly_initialized_nodes == 1


def test_filter_checkpoint_preserves_exact_temporal_state() -> None:
    source = WallPressureGradientFilter(time_constant_steps=7.0)
    source.update(_vector(0.0))
    source.update(_vector(2.0), delta_steps=3.0)
    state = source.state_dict()
    restored = WallPressureGradientFilter(time_constant_steps=7.0)
    restored.load_state_dict(state)
    expected, _, _ = source.update(_vector(-1.0), delta_steps=2.5)
    actual, _, _ = restored.update(_vector(-1.0), delta_steps=2.5)
    torch.testing.assert_close(actual, expected)


def test_filter_checkpoint_and_shape_mismatches_fail_closed() -> None:
    source = WallPressureGradientFilter(time_constant_steps=7.0)
    source.update(_vector(1.0))
    with pytest.raises(ValueError, match="configuration"):
        WallPressureGradientFilter(time_constant_steps=8.0).load_state_dict(
            source.state_dict(),
        )
    with pytest.raises(ValueError, match="reset"):
        source.update(torch.zeros((2, 3), dtype=torch.float64))


def test_uninitialised_filter_checkpoint_round_trip() -> None:
    source = WallPressureGradientFilter(time_constant_steps=4.0)
    restored = WallPressureGradientFilter(time_constant_steps=4.0)
    restored.load_state_dict(source.state_dict())
    assert restored.mean is None
    assert restored.initialized is None
