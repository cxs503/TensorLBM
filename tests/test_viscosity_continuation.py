from __future__ import annotations

import pytest

from tensorlbm.viscosity_continuation import ResolvedReynoldsContinuation


def test_inverse_reynolds_continuation_has_exact_endpoints_and_midpoint() -> None:
    schedule = ResolvedReynoldsContinuation(5_000.0, 100_000.0, 100, 300)

    assert schedule.reynolds_at(0) == 5_000.0
    assert schedule.reynolds_at(100) == 5_000.0
    assert schedule.reynolds_at(300) == 100_000.0
    assert schedule.reynolds_at(500) == 100_000.0
    assert schedule.reynolds_at(200) == pytest.approx(
        1.0 / (0.5 / 5_000.0 + 0.5 / 100_000.0),
    )


def test_tau_chain_preserves_convective_scaling() -> None:
    schedule = ResolvedReynoldsContinuation(5_000.0, 100_000.0, 100, 300)
    tau = schedule.tau_by_level(
        100,
        lattice_speed=0.06,
        root_hull_length=90.0,
        levels=3,
    )
    assert tau == pytest.approx((0.50324, 0.50648, 0.51296))


def test_constant_schedule_does_not_require_a_ramp() -> None:
    schedule = ResolvedReynoldsContinuation(20_000.0, 20_000.0, 0, 0)
    assert schedule.reynolds_at(99) == 20_000.0


def test_invalid_continuation_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive ramp"):
        ResolvedReynoldsContinuation(5_000.0, 100_000.0, 10, 10)
    with pytest.raises(ValueError, match="0 <= start"):
        ResolvedReynoldsContinuation(5_000.0, 100_000.0, -1, 10)
