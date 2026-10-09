"""Tests for the SI icebreaking force ledger."""

import math

import pytest

from tensorlbm.icebreaking import IcebreakingForceLedger, WrenchSI


def test_force_decomposition_and_positive_resistance() -> None:
    ledger = IcebreakingForceLedger(
        time_s=1.25,
        fluid_on_hull=WrenchSI(fx_n=-120.0, mz_nm=4.0),
        ice_contact_on_hull=WrenchSI(fx_n=-380.0, fy_n=10.0, mz_nm=6.0),
        other_on_hull=WrenchSI(fx_n=-5.0),
    )

    assert ledger.total_on_hull.fx_n == pytest.approx(-505.0)
    assert ledger.total_on_hull.fy_n == pytest.approx(10.0)
    assert ledger.total_on_hull.mz_nm == pytest.approx(10.0)
    assert ledger.resistance_n == pytest.approx(505.0)


def test_resistance_uses_configured_forward_axis() -> None:
    ledger = IcebreakingForceLedger(
        time_s=0.0,
        fluid_on_hull=WrenchSI(fx_n=-3.0, fy_n=8.0),
        forward_axis=(0.0, 1.0, 0.0),
    )

    assert ledger.resistance_n == pytest.approx(-8.0)


def test_record_contains_separate_and_total_contributions() -> None:
    ledger = IcebreakingForceLedger(
        time_s=2.0,
        fluid_on_hull=WrenchSI(fx_n=-2.0),
        ice_contact_on_hull=WrenchSI(fx_n=-7.0),
    )

    record = ledger.as_record()

    assert record["time_s"] == pytest.approx(2.0)
    assert record["fluid_fx_n"] == pytest.approx(-2.0)
    assert record["ice_contact_fx_n"] == pytest.approx(-7.0)
    assert record["total_fx_n"] == pytest.approx(-9.0)
    assert record["resistance_n"] == pytest.approx(9.0)


def test_wrench_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        WrenchSI(fx_n=math.nan)

    with pytest.raises(ValueError, match="finite"):
        WrenchSI(mz_nm=math.inf)


@pytest.mark.parametrize("time_s", [-1.0, math.nan, math.inf])
def test_ledger_rejects_invalid_time(time_s: float) -> None:
    with pytest.raises(ValueError, match="time_s"):
        IcebreakingForceLedger(time_s=time_s)


@pytest.mark.parametrize(
    "axis",
    [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (1.0, 1.0),
        (1.0, math.nan, 0.0),
    ],
)
def test_ledger_rejects_invalid_forward_axis(axis: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="forward_axis"):
        IcebreakingForceLedger(time_s=0.0, forward_axis=axis)  # type: ignore[arg-type]


def test_wrench_addition_and_scaling() -> None:
    result = WrenchSI(fx_n=2.0, my_nm=3.0) + WrenchSI(fx_n=-5.0, my_nm=1.0)

    assert result.fx_n == pytest.approx(-3.0)
    assert result.my_nm == pytest.approx(4.0)
    assert result.scaled(2.0).fx_n == pytest.approx(-6.0)
    with pytest.raises(ValueError, match="finite"):
        result.scaled(math.inf)
