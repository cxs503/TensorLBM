from __future__ import annotations

import math

import pytest

from tensorlbm.force_convergence import assess_force_stationarity


def test_stationary_force_history_is_admitted() -> None:
    samples = [10.0 + 0.001 * math.sin(index) for index in range(400)]
    report = assess_force_stationarity(samples, block_size=100)
    assert report.block_count == 4
    assert report.meets(0.1)


def test_reference_crossing_does_not_hide_oscillation() -> None:
    samples = [value for block in (8.0, 12.0, 8.0, 12.0) for value in [block] * 50]
    report = assess_force_stationarity(samples, block_size=50)
    assert report.mean == pytest.approx(10.0)
    assert report.relative_range_pct == pytest.approx(40.0)
    assert not report.meets(5.0)


def test_monotone_force_history_reports_trend_and_half_drift() -> None:
    samples = [float(index) for index in range(400)]
    report = assess_force_stationarity(samples, block_size=100)
    assert report.linear_trend_pct > 100.0
    assert report.half_mean_drift_pct > 100.0
    assert not report.meets(10.0)


def test_too_short_or_nonfinite_history_fails_closed() -> None:
    short = assess_force_stationarity([1.0] * 30, block_size=10)
    bad = assess_force_stationarity([1.0, math.nan] * 20, block_size=10)
    assert not short.sufficiently_sampled
    assert not short.meets(1.0)
    assert not bad.finite
    assert not bad.meets(1.0)


def test_single_complete_block_fails_closed_without_division_by_zero() -> None:
    report = assess_force_stationarity([2.0] * 10, block_size=10)
    assert report.block_count == 1
    assert report.mean == pytest.approx(2.0)
    assert report.relative_range_pct == pytest.approx(0.0)
    assert math.isinf(report.half_mean_drift_pct)
    assert math.isinf(report.linear_trend_pct)
    assert not report.meets(1.0)
