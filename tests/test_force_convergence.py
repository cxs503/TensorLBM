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


def test_resolved_periodic_force_can_converge_in_mean() -> None:
    samples = [
        value
        for block in range(100)
        for value in [8.0 if block % 2 == 0 else 12.0] * 10
    ]
    report = assess_force_stationarity(samples, block_size=10)

    assert report.relative_range_pct == pytest.approx(40.0)
    assert report.confidence95_half_width_pct < 5.0
    assert report.dominant_period_steps == pytest.approx(20.0)
    assert report.effective_sample_count < report.sample_count
    assert report.meets(5.0)


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


def test_autocorrelation_diagnostics_reduce_repeated_sample_count() -> None:
    samples = [value for value in range(40) for _ in range(10)]
    report = assess_force_stationarity(samples, block_size=100)

    assert report.autocorrelation_zero_crossing_lag is not None
    assert report.integrated_autocorrelation_time_steps > 1.0
    assert report.effective_sample_count < len(samples) / 5.0
    assert report.autocorrelation_standard_error_pct > 0.0
