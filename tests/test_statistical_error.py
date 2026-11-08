"""Tests for :mod:`tensorlbm.validation.statistical_error`.

Every estimator is pinned against a synthetic case with a *known* answer:
* the Student-t quantile against published table values,
* the autocorrelation integral time against AR(1)'s analytic ``T_int``,
* batch means against a closed-form Gaussian mean interval,
* the attribution verdict against a constructed bias / constructed noise case,
* the budget fit against a synthetic ``T^-1/2`` law.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.validation.statistical_error import (
    batch_means_summary,
    budget_extrapolation,
    convergence_eligibility,
    decompose_error,
    domain_length_check,
    half_window_max_diff,
    integral_time,
    metric_from_batches,
    negative_control_check,
    relative_l2_error,
    student_t_quantile,
)


def test_student_t_quantile_matches_published_table():
    # 97.5 % two-sided critical values, standard tables
    table = {1: 12.706, 2: 4.303, 5: 2.571, 10: 2.228, 30: 2.042}
    for dof, expected in table.items():
        assert student_t_quantile(0.975, dof) == pytest.approx(expected, rel=2.0e-3)
    # and the symmetric property / large-dof limit
    assert student_t_quantile(0.5, 7) == pytest.approx(0.0, abs=1.0e-12)
    assert student_t_quantile(0.975, 10**6) == pytest.approx(1.959964, rel=1.0e-3)


def test_integral_time_recovers_ar1_value():
    rng = np.random.default_rng(20261108)
    phi = 0.9
    n = 400_000
    noise = rng.standard_normal(n)
    series = np.empty(n)
    series[0] = noise[0]
    for i in range(1, n):  # AR(1): rho(k) = phi**k
        series[i] = phi * series[i - 1] + noise[i]
    expected = 0.5 + phi / (1.0 - phi)  # exact T_int of an AR(1) process
    result = integral_time(series, method="sokal")
    assert result["t_int_samples"] == pytest.approx(expected, rel=0.10)
    assert result["n_effective_samples"] == pytest.approx(n / (2.0 * expected), rel=0.15)
    # white noise: T_int -> 0.5 samples
    white = integral_time(rng.standard_normal(100_000))
    assert white["t_int_samples"] == pytest.approx(0.5, abs=0.1)


def test_integral_time_rejects_degenerate_input():
    with pytest.raises(ValueError):
        integral_time([1.0] * 100)  # zero variance
    with pytest.raises(ValueError):
        integral_time([1.0, 2.0, 3.0])  # too short


def test_batch_means_ci_covers_the_mean_at_the_stated_rate():
    rng = np.random.default_rng(7)
    trials = 400
    interval_count = 0
    covered = 0
    for _ in range(trials):
        batches = rng.normal(loc=18.0, scale=1.0, size=(16, 3))
        summary = batch_means_summary(batches)
        low = np.asarray(summary["mean"]) - np.asarray(summary["ci_half"])
        high = np.asarray(summary["mean"]) + np.asarray(summary["ci_half"])
        covered += int(np.sum((low <= 18.0) & (high >= 18.0)))
        interval_count += low.size
    rate = covered / interval_count
    assert 0.92 <= rate <= 0.98, rate  # nominal 95 %


def test_batch_means_reports_relative_half_width():
    batches = [[10.0, 20.0], [12.0, 22.0], [14.0, 24.0], [16.0, 26.0]]
    summary = batch_means_summary(batches)
    assert summary["n_batches"] == 4 and summary["dof"] == 3
    assert summary["t_critical"] == pytest.approx(3.182, rel=2.0e-3)
    # mean 13, sd 2.582, sem 1.291, half 4.108 -> 31.6 %
    assert summary["max_ci_half"] / 13.0 == pytest.approx(0.316, rel=0.02)
    with pytest.raises(ValueError):
        batch_means_summary([[1.0]])  # a single batch carries no variance


def test_half_window_gate_separates_drift_from_stationarity():
    stationary = np.tile(np.linspace(0.0, 20.0, 6), (8, 1))
    gate = half_window_max_diff(stationary)
    assert gate["half_window_max_diff"] < 1.0e-12
    drift = stationary + np.linspace(0.0, 4.0, 8)[:, None]
    gate = half_window_max_diff(drift)
    assert gate["half_window_max_diff"] > 1.0  # a 4 u+ drift over the window
    with pytest.raises(ValueError):
        half_window_max_diff(np.zeros((2, 5)))


def test_decompose_error_calls_noise_statistics_and_bias_model():
    reference = np.linspace(0.0, 20.0, 40)
    rng = np.random.default_rng(11)
    batches = reference[None, :] + rng.normal(scale=0.02, size=(24, 40))
    noisy = batches.mean(axis=0)
    metric = metric_from_batches(batches, reference)
    verdict = decompose_error(noisy, reference, metric, tolerance_relative=0.03)
    assert verdict["verdict"] == "statistics_dominated"
    assert verdict["significant"] is False
    assert verdict["u_sn"] > 0.0

    biased = 1.2 * reference
    biased_batches = biased[None, :] + rng.normal(scale=0.02, size=(24, 40))
    metric_bias = metric_from_batches(biased_batches, reference)
    verdict_bias = decompose_error(
        biased_batches.mean(axis=0), reference, metric_bias, tolerance_relative=0.03
    )
    assert verdict_bias["observed_relative_l2"] == pytest.approx(0.2, rel=0.05)
    assert verdict_bias["verdict"] == "model_or_domain_dominated"
    assert verdict_bias["significant"] is True
    # the residual beyond the CI must be close to the full bias, not the noise
    assert verdict_bias["residual_beyond_ci"] > 0.15


def test_decompose_error_subtracts_a_resolved_domain_shift():
    reference = np.linspace(0.0, 20.0, 40)
    rng = np.random.default_rng(13)
    biased = 1.10 * reference
    batches = biased[None, :] + rng.normal(scale=0.02, size=(16, 40))
    metric = metric_from_batches(batches, reference)
    plain = decompose_error(batches.mean(axis=0), reference, metric)
    with_domain = decompose_error(
        batches.mean(axis=0), reference, metric, domain_shift_relative=0.05
    )
    assert with_domain["domain_shift_relative"] == pytest.approx(0.05)
    assert with_domain["unexplained_model_relative"] == pytest.approx(
        max(plain["residual_beyond_ci"] - 0.05, 0.0)
    )


def test_domain_length_and_negative_control_checks():
    long_profile = np.linspace(0.0, 20.0, 30)
    short_profile = long_profile + 0.05 * np.abs(long_profile)  # 5 % box effect
    ci = np.full(30, 0.1)
    domain = domain_length_check(short_profile, long_profile, short_ci_half=ci, long_ci_half=ci)
    assert domain["within_tolerance"] is False
    assert domain["resolved"] is True
    same = domain_length_check(long_profile, long_profile, short_ci_half=ci, long_ci_half=ci)
    assert same["within_tolerance"] is True and same["resolved"] is False

    control = np.linspace(0.0, 20.0, 30)
    variant = control * 1.01  # 1 % SGS effect, larger than the 0.1 u+ half-width
    negative = negative_control_check(control, variant, control_ci_half=ci, variant_ci_half=ci)
    assert negative["separated"] is True
    flat = negative_control_check(control, control + 1.0e-3, control_ci_half=ci, variant_ci_half=ci)
    assert flat["separated"] is False


def test_relative_l2_uses_the_reference_norm():
    assert relative_l2_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert relative_l2_error([2.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        relative_l2_error([1.0], [1.0, 2.0])


def test_convergence_eligibility_is_fail_closed():
    assert convergence_eligibility(2.9)["status"] == "WITHHELD"
    assert convergence_eligibility(2.9)["eligible"] is False
    assert convergence_eligibility(3.0)["status"] == "OK"


def test_budget_extrapolation_recovers_sqrt_law_and_box_cost():
    t_obs = [4.0, 8.0, 16.0, 32.0]
    errors = [0.20 * math.sqrt(4.0 / t) for t in t_obs]  # err = 0.4 T^-0.5
    budget = budget_extrapolation(
        t_obs,
        errors,
        target_relative=0.03,
        throughput_cell_updates_per_s=5.0e5,
        reference_box_cells=400 * 64 * 132,
        reference_box_steps=120_000,
        own_grid_cells=150_000,
        eddy_turnover_steps=1500.0,
    )
    assert budget["fitted_exponent"] == pytest.approx(0.5, rel=1.0e-3)
    assert budget["exponent_assumed"] is False
    assert budget["predicted_turnovers_to_target"] == pytest.approx(
        (0.4 / 0.03) ** 2, rel=1.0e-3
    )
    assert budget["reference_box"]["predicted_hours"] == pytest.approx(
        400 * 64 * 132 * 120_000 / 5.0e5 / 3600.0, rel=1.0e-9
    )
    single = budget_extrapolation([10.0], [0.1])
    assert single["exponent_assumed"] is True

def test_budget_extrapolation_refuses_to_promise_a_systematic_floor():
    # error independent of the averaging window: a bias floor, not noise
    budget = budget_extrapolation(
        [4.0, 8.0, 16.0, 32.0], [0.151, 0.152, 0.151, 0.152], target_relative=0.03
    )
    assert budget["predicted_turnovers_to_target"] is None
    assert "systematic floor" in budget["extrapolation_refused_reason"]
