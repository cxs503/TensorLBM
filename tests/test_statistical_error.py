"""Tests for the statistical error estimation module.

Verifies:
- Autocorrelation time computation
- Batch-means confidence interval
- u+ profile CI computation
- Stationarity check
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.statistical_error import (
    AutocorrelationResult,
    BatchMeansResult,
    compute_autocorrelation_time,
    compute_batch_means_ci,
    compute_u_plus_ci,
    stationarity_check,
)


class TestAutocorrelationTime:
    """Tests for compute_autocorrelation_time."""

    def test_white_noise_t_int_near_one(self):
        """White noise should have T_int ≈ 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        result = compute_autocorrelation_time(x)
        assert 0.8 < result.t_int < 1.5, f"T_int={result.t_int} for white noise"
        assert result.n_samples == 10000

    def test_ar1_process_t_int(self):
        """AR(1) process with phi=0.8 should have T_int ≈ (1+phi)/(1-phi) = 9."""
        rng = np.random.default_rng(42)
        n = 50000
        phi = 0.8
        x = np.zeros(n)
        x[0] = rng.standard_normal()
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.standard_normal() * math.sqrt(1 - phi**2)

        result = compute_autocorrelation_time(x)
        # Theoretical T_int = (1+phi)/(1-phi) = 9
        # With finite samples and first-negative truncation, expect ~5-12
        assert 3.0 < result.t_int < 15.0, f"T_int={result.t_int} for AR(1) phi=0.8"

    def test_constant_series(self):
        """Constant series should have T_int = 1 and n_eff = n/2."""
        x = np.ones(100)
        result = compute_autocorrelation_time(x)
        assert result.t_int == 1.0
        assert result.n_eff == 50.0

    def test_short_series(self):
        """Very short series should not crash."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_autocorrelation_time(x)
        assert result.n_samples == 5
        assert result.t_int >= 1.0


class TestBatchMeansCI:
    """Tests for compute_batch_means_ci."""

    def test_known_mean(self):
        """CI should contain the true mean for a known distribution."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        x = rng.normal(true_mean, 1.0, 10000)
        result = compute_batch_means_ci(x, n_batches=50)

        assert abs(result.mean - true_mean) < 0.1
        assert result.ci_95_half_width > 0
        assert result.n_batches == 50
        assert result.n_samples == 10000

    def test_ci_relative_pct(self):
        """CI relative percentage should be reasonable."""
        rng = np.random.default_rng(42)
        x = rng.normal(100.0, 5.0, 10000)
        result = compute_batch_means_ci(x, n_batches=50)

        # With 10000 samples and std=5, CI should be small
        assert result.ci_relative_pct < 1.0  # < 1%

    def test_autocorrelated_series_wider_ci(self):
        """Autocorrelated series should have wider CI than iid."""
        rng = np.random.default_rng(42)
        n = 10000

        # IID
        x_iid = rng.standard_normal(n)
        ci_iid = compute_batch_means_ci(x_iid, n_batches=50)

        # AR(1) with phi=0.9
        phi = 0.9
        x_ar = np.zeros(n)
        x_ar[0] = rng.standard_normal()
        for i in range(1, n):
            x_ar[i] = phi * x_ar[i - 1] + rng.standard_normal() * math.sqrt(1 - phi**2)
        ci_ar = compute_batch_means_ci(x_ar, n_batches=50)

        # AR(1) should have wider CI
        assert ci_ar.ci_95_half_width > ci_iid.ci_95_half_width

    def test_constant_series_zero_ci(self):
        """Constant series should have zero CI."""
        x = np.ones(100) * 42.0
        result = compute_batch_means_ci(x, n_batches=10)
        assert result.ci_95_half_width == 0.0
        assert result.mean == 42.0


class TestUPlusCI:
    """Tests for compute_u_plus_ci."""

    def test_basic_shape(self):
        """Output should have correct shapes."""
        n_samples = 500
        n_y = 30
        rng = np.random.default_rng(42)
        u_plus_samples = rng.uniform(10, 20, (n_samples, n_y))
        y_plus = np.linspace(1, 100, n_y)

        result = compute_u_plus_ci(u_plus_samples, y_plus=y_plus, n_batches=30)

        assert result["mean"].shape == (n_y,)
        assert result["ci_lower"].shape == (n_y,)
        assert result["ci_upper"].shape == (n_y,)
        assert result["ci_half_width"].shape == (n_y,)
        assert result["n_samples"] == n_samples
        assert result["n_y"] == n_y

    def test_ci_bounds(self):
        """CI lower should be below mean, CI upper above."""
        rng = np.random.default_rng(42)
        u_plus_samples = rng.uniform(10, 20, (200, 20))

        result = compute_u_plus_ci(u_plus_samples, n_batches=20)

        assert (result["ci_lower"] <= result["mean"]).all()
        assert (result["ci_upper"] >= result["mean"]).all()

    def test_laminar_profile_zero_ci(self):
        """Laminar (constant) profile should have zero CI."""
        n_samples = 100
        n_y = 10
        u_plus_samples = np.ones((n_samples, n_y)) * 15.0

        result = compute_u_plus_ci(u_plus_samples, n_batches=10)

        assert (result["ci_half_width"] == 0.0).all()


class TestStationarityCheck:
    """Tests for stationarity_check."""

    def test_stationary_series_passes(self):
        """Stationary series should pass the check."""
        rng = np.random.default_rng(42)
        n_samples = 1000
        n_y = 20
        # Same mean throughout
        u_plus_samples = rng.normal(15.0, 1.0, (n_samples, n_y))

        result = stationarity_check(u_plus_samples, threshold=1.0)

        assert result["passed"] is True
        assert result["max_diff"] < 1.0

    def test_non_stationary_series_fails(self):
        """Series with drift should fail the check."""
        n_samples = 1000
        n_y = 20
        # Linear drift
        t = np.linspace(0, 10, n_samples)[:, None]
        u_plus_samples = 10.0 + t + np.random.default_rng(42).standard_normal((n_samples, n_y)) * 0.1

        result = stationarity_check(u_plus_samples, threshold=1.0)

        assert result["passed"] is False
        assert result["max_diff"] > 1.0

    def test_constant_series_passes(self):
        """Constant series should pass trivially."""
        u_plus_samples = np.ones((100, 10)) * 15.0
        result = stationarity_check(u_plus_samples, threshold=1.0)
        assert result["passed"] is True
        assert result["max_diff"] == 0.0
