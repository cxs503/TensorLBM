"""Statistical error estimation for turbulent channel flow via batch means.

Implements the batch-means method with autocorrelation integral time (T_int)
to produce 95% confidence intervals on u+(y+) profiles.  This enables the
decomposition of the P20-N01 15.19% u+ gap into:

* **Statistical error** (sampling noise, quantified by CI half-width)
* **Model/domain-length bias** (systematic deviation beyond the CI)

References
----------
* Flyvbjerg & Petersen (1989), J. Chem. Phys. 91, 461 – error estimates
  from Monte Carlo simulations via block averaging.
* Madras & Sokal (1988), J. Stat. Phys. 52, 109 – autocorrelation time
  and optimal batch size.
* Kim, Moin & Moser (1987), JFM 177, 133 – DNS reference for Re_tau=180.

Key quantities
--------------
* ``T_int`` : integrated autocorrelation time (in samples)
* ``T_obs`` : total observation window (in samples)
* ``N_eff = T_obs / (2 * T_int)`` : effective number of independent samples
* ``CI_95 = 1.96 * sigma / sqrt(N_eff)`` : 95% confidence half-width
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BatchMeansResult",
    "AutocorrelationResult",
    "compute_autocorrelation_time",
    "compute_batch_means_ci",
    "compute_u_plus_ci",
    "stationarity_check",
]


@dataclass(frozen=True)
class AutocorrelationResult:
    """Result of autocorrelation analysis on a scalar time series."""

    acf: np.ndarray
    """Autocorrelation function values (lag 0 = 1.0)."""
    t_int: float
    """Integrated autocorrelation time (sum of ACF up to first negative crossing)."""
    n_samples: int
    """Total number of samples in the series."""
    n_eff: float
    """Effective number of independent samples: n_samples / (2 * t_int)."""


@dataclass(frozen=True)
class BatchMeansResult:
    """Result of batch-means confidence-interval analysis."""

    mean: float
    """Sample mean."""
    std: float
    """Sample standard deviation."""
    ci_95_half_width: float
    """95% confidence interval half-width."""
    t_int: float
    """Integrated autocorrelation time."""
    n_eff: float
    """Effective number of independent samples."""
    n_batches: int
    """Number of batches used."""
    batch_size: int
    """Samples per batch."""
    n_samples: int
    """Total samples."""
    ci_relative_pct: float
    """CI half-width as percentage of the mean."""


def _autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute the autocorrelation function of a zero-mean series.

    Uses FFT-based computation for O(N log N) efficiency.

    Parameters
    ----------
    x : np.ndarray
        Zero-mean time series.
    max_lag : int, optional
        Maximum lag to compute.  Default = len(x) // 4.

    Returns
    -------
    np.ndarray
        ACF values from lag 0 to max_lag.
    """
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n - 1)

    # FFT-based autocorrelation
    fft_size = 2 * n
    fx = np.fft.rfft(x, n=fft_size)
    acf_full = np.fft.irfft(fx * np.conj(fx), n=fft_size)[:n]

    var = acf_full[0]
    if var < 1e-30:
        # Constant or near-constant series: ACF is undefined, return all ones
        return np.ones(max_lag + 1)

    acf_full /= var  # normalize
    return acf_full[: max_lag + 1]


def compute_autocorrelation_time(
    x: np.ndarray,
    *,
    max_lag: int | None = None,
    method: str = "first_negative",
) -> AutocorrelationResult:
    """Compute the integrated autocorrelation time of a scalar series.

    Parameters
    ----------
    x : np.ndarray
        Time series (any mean; will be demeaned internally).
    max_lag : int, optional
        Maximum lag for ACF computation.
    method : str
        How to truncate the ACF sum:
        - ``"first_negative"``: sum until first negative ACF value (default)
        - ``"window"``: use a fixed window of max_lag

    Returns
    -------
    AutocorrelationResult
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)

    if n < 10:
        return AutocorrelationResult(
            acf=np.array([1.0]),
            t_int=1.0,
            n_samples=n,
            n_eff=float(n) / 2.0,
        )

    acf = _autocorrelation(x, max_lag=max_lag)

    # Check if series is constant (ACF all ones or NaN)
    if np.isnan(acf).any() or (acf == 1.0).all():
        return AutocorrelationResult(
            acf=np.ones(min(n // 4 + 1, n)),
            t_int=1.0,
            n_samples=n,
            n_eff=float(n) / 2.0,
        )

    # Integrated autocorrelation time: sum ACF until first negative crossing
    # (or until max_lag if no negative crossing)
    if method == "first_negative":
        t_int = 1.0  # lag 0 contributes 1
        for lag in range(1, len(acf)):
            if acf[lag] < 0:
                break
            t_int += 2.0 * acf[lag]  # factor 2 for positive/negative lags
    else:
        t_int = 1.0 + 2.0 * acf[1:].sum()

    t_int = max(t_int, 1.0)  # at least 1
    n_eff = n / (2.0 * t_int)

    return AutocorrelationResult(
        acf=acf,
        t_int=t_int,
        n_samples=n,
        n_eff=n_eff,
    )


def compute_batch_means_ci(
    x: np.ndarray,
    *,
    n_batches: int | None = None,
    confidence: float = 0.95,
) -> BatchMeansResult:
    """Compute confidence interval via the batch-means method.

    The series is divided into ``n_batches`` contiguous blocks.  The variance
    of the batch means estimates the variance of the sample mean accounting
    for autocorrelation.

    Parameters
    ----------
    x : np.ndarray
        Time series.
    n_batches : int, optional
        Number of batches.  If None, uses the rule of thumb
        ``max(30, int(sqrt(n)))``.
    confidence : float
        Confidence level (default 0.95 → z=1.96).

    Returns
    -------
    BatchMeansResult
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n_batches is None:
        n_batches = max(30, int(math.sqrt(n)))
    n_batches = min(n_batches, n // 2)  # at least 2 samples per batch

    batch_size = n // n_batches
    trimmed = x[: n_batches * batch_size]
    batches = trimmed.reshape(n_batches, batch_size)
    batch_means = batches.mean(axis=1)

    overall_mean = float(x.mean())
    overall_std = float(x.std(ddof=1))
    bm_mean = float(batch_means.mean())
    bm_std = float(batch_means.std(ddof=1))

    # CI from batch means variance
    z = 1.96 if confidence == 0.95 else 1.645  # 90%
    ci_half_width = z * bm_std / math.sqrt(n_batches)

    # Estimate T_int from batch means variance
    # Var(batch_means) ≈ sigma^2 * 2 * T_int / batch_size
    if overall_std > 1e-30 and batch_size > 0:
        t_int_est = (bm_std**2 * batch_size) / (2.0 * overall_std**2)
        t_int_est = max(t_int_est, 1.0)
    else:
        t_int_est = 1.0

    n_eff = n / (2.0 * t_int_est)
    ci_rel = abs(ci_half_width / max(abs(bm_mean), 1e-30)) * 100.0

    return BatchMeansResult(
        mean=bm_mean,
        std=overall_std,
        ci_95_half_width=ci_half_width,
        t_int=t_int_est,
        n_eff=n_eff,
        n_batches=n_batches,
        batch_size=batch_size,
        n_samples=n,
        ci_relative_pct=ci_rel,
    )


def compute_u_plus_ci(
    u_plus_samples: np.ndarray,
    *,
    y_plus: np.ndarray | None = None,
    n_batches: int | None = None,
    confidence: float = 0.95,
) -> dict:
    """Compute 95% CI on u+(y+) profile from time-series samples.

    Parameters
    ----------
    u_plus_samples : np.ndarray
        Shape ``(n_samples, n_y)`` – u+ values at each wall-normal location
        for each sampling instant.
    y_plus : np.ndarray, optional
        Wall-normal coordinate (for labeling).
    n_batches : int, optional
        Number of batches for batch-means analysis.
    confidence : float
        Confidence level.

    Returns
    -------
    dict
        Keys: ``mean``, ``ci_lower``, ``ci_upper``, ``ci_half_width``,
        ``ci_relative_pct``, ``t_int``, ``n_eff``, ``n_batches``, ``batch_size``.
        Each value is an array of shape ``(n_y,)`` except scalars.
    """
    u_plus_samples = np.asarray(u_plus_samples, dtype=np.float64)
    if u_plus_samples.ndim != 2:
        raise ValueError("u_plus_samples must be 2D: (n_samples, n_y)")

    n_samples, n_y = u_plus_samples.shape

    mean_profile = u_plus_samples.mean(axis=0)
    ci_half_width = np.zeros(n_y)
    t_int_profile = np.zeros(n_y)
    n_eff_profile = np.zeros(n_y)

    for j in range(n_y):
        series = u_plus_samples[:, j]
        if series.std() < 1e-30:
            # Laminar or frozen – no statistical error
            ci_half_width[j] = 0.0
            t_int_profile[j] = 1.0
            n_eff_profile[j] = float(n_samples)
            continue

        result = compute_batch_means_ci(series, n_batches=n_batches, confidence=confidence)
        ci_half_width[j] = result.ci_95_half_width
        t_int_profile[j] = result.t_int
        n_eff_profile[j] = result.n_eff

    ci_lower = mean_profile - ci_half_width
    ci_upper = mean_profile + ci_half_width
    ci_rel = np.abs(ci_half_width / np.maximum(np.abs(mean_profile), 1e-30)) * 100.0

    # Overall T_int: average over the log-law region (y+ > 30)
    if y_plus is not None:
        log_law_mask = y_plus > 30.0
        if log_law_mask.any():
            t_int_overall = float(np.mean(t_int_profile[log_law_mask]))
        else:
            t_int_overall = float(np.mean(t_int_profile))
    else:
        t_int_overall = float(np.mean(t_int_profile))

    return {
        "mean": mean_profile,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_half_width": ci_half_width,
        "ci_relative_pct": ci_rel,
        "t_int_profile": t_int_profile,
        "t_int_overall": t_int_overall,
        "n_eff_profile": n_eff_profile,
        "n_batches": n_batches or max(30, int(math.sqrt(n_samples))),
        "batch_size": n_samples // (n_batches or max(30, int(math.sqrt(n_samples)))),
        "n_samples": n_samples,
        "n_y": n_y,
    }


def stationarity_check(
    u_plus_samples: np.ndarray,
    *,
    window_fraction: float = 0.5,
    threshold: float = 1.0,
) -> dict:
    """Check stationarity by comparing two half-windows of the u+ profile.

    This implements criterion C1: adjacent half-window u+ profile max diff
    must be ≤ 1 u+ unit.

    Parameters
    ----------
    u_plus_samples : np.ndarray
        Shape ``(n_samples, n_y)``.
    window_fraction : float
        Fraction of samples in each half-window (default 0.5).
    threshold : float
        Maximum allowed difference in u+ units (default 1.0).

    Returns
    -------
    dict
        Keys: ``passed``, ``max_diff``, ``first_half_mean``, ``second_half_mean``,
        ``threshold``.
    """
    u_plus_samples = np.asarray(u_plus_samples, dtype=np.float64)
    n = u_plus_samples.shape[0]
    split = int(n * window_fraction)

    first_half = u_plus_samples[:split].mean(axis=0)
    second_half = u_plus_samples[split:].mean(axis=0)
    diff = np.abs(second_half - first_half)
    max_diff = float(diff.max())

    return {
        "passed": max_diff <= threshold,
        "max_diff": max_diff,
        "threshold": threshold,
        "first_half_mean": first_half,
        "second_half_mean": second_half,
        "first_half_n": split,
        "second_half_n": n - split,
    }
