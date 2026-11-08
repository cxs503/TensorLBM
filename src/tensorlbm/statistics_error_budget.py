"""Sampling-error budget for time-averaged turbulence statistics.

Wall-resolved turbulence statistics (``u+``, Reynolds stresses, ...) are time
averages of a correlated stochastic signal.  Treating consecutive samples as
independent under-estimates the confidence interval by ``sqrt(2 * T_int)``,
which is precisely the failure mode that makes "u+ matches the log law within
1%" claims unfalsifiable.  This module supplies the missing ledger:

* :func:`autocorrelation_function` - normalised autocorrelation via FFT;
* :func:`integrated_autocorrelation_time` - Sokal-windowed ``T_int``;
* :func:`batch_means_estimate` - batch-means mean with Student-t interval;
* :func:`profile_statistics` - per-column (per wall-normal cell) mean, 95%
  confidence half-width and ``T_obs / T_int`` ledger for a (time, space)
  sample matrix.

The estimators are deliberately model-light: no assumption on the shape of the
autocorrelation function beyond "finite ``T_int`` and ``T_obs >> T_int``".
Both are checked and reported instead of silently trusted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

# Two-sided Student-t critical values at the 95% level, indexed by degrees of
# freedom.  Used instead of a hard scipy dependency; ``_t_critical`` falls back
# to the normal quantile for large ``dof`` and interpolates in between.
_T95_TABLE: tuple[tuple[int, float], ...] = (
    (1, 12.706204736174698),
    (2, 4.302652729749464),
    (3, 3.1824463052837086),
    (4, 2.7764451051977934),
    (5, 2.570581835636312),
    (6, 2.4469118487916707),
    (7, 2.3646242515927853),
    (8, 2.306004135204166),
    (9, 2.262157162798205),
    (10, 2.228138851986273),
    (12, 2.1788128296672282),
    (15, 2.131449545559323),
    (20, 2.0859634472658626),
    (25, 2.059538552753294),
    (30, 2.0422724563012373),
    (40, 2.0210753836551926),
    (60, 2.0002978220142578),
    (120, 1.9799304051840944),
)
_NORMAL_95 = 1.959963984540054


def _t_critical(dof: int, level: float = 0.95) -> float:
    """Two-sided Student-t critical value (table for 95%, normal otherwise)."""
    if dof < 1:
        raise ValueError("degrees of freedom must be positive")
    if not 0.5 < level < 1.0:
        raise ValueError("level must lie in (0.5, 1)")
    if abs(level - 0.95) > 1e-12:
        # Fall back to the large-sample normal quantile; conservative enough
        # for the bookkeeping role this value plays in the ledger.
        return _NORMAL_95 if level == 0.95 else _normal_quantile(level)
    dofs = [d for d, _ in _T95_TABLE]
    values = [v for _, v in _T95_TABLE]
    if dof in dofs:
        return values[dofs.index(dof)]
    if dof > dofs[-1]:
        return _NORMAL_95
    upper = next(d for d in dofs if d > dof)
    lower = max(d for d in dofs if d < dof)
    weight = (dof - lower) / (upper - lower)
    lo = values[dofs.index(lower)]
    hi = values[dofs.index(upper)]
    return float(lo + weight * (hi - lo))


def _normal_quantile(level: float) -> float:
    # Acklam's rational approximation; |error| < 1.2e-9 over the whole range.
    p = 0.5 + 0.5 * level
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def _as_time_series(samples: np.ndarray, *, name: str = "samples") -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D (time, column)")
    if array.shape[0] < 4:
        raise ValueError(f"{name} needs at least 4 time samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class IntegratedTimeEstimate:
    """Result of an integrated-autocorrelation-time estimate."""

    t_int: float
    window: int
    n_samples: int
    method: str
    t_obs: float
    ratio: float
    converged: bool
    notes: str = ""

    @property
    def n_effective(self) -> float:
        """Effective independent sample count ``n / (2 T_int)``."""
        return float(self.n_samples / (2.0 * self.t_int))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["n_effective"] = self.n_effective
        return payload


def autocorrelation_function(
    samples: np.ndarray,
    *,
    max_lag: int | None = None,
) -> np.ndarray:
    """Normalised autocorrelation ``rho(lag)`` with ``rho(0) = 1``.

    Computed by FFT with the biased (``1/n``) normalisation, which is the
    correct estimator for a finite record: it damps the noisy tail and keeps
    ``rho`` bounded by 1.
    """
    array = _as_time_series(samples)
    n = array.shape[0]
    centred = array - array.mean(axis=0, keepdims=True)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    spectrum = np.fft.rfft(centred, n=nfft, axis=0)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft, axis=0)[:n]
    variance = raw[0]
    if np.any(variance <= 0.0):
        raise ValueError("at least one column has zero variance")
    rho = raw / variance
    limit = n if max_lag is None else min(int(max_lag), n)
    return rho[:limit]


def integrated_autocorrelation_time(
    samples: np.ndarray,
    *,
    method: str = "sokal",
    c: float = 5.0,
    max_lag: int | None = None,
) -> IntegratedTimeEstimate:
    """Integrated autocorrelation time ``T_int`` in sample intervals.

    ``method="sokal"`` applies Sokal's automatic window: the smallest lag ``M``
    with ``M >= c * tau(M)`` where

        ``tau(M) = 0.5 + sum_{t=1..M} rho(t)``

    (the ``0.5`` is the standard self-counting correction for a discrete time
    series).  If no window satisfies the criterion before the record ends the
    estimate is flagged ``converged=False`` - the honest answer for a series
    shorter than its own correlation time.

    ``method="initial_positive"`` truncates at the first non-positive ``rho``,
    which is the cheapest and most robust variant for oversampled records.
    """
    array = _as_time_series(samples)
    if array.shape[1] != 1:
        raise ValueError(
            "integrated_autocorrelation_time handles one column; loop over columns",
        )
    n = array.shape[0]
    if method not in ("sokal", "initial_positive"):
        raise ValueError("method must be 'sokal' or 'initial_positive'")
    if c <= 0.0:
        raise ValueError("c must be positive")
    limit = n if max_lag is None else min(int(max_lag), n)
    rho = autocorrelation_function(array, max_lag=limit)[:, 0]
    cumulative = 0.5 + np.cumsum(rho[1:])
    if method == "initial_positive":
        non_positive = np.nonzero(rho[1:] <= 0.0)[0]
        window = int(non_positive[0]) if non_positive.size else int(rho.size - 1)
        t_int = float(max(cumulative[max(window - 1, 0)] if window > 0 else 0.5, 0.5))
        return IntegratedTimeEstimate(
            t_int=t_int,
            window=window,
            n_samples=n,
            method=method,
            t_obs=float(n),
            ratio=float(n / t_int),
            converged=bool(non_positive.size > 0),
            notes="truncated at first non-positive autocorrelation",
        )
    lags = np.arange(1, rho.size)
    accepted = np.nonzero(lags >= c * cumulative)[0]
    # A Sokal window is only meaningful inside the first half of the record;
    # beyond that the autocorrelation tail is dominated by its own noise and
    # the criterion closes spuriously early.
    half = max(int(n // 2), 2)
    if accepted.size:
        index = int(accepted[0])
        t_int = float(cumulative[index])
        window = int(lags[index])
        converged = bool(window <= half)
        notes = (
            ""
            if converged
            else (
                "Sokal window closed beyond n/2; T_int is a lower bound and the "
                "ratio T_obs/T_int is optimistic"
            )
        )
    else:
        t_int = float(cumulative[-1])
        window = int(lags[-1])
        converged = False
        notes = (
            "Sokal window not satisfied within the record; T_int is a lower "
            "bound and the ratio T_obs/T_int is optimistic"
        )
    return IntegratedTimeEstimate(
        t_int=max(t_int, 0.5),
        window=window,
        n_samples=n,
        method=method,
        t_obs=float(n),
        ratio=float(n / max(t_int, 0.5)),
        converged=converged,
        notes=notes,
    )


@dataclass(frozen=True)
class BatchMeansEstimate:
    """Batch-means mean estimate with a Student-t confidence interval."""

    mean: float
    stderr: float
    ci_half_width: float
    level: float
    n_samples: int
    n_batches: int
    batch_length: int
    dof: int
    naive_stderr: float | None = None
    t_int: float | None = None

    @property
    def ci(self) -> tuple[float, float]:
        return (self.mean - self.ci_half_width, self.mean + self.ci_half_width)

    @property
    def relative_ci_width(self) -> float:
        """Width of the 95% interval relative to ``|mean|``."""
        if self.mean == 0.0:
            return float("inf")
        return float(2.0 * self.ci_half_width / abs(self.mean))

    @property
    def variance_inflation(self) -> float | None:
        """How much the naive iid error under-states the true error."""
        if not self.naive_stderr or self.naive_stderr == 0.0:
            return None
        return float(self.stderr / self.naive_stderr)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ci"] = list(self.ci)
        payload["relative_ci_width"] = self.relative_ci_width
        payload["variance_inflation"] = self.variance_inflation
        return payload


def batch_means_estimate(
    samples: np.ndarray,
    *,
    n_batches: int = 20,
    level: float = 0.95,
) -> BatchMeansEstimate:
    """Batch-means estimator of the standard error of a time average.

    The record is cut into ``n_batches`` contiguous blocks; block means are
    approximately independent once the block length exceeds ``T_int``, so their
    sample standard deviation scaled by ``1/sqrt(n_batches)`` is a
    correlation-aware standard error.  The interval uses ``n_batches - 1``
    degrees of freedom.
    """
    array = _as_time_series(samples)
    n = array.shape[0]
    if not 2 <= n_batches <= n:
        raise ValueError("n_batches must satisfy 2 <= n_batches <= n_samples")
    usable = (n // n_batches) * n_batches
    if usable < 2 * n_batches:
        raise ValueError("record too short for the requested number of batches")
    trimmed = array[:usable]
    blocks = trimmed.reshape(n_batches, usable // n_batches, trimmed.shape[1])
    block_means = blocks.mean(axis=1)
    if block_means.shape[1] != 1:
        raise ValueError("batch_means_estimate expects a single column")
    values = block_means[:, 0]
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(n_batches))
    dof = n_batches - 1
    half = float(_t_critical(dof, level) * stderr)
    naive = float(trimmed[:, 0].std(ddof=1) / math.sqrt(usable))
    return BatchMeansEstimate(
        mean=mean,
        stderr=stderr,
        ci_half_width=half,
        level=level,
        n_samples=int(usable),
        n_batches=int(n_batches),
        batch_length=int(usable // n_batches),
        dof=int(dof),
        naive_stderr=naive,
    )


@dataclass(frozen=True)
class ProfileStatistics:
    """Per-column time-average statistics with an autocorrelation ledger.

    ``samples`` is interpreted as ``(n_time, n_column)``; a column is normally a
    wall-normal cell whose mean is taken over the homogeneous directions.
    """

    mean: np.ndarray
    ci_half_width: np.ndarray
    stderr: np.ndarray
    n_samples: int
    n_batches: int
    level: float
    t_int: np.ndarray
    t_obs: float
    ratio: np.ndarray
    converged: np.ndarray
    variance_inflation: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def relative_ci_width(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * self.ci_half_width / np.abs(self.mean)

    def ledger(self, labels: list[str] | None = None) -> list[dict[str, object]]:
        """JSON-ready per-column ledger rows."""
        rows: list[dict[str, object]] = []
        rel = self.relative_ci_width()
        for index in range(self.mean.size):
            row: dict[str, object] = {
                "index": index,
                "mean": float(self.mean[index]),
                "ci95_half_width": float(self.ci_half_width[index]),
                "relative_ci_width": (
                    float(rel[index]) if np.isfinite(rel[index]) else None
                ),
                "t_int": float(self.t_int[index]),
                "t_obs": float(self.t_obs),
                "obs_over_t_int": float(self.ratio[index]),
                "converged": bool(self.converged[index]),
                "variance_inflation": float(self.variance_inflation[index]),
            }
            if labels is not None and index < len(labels):
                row["label"] = labels[index]
            rows.append(row)
        return rows

    def to_dict(self, labels: list[str] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "n_samples": self.n_samples,
            "n_batches": self.n_batches,
            "level": self.level,
            "t_obs": self.t_obs,
            "mean": self.mean.tolist(),
            "ci95_half_width": self.ci_half_width.tolist(),
            "t_int": self.t_int.tolist(),
            "obs_over_t_int": self.ratio.tolist(),
            "converged": self.converged.tolist(),
        }
        if self.variance_inflation.size:
            payload["variance_inflation"] = self.variance_inflation.tolist()
        rows = self.ledger(labels)
        if rows:
            payload["ledger"] = rows
        return payload


def profile_statistics(
    samples: np.ndarray,
    *,
    n_batches: int = 20,
    level: float = 0.95,
    t_int_kwargs: dict[str, object] | None = None,
) -> ProfileStatistics:
    """Time-average every column with a correlation-aware 95% interval.

    Combines batch means (for the interval) with the Sokal window estimate (for
    the ``T_obs / T_int`` ledger), so the two independent error models can be
    cross-checked: for consistent statistics
    ``stderr_batch / stderr_naive ~ sqrt(2 T_int)``.
    """
    array = _as_time_series(samples)
    n, n_columns = array.shape
    if n_batches > n:
        raise ValueError("n_batches cannot exceed the number of samples")
    options = dict(t_int_kwargs or {})
    means = array.mean(axis=0)
    t_int = np.empty(n_columns, dtype=np.float64)
    converged = np.empty(n_columns, dtype=bool)
    ratio = np.empty(n_columns, dtype=np.float64)
    for column in range(n_columns):
        estimate = integrated_autocorrelation_time(
            array[:, column : column + 1],
            **options,
        )
        t_int[column] = estimate.t_int
        converged[column] = estimate.converged
        ratio[column] = n / estimate.t_int
    half = np.empty(n_columns, dtype=np.float64)
    stderr = np.empty(n_columns, dtype=np.float64)
    inflation = np.empty(n_columns, dtype=np.float64)
    for column in range(n_columns):
        estimate = batch_means_estimate(array[:, column : column + 1], n_batches=n_batches, level=level)
        half[column] = estimate.ci_half_width
        stderr[column] = estimate.stderr
        inflation[column] = estimate.variance_inflation or 1.0
    return ProfileStatistics(
        mean=means,
        ci_half_width=half,
        stderr=stderr,
        n_samples=int(n),
        n_batches=int(n_batches),
        level=float(level),
        t_int=t_int,
        t_obs=float(n),
        ratio=ratio,
        converged=converged,
        variance_inflation=inflation,
    )


def wall_units(
    mean_velocity: np.ndarray,
    wall_distance: np.ndarray,
    *,
    u_tau: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a mean profile to wall units ``(u+, y+)``."""
    if u_tau <= 0.0 or nu <= 0.0:
        raise ValueError("u_tau and nu must be positive")
    velocities = np.asarray(mean_velocity, dtype=np.float64)
    distances = np.asarray(wall_distance, dtype=np.float64)
    if velocities.shape != distances.shape:
        raise ValueError("mean_velocity and wall_distance must share a shape")
    if np.any(distances < 0.0):
        raise ValueError("wall_distance must be non-negative")
    return velocities / u_tau, distances * u_tau / nu


def peak_relative_ci_width(
    u_plus: np.ndarray,
    ci_half_width: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Largest 95%-interval width of ``u+`` relative to its local value."""
    values = np.asarray(u_plus, dtype=np.float64)
    half = np.asarray(ci_half_width, dtype=np.float64)
    if values.shape != half.shape:
        raise ValueError("u_plus and ci_half_width must share a shape")
    select = np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not np.any(select):
        raise ValueError("mask selects no samples")
    with np.errstate(divide="ignore", invalid="ignore"):
        widths = 2.0 * half[select] / np.abs(values[select])
    if not np.any(np.isfinite(widths)):
        raise ValueError("no finite relative interval width")
    return float(np.nanmax(widths))


__all__ = [
    "BatchMeansEstimate",
    "IntegratedTimeEstimate",
    "ProfileStatistics",
    "autocorrelation_function",
    "batch_means_estimate",
    "integrated_autocorrelation_time",
    "peak_relative_ci_width",
    "profile_statistics",
    "wall_units",
]