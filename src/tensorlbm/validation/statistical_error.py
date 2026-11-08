"""Statistical-vs-model error decomposition for time-averaged channel profiles.

Why this module exists
----------------------
A wall-resolved channel comparison reports something like "the u+ profile is
15.19 % away from Moser-Kim-Mansour (MKM) in relative L2".  That number has two
very different ingredients and they demand opposite responses:

* **statistical uncertainty** -- a finite averaging window only samples
  ``N_eff ~ T_obs / (2 T_int)`` independent large-scale structures, so the
  profile itself is uncertain.  Its cure is *more machine time*.
* **model / domain bias** -- the discretisation, the sub-grid closure, or a
  computational box shorter than the physical integral scale shifts the profile
  systematically.  Its cure is *a different model or box*, and more machine
  time alone will never remove it.

Telling the two apart is a metrology problem, not a plotting problem.  This
module provides the small, dependency-light toolkit that makes the split
machine-checkable:

``integral_time``
    Autocorrelation integral time ``T_int`` of a sampled signal (Sokal automatic
    windowing), the effective sample count and the resulting relative standard
    error of the mean.  ``T_int`` is what converts "I ran 30 000 steps" into
    "I have N_eff independent samples".
``batch_means_summary``
    Batch-means confidence interval of a time-averaged profile: split the
    averaging window into ``B`` *consecutive* batches, treat the batch means as
    the sample, and form a Student-t interval.  This is the only variance
    estimator that stays honest when the samples are correlated.
``half_window_max_diff``
    The steady-state gate: the means of the two adjacent halves of the window
    must agree within a stated tolerance, otherwise the window straddles a
    transient and no CI is meaningful.
``metric_from_batches``
    Propagates the batch variance onto the *error metric itself*, giving
    ``U_SN`` -- the 95 % half-width of the reported relative L2 error.  This is
    the yardstick for the attribution verdict.
``decompose_error``
    The verdict: of the observed error, how much is inside ``U_SN``
    (statistics), how much is a resolved but finite domain-length effect, and
    how much is left over (model / discretisation).  Binary, fail-closed.
``domain_length_check`` / ``negative_control_check``
    Two designed contrasts that make bias *visible*: same grid twice with the
    box length changed (domain), and the same grid with the sub-grid closure
    switched on (SGS).  A contrast only counts if it exceeds the CI of the pair.
``budget_extrapolation``
    Error-vs-averaging-window fit ``err = a T^-p`` plus the machine-time
    forecast for a reference DNS box.

All estimators are NumPy-only and side-effect free.  The channel kernel records
raw material (per-batch moment sums and a wall-shear time series); every number
below is reproducible from that raw record.

References
----------
Moser, Kim & Mansour, Phys. Fluids 11, 943 (1999).  Sokal, *Monte Carlo
Methods in Statistical Mechanics* (1989) for the automatic ACF window.  Flyvbjerg
& Petersen, J. Chem. Phys. 91, 461 (1989).  Oliver et al., "Estimating
uncertainties in statistics from correlated data", J. Comput. Phys. 228 (2014);
Bendat & Piersol, *Random Data* (2010), ch. 8 for batch means.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "student_t_quantile",
    "integral_time",
    "batch_means_summary",
    "half_window_max_diff",
    "relative_l2_error",
    "metric_from_batches",
    "decompose_error",
    "domain_length_check",
    "negative_control_check",
    "budget_extrapolation",
    "convergence_eligibility",
]


# --------------------------------------------------------------------------
# Student-t quantile (no SciPy dependency)
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3.0e-14) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, dof: float) -> float:
    if dof <= 0.0:
        raise ValueError("dof must be positive")
    x = dof / (dof + t * t)
    tail = 0.5 * _betai(0.5 * dof, 0.5, x)
    return 1.0 - tail if t > 0.0 else tail


def student_t_quantile(p: float, dof: float) -> float:
    """Quantile of Student's t distribution by bisection on the exact CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0,1)")
    if dof <= 0.0:
        raise ValueError("dof must be positive")
    if p == 0.5:
        return 0.0
    lo, hi = 0.0, 1.0
    # expand the bracket until the CDF is bracketed (t can be huge for tiny dof)
    while _student_t_cdf(hi, dof) < p:
        hi *= 2.0
        if hi > 1.0e12:
            raise RuntimeError("t quantile bracket failed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Autocorrelation integral time
# --------------------------------------------------------------------------
def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    n = values.size
    centered = values - values.mean()
    variance = float(np.dot(centered, centered) / n)
    if variance <= 0.0:
        raise ValueError("signal has zero variance")
    nfft = 1 << int(math.ceil(math.log2(max(2 * n, 4))))
    spectrum = np.fft.rfft(centered, nfft)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), nfft)[: max_lag + 1]
    # unbiased normalisation: lag k has n-k contributions
    lags = np.arange(acf.size)
    return acf / (variance * np.maximum(n - lags, 1))


def integral_time(
    series: Iterable[float],
    *,
    method: str = "sokal",
    window_constant: float = 5.0,
    max_lag: int | None = None,
) -> dict:
    """Autocorrelation integral time of a sampled scalar signal.

    ``T_int`` is returned in *samples* (multiply by the sampling interval to get
    physical steps).  The relative standard error of the mean follows from
    ``sigma/sqrt(N_eff)`` with ``N_eff = N / (2 T_int)``; the mean itself is
    **not** corrected by ``2 T_int`` (that common prefactor cancels here because
    ``N_eff`` already carries it).

    Parameters
    ----------
    method
        ``"sokal"`` (automatic window: smallest ``M`` with ``M >= c tau_int(M)``)
        or ``"first_negative"`` (truncate the ACF sum at its first non-positive
        value -- noisier but assumption-free).
    """
    values = np.asarray([float(item) for item in series], dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 16:
        raise ValueError("need at least 16 samples for an integral time")
    lag_cap = n // 2 if max_lag is None else int(min(max_lag, n // 2))
    if lag_cap < 2:
        raise ValueError("max_lag too small")
    acf = _autocorrelation(values, lag_cap)
    if method not in {"sokal", "first_negative"}:
        raise ValueError("method must be 'sokal' or 'first_negative'")

    clamped = False
    if method == "sokal":
        tau = 0.5
        window = lag_cap
        for m in range(1, lag_cap + 1):
            tau += float(acf[m])
            if tau < 0.0:
                # Strongly anti-correlated signal (e.g. an acoustic oscillation
                # dominating a wall-shear trace): the ACF sum is meaningless.
                # Report the uncorrelated floor instead of a negative T_int.
                tau = 0.5
                window = m
                clamped = True
                break
            if m >= window_constant * tau:
                window = m
                break
        t_int = float(max(tau, 0.5))
        truncated = False
    else:
        window = lag_cap
        t_int = 0.5
        truncated = True
        for m in range(1, lag_cap + 1):
            value = float(acf[m])
            if value <= 0.0:
                window = m - 1
                truncated = False
                break
            t_int += value
        if t_int < 0.5:
            t_int = 0.5
            clamped = True

    n_eff = n / (2.0 * t_int)
    sigma = float(values.std(ddof=1))
    mean = float(values.mean())
    return {
        "method": method,
        "n_samples": int(n),
        "t_int_samples": t_int,
        "acf_window": int(window),
        "acf_window_truncated_at_cap": bool(truncated),
        "clamped_at_uncorrelated_floor": bool(clamped),
        "n_effective_samples": float(n_eff),
        "signal_mean": mean,
        "signal_sigma": sigma,
        "relative_standard_error_of_mean": float(
            sigma / math.sqrt(max(n_eff, 1.0)) / abs(mean)
        )
        if mean != 0.0
        else None,
        "standard_error_of_mean": float(sigma / math.sqrt(max(n_eff, 1.0))),
        "autocorrelation": [float(value) for value in acf[: min(window + 1, 64)]],
    }


# --------------------------------------------------------------------------
# Batch means confidence interval
# --------------------------------------------------------------------------
def batch_means_summary(
    batches: Iterable[Iterable[float]],
    *,
    confidence: float = 0.95,
) -> dict:
    """Student-t batch-means interval for one or many time-averaged quantities.

    ``batches`` is ``(n_batches, n_points)`` in **time order** (each row a
    consecutive batch mean); a 1-D input is read as ``(n_batches, 1)``.
    """
    data = np.asarray(batches, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("batches must be 1-D (time axis) or 2-D (time x points)")
    n_batches = data.shape[0]
    if n_batches < 2:
        raise ValueError("need at least two batches for a confidence interval")
    if not np.all(np.isfinite(data)):
        raise ValueError("batch means must be finite")
    dof = n_batches - 1
    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / math.sqrt(n_batches)
    t_critical = student_t_quantile(0.5 + 0.5 * confidence, dof)
    half = t_critical * sem
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(np.abs(mean) > 0.0, half / np.abs(mean), np.inf)
    finite_rel = relative[np.isfinite(relative)]
    return {
        "confidence": confidence,
        "n_batches": int(n_batches),
        "dof": int(dof),
        "t_critical": float(t_critical),
        "mean": [float(value) for value in mean],
        "sem": [float(value) for value in sem],
        "ci_half": [float(value) for value in half],
        "ci_half_relative": [float(value) for value in relative],
        "max_ci_half": float(np.max(half)),
        "max_ci_half_relative": float(np.max(finite_rel)) if finite_rel.size else None,
    }


def half_window_max_diff(batches: Iterable[Iterable[float]]) -> dict:
    """Steady-state gate between the two adjacent halves of the window.

    The batch means are split in the middle of the *time* axis; the metric is
    ``max_i | mean(first half)_i - mean(second half)_i |`` in the units of the
    input (u+ units when wall-unit profiles are supplied).  A drifting window
    (transient not finished) shows up here long before it shows up in the mean.
    """
    data = np.asarray(batches, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("batches must be 1-D (time axis) or 2-D (time x points)")
    n_batches = data.shape[0]
    if n_batches < 4:
        raise ValueError("need at least four batches to form adjacent halves")
    split = n_batches // 2
    first = data[:split].mean(axis=0)
    second = data[split:].mean(axis=0)
    diff = np.abs(first - second)
    # secondary, noise-dominated view: worst consecutive-batch jump
    consecutive = np.abs(np.diff(data, axis=0))
    return {
        "n_batches": int(n_batches),
        "split_index": int(split),
        "half_window_max_diff": float(np.max(diff)),
        "half_window_rms_diff": float(np.sqrt(np.mean(diff**2))),
        "half_window_max_diff_point": int(np.argmax(diff)),
        "max_consecutive_batch_diff": float(np.max(consecutive)),
        "first_half_mean": [float(value) for value in first],
        "second_half_mean": [float(value) for value in second],
    }


def relative_l2_error(simulation: Iterable[float], reference: Iterable[float]) -> float:
    """``||sim - ref||_2 / ||ref||_2`` -- the convention used by the MKM gate."""
    sim = np.asarray([float(item) for item in simulation], dtype=float)
    ref = np.asarray([float(item) for item in reference], dtype=float)
    if sim.shape != ref.shape:
        raise ValueError("simulation and reference profiles differ in length")
    denominator = float(np.linalg.norm(ref))
    if denominator <= 0.0:
        raise ValueError("reference profile has zero norm")
    return float(np.linalg.norm(sim - ref) / denominator)


def metric_from_batches(
    batches: Iterable[Iterable[float]],
    reference: Iterable[float],
    *,
    confidence: float = 0.95,
) -> dict:
    """Batch-wise relative L2 error -> ``U_SN``, the simulation-side uncertainty.

    Each batch mean gives one realisation of the *same* metric that is reported
    against the (fixed, error-free) reference.  Their scatter is therefore the
    uncertainty of the reported error caused by finite averaging -- the yardstick
    ``U_SN`` used by :func:`decompose_error`.
    """
    data = np.asarray(batches, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("batches must be 1-D (time axis) or 2-D (time x points)")
    ref = np.asarray([float(item) for item in reference], dtype=float)
    if data.shape[1] != ref.size:
        raise ValueError("batch width and reference length differ")
    per_batch = [relative_l2_error(row, ref) for row in data]
    summary = batch_means_summary(per_batch, confidence=confidence)
    pointwise = batch_means_summary(data, confidence=confidence)
    return {
        "metric": "relative_l2",
        "n_batches": summary["n_batches"],
        "dof": summary["dof"],
        "t_critical": summary["t_critical"],
        "per_batch": per_batch,
        "error_mean": summary["mean"][0],
        "error_ci_half": summary["ci_half"][0],
        "error_ci_low": summary["mean"][0] - summary["ci_half"][0],
        "error_ci_high": summary["mean"][0] + summary["ci_half"][0],
        "u_sn": summary["ci_half"][0],
        "pointwise_max_ci_half": pointwise["max_ci_half"],
        "pointwise_ci_half": pointwise["ci_half"],
        "profile_mean": pointwise["mean"],
    }


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def decompose_error(
    simulation: Iterable[float],
    reference: Iterable[float],
    metric: dict,
    *,
    tolerance_relative: float = 0.03,
    domain_shift_relative: float | None = None,
) -> dict:
    """Binary attribution of an observed profile error.

    Definitions (recorded in the artifact so the verdict is auditable):

    ``u_sn``
        the 95 % half-width of the error metric itself (from
        :func:`metric_from_batches`) -- the simulation-side numerical/statistical
        yardstick;
    ``residual_beyond_ci``
        ``error_mean - u_sn`` (clamped at 0): the part of the error that finite
        averaging cannot explain;
    ``domain_shift``
        the *measured* relative L2 shift between the two box lengths (clamped at
        0), only counted when :func:`domain_length_check` calls it resolved;
    ``unexplained``
        ``residual_beyond_ci - domain_shift`` (clamped at 0): attributed to the
        model (closure + discretisation), under the stated assumption that the
        domain and model effects add linearly.

    Verdict (fail-closed, exactly one of):
    ``"statistics_dominated"`` when ``residual_beyond_ci <= tolerance_relative``;
    ``"model_or_domain_dominated"`` otherwise.
    """
    error_mean = float(metric["error_mean"])
    u_sn = float(metric["u_sn"])
    observed = relative_l2_error(simulation, reference)
    residual = max(error_mean - u_sn, 0.0)
    domain = 0.0
    if domain_shift_relative is not None and math.isfinite(domain_shift_relative):
        domain = max(float(domain_shift_relative), 0.0)
    unexplained = max(residual - domain, 0.0)
    significant = residual > tolerance_relative
    return {
        "observed_relative_l2": observed,
        "observed_relative_l2_from_batches": error_mean,
        "u_sn": u_sn,
        "u_sn_over_error": u_sn / error_mean if error_mean > 0.0 else None,
        "error_ci_low": float(metric["error_ci_low"]),
        "error_ci_high": float(metric["error_ci_high"]),
        "residual_beyond_ci": residual,
        "residual_in_u_sn_units": residual / u_sn if u_sn > 0.0 else None,
        "domain_shift_relative": domain if domain_shift_relative is not None else None,
        "unexplained_model_relative": unexplained,
        "tolerance_relative": tolerance_relative,
        "significant": bool(significant),
        "verdict": "model_or_domain_dominated" if significant else "statistics_dominated",
        "assumption": (
            "domain and model effects are additive in the relative L2 metric; "
            "unexplained = max(error - u_sn - domain_shift, 0)"
        ),
    }


def domain_length_check(
    short_profile: Iterable[float],
    long_profile: Iterable[float],
    *,
    short_ci_half: Iterable[float] | None = None,
    long_ci_half: Iterable[float] | None = None,
    tolerance_relative: float = 0.03,
) -> dict:
    """Same-grid, two-box-length cross-check (``Lx = 2 pi delta`` vs ``4 pi delta``).

    ``within_tolerance`` is the domain gate; ``resolved`` says whether the
    difference is larger than the combined statistical CI, i.e. whether the box
    effect is a *signal* rather than noise (a contrast that hides inside its own
    error bars cannot be used for attribution).
    """
    short = np.asarray([float(item) for item in short_profile], dtype=float)
    long = np.asarray([float(item) for item in long_profile], dtype=float)
    if short.shape != long.shape:
        raise ValueError("the two profiles must share the comparison grid")
    diff = short - long
    metric = relative_l2_error(short, long)
    result = {
        "relative_l2_difference": metric,
        "max_abs_difference": float(np.max(np.abs(diff))),
        "rms_abs_difference": float(np.sqrt(np.mean(diff**2))),
        "signed_mean_difference": float(np.mean(diff)),
        "within_tolerance": bool(metric <= tolerance_relative),
        "tolerance_relative": tolerance_relative,
        "resolved": None,
        "combined_ci_half": None,
    }
    if short_ci_half is not None and long_ci_half is not None:
        ci_short = np.asarray([float(item) for item in short_ci_half], dtype=float)
        ci_long = np.asarray([float(item) for item in long_ci_half], dtype=float)
        combined = np.sqrt(ci_short**2 + ci_long**2)
        result["combined_ci_half"] = [float(value) for value in combined]
        result["max_combined_ci_half"] = float(np.max(combined))
        result["points_beyond_ci"] = int(np.sum(np.abs(diff) > combined))
        result["fraction_points_beyond_ci"] = float(np.mean(np.abs(diff) > combined))
        result["resolved"] = bool(np.max(np.abs(diff)) > np.max(combined))
    return result


def negative_control_check(
    control_profile: Iterable[float],
    variant_profile: Iterable[float],
    *,
    control_ci_half: Iterable[float] | None = None,
    variant_ci_half: Iterable[float] | None = None,
) -> dict:
    """SGS negative control: the ``C_s = 0`` and ``C_s = 0.1`` u+ profiles must differ
    by more than the statistical CI of the pair (``separated``)."""
    control = np.asarray([float(item) for item in control_profile], dtype=float)
    variant = np.asarray([float(item) for item in variant_profile], dtype=float)
    if control.shape != variant.shape:
        raise ValueError("the two profiles must share the comparison grid")
    diff = variant - control
    result = {
        "max_abs_difference": float(np.max(np.abs(diff))),
        "rms_abs_difference": float(np.sqrt(np.mean(diff**2))),
        "signed_mean_difference": float(np.mean(diff)),
        "relative_l2_difference": relative_l2_error(variant, control),
        "separated": None,
    }
    if control_ci_half is not None and variant_ci_half is not None:
        ci_control = np.asarray([float(item) for item in control_ci_half], dtype=float)
        ci_variant = np.asarray([float(item) for item in variant_ci_half], dtype=float)
        combined = np.sqrt(ci_control**2 + ci_variant**2)
        result["max_combined_ci_half"] = float(np.max(combined))
        result["points_beyond_ci"] = int(np.sum(np.abs(diff) > combined))
        result["fraction_points_beyond_ci"] = float(np.mean(np.abs(diff) > combined))
        result["separation_in_ci_units"] = float(
            np.max(np.abs(diff)) / max(float(np.max(combined)), 1.0e-300)
        )
        result["separated"] = bool(np.max(np.abs(diff)) > np.max(combined))
    return result


def convergence_eligibility(
    observed_eddy_turnovers: float,
    *,
    minimum_eddy_turnovers: float = 3.0,
) -> dict:
    """Fail-closed guard: below three turnover times a CI is not trustworthy."""
    eligible = float(observed_eddy_turnovers) >= float(minimum_eddy_turnovers)
    return {
        "observed_eddy_turnovers": float(observed_eddy_turnovers),
        "minimum_eddy_turnovers": float(minimum_eddy_turnovers),
        "eligible": bool(eligible),
        "status": "OK" if eligible else "WITHHELD",
        "reason": (
            "averaging window covers at least three eddy turnovers"
            if eligible
            else "fewer than three eddy turnovers: the batch-means CI is not estimable"
        ),
    }


# --------------------------------------------------------------------------
# Budget extrapolation
# --------------------------------------------------------------------------
def budget_extrapolation(
    t_obs_turnovers: Sequence[float],
    errors: Sequence[float],
    *,
    target_relative: float = 0.03,
    throughput_cell_updates_per_s: float | None = None,
    reference_box_cells: int | None = None,
    reference_box_steps: int | None = None,
    own_grid_cells: int | None = None,
    eddy_turnover_steps: float | None = None,
) -> dict:
    """Fit ``err = a T_obs^-p`` and forecast the machine time to reach ``target``.

    With two or more points the exponent is fitted (log-log least squares, so
    the scaling is *measured*, not assumed); with one point ``p = 1/2`` (the
    pure random-error law) is assumed and reported as such.  The machine-time
    forecast uses the measured throughput of this campaign, so it inherits the
    same hardware/compiler caveat.
    """
    t = np.asarray([float(item) for item in t_obs_turnovers], dtype=float)
    e = np.asarray([float(item) for item in errors], dtype=float)
    keep = (t > 0.0) & (e > 0.0) & np.isfinite(t) & np.isfinite(e)
    t, e = t[keep], e[keep]
    if t.size == 0:
        raise ValueError("no usable (T_obs, error) points")
    if t.size >= 2:
        slope, intercept = np.polyfit(np.log(t), np.log(e), 1)
        exponent = float(-slope)
        coefficient = float(math.exp(intercept))
        assumed = False
    else:
        exponent = 0.5
        coefficient = float(e[0] * math.sqrt(t[0]))
        assumed = True
    fit = coefficient * np.power(t, -exponent)
    # A weak (or negative) exponent means the error does not shrink with the
    # averaging window at all -- the signature of a systematic floor.  Refusing
    # to extrapolate is the honest answer there: no finite T_obs reaches the
    # target, and pretending otherwise would promise machine time that cannot
    # deliver.
    predicted_t = None
    refusal = None
    if exponent <= 0.05:
        refusal = (
            "fitted exponent <= 0.05: the error does not decrease with the "
            "averaging window (systematic floor), so no finite T_obs reaches the target"
        )
    else:
        try:
            candidate = (coefficient / target_relative) ** (1.0 / exponent)
        except OverflowError:
            candidate = float("inf")
        if not math.isfinite(candidate) or candidate > 1.0e9:
            refusal = "extrapolated averaging window exceeds 1e9 turnovers"
        else:
            predicted_t = float(candidate)
    result = {
        "target_relative": target_relative,
        "points": [{"t_obs_turnovers": float(a), "relative_error": float(b)} for a, b in zip(t, e)],
        "fitted_coefficient": coefficient,
        "fitted_exponent": exponent,
        "exponent_assumed": assumed,
        "fitted_errors": [float(value) for value in fit],
        "predicted_turnovers_to_target": predicted_t,
        "extrapolation_refused_reason": refusal,
        "throughput_cell_updates_per_s": throughput_cell_updates_per_s,
    }
    if (
        predicted_t is not None
        and throughput_cell_updates_per_s
        and own_grid_cells
        and eddy_turnover_steps
    ):
        steps = predicted_t * float(eddy_turnover_steps)
        result["own_box_predicted_hours"] = (
            float(own_grid_cells) * steps / float(throughput_cell_updates_per_s) / 3600.0
        )
        result["own_box_predicted_steps"] = float(steps)
    if (
        throughput_cell_updates_per_s
        and reference_box_cells
        and reference_box_steps
    ):
        total = float(reference_box_cells) * float(reference_box_steps)
        result["reference_box"] = {
            "cells": int(reference_box_cells),
            "steps": int(reference_box_steps),
            "cell_updates": total,
            "predicted_hours": total / float(throughput_cell_updates_per_s) / 3600.0,
        }
    return result