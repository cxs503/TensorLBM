"""B4-P4 — UQ calibration + guardrail audit for the drag-surrogate service.

The B4 serving contract (``DragSurrogateService``) sells two honest
numbers per query: the deep-ensemble ``std`` (exposed as a sigma-like UQ
band) and a guard ``verdict`` (``ok`` / ``review`` / ``reject`` from
:class:`.inference_service.EnvelopeMahalanobisGuardrail`).  Neither has
ever been audited: *is the served std actually a calibrated sigma*, and
*does ``ok`` actually imply small error*?  This module is the metric
layer that answers both, applied offline over corpora / archived fold
predictions.  It contains no I/O and no training — drivers live in the
run directories (see ``docs/uq_calibration_20260827.md``).

Two views, never mixed
----------------------
Calibration numbers are only meaningful relative to how the evaluated
points relate to the ensemble's training data:

* **fit / residual view** — the deployed ensemble evaluated on points it
  was trained on.  Under-fit std there is a *residual scale*, not a
  generalisation error scale; it is reported because a user querying an
  in-corpus design implicitly receives exactly this quantity.
* **leave-out / generalisation view** — predictions from members that
  never saw the evaluated point (out-of-fold, leave-one-hull-out,
  leave-one-family-out).  This is the view a new query actually lives in.

Every table produced from this module must carry the view label; the
grouping helpers below take explicit group indices so the caller cannot
accidentally pool the two views into one number.

Metrics
-------
For ``z = (y - mu) / sigma`` against ``N(0, 1)``:

* coverage at 1 / 1.96 / 2.5758 sigma (nominal 68.27 / 95 / 99 %),
* sharpness (mean sigma — smaller is better at equal coverage),
* Gaussian negative log-likelihood and closed-form Gaussian CRPS,
* z-distribution normality: sample skewness, excess kurtosis and a
  one-sample Kolmogorov-Smirnov statistic with asymptotic p-value
  (Stephens-corrected; no scipy dependency).

Guard audit
-----------
The guard verdict is treated as a classifier for "absolute relative
error above an operating threshold":

* :func:`roc_auc` / :func:`guard_roc` sweep the guard severity score
  against large-error labels and report AUC plus operating points
  (capture / false-alarm / precision), including the guard's own
  ``review`` / ``reject`` cut-offs;
* :func:`row_verdicts` applies the service's aggregate
  :meth:`Guardrail.check` row-by-row (exact semantics, no re-derived
  envelope formula) so per-point flags are comparable with the verdict a
  single-design query would receive;
* :func:`verdict_confusion` cross-tabulates flags with error bands and
  gives ``P(|err| <= t | flag)`` — the honest meaning of ``ok``.

Temperature scaling
-------------------
:func:`fit_temperature` returns the closed-form NLL-optimal scalar
``T`` with ``sigma_cal = T * sigma`` (``T = rms(z)``, exact for a
Gaussian likelihood).  If the ensemble is already calibrated ``T ~= 1``
and the honest recommendation is to not deploy a temperature — the
audit reports both halves (fit on one half of the corpus, validated on
the held-out half) instead of a single fitted number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .inference_service import FLAG_OK, FLAG_REJECT, FLAG_REVIEW, EnvelopeMahalanobisGuardrail

__all__ = [
    "FLAG_ORDER",
    "GUARDROC_FIELDS",
    "UQMETRIC_FIELDS",
    "GuardOperatingPoint",
    "GuardRocReport",
    "UqMetrics",
    "VerdictConfusion",
    "apply_temperature",
    "calibration_metrics",
    "error_summary_by_flag",
    "fit_temperature",
    "grouped_calibration",
    "guard_roc",
    "roc_auc",
    "row_verdicts",
    "verdict_confusion",
]

#: Verdict row order of confusion tables (service flag levels).
FLAG_ORDER: tuple[str, ...] = (FLAG_OK, FLAG_REVIEW, FLAG_REJECT)

#: Two-sided z quantiles of the reported coverage levels.
_COVERAGE_Z: tuple[tuple[str, float], ...] = (
    ("coverage_68", 1.0),  # nominal 0.6827
    ("coverage_95", 1.959963984540054),  # nominal 0.95
    ("coverage_99", 2.5758293035489004),  # nominal 0.99
)

#: Nominal coverage fractions matching ``_COVERAGE_Z`` (for report tables).
COVERAGE_NOMINAL: dict[str, float] = {
    "coverage_68": 0.6826894921370859,
    "coverage_95": 0.95,
    "coverage_99": 0.99,
}

#: Machine-facing protocol fields of :meth:`UqMetrics.as_dict`.
UQMETRIC_FIELDS: tuple[str, ...] = (
    "n",
    "mean_sigma",
    "median_sigma",
    "mean_abs_err",
    "mean_abs_rel_err_pct",
    "coverage_68",
    "coverage_95",
    "coverage_99",
    "mean_abs_z",
    "rms_z",
    "gaussian_nll",
    "crps_gauss",
    "z_skew",
    "z_excess_kurtosis",
    "z_ks_stat",
    "z_ks_pvalue",
)

#: Machine-facing protocol fields of :meth:`GuardRocReport.as_dict`.
GUARDROC_FIELDS: tuple[str, ...] = (
    "error_threshold",
    "auc",
    "n_large",
    "n_small",
    "points",
)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _ks_pvalue(d: float, n: int) -> float:
    """Asymptotic Kolmogorov Q(lambda) with the Stephens small-n correction.

    ``lambda = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * D_n``; the series is
    summed to j = 100 (converged far below float64 noise for any lambda
    that matters).  Deliberately scipy-free, matching the dependency
    policy of :mod:`.inference_service`.
    """
    if n <= 0:
        raise ValueError(f"ks p-value needs n >= 1, got {n}")
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * float(d)
    s = 0.0
    for j in range(1, 101):
        s += (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
    return float(min(1.0, max(0.0, 2.0 * s)))


def _as_1d(arr: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    return out


def _check_aligned(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError(f"y/mu/sigma shapes differ: {y.shape}, {mu.shape}, {sigma.shape}")
    if y.size == 0:
        raise ValueError("y/mu/sigma must be non-empty")
    if not np.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError("sigma must be finite and strictly positive")


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UqMetrics:
    """Calibration of ``(mu, sigma)`` against observations ``y``.

    All error-scale quantities live in the space the caller passed (the
    audit reports both linear ``C_D`` — the space the service exposes —
    and ``log10 C_D`` — the space the members were regressed in).
    """

    n: int
    mean_sigma: float
    median_sigma: float
    mean_abs_err: float
    mean_abs_rel_err_pct: float
    coverage_68: float
    coverage_95: float
    coverage_99: float
    mean_abs_z: float
    rms_z: float
    gaussian_nll: float
    crps_gauss: float
    z_skew: float
    z_excess_kurtosis: float
    z_ks_stat: float
    z_ks_pvalue: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "mean_sigma": float(self.mean_sigma),
            "median_sigma": float(self.median_sigma),
            "mean_abs_err": float(self.mean_abs_err),
            "mean_abs_rel_err_pct": float(self.mean_abs_rel_err_pct),
            "coverage_68": float(self.coverage_68),
            "coverage_95": float(self.coverage_95),
            "coverage_99": float(self.coverage_99),
            "mean_abs_z": float(self.mean_abs_z),
            "rms_z": float(self.rms_z),
            "gaussian_nll": float(self.gaussian_nll),
            "crps_gauss": float(self.crps_gauss),
            "z_skew": float(self.z_skew),
            "z_excess_kurtosis": float(self.z_excess_kurtosis),
            "z_ks_stat": float(self.z_ks_stat),
            "z_ks_pvalue": float(self.z_ks_pvalue),
        }


def calibration_metrics(
    y: Sequence[float] | np.ndarray,
    mu: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray,
) -> UqMetrics:
    """Coverage / sharpness / NLL / CRPS / z-normality of one point set.

    ``z = (y - mu) / sigma`` is compared against ``N(0, 1)``: a perfectly
    calibrated predictive Gaussian has coverage_68/95/99 at the nominal
    levels, ``gaussian_nll = 0.5*log(2*pi*e) ~= 1.4189``, zero z-skew /
    excess kurtosis and a non-significant KS statistic.  ``sigma`` must be
    strictly positive (a single-member "ensemble" has no std and cannot
    be audited — call that out instead of passing zeros).
    """
    y = _as_1d(y, "y")
    mu = _as_1d(mu, "mu")
    sigma = _as_1d(sigma, "sigma")
    _check_aligned(y, mu, sigma)
    n = int(y.size)
    z = (y - mu) / sigma
    abs_err = np.abs(y - mu)
    rel = abs_err / np.abs(y) if (y != 0).all() else None

    cov = {name: float(np.mean(np.abs(z) <= k)) for name, k in _COVERAGE_Z}
    nll = float(np.mean(0.5 * np.log(2.0 * math.pi * sigma**2) + 0.5 * z**2))
    # Gneiting closed form: CRPS(N(mu, sigma), y) =
    #   sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ].
    crps = float(
        np.mean(
            sigma
            * (
                z * (2.0 * np.vectorize(_norm_cdf)(z) - 1.0)
                + 2.0 * np.vectorize(_norm_pdf)(z)
                - 1.0 / math.sqrt(math.pi)
            )
        )
    )

    if n >= 3:
        m2 = float(np.mean((z - z.mean()) ** 2))
        zc = z - z.mean()
        skew = float(np.mean(zc**3) / m2**1.5) if m2 > 0 else float("nan")
        exkurt = float(np.mean(zc**4) / m2**2 - 3.0) if m2 > 0 else float("nan")
    else:
        skew = float("nan")
        exkurt = float("nan")

    if n >= 2:
        # One-sample KS distance of z against the fully specified N(0, 1)
        # (mu/sigma are model outputs, not sample estimates of y).
        zs = np.sort(z)
        cdf = np.vectorize(_norm_cdf)(zs)
        upper = np.arange(1, n + 1) / n
        lower = np.arange(0, n) / n
        ks = float(max(np.max(upper - cdf), np.max(cdf - lower)))
        ks_p = _ks_pvalue(ks, n)
    else:
        ks = float("nan")
        ks_p = float("nan")

    return UqMetrics(
        n=n,
        mean_sigma=float(sigma.mean()),
        median_sigma=float(np.median(sigma)),
        mean_abs_err=float(abs_err.mean()),
        mean_abs_rel_err_pct=float(rel.mean() * 100.0) if rel is not None else float("nan"),
        coverage_68=cov["coverage_68"],
        coverage_95=cov["coverage_95"],
        coverage_99=cov["coverage_99"],
        mean_abs_z=float(np.abs(z).mean()),
        rms_z=float(np.sqrt(np.mean(z**2))),
        gaussian_nll=nll,
        crps_gauss=crps,
        z_skew=skew,
        z_excess_kurtosis=exkurt,
        z_ks_stat=ks,
        z_ks_pvalue=ks_p,
    )


def grouped_calibration(
    y: Sequence[float] | np.ndarray,
    mu: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray,
    groups: Mapping[str, Sequence[int] | np.ndarray],
) -> dict[str, UqMetrics]:
    """Per-group :func:`calibration_metrics` over integer index groups.

    Groups are the split views of the audit (fit / val / random-holdout,
    LOFO per family, LOHO per hull).  Indices must be in range and the
    caller keeps responsibility for not pooling fit and leave-out views
    into one group.
    """
    y = _as_1d(y, "y")
    mu = _as_1d(mu, "mu")
    sigma = _as_1d(sigma, "sigma")
    _check_aligned(y, mu, sigma)
    out: dict[str, UqMetrics] = {}
    for name, idx in groups.items():
        ii = np.asarray(idx, dtype=np.int64)
        if ii.ndim != 1:
            raise ValueError(f"group {name!r} indices must be 1-D")
        if ii.size and (ii.min() < 0 or ii.max() >= y.size):
            raise ValueError(f"group {name!r} indices out of range for n={y.size}")
        out[name] = calibration_metrics(y[ii], mu[ii], sigma[ii])
    return out


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def fit_temperature(
    y: Sequence[float] | np.ndarray,
    mu: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray,
) -> float:
    """Closed-form NLL-optimal scalar sigma multiplier ``T`` (``= rms z``).

    Minimising ``mean[ 0.5*log(2*pi*(T*sigma)^2) + 0.5*z^2/T^2 ]`` over
    ``T > 0`` gives ``T* = sqrt(mean(z^2))`` exactly, so no numerical
    optimiser is involved.  ``T > 1`` means the ensemble is overconfident
    (sigma too small), ``T < 1`` underconfident.  Fit on a train half and
    validate on a held-out half; a value near 1 on the fit half means the
    raw ensemble sigma is already the right scale and no temperature
    should be deployed.
    """
    y = _as_1d(y, "y")
    mu = _as_1d(mu, "mu")
    sigma = _as_1d(sigma, "sigma")
    _check_aligned(y, mu, sigma)
    z2 = ((y - mu) / sigma) ** 2
    t = float(math.sqrt(z2.mean()))
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"degenerate temperature {t!r} (check y/mu/sigma inputs)")
    return t


def apply_temperature(
    sigma: Sequence[float] | np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return ``temperature * sigma`` (the recalibrated predictive sigma)."""
    sigma = _as_1d(sigma, "sigma")
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    return sigma * t


# ---------------------------------------------------------------------------
# Guardrail ROC
# ---------------------------------------------------------------------------


def roc_auc(scores: Sequence[float] | np.ndarray, positive: Sequence[bool] | np.ndarray) -> float:
    """Rank AUC of ``scores`` against boolean labels (ties get midranks).

    ``nan`` when either class is empty (AUC undefined) — callers report
    the degeneracy instead of pretending a number.
    """
    s = _as_1d(scores, "scores")
    lab = np.asarray(positive)
    if lab.ndim != 1 or lab.shape != s.shape:
        raise ValueError(f"positive must be 1-D aligned with scores, got {lab.shape}")
    lab = lab.astype(bool)
    n_pos = int(lab.sum())
    n_neg = int(lab.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    ss = s[order]
    while i < s.size:
        j = i
        while j + 1 < s.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # midrank (1-based)
        i = j + 1
    sum_pos = float(ranks[lab].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class GuardOperatingPoint:
    """One (error threshold, score cut) operating point of the guard."""

    error_threshold: float
    score_threshold: float
    n_flagged: int
    n_large: int
    tpr: float  # fraction of large errors flagged (capture)
    fpr: float  # fraction of small errors flagged (false alarm)
    precision: float  # flagged points that are actually large errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_threshold": float(self.error_threshold),
            "score_threshold": float(self.score_threshold),
            "n_flagged": int(self.n_flagged),
            "n_large": int(self.n_large),
            "tpr": float(self.tpr),
            "fpr": float(self.fpr),
            "precision": float(self.precision),
        }


@dataclass(frozen=True)
class GuardRocReport:
    """ROC summary of the guard severity score for one error threshold."""

    error_threshold: float
    auc: float
    n_large: int
    n_small: int
    points: tuple[GuardOperatingPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_threshold": float(self.error_threshold),
            "auc": float(self.auc),
            "n_large": int(self.n_large),
            "n_small": int(self.n_small),
            "points": [p.as_dict() for p in self.points],
        }


def guard_roc(
    scores: Sequence[float] | np.ndarray,
    errors: Sequence[float] | np.ndarray,
    *,
    error_thresholds: Sequence[float],
    score_thresholds: Sequence[float] | None = None,
) -> list[GuardRocReport]:
    """Guard-severity ROC against "absolute error above threshold" labels.

    The service contract bins severity into ``ok`` / ``review`` /
    ``reject``; the ROC treats the underlying severity score as the
    continuous discriminator it is.  For each ``error_threshold`` (a
    relative-error cut, e.g. 0.01 / 0.05 / 0.10) points are labelled
    large-error, the AUC of the severity score is computed, and operating
    points report capture / false-alarm / precision over a sweep of score
    cuts (defaults: 21 score quantiles; pass the guard's own
    ``review_threshold`` / ``mahal_threshold`` to see the deployed cut
    among them).
    """
    s = _as_1d(scores, "scores")
    err = np.abs(_as_1d(errors, "errors"))
    if s.shape != err.shape or s.size == 0:
        raise ValueError("scores and errors must be same-length, non-empty")
    if not error_thresholds:
        raise ValueError("error_thresholds must be non-empty")
    sts: np.ndarray
    if score_thresholds is None:
        sts = np.quantile(s, np.linspace(0.0, 1.0, 21))
    else:
        sts = np.asarray(_as_1d(score_thresholds, "score_thresholds"))
    reports: list[GuardRocReport] = []
    for et in error_thresholds:
        if not (float(et) > 0):
            raise ValueError(f"error thresholds must be positive, got {et!r}")
        large = err > float(et)
        n_large = int(large.sum())
        n_small = int(large.size - n_large)
        auc = roc_auc(s, large)
        pts: list[GuardOperatingPoint] = []
        for st in np.unique(sts):
            flagged = s > float(st)
            n_flag = int(flagged.sum())
            captured = int((flagged & large).sum())
            pts.append(
                GuardOperatingPoint(
                    error_threshold=float(et),
                    score_threshold=float(st),
                    n_flagged=n_flag,
                    n_large=n_large,
                    tpr=captured / n_large if n_large else float("nan"),
                    fpr=int((flagged & ~large).sum()) / n_small if n_small else float("nan"),
                    precision=captured / n_flag if n_flag else float("nan"),
                )
            )
        reports.append(
            GuardRocReport(
                error_threshold=float(et),
                auc=auc,
                n_large=n_large,
                n_small=n_small,
                points=tuple(pts),
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Verdict semantics
# ---------------------------------------------------------------------------


def row_verdicts(
    guard: EnvelopeMahalanobisGuardrail, features: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Per-row service verdicts (``ok`` / ``review`` / ``reject``).

    Calls :meth:`EnvelopeMahalanobisGuardrail.check` on each row
    separately, so the flag of row ``i`` is exactly the verdict a
    single-design service query carrying that row would receive (the
    aggregate ``check`` is worst-row driven and would report one flag for
    a whole batch).  No envelope formula is re-derived here — semantics
    stay defined by the guard class itself.
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {feats.shape}")
    return np.array(
        [guard.check(feats[i : i + 1]).flag for i in range(feats.shape[0])],
        dtype=f"<U{max(len(f) for f in FLAG_ORDER)}",
    )


def error_summary_by_flag(
    flags: Sequence[str] | np.ndarray, errors: Sequence[float] | np.ndarray
) -> dict[str, dict[str, float]]:
    """|error| distribution per verdict level (n / mean / q50 / q90 / q95 / max)."""
    fl = np.asarray(flags, dtype=object)
    err = np.abs(_as_1d(errors, "errors"))
    if fl.ndim != 1 or fl.shape != err.shape:
        raise ValueError(f"flags must be 1-D aligned with errors, got {fl.shape}")
    out: dict[str, dict[str, float]] = {}
    for name in FLAG_ORDER:
        sel = err[fl == name]
        if sel.size == 0:
            continue
        q50, q90, q95 = np.quantile(sel, (0.50, 0.90, 0.95))
        out[name] = {
            "n": float(sel.size),
            "mean": float(sel.mean()),
            "q50": float(q50),
            "q90": float(q90),
            "q95": float(q95),
            "max": float(sel.max()),
        }
    return out


def _band_label(lo: float | None, hi: float | None) -> str:
    if lo is None:
        return f"<{hi:.0%}" if hi is not None else "all"
    if hi is None:
        return f">={lo:.0%}"
    return f"{lo:.0%}-{hi:.0%}"


@dataclass(frozen=True)
class VerdictConfusion:
    """Verdict x error-band cross-tabulation with per-flag error CDF."""

    flags: tuple[str, ...]
    bands: tuple[str, ...]
    counts: np.ndarray  # (n_flags, n_bands) integers
    errors_by_flag: dict[str, dict[str, float]]

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    def row(self, flag: str) -> dict[str, int]:
        if flag not in self.flags:
            raise KeyError(f"unknown flag {flag!r}; have {self.flags}")
        i = self.flags.index(flag)
        return {b: int(self.counts[i, j]) for j, b in enumerate(self.bands)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "flags": list(self.flags),
            "bands": list(self.bands),
            "counts": self.counts.tolist(),
            "n": self.n,
            "errors_by_flag": self.errors_by_flag,
        }


def verdict_confusion(
    flags: Sequence[str] | np.ndarray,
    errors: Sequence[float] | np.ndarray,
    *,
    band_edges: Sequence[float],
) -> VerdictConfusion:
    """Cross-tabulate verdict flags against absolute-error bands.

    ``band_edges`` are relative-error cuts in (0, 1] (e.g.
    ``(0.01, 0.02, 0.05, 0.15)``); bands are labelled with rounded
    percents.  The companion :func:`p_error_below` carries the exact
    ``P(|err| <= t | flag)`` statements used in the report.
    """
    fl = np.asarray(flags, dtype=object)
    err = np.abs(_as_1d(errors, "errors"))
    if fl.ndim != 1 or fl.shape != err.shape:
        raise ValueError(f"flags must be 1-D aligned with errors, got {fl.shape}")
    if fl.size == 0:
        raise ValueError("flags/errors must be non-empty")
    edges = [float(e) for e in band_edges]
    if not edges or any(not (e > 0) for e in edges) or edges != sorted(edges):
        raise ValueError(f"band_edges must be ascending positive cuts, got {band_edges!r}")
    unknown = sorted({str(f) for f in fl.tolist()} - set(FLAG_ORDER))
    if unknown:
        raise ValueError(f"flags outside ok/review/reject: {unknown}")

    labels: list[str] = []
    lo: float | None = None
    for hi in edges:
        labels.append(_band_label(lo, hi))
        lo = hi
    labels.append(_band_label(lo, None))

    present = [f for f in FLAG_ORDER if np.any(fl == f)]
    counts = np.zeros((len(present), len(labels)), dtype=np.int64)
    for i, f in enumerate(present):
        sel = err[fl == f]
        idx = np.searchsorted(np.asarray(edges), sel, side="left")
        for b in idx:
            counts[i, int(b)] += 1
    return VerdictConfusion(
        flags=tuple(present),
        bands=tuple(labels),
        counts=counts,
        errors_by_flag=error_summary_by_flag(fl, err),
    )


def p_error_below(
    flags: Sequence[str] | np.ndarray,
    errors: Sequence[float] | np.ndarray,
    flag: str,
    error_threshold: float,
) -> float:
    """``P(|err| <= error_threshold | verdict == flag)`` — the deployed
    meaning of a verdict: what an ``ok`` actually promises about error."""
    fl = np.asarray(flags, dtype=object)
    err = np.abs(_as_1d(errors, "errors"))
    if fl.ndim != 1 or fl.shape != err.shape:
        raise ValueError(f"flags must be 1-D aligned with errors, got {fl.shape}")
    sel = err[fl == flag]
    if sel.size == 0:
        raise ValueError(f"no points with flag {flag!r}")
    return float(np.mean(sel <= float(error_threshold)))
