"""Arbitrable error decomposition for wall-law (``u+`` vs ``y+``) evidence.

A statement such as "the simulated ``u+`` profile agrees with the log law within
1%" mixes three physically different error sources that must be separated
before any verdict is possible:

1. **Statistical (sampling) error** - the finite averaging window.  Quantified as
   the batch-means 95% confidence half-width of ``u+``, with the
   ``T_obs / T_int`` ledger from :mod:`tensorlbm.statistics_error_budget`.
2. **Domain-length bias** - the streamwise extent ``Lx`` is not statistically
   independent of the largest energy-containing structure.  Isolated by
   comparing two box lengths on the *same* wall-normal grid: everything except
   box-size effects cancels.
3. **Model (sub-grid) bias** - the closure's contribution.  Isolated by a
   negative control in which the sub-grid model is switched off
   (``C_s = 0``) at identical numerics; a large control deviation proves the
   closure is load-bearing rather than decorative.

The three components are reported separately and combined in quadrature; the
result is written as a machine-readable ledger
(``u+_error_decomposition.json``) so each claim carries the number, the
comparison band and the criterion it is judged against.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_LOG_LAW_LOWER: float = 30.0
DEFAULT_LOG_LAW_UPPER_FRACTION: float = 0.5
DEFAULT_DOMAIN_TOLERANCE: float = 0.01
DEFAULT_SGS_THRESHOLD: float = 0.05


@dataclass(frozen=True)
class CriterionVerdict:
    """A single machine-checkable acceptance criterion."""

    name: str
    criterion: str
    value: float
    threshold: float
    comparator: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_law_band(
    y_plus: np.ndarray,
    re_tau: float,
    *,
    lower: float = DEFAULT_LOG_LAW_LOWER,
    upper_fraction: float = DEFAULT_LOG_LAW_UPPER_FRACTION,
) -> np.ndarray:
    """Boolean mask of the logarithmic-layer comparison band.

    Defaults to ``30 <= y+ <= 0.5 Re_tau`` - the classic overlap layer, kept
    clear of both the buffer layer (where a log fit is not expected to hold) and
    the channel centreline (where the wake departs from the log law).
    """
    values = np.asarray(y_plus, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("y_plus must be one-dimensional")
    if re_tau <= 0.0:
        raise ValueError("re_tau must be positive")
    if lower <= 0.0:
        raise ValueError("lower must be positive")
    if not 0.0 < upper_fraction <= 1.0:
        raise ValueError("upper_fraction must lie in (0, 1]")
    upper = upper_fraction * re_tau
    if upper <= lower:
        raise ValueError(
            f"band is empty for Re_tau={re_tau}: 0.5 Re_tau is below the {lower} lower limit",
        )
    return (values >= lower) & (values <= upper)


def _prepare_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    y_plus: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    y = np.asarray(y_plus, dtype=np.float64)
    if not (a.shape == b.shape == y.shape):
        raise ValueError("reference, candidate and y_plus must share a shape")
    if a.ndim != 1:
        raise ValueError("profiles must be one-dimensional")
    select = np.asarray(mask, dtype=bool)
    if select.shape != a.shape:
        raise ValueError("mask must match the profile shape")
    if not np.any(select):
        raise ValueError("mask selects no samples")
    if not (np.all(np.isfinite(a[select])) and np.all(np.isfinite(b[select]))):
        raise ValueError("profiles contain non-finite values inside the band")
    return a, b, select


def band_max_relative_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    y_plus: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Largest and mean relative difference between two profiles in a band.

    The reference profile is the denominator, so the number reads as "how far
    the candidate is from the accepted profile, in percent of the local value".
    """
    a, b, select = _prepare_pair(reference, candidate, y_plus, mask)
    denominator = np.abs(a[select])
    if np.any(denominator <= 0.0):
        raise ValueError("reference profile vanishes inside the band")
    rel = np.abs(b[select] - a[select]) / denominator
    return {
        "max_relative_difference": float(rel.max()),
        "mean_relative_difference": float(rel.mean()),
        "band_lower_y_plus": float(np.asarray(y_plus, dtype=np.float64)[select].min()),
        "band_upper_y_plus": float(np.asarray(y_plus, dtype=np.float64)[select].max()),
        "band_points": int(select.sum()),
    }


def arbitrate_domain_length(
    short_domain: np.ndarray,
    long_domain: np.ndarray,
    y_plus: np.ndarray,
    re_tau: float,
    *,
    tolerance: float = DEFAULT_DOMAIN_TOLERANCE,
    short_domain_length: str = "Lx=2*pi*delta",
    long_domain_length: str = "Lx=4*pi*delta",
    lower: float = DEFAULT_LOG_LAW_LOWER,
    upper_fraction: float = DEFAULT_LOG_LAW_UPPER_FRACTION,
    criterion_name: str = "C2_domain_length_arbitration",
) -> CriterionVerdict:
    """Judge whether the streamwise box is long enough for ``u+``.

    Both boxes must share the same wall-normal grid and ``Re_tau``; the only
    intended difference is the box length, so the residual difference is the
    domain-length bias.
    """
    mask = log_law_band(y_plus, re_tau, lower=lower, upper_fraction=upper_fraction)
    stats = band_max_relative_difference(long_domain, short_domain, y_plus, mask)
    value = stats["max_relative_difference"]
    return CriterionVerdict(
        name=criterion_name,
        criterion=f"max |u+({short_domain_length}) - u+({long_domain_length})| / u+ < {tolerance:g}",
        value=value,
        threshold=float(tolerance),
        comparator="<",
        passed=bool(value < tolerance),
        detail={
            **stats,
            "short_domain_length": short_domain_length,
            "long_domain_length": long_domain_length,
            "reference": long_domain_length,
            "measure": "relative difference normalised by the long-domain profile",
            "mask_rule": f"{lower} <= y+ <= {upper_fraction}*Re_tau",
        },
    )


def sgs_negative_control(
    without_sgs: np.ndarray,
    with_sgs: np.ndarray,
    y_plus: np.ndarray,
    re_tau: float,
    *,
    threshold: float = DEFAULT_SGS_THRESHOLD,
    lower: float = DEFAULT_LOG_LAW_LOWER,
    upper_fraction: float = DEFAULT_LOG_LAW_UPPER_FRACTION,
    criterion_name: str = "C3_sgs_negative_control",
) -> CriterionVerdict:
    """Judge whether the sub-grid closure is load-bearing.

    ``with_sgs`` is the production profile (equivalent-viscosity closure on);
    ``without_sgs`` is the negative control at ``C_s = 0``.  The criterion
    *fails* when the two agree, because that means the closure contributes
    nothing measurable at this resolution and the "SGS model" label is empty.
    """
    mask = log_law_band(y_plus, re_tau, lower=lower, upper_fraction=upper_fraction)
    stats = band_max_relative_difference(with_sgs, without_sgs, y_plus, mask)
    value = stats["max_relative_difference"]
    return CriterionVerdict(
        name=criterion_name,
        criterion=f"max |u+(C_s=0) - u+(C_s=cs)| / u+(C_s=cs) > {threshold:g}",
        value=value,
        threshold=float(threshold),
        comparator=">",
        passed=bool(value > threshold),
        detail={
            **stats,
            "reference": "production C_s profile",
            "measure": "relative deviation of the C_s=0 control",
            "mask_rule": f"{lower} <= y+ <= {upper_fraction}*Re_tau",
        },
    )


def statistical_error_verdict(
    u_plus: np.ndarray,
    ci_half_width: np.ndarray,
    y_plus: np.ndarray,
    re_tau: float,
    *,
    threshold: float = 0.02,
    obs_over_t_int: float | None = None,
    required_ratio: float = 50.0,
    lower: float = DEFAULT_LOG_LAW_LOWER,
    upper_fraction: float = DEFAULT_LOG_LAW_UPPER_FRACTION,
    criterion_name: str = "C1_statistical_error",
) -> CriterionVerdict:
    """Judge whether the time average is converged enough to be quoted.

    The value is the widest 95% interval of ``u+`` inside the log-law band,
    relative to the local ``u+``; the criterion additionally requires
    ``T_obs / T_int >= required_ratio`` because a narrow interval computed from a
    record shorter than ~50 correlation times is not trustworthy.
    """
    values = np.asarray(u_plus, dtype=np.float64)
    half = np.asarray(ci_half_width, dtype=np.float64)
    mask = log_law_band(y_plus, re_tau, lower=lower, upper_fraction=upper_fraction)
    if values.shape != half.shape or values.shape != mask.shape:
        raise ValueError("u_plus, ci_half_width and y_plus must share a shape")
    denominator = np.abs(values[mask])
    if np.any(denominator <= 0.0):
        raise ValueError("u_plus vanishes inside the band")
    widths = 2.0 * half[mask] / denominator
    value = float(np.max(widths))
    ratio_ok = True if obs_over_t_int is None else bool(obs_over_t_int >= required_ratio)
    return CriterionVerdict(
        name=criterion_name,
        criterion=(
            f"max 95% CI width of u+ in the log-law band < {threshold:g} "
            f"and T_obs/T_int >= {required_ratio:g}"
        ),
        value=value,
        threshold=float(threshold),
        comparator="<",
        passed=bool(value < threshold and ratio_ok),
        detail={
            "max_relative_ci_width": value,
            "mean_relative_ci_width": float(np.mean(widths)),
            "obs_over_t_int": obs_over_t_int,
            "required_obs_over_t_int": float(required_ratio),
            "ratio_satisfied": ratio_ok,
            "band_points": int(mask.sum()),
        },
    )


def decompose_error_budget(
    components: dict[str, float | None],
) -> dict[str, Any]:
    """Combine independent error components into a quadrature budget.

    ``components`` maps a component name to a relative error (or ``None`` when
    that component was not measured - never silently replaced by zero).
    """
    measured = {
        name: float(value)
        for name, value in components.items()
        if value is not None and math.isfinite(float(value))
    }
    missing = [name for name, value in components.items() if name not in measured]
    total = math.sqrt(sum(value**2 for value in measured.values()))
    dominant = max(measured, key=lambda name: measured[name]) if measured else None
    return {
        "components": measured,
        "unmeasured_components": missing,
        "quadrature_total": total,
        "dominant_component": dominant,
        "dominant_share": (
            measured[dominant] ** 2 / total**2 if dominant and total > 0.0 else None
        ),
    }


@dataclass
class ErrorDecompositionReport:
    """Container for the full ``u+`` error decomposition ledger."""

    case: dict[str, Any]
    statistics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_verdict(self, verdict: CriterionVerdict) -> None:
        self.verdicts.append(verdict)

    def rebuild_budget(self) -> dict[str, Any]:
        components: dict[str, float | None] = {}
        for verdict in self.verdicts:
            components[verdict.name] = verdict.value
        self.budget = decompose_error_budget(components)
        return self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "statistics": self.statistics,
            "comparison": self.comparison,
            "budget": self.budget,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
            "all_passed": all(verdict.passed for verdict in self.verdicts) if self.verdicts else None,
            "notes": self.notes,
        }


def write_error_decomposition(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the decomposition ledger as stable, sorted JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_error_decomposition(path: str | Path) -> dict[str, Any]:
    """Read a decomposition ledger back (round-trip check for audits)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CriterionVerdict",
    "DEFAULT_DOMAIN_TOLERANCE",
    "DEFAULT_LOG_LAW_LOWER",
    "DEFAULT_LOG_LAW_UPPER_FRACTION",
    "DEFAULT_SGS_THRESHOLD",
    "ErrorDecompositionReport",
    "arbitrate_domain_length",
    "band_max_relative_difference",
    "decompose_error_budget",
    "load_error_decomposition",
    "log_law_band",
    "sgs_negative_control",
    "statistical_error_verdict",
    "write_error_decomposition",
]