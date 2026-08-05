"""Typed, fail-closed admission gate for physical-accuracy recommendations.

This module deliberately accepts only :class:`PhysicalAccuracyEvidence` records.
Collision contracts, capability matrices, solver self-consistency checks, and
untyped campaign artifacts are not physical-accuracy evidence and cannot enter
the comparison path.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal, Sequence

_WITHHELD = "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"
_RECOMMENDED = "RECOMMENDED_FROM_PHYSICAL_ACCURACY_EVIDENCE"
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class KPIDefinition:
    """Identity of a physically compared quantity, including its sampling rule."""

    name: str
    units: str
    aggregation: str
    sampling_window: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.name, self.units, self.aggregation, self.sampling_window,
        )):
            raise ValueError("KPI definition fields must be non-empty strings")


@dataclass(frozen=True)
class ErrorMetric:
    """Declared error result against the named external/reference source."""

    name: str
    normalization: str
    value: float
    uncertainty: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("error metric name must be a non-empty string")
        if not isinstance(self.normalization, str) or not self.normalization.strip():
            raise ValueError("error metric normalization must be a non-empty string")
        if not isinstance(self.value, Real) or isinstance(self.value, bool):
            raise TypeError("error metric value must be a real number (not bool)")
        if not isfinite(self.value) or self.value < 0.0:
            raise ValueError("error metric value must be finite and >= 0")
        object.__setattr__(self, "value", float(self.value))
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, Real) or isinstance(self.uncertainty, bool):
                raise TypeError("error metric uncertainty must be a real number (not bool)")
            if not isfinite(self.uncertainty) or self.uncertainty < 0.0:
                raise ValueError("error metric uncertainty must be finite and >= 0")
            object.__setattr__(self, "uncertainty", float(self.uncertainty))


@dataclass(frozen=True)
class ConvergenceEvidence:
    """Explicit evidence that the reported physical error has converged."""

    grid: bool
    time: bool
    domain: bool

    def __post_init__(self) -> None:
        for field_name in ("grid", "time", "domain"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"convergence {field_name} must be bool")


@dataclass(frozen=True)
class PhysicalAccuracyEvidence:
    """One configuration's traceable physical-accuracy result.

    ``reference_source_id`` identifies the source used for the reference value,
    not merely a similarly named benchmark.  The two hashes bind the exact
    solver configuration and the collected evidence/provenance manifest.
    """

    candidate_id: str
    case_id: str
    reference_id: str
    reference_source_id: str
    configuration_hash: str
    provenance_hash: str
    kpi: KPIDefinition
    error: ErrorMetric
    convergence: ConvergenceEvidence

    def __post_init__(self) -> None:
        for name in ("candidate_id", "case_id", "reference_id", "reference_source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.kpi, KPIDefinition):
            raise TypeError("kpi must be a KPIDefinition")
        if not isinstance(self.error, ErrorMetric):
            raise TypeError("error must be an ErrorMetric")
        if not isinstance(self.convergence, ConvergenceEvidence):
            raise TypeError("convergence must be a ConvergenceEvidence")


@dataclass(frozen=True)
class AccuracyRecommendation:
    """A recommendation, or a fail-closed withholding decision."""

    status: Literal[
        "RECOMMENDED_FROM_PHYSICAL_ACCURACY_EVIDENCE",
        "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE",
    ]
    recommended_candidate_id: str | None
    missing_requirements: tuple[str, ...]
    reason_codes: tuple[str, ...]
    compared_candidate_ids: tuple[str, ...]


def _is_sha256(value: str) -> bool:
    return (isinstance(value, str) and len(value) == _SHA256_LENGTH
            and all(character in "0123456789abcdef" for character in value.lower()))


def _validate(evidence: PhysicalAccuracyEvidence) -> set[str]:
    missing: set[str] = set()
    if not _is_sha256(evidence.configuration_hash):
        missing.add("configuration/provenance hash")
    if not _is_sha256(evidence.provenance_hash):
        missing.add("configuration/provenance hash")
    if evidence.error.uncertainty is None:
        missing.add("uncertainty/error metric")
    if not evidence.convergence.grid:
        missing.add("grid convergence")
    if not evidence.convergence.time:
        missing.add("time convergence")
    if not evidence.convergence.domain:
        missing.add("domain convergence")
    return missing


def _reason_codes(requirements: set[str]) -> tuple[str, ...]:
    mapping = {
        "same-case reference/source": "MISSING_SAME_CASE_REFERENCE_SOURCE",
        "grid convergence": "MISSING_GRID_CONVERGENCE",
        "time convergence": "MISSING_TIME_CONVERGENCE",
        "domain convergence": "MISSING_DOMAIN_CONVERGENCE",
        "matching KPI definition": "MISSING_MATCHING_KPI_DEFINITION",
        "matching error metric definition": "MISSING_MATCHING_ERROR_METRIC_DEFINITION",
        "configuration/provenance hash": "MISSING_CONFIGURATION_OR_PROVENANCE_HASH",
        "uncertainty/error metric": "MISSING_UNCERTAINTY_OR_ERROR_METRIC",
        "typed physical accuracy evidence": "MISSING_TYPED_PHYSICAL_ACCURACY_EVIDENCE",
    }
    return tuple(mapping[item] for item in sorted(requirements))


def _error_upper_bound(evidence: PhysicalAccuracyEvidence) -> float:
    """Return a bound only after admission has established uncertainty."""
    uncertainty = evidence.error.uncertainty
    if uncertainty is None:  # Defensive: callers must run admission first.
        raise ValueError("physical-accuracy evidence lacks uncertainty")
    return evidence.error.value + uncertainty


def recommend_by_physical_accuracy(
    evidence_data: Sequence[PhysicalAccuracyEvidence] | object,
) -> AccuracyRecommendation:
    """Recommend only the lowest uncertainty-bounded physical error.

    All records must be typed physical evidence for *one* case, reference and
    reference source, and must define exactly the same KPI and error metric
    definition (name and normalization). Ranking uses the declared physical
    error upper bound (``error.value + error.uncertainty``), never a
    D3Q19/D3Q27 fixed-point difference or a collision capability claim.
    """
    if (not isinstance(evidence_data, Sequence)
            or isinstance(evidence_data, (str, bytes))
            or not evidence_data
            or any(not isinstance(item, PhysicalAccuracyEvidence) for item in evidence_data)):
        requirements = {"typed physical accuracy evidence"}
        return AccuracyRecommendation(
            _WITHHELD, None, tuple(sorted(requirements)), _reason_codes(requirements), (),
        )

    evidence = tuple(evidence_data)
    requirements: set[str] = set()
    for item in evidence:
        requirements.update(_validate(item))
    baseline = evidence[0]
    if any((item.case_id, item.reference_id, item.reference_source_id) != (
        baseline.case_id, baseline.reference_id, baseline.reference_source_id,
    ) for item in evidence[1:]):
        requirements.add("same-case reference/source")
    if any(item.kpi != baseline.kpi for item in evidence[1:]):
        requirements.add("matching KPI definition")
    if any((item.error.name, item.error.normalization) != (
        baseline.error.name, baseline.error.normalization,
    ) for item in evidence[1:]):
        requirements.add("matching error metric definition")

    candidate_ids = tuple(item.candidate_id for item in evidence)
    if requirements:
        return AccuracyRecommendation(
            _WITHHELD, None, tuple(sorted(requirements)), _reason_codes(requirements), candidate_ids,
        )

    best = min(evidence, key=lambda item: (_error_upper_bound(item), item.candidate_id))
    return AccuracyRecommendation(_RECOMMENDED, best.candidate_id, (), (), candidate_ids)


__all__ = [
    "AccuracyRecommendation", "ConvergenceEvidence", "ErrorMetric", "KPIDefinition",
    "PhysicalAccuracyEvidence", "recommend_by_physical_accuracy",
]
