"""TDD for the physical-accuracy recommendation evidence admission gate."""
from __future__ import annotations

import pytest
from typing import Any

from tensorlbm.accuracy_recommendation import (
    ConvergenceEvidence,
    ErrorMetric,
    KPIDefinition,
    PhysicalAccuracyEvidence,
    recommend_by_physical_accuracy,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_KPI = KPIDefinition("Ct_total", "1", "time_mean", "post-transient steps 5000-10000")


def _evidence(candidate: str, *, error: float = 0.04, uncertainty: float | None = 0.01,
              case: str = "SUBOFF-full-wet-Re1e7", reference: str = "SUBOFF-2024-run-17",
              source: str = "doi:10.0000/suboff.reference", kpi: KPIDefinition = _KPI,
              grid: bool = True, time: bool = True, domain: bool = True,
              error_name: str = "absolute relative error",
              normalization: str = "reference Ct_total",
              configuration_hash: str = _HASH_A, provenance_hash: str = _HASH_B) -> PhysicalAccuracyEvidence:
    return PhysicalAccuracyEvidence(
        candidate, case, reference, source, configuration_hash, provenance_hash, kpi,
        ErrorMetric(error_name, normalization, error, uncertainty),
        ConvergenceEvidence(grid, time, domain),
    )


def test_collision_only_artifact_is_withheld_not_ranked_as_physical_accuracy() -> None:
    collision_only = {"lattice": "D3Q27", "collision": "cumulant", "fixed_point_error": 0.0}

    result = recommend_by_physical_accuracy(collision_only)

    assert result.status == "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"
    assert result.recommended_candidate_id is None
    assert result.missing_requirements == ("typed physical accuracy evidence",)
    assert result.reason_codes == ("MISSING_TYPED_PHYSICAL_ACCURACY_EVIDENCE",)


def test_capability_only_evidence_is_withheld_not_ranked_as_physical_accuracy() -> None:
    capability_only = [{"lattice": "D3Q19", "supports_mrt": True}, {"lattice": "D3Q27", "supports_mrt": True}]

    result = recommend_by_physical_accuracy(capability_only)

    assert result.status == "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"
    assert result.recommended_candidate_id is None
    assert result.reason_codes == ("MISSING_TYPED_PHYSICAL_ACCURACY_EVIDENCE",)


def test_missing_all_physical_admission_requirements_withholds_with_reasons() -> None:
    result = recommend_by_physical_accuracy([_evidence(
        "D3Q19-MRT", uncertainty=None, grid=False, time=False, domain=False,
        configuration_hash="not-a-hash", provenance_hash="also-not-a-hash",
    )])

    assert result.status == "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"
    assert result.recommended_candidate_id is None
    assert result.missing_requirements == (
        "configuration/provenance hash", "domain convergence", "grid convergence",
        "time convergence", "uncertainty/error metric",
    )
    assert result.reason_codes == (
        "MISSING_CONFIGURATION_OR_PROVENANCE_HASH", "MISSING_DOMAIN_CONVERGENCE",
        "MISSING_GRID_CONVERGENCE", "MISSING_TIME_CONVERGENCE",
        "MISSING_UNCERTAINTY_OR_ERROR_METRIC",
    )


def test_mixed_case_source_or_kpi_cannot_be_compared() -> None:
    changed_kpi = KPIDefinition("Cd", "1", "time_mean", "post-transient steps 5000-10000")
    result = recommend_by_physical_accuracy([
        _evidence("D3Q19-MRT"),
        _evidence("D3Q27-cumulant", case="different-case", source="different-source", kpi=changed_kpi),
    ])

    assert result.status == "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"
    assert result.recommended_candidate_id is None
    assert result.missing_requirements == ("matching KPI definition", "same-case reference/source")
    assert result.reason_codes == (
        "MISSING_MATCHING_KPI_DEFINITION", "MISSING_SAME_CASE_REFERENCE_SOURCE",
    )


@pytest.mark.parametrize(
    ("error_name", "normalization"),
    [
        ("root mean square relative error", "reference Ct_total"),
        ("absolute relative error", "dynamic pressure"),
    ],
)
def test_different_error_metric_definition_is_withheld_before_score_ordering(
    error_name: str, normalization: str,
) -> None:
    """A superficially lower error must not win when its definition differs."""
    result = recommend_by_physical_accuracy([
        _evidence("D3Q19-MRT", error=0.04, uncertainty=0.01),
        _evidence(
            "D3Q27-cumulant", error=0.001, uncertainty=0.0001,
            error_name=error_name, normalization=normalization,
        ),
    ])

    assert result.status == "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"
    assert result.recommended_candidate_id is None
    assert result.missing_requirements == ("matching error metric definition",)
    assert result.reason_codes == ("MISSING_MATCHING_ERROR_METRIC_DEFINITION",)


@pytest.mark.parametrize("field", ["grid", "time", "domain"])
@pytest.mark.parametrize("invalid_value", [1, "true", object()])
def test_convergence_evidence_rejects_truthy_non_bool_values(field: str, invalid_value: object) -> None:
    values: dict[str, Any] = {"grid": True, "time": True, "domain": True}
    values[field] = invalid_value

    with pytest.raises(TypeError, match=rf"convergence {field} must be bool"):
        ConvergenceEvidence(**values)


@pytest.mark.parametrize("invalid_value", ["0.04", True])
def test_error_metric_rejects_non_real_or_bool_values(invalid_value: object) -> None:
    with pytest.raises(TypeError, match="error metric value must be a real number"):
        ErrorMetric("absolute relative error", "reference Ct_total", invalid_value, 0.01)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="error metric uncertainty must be a real number"):
        ErrorMetric("absolute relative error", "reference Ct_total", 0.04, invalid_value)  # type: ignore[arg-type]


def test_admitted_same_case_evidence_recommends_lowest_uncertainty_bounded_error() -> None:
    result = recommend_by_physical_accuracy([
        _evidence("D3Q19-MRT", error=0.03, uncertainty=0.02),
        _evidence("D3Q27-cumulant", error=0.04, uncertainty=0.005),
    ])

    assert result.status == "RECOMMENDED_FROM_PHYSICAL_ACCURACY_EVIDENCE"
    assert result.recommended_candidate_id == "D3Q27-cumulant"
    assert result.missing_requirements == ()
    assert result.reason_codes == ()
