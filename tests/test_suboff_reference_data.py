"""TDD tests for the SUBOFF reference resistance-data compilation module.

These tests verify that the compiled reference data:
  1. Has the required typed structure (case_id, reference_id,
     reference_source_id, Ct_reference, Re, uncertainty, source_citation).
  2. Every non-withheld entry has a finite, positive Ct_reference and Re.
  3. Every non-withheld entry has a finite, non-negative uncertainty.
  4. Every entry has a non-empty source_citation.
  5. WITHHELD entries are explicitly marked and carry no numeric value.
  6. The registry is non-empty and look-up by case_id works.
  7. Reference data can be used to compute an error metric for the
     accuracy_recommendation gate.
"""
from __future__ import annotations

from math import isfinite

import pytest

from tensorlbm.suboff_reference_data import (
    SUBOFF_REFERENCE_REGISTRY,
    SuboffReferenceDatum,
    get_reference_data,
    get_reference_data_by_case,
    list_available_case_ids,
    list_available_reference_ids,
)


# ---------------------------------------------------------------------------
# 1. Typed data structure
# ---------------------------------------------------------------------------

class TestSuboffReferenceDatumStructure:
    """Verify the typed data structure has all required fields and validation."""

    def test_datum_has_all_required_fields(self) -> None:
        d = SuboffReferenceDatum(
            case_id="SUBOFF-AFF1-bare-hull-Re1.2e7",
            reference_id="ITTC-1957-Cf-Re1.2e7",
            reference_source_id="ITTC-1957-model-ship-correlation-line",
            Ct_reference=0.00291,
            Re=1.2e7,
            uncertainty=0.00015,
            source_citation="ITTC 1957 Model-Ship Correlation Line",
            hull_type="bare_hull",
            reference_area_basis="wetted_surface",
            applicable_conditions="Re=1.2e7, single-phase, deep water",
            notes="Frictional resistance coefficient only.",
        )
        assert d.case_id == "SUBOFF-AFF1-bare-hull-Re1.2e7"
        assert d.reference_id == "ITTC-1957-Cf-Re1.2e7"
        assert d.reference_source_id == "ITTC-1957-model-ship-correlation-line"
        assert d.Ct_reference == pytest.approx(0.00291)
        assert d.Re == pytest.approx(1.2e7)
        assert d.uncertainty == pytest.approx(0.00015)
        assert d.source_citation == "ITTC 1957 Model-Ship Correlation Line"
        assert d.hull_type == "bare_hull"
        assert d.reference_area_basis == "wetted_surface"
        assert d.is_withheld is False

    def test_datum_is_frozen(self) -> None:
        d = SuboffReferenceDatum(
            case_id="c", reference_id="r", reference_source_id="s",
            Ct_reference=0.004, Re=2.0e6, uncertainty=0.0004,
            source_citation="cite", hull_type="bare_hull",
            reference_area_basis="wetted_surface",
            applicable_conditions="cond", notes="n",
        )
        with pytest.raises(AttributeError, match="cannot assign"):
            d.Ct_reference = 0.999  # type: ignore[misc]

    def test_datum_rejects_empty_case_id(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            SuboffReferenceDatum(
                case_id="", reference_id="r", reference_source_id="s",
                Ct_reference=0.004, Re=2.0e6, uncertainty=0.0004,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_empty_reference_id(self) -> None:
        with pytest.raises(ValueError, match="reference_id"):
            SuboffReferenceDatum(
                case_id="c", reference_id="", reference_source_id="s",
                Ct_reference=0.004, Re=2.0e6, uncertainty=0.0004,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_empty_source_citation(self) -> None:
        with pytest.raises(ValueError, match="source_citation"):
            SuboffReferenceDatum(
                case_id="c", reference_id="r", reference_source_id="s",
                Ct_reference=0.004, Re=2.0e6, uncertainty=0.0004,
                source_citation="", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_non_positive_Ct_reference(self) -> None:
        with pytest.raises(ValueError, match="Ct_reference"):
            SuboffReferenceDatum(
                case_id="c", reference_id="r", reference_source_id="s",
                Ct_reference=0.0, Re=2.0e6, uncertainty=0.0004,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_non_positive_Re(self) -> None:
        with pytest.raises(ValueError, match="Re"):
            SuboffReferenceDatum(
                case_id="c", reference_id="r", reference_source_id="s",
                Ct_reference=0.004, Re=0.0, uncertainty=0.0004,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_negative_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            SuboffReferenceDatum(
                case_id="c", reference_id="r", reference_source_id="s",
                Ct_reference=0.004, Re=2.0e6, uncertainty=-0.001,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_non_finite_Ct_reference(self) -> None:
        with pytest.raises(ValueError, match="Ct_reference"):
            SuboffReferenceDatum(
                case_id="c", reference_id="r", reference_source_id="s",
                Ct_reference=float("nan"), Re=2.0e6, uncertainty=0.0004,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_datum_rejects_bool_Ct_reference(self) -> None:
        with pytest.raises(TypeError, match="Ct_reference"):
            SuboffReferenceDatum(
                case_id="c", reference_id="r", reference_source_id="s",
                Ct_reference=True,  # type: ignore[arg-type]
                Re=2.0e6, uncertainty=0.0004,
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )


# ---------------------------------------------------------------------------
# 2. WITHHELD entries
# ---------------------------------------------------------------------------

class TestWithheldEntries:
    """Verify WITHHELD entries are correctly marked and carry no numeric value."""

    def test_withheld_datum_has_marker(self) -> None:
        d = SuboffReferenceDatum.withheld(
            case_id="SUBOFF-AFF1-experimental",
            reference_id="DARPA-SUBOFF-AFF1-experimental",
            reference_source_id="DARPA-SUBOFF-experimental",
            source_citation=(
                "DARPA SUBOFF AFF-1 bare hull experimental data; "
                "specific Ct values not independently verified."
            ),
            hull_type="bare_hull",
            reference_area_basis="wetted_surface",
            applicable_conditions="Re=1.2e7, single-phase, deep water",
            notes="WITHHELD: specific experimental Ct values not confirmed.",
        )
        assert d.is_withheld is True
        assert d.Ct_reference is None
        assert d.Re is None
        assert d.uncertainty is None

    def test_withheld_datum_rejects_non_empty_case_id(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            SuboffReferenceDatum.withheld(
                case_id="", reference_id="r", reference_source_id="s",
                source_citation="cite", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )

    def test_withheld_datum_rejects_empty_source_citation(self) -> None:
        with pytest.raises(ValueError, match="source_citation"):
            SuboffReferenceDatum.withheld(
                case_id="c", reference_id="r", reference_source_id="s",
                source_citation="", hull_type="bare_hull",
                reference_area_basis="wetted_surface",
                applicable_conditions="cond", notes="n",
            )


# ---------------------------------------------------------------------------
# 3. Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Verify the compiled reference-data registry."""

    def test_registry_is_non_empty(self) -> None:
        assert len(SUBOFF_REFERENCE_REGISTRY) > 0

    def test_registry_contains_only_typed_datums(self) -> None:
        for d in SUBOFF_REFERENCE_REGISTRY:
            assert isinstance(d, SuboffReferenceDatum)

    def test_every_non_withheld_entry_has_finite_positive_values(self) -> None:
        for d in SUBOFF_REFERENCE_REGISTRY:
            if d.is_withheld:
                continue
            assert isfinite(d.Ct_reference) and d.Ct_reference > 0.0, (
                f"{d.reference_id}: Ct_reference must be finite and positive"
            )
            assert isfinite(d.Re) and d.Re > 0.0, (
                f"{d.reference_id}: Re must be finite and positive"
            )
            assert isfinite(d.uncertainty) and d.uncertainty >= 0.0, (
                f"{d.reference_id}: uncertainty must be finite and >= 0"
            )

    def test_every_entry_has_non_empty_source_citation(self) -> None:
        for d in SUBOFF_REFERENCE_REGISTRY:
            assert d.source_citation and d.source_citation.strip()

    def test_every_entry_has_non_empty_applicable_conditions(self) -> None:
        for d in SUBOFF_REFERENCE_REGISTRY:
            assert d.applicable_conditions and d.applicable_conditions.strip()

    def test_registry_contains_ittc_reference(self) -> None:
        """The ITTC-1957 friction line must be present as a reference source."""
        source_ids = {d.reference_source_id for d in SUBOFF_REFERENCE_REGISTRY}
        assert any("ITTC" in sid for sid in source_ids)

    def test_registry_contains_withheld_entry(self) -> None:
        """At least one WITHHELD entry must exist for unconfirmable data."""
        assert any(d.is_withheld for d in SUBOFF_REFERENCE_REGISTRY)

    def test_reference_ids_are_unique(self) -> None:
        ids = [d.reference_id for d in SUBOFF_REFERENCE_REGISTRY]
        assert len(ids) == len(set(ids)), "reference_id values must be unique"


# ---------------------------------------------------------------------------
# 4. Look-up functions
# ---------------------------------------------------------------------------

class TestLookUp:
    """Verify look-up by case_id and reference_id."""

    def test_get_reference_data_returns_matching_datum(self) -> None:
        any_id = SUBOFF_REFERENCE_REGISTRY[0].reference_id
        d = get_reference_data(any_id)
        assert d is not None
        assert d.reference_id == any_id

    def test_get_reference_data_returns_none_for_unknown(self) -> None:
        assert get_reference_data("nonexistent-id-12345") is None

    def test_get_reference_data_by_case_returns_list(self) -> None:
        any_case = SUBOFF_REFERENCE_REGISTRY[0].case_id
        results = get_reference_data_by_case(any_case)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(d.case_id == any_case for d in results)

    def test_get_reference_data_by_case_returns_empty_for_unknown(self) -> None:
        assert get_reference_data_by_case("nonexistent-case-12345") == []

    def test_list_available_case_ids_is_non_empty(self) -> None:
        ids = list_available_case_ids()
        assert len(ids) > 0
        assert all(isinstance(i, str) and i for i in ids)

    def test_list_available_reference_ids_is_non_empty(self) -> None:
        ids = list_available_reference_ids()
        assert len(ids) > 0
        assert all(isinstance(i, str) and i for i in ids)


# ---------------------------------------------------------------------------
# 5. Integration with accuracy_recommendation gate
# ---------------------------------------------------------------------------

class TestAccuracyGateIntegration:
    """Verify reference data can feed the accuracy_recommendation gate."""

    def test_reference_data_provides_fields_for_error_metric(self) -> None:
        """A non-withheld datum must provide Ct_reference and uncertainty
        sufficient to construct an ErrorMetric for the gate."""
        non_withheld = [d for d in SUBOFF_REFERENCE_REGISTRY if not d.is_withheld]
        assert len(non_withheld) > 0
        d = non_withheld[0]
        # The gate's ErrorMetric needs a finite value and uncertainty.
        assert isinstance(d.Ct_reference, float)
        assert isfinite(d.Ct_reference) and d.Ct_reference > 0.0
        assert isinstance(d.uncertainty, float)
        assert isfinite(d.uncertainty) and d.uncertainty >= 0.0
        # The gate's PhysicalAccuracyEvidence needs case_id, reference_id,
        # reference_source_id as non-empty strings.
        assert d.case_id and d.reference_id and d.reference_source_id

    def test_withheld_datum_cannot_feed_error_metric(self) -> None:
        """A withheld datum must not provide a numeric Ct_reference."""
        withheld = [d for d in SUBOFF_REFERENCE_REGISTRY if d.is_withheld]
        assert len(withheld) > 0
        d = withheld[0]
        assert d.Ct_reference is None
        assert d.uncertainty is None


# ---------------------------------------------------------------------------
# 6. End-to-end integration with accuracy_recommendation gate
# ---------------------------------------------------------------------------

class TestEndToEndGateIntegration:
    """Demonstrate that compiled reference data can construct
    PhysicalAccuracyEvidence and drive recommend_by_physical_accuracy()."""

    def test_reference_data_feeds_gate_and_produces_recommendation(self) -> None:
        """Use a non-withheld reference datum to build PhysicalAccuracyEvidence
        for two candidates and verify the gate recommends the lower-error one."""
        from tensorlbm.accuracy_recommendation import (
            ConvergenceEvidence,
            ErrorMetric,
            KPIDefinition,
            PhysicalAccuracyEvidence,
            recommend_by_physical_accuracy,
        )

        # Pick a non-withheld reference datum.
        ref = next(
            d for d in SUBOFF_REFERENCE_REGISTRY if not d.is_withheld
        )
        assert ref.Ct_reference is not None
        assert ref.uncertainty is not None

        # Simulated measured Ct for two candidates.
        ct_candidate_a = ref.Ct_reference * 1.03   # 3% error
        ct_candidate_b = ref.Ct_reference * 1.08   # 8% error

        # Compute absolute relative error against the reference.
        err_a = abs(ct_candidate_a - ref.Ct_reference) / ref.Ct_reference
        err_b = abs(ct_candidate_b - ref.Ct_reference) / ref.Ct_reference

        kpi = KPIDefinition(
            "Ct_total", "1", "time_mean",
            "post-transient steady-state window",
        )

        _HASH = "a" * 64

        evidence = [
            PhysicalAccuracyEvidence(
                candidate_id="D3Q19-MRT",
                case_id=ref.case_id,
                reference_id=ref.reference_id,
                reference_source_id=ref.reference_source_id,
                configuration_hash=_HASH,
                provenance_hash=_HASH,
                kpi=kpi,
                error=ErrorMetric(
                    "absolute relative error",
                    "reference Ct_total",
                    err_a,
                    ref.uncertainty / ref.Ct_reference,
                ),
                convergence=ConvergenceEvidence(True, True, True),
            ),
            PhysicalAccuracyEvidence(
                candidate_id="D3Q27-cumulant",
                case_id=ref.case_id,
                reference_id=ref.reference_id,
                reference_source_id=ref.reference_source_id,
                configuration_hash=_HASH,
                provenance_hash=_HASH,
                kpi=kpi,
                error=ErrorMetric(
                    "absolute relative error",
                    "reference Ct_total",
                    err_b,
                    ref.uncertainty / ref.Ct_reference,
                ),
                convergence=ConvergenceEvidence(True, True, True),
            ),
        ]

        result = recommend_by_physical_accuracy(evidence)
        assert result.status == "RECOMMENDED_FROM_PHYSICAL_ACCURACY_EVIDENCE"
        assert result.recommended_candidate_id == "D3Q19-MRT"

    def test_withheld_reference_cannot_produce_admitted_evidence(self) -> None:
        """A WITHHELD reference datum must not be usable to construct
        a valid PhysicalAccuracyEvidence (no Ct_reference to compute error)."""
        withheld = next(
            d for d in SUBOFF_REFERENCE_REGISTRY if d.is_withheld
        )
        assert withheld.Ct_reference is None
        assert withheld.uncertainty is None
        # Without a numeric Ct_reference, no error metric can be computed.
        # This is the intended fail-closed behaviour.
