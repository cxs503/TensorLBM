"""TDD specification for the unified cross-module composition admission matrix.

This module cross-queries every merged single-dimension capability contract and
aggregates their decisions into a single fail-closed composition decision.

Aggregation rules:
  - All dimensions ADMITTED (or NOT_APPLICABLE) → SUPPORTED
  - Any dimension WITHHELD (and none NOT_SUPPORTED) → WITHHELD
  - Any dimension NOT_SUPPORTED → NOT_SUPPORTED
"""
from __future__ import annotations

import pytest

from tensorlbm.cross_module_composition_matrix import (
    CompositionDecision,
    CompositionRequest,
    CompositionStatus,
    MATRIX_VERSION,
    SubContractResult,
    SubContractStatus,
    assess_composition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def baseline(**changes: str) -> CompositionRequest:
    """Return the R1 baseline request with optional overrides."""
    defaults: dict[str, object] = dict(
        lattice="d3q19",
        collision="mrt",
        turbulence="none",
        multiphase="single_phase",
        boundary="static_wall",
        geometry="static_solid_mask",
        wall_treatment="bounce_back",
        refinement="none",
        backend="torch",
        outputs=("rho", "velocity"),
    )
    defaults.update(changes)
    return CompositionRequest(**defaults)  # type: ignore[arg-type]


def _names(results: tuple[SubContractResult, ...]) -> set[str]:
    return {r.contract_name for r in results}


def _by_contract(results: tuple[SubContractResult, ...], name: str) -> SubContractResult:
    for r in results:
        if r.contract_name == name:
            return r
    raise KeyError(name)


# ---------------------------------------------------------------------------
# RED: baseline assessment
# ---------------------------------------------------------------------------

class TestBaseline:
    """The R1 baseline D3Q19/MRT single-phase static-wall configuration."""

    def test_returns_composition_decision(self) -> None:
        decision = assess_composition(baseline())
        assert isinstance(decision, CompositionDecision)

    def test_baseline_is_withheld_due_to_boundary_no_complete_composition(self) -> None:
        """The boundary contract withholds every combination because no complete
        composition evidence exists.  Therefore even the R1 baseline is WITHHELD."""
        decision = assess_composition(baseline())
        assert decision.status is CompositionStatus.WITHHELD
        boundary = _by_contract(decision.sub_contract_results, "boundary_capability_contract")
        assert boundary.status is SubContractStatus.WITHHELD
        assert "WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE" in boundary.reason_codes

    def test_collision_dimension_is_admitted_for_baseline(self) -> None:
        decision = assess_composition(baseline())
        collision = _by_contract(decision.sub_contract_results, "advanced_collision_contract")
        assert collision.status is SubContractStatus.ADMITTED

    def test_wall_refinement_gate_is_admitted_for_baseline(self) -> None:
        decision = assess_composition(baseline())
        gate = _by_contract(decision.sub_contract_results, "wall_refinement_combination_gate")
        assert gate.status is SubContractStatus.ADMITTED

    def test_general_capability_matrix_is_admitted_for_baseline(self) -> None:
        decision = assess_composition(baseline())
        gmc = _by_contract(decision.sub_contract_results, "general_capability_matrix")
        assert gmc.status is SubContractStatus.ADMITTED

    def test_turbulence_is_not_applicable_when_none(self) -> None:
        decision = assess_composition(baseline())
        turb = _by_contract(decision.sub_contract_results, "turbulence_capability_contract")
        assert turb.status is SubContractStatus.NOT_APPLICABLE

    def test_amr_is_not_applicable_when_none(self) -> None:
        decision = assess_composition(baseline())
        amr = _by_contract(decision.sub_contract_results, "amr_capability_contract")
        assert amr.status is SubContractStatus.NOT_APPLICABLE

    def test_wall_function_is_not_applicable_when_bounce_back(self) -> None:
        decision = assess_composition(baseline())
        wf = _by_contract(decision.sub_contract_results, "wall_function_contract")
        assert wf.status is SubContractStatus.NOT_APPLICABLE

    def test_wall_function_admission_is_not_applicable_when_bounce_back(self) -> None:
        decision = assess_composition(baseline())
        wfa = _by_contract(decision.sub_contract_results, "wall_function_admission")
        assert wfa.status is SubContractStatus.NOT_APPLICABLE

    def test_accuracy_is_not_applicable_without_evidence(self) -> None:
        decision = assess_composition(baseline())
        acc = _by_contract(decision.sub_contract_results, "accuracy_recommendation")
        assert acc.status is SubContractStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# RED: D3Q27
# ---------------------------------------------------------------------------

class TestD3Q27:
    def test_d3q27_mrt_is_withheld(self) -> None:
        decision = assess_composition(baseline(lattice="d3q27"))
        assert decision.status is CompositionStatus.WITHHELD

    def test_d3q27_collision_is_admitted(self) -> None:
        decision = assess_composition(baseline(lattice="d3q27"))
        collision = _by_contract(decision.sub_contract_results, "advanced_collision_contract")
        assert collision.status is SubContractStatus.ADMITTED

    def test_d3q27_general_matrix_is_withheld(self) -> None:
        decision = assess_composition(baseline(lattice="d3q27"))
        gmc = _by_contract(decision.sub_contract_results, "general_capability_matrix")
        assert gmc.status is SubContractStatus.WITHHELD
        assert any("WITHHELD_D3Q27_COMPOSITION" in c for c in gmc.reason_codes)

    def test_d3q27_boundary_is_withheld(self) -> None:
        decision = assess_composition(baseline(lattice="d3q27"))
        boundary = _by_contract(decision.sub_contract_results, "boundary_capability_contract")
        assert boundary.status is SubContractStatus.WITHHELD


# ---------------------------------------------------------------------------
# RED: wall function
# ---------------------------------------------------------------------------

class TestWallFunction:
    def test_wall_function_is_withheld(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        assert decision.status is CompositionStatus.WITHHELD

    def test_wall_function_contract_is_queried(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        wf = _by_contract(decision.sub_contract_results, "wall_function_contract")
        # The wall function contract admits the implementation-only baseline
        assert wf.status is SubContractStatus.ADMITTED

    def test_wall_function_admission_is_queried(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        wfa = _by_contract(decision.sub_contract_results, "wall_function_admission")
        assert wfa.status is SubContractStatus.ADMITTED

    def test_wall_refinement_gate_withholds_wall_function(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        gate = _by_contract(decision.sub_contract_results, "wall_refinement_combination_gate")
        assert gate.status is SubContractStatus.WITHHELD
        assert any("WITHHELD" in c for c in gate.reason_codes)

    def test_general_matrix_withholds_wall_function(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        gmc = _by_contract(decision.sub_contract_results, "general_capability_matrix")
        assert gmc.status is SubContractStatus.WITHHELD


# ---------------------------------------------------------------------------
# RED: AMR
# ---------------------------------------------------------------------------

class TestAMR:
    def test_amr_is_withheld(self) -> None:
        decision = assess_composition(baseline(refinement="amr"))
        assert decision.status is CompositionStatus.WITHHELD

    def test_amr_contract_is_queried_and_withheld(self) -> None:
        decision = assess_composition(baseline(refinement="amr"))
        amr = _by_contract(decision.sub_contract_results, "amr_capability_contract")
        assert amr.status is SubContractStatus.WITHHELD
        assert any("WITHHELD" in c for c in amr.reason_codes)

    def test_wall_refinement_gate_withholds_amr(self) -> None:
        decision = assess_composition(baseline(refinement="amr"))
        gate = _by_contract(decision.sub_contract_results, "wall_refinement_combination_gate")
        assert gate.status is SubContractStatus.WITHHELD

    def test_amr_reports_missing_evidence(self) -> None:
        decision = assess_composition(baseline(refinement="amr"))
        amr = _by_contract(decision.sub_contract_results, "amr_capability_contract")
        assert len(amr.missing_evidence) > 0


# ---------------------------------------------------------------------------
# RED: turbulence
# ---------------------------------------------------------------------------

class TestTurbulence:
    def test_smagorinsky_is_withheld(self) -> None:
        decision = assess_composition(baseline(turbulence="smagorinsky"))
        assert decision.status is CompositionStatus.WITHHELD

    def test_turbulence_contract_is_queried(self) -> None:
        decision = assess_composition(baseline(turbulence="smagorinsky"))
        turb = _by_contract(decision.sub_contract_results, "turbulence_capability_contract")
        assert turb.status is SubContractStatus.WITHHELD
        assert any("WITHHELD" in c for c in turb.reason_codes)

    def test_general_matrix_withholds_smagorinsky(self) -> None:
        decision = assess_composition(baseline(turbulence="smagorinsky"))
        gmc = _by_contract(decision.sub_contract_results, "general_capability_matrix")
        assert gmc.status is SubContractStatus.WITHHELD


# ---------------------------------------------------------------------------
# RED: NOT_SUPPORTED
# ---------------------------------------------------------------------------

class TestNotSupported:
    def test_unknown_lattice_is_not_supported(self) -> None:
        decision = assess_composition(baseline(lattice="d3q99"))
        assert decision.status is CompositionStatus.NOT_SUPPORTED

    def test_unknown_collision_is_not_supported(self) -> None:
        decision = assess_composition(baseline(collision="unknown"))
        assert decision.status is CompositionStatus.NOT_SUPPORTED

    def test_not_supported_dominates_over_withheld(self) -> None:
        """If one dimension is NOT_SUPPORTED and another is WITHHELD,
        the overall is NOT_SUPPORTED."""
        decision = assess_composition(baseline(lattice="d3q99", turbulence="smagorinsky"))
        assert decision.status is CompositionStatus.NOT_SUPPORTED

    def test_unknown_lattice_collision_contract_is_not_supported(self) -> None:
        decision = assess_composition(baseline(lattice="d3q99"))
        collision = _by_contract(decision.sub_contract_results, "advanced_collision_contract")
        assert collision.status is SubContractStatus.NOT_SUPPORTED


# ---------------------------------------------------------------------------
# RED: non-single-phase
# ---------------------------------------------------------------------------

class TestNonSinglePhase:
    def test_free_surface_is_withheld(self) -> None:
        decision = assess_composition(baseline(multiphase="free_surface"))
        assert decision.status is CompositionStatus.WITHHELD

    def test_boundary_withholds_free_surface_physics(self) -> None:
        decision = assess_composition(baseline(multiphase="free_surface"))
        boundary = _by_contract(decision.sub_contract_results, "boundary_capability_contract")
        assert boundary.status is SubContractStatus.WITHHELD
        assert any("PHYSICS" in c for c in boundary.reason_codes)

    def test_phase_field_is_withheld(self) -> None:
        decision = assess_composition(baseline(multiphase="phase_field"))
        assert decision.status is CompositionStatus.WITHHELD


# ---------------------------------------------------------------------------
# RED: all sub-contracts queried
# ---------------------------------------------------------------------------

class TestAllSubContractsQueried:
    EXPECTED = {
        "advanced_collision_contract",
        "general_capability_matrix",
        "wall_function_contract",
        "wall_function_admission",
        "wall_refinement_combination_gate",
        "amr_capability_contract",
        "boundary_capability_contract",
        "turbulence_capability_contract",
        "accuracy_recommendation",
    }

    def test_all_nine_sub_contracts_are_queried(self) -> None:
        decision = assess_composition(baseline())
        assert _names(decision.sub_contract_results) == self.EXPECTED

    def test_each_result_has_contract_name_and_dimension(self) -> None:
        decision = assess_composition(baseline())
        for r in decision.sub_contract_results:
            assert r.contract_name
            assert r.dimension
            assert isinstance(r.status, SubContractStatus)
            assert isinstance(r.reason_codes, tuple)
            assert isinstance(r.missing_evidence, tuple)
            assert isinstance(r.note, str)


# ---------------------------------------------------------------------------
# RED: reason codes and missing dimensions
# ---------------------------------------------------------------------------

class TestReasonCodesAndMissing:
    def test_withheld_results_contain_reason_codes(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        withheld = [r for r in decision.sub_contract_results if r.status is SubContractStatus.WITHHELD]
        assert len(withheld) > 0
        for r in withheld:
            assert len(r.reason_codes) > 0

    def test_decision_has_aggregated_reason_codes(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        assert len(decision.reason_codes) > 0

    def test_missing_dimensions_are_reported_for_amr(self) -> None:
        decision = assess_composition(baseline(refinement="amr"))
        assert len(decision.missing_dimensions) > 0

    def test_missing_dimensions_are_reported_for_wall_function_amr(self) -> None:
        decision = assess_composition(
            baseline(wall_treatment="wall_function", refinement="amr")
        )
        assert len(decision.missing_dimensions) > 0
        # Wall refinement gate reports cross-level evidence requirements
        assert any("wall_distance" in d or "y_plus" in d for d in decision.missing_dimensions)


# ---------------------------------------------------------------------------
# RED: normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_case_insensitive(self) -> None:
        d1 = assess_composition(baseline(lattice="D3Q19"))
        d2 = assess_composition(baseline(lattice="d3q19"))
        assert d1.normalized_request.lattice == d2.normalized_request.lattice

    def test_alias_resolution(self) -> None:
        d1 = assess_composition(baseline(turbulence="les"))
        d2 = assess_composition(baseline(turbulence="smagorinsky"))
        assert d1.normalized_request.turbulence == d2.normalized_request.turbulence

    def test_outputs_are_sorted(self) -> None:
        d = assess_composition(baseline(outputs=("velocity", "rho")))
        assert d.normalized_request.outputs == ("rho", "velocity")

    def test_mapping_input_accepted(self) -> None:
        d = assess_composition({"lattice": "d3q19", "collision": "mrt"})
        assert d.normalized_request.lattice == "d3q19"

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            assess_composition({"lattice": "d3q19", "collision": "mrt", "bogus": "x"})


# ---------------------------------------------------------------------------
# RED: to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_is_json_ready(self) -> None:
        import json

        decision = assess_composition(baseline())
        d = decision.to_dict()
        # Must be JSON-serializable
        json.dumps(d)
        assert d["status"] == decision.status.value
        assert len(d["sub_contract_results"]) == 9
        assert "missing_dimensions" in d
        assert "reason_codes" in d
        assert "normalized_request" in d

    def test_to_dict_contains_sub_contract_details(self) -> None:
        decision = assess_composition(baseline(wall_treatment="wall_function"))
        d = decision.to_dict()
        for r in d["sub_contract_results"]:
            assert "contract_name" in r
            assert "dimension" in r
            assert "status" in r
            assert "reason_codes" in r
            assert "missing_evidence" in r
            assert "note" in r


# ---------------------------------------------------------------------------
# RED: matrix version
# ---------------------------------------------------------------------------

class TestMatrixVersion:
    def test_version_is_stable_string(self) -> None:
        assert isinstance(MATRIX_VERSION, str)
        assert "cross-module" in MATRIX_VERSION
