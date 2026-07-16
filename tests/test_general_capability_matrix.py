"""Contract tests for the pure, fail-closed general capability matrix R1."""
from __future__ import annotations

from tensorlbm.general_capability_matrix import (
    CapabilityRequest,
    CapabilityStatus,
    EvidenceTier,
    assess_capability,
    capability_matrix,
)


def test_known_d3q19_mrt_single_phase_static_wall_no_amr_is_supported() -> None:
    result = assess_capability(CapabilityRequest(lattice="D3Q19", collision="MRT"))

    assert result.status is CapabilityStatus.SUPPORTED
    assert result.evidence_tier is EvidenceTier.EXECUTABLE_CONTRACT
    assert result.supported
    assert result.capability_hash
    assert len(result.config_hash) == 64
    assert result.to_dict()["status"] == "supported"


def test_d3q27_mrt_is_withheld_without_complete_composition_evidence() -> None:
    result = assess_capability({"lattice": "D3Q27", "collision": "MRT"})

    assert result.status is CapabilityStatus.WITHHELD
    assert result.evidence_tier is EvidenceTier.NO_COMPOSITION_EVIDENCE
    assert any(reason.code == "WITHHELD_D3Q27_COMPOSITION" for reason in result.reasons)


def test_cm_and_kbc_are_explicitly_withheld_not_advertised_from_legacy_names() -> None:
    for collision in ("CM", "KBC"):
        result = assess_capability({"lattice": "D3Q19", "collision": collision})
        assert result.status is CapabilityStatus.WITHHELD
        assert result.evidence_tier is EvidenceTier.UNIMPLEMENTED
        assert any(reason.code == "WITHHELD_COLLISION_FAMILY" for reason in result.reasons)


def test_wall_function_and_amr_candidates_are_withheld_as_unverified_compositions() -> None:
    result = assess_capability({
        "lattice": "D3Q19",
        "collision": "MRT",
        "wall_treatment": "wall_function",
        "refinement": "amr",
    })

    assert result.status is CapabilityStatus.WITHHELD
    assert result.evidence_tier is EvidenceTier.NO_COMPOSITION_EVIDENCE
    assert {reason.field for reason in result.reasons} >= {"wall_treatment", "refinement"}


def test_unknown_values_are_not_supported_and_hashes_are_normalized() -> None:
    result = assess_capability({"lattice": "D3Q99", "collision": "MRT"})
    equivalent = assess_capability({"lattice": " d3q19 ", "collision": "mrt"})
    baseline = assess_capability({"lattice": "D3Q19", "collision": "MRT"})

    assert result.status is CapabilityStatus.NOT_SUPPORTED
    assert result.evidence_tier is EvidenceTier.UNKNOWN_REQUEST
    assert equivalent.config_hash == baseline.config_hash
    assert equivalent.capability_hash == baseline.capability_hash


def test_unknown_values_in_every_field_are_not_supported_before_composition_assessment() -> None:
    unknown_by_field = {
        "lattice": "d3q99",
        "collision": "unknown_collision",
        "turbulence": "unknown_turbulence",
        "multiphase": "unknown_multiphase",
        "boundary": "unknown_boundary",
        "geometry": "unknown_geometry",
        "wall_treatment": "unknown_wall_treatment",
        "refinement": "unknown_refinement",
        "backend": "unknown_backend",
        "outputs": ["rho", "unknown_output"],
    }

    for field, value in unknown_by_field.items():
        result = assess_capability({"lattice": "d3q19", "collision": "mrt", field: value})
        assert result.status is CapabilityStatus.NOT_SUPPORTED
        assert result.evidence_tier is EvidenceTier.UNKNOWN_REQUEST
        assert any(reason.code == "UNKNOWN_VALUE" and reason.field == field for reason in result.reasons)
        assert not any(reason.code.startswith("WITHHELD_") for reason in result.reasons)


def test_known_but_unverified_values_remain_withheld() -> None:
    result = assess_capability({
        "lattice": "d3q19", "collision": "mrt", "turbulence": "les",
        "outputs": ["rho", "pressure"],
    })

    assert result.status is CapabilityStatus.WITHHELD
    assert result.evidence_tier is EvidenceTier.NO_COMPOSITION_EVIDENCE
    assert {reason.field for reason in result.reasons} >= {"turbulence", "outputs"}


def test_component_registry_exposes_only_audited_collision_availability() -> None:
    matrix = capability_matrix()

    assert matrix["collision"]["mrt"]["available"] is True
    assert matrix["collision"]["cm"]["available"] is False
    assert matrix["collision"]["kbc"]["available"] is False
