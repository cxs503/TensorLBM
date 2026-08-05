"""Fail-closed contract tests for the audited acoustics post-processing capability.

These tests verify the audit boundary, not acoustic physics correctness.  The
acoustics module is a post-processing layer that operates on saved density /
pressure history; it does not enter the timestep hot path and can be composed
with any collision / turbulence / boundary configuration as a post-processing
step.
"""
from __future__ import annotations

import pytest

from tensorlbm.acoustics_capability_contract import (
    NO_IMPLEMENTATION,
    VERIFICATION_CONTRACT_TESTED,
    VERIFICATION_NO_IMPLEMENTATION,
    WITHHELD_NO_IMPLEMENTATION,
    WITHHELD_NO_PHYSICS_VALIDATION,
    WITHHELD_UNKNOWN_FUNCTION,
    WITHHELD_UNKNOWN_LATTICE,
    AcousticsCapability,
    AcousticsWithheldError,
    PostProcessingAudit,
    acoustics_capability_matrix,
    acoustics_post_processing_audit,
    require_acoustics_capability,
)


# ---------------------------------------------------------------------------
# Matrix structure
# ---------------------------------------------------------------------------

_AUDITED_FUNCTIONS = (
    "fwh_far_field",
    "spl_spectrum",
    "surface_pressure_extraction",
    "oaspl",
    "fwh_result_wrapper",
)
_AUDITED_LATTICES = ("D2Q9", "D3Q19", "D3Q27", "N/A")


def test_matrix_covers_all_audited_functions_and_lattices() -> None:
    matrix = acoustics_capability_matrix()
    assert set(matrix) == set(_AUDITED_FUNCTIONS)
    for func in _AUDITED_FUNCTIONS:
        assert set(matrix[func]) == set(_AUDITED_LATTICES), func


# ---------------------------------------------------------------------------
# FWH far-field: lattice-agnostic (N/A), implemented + contract-tested
# ---------------------------------------------------------------------------

class TestFWHFarFieldCapability:
    def test_na_implemented_and_contract_tested(self) -> None:
        cap = acoustics_capability_matrix()["fwh_far_field"]["N/A"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint == "tensorlbm.acoustics.compute_fwh_far_field"
        assert cap.test_evidence is not None
        assert "test_acoustics.py" in cap.test_evidence

    def test_lattice_specific_not_implemented(self) -> None:
        """FWH far-field is lattice-agnostic; lattice-specific entries are N/A."""
        for lattice in ("D2Q9", "D3Q19", "D3Q27"):
            cap = acoustics_capability_matrix()["fwh_far_field"][lattice]
            assert cap.implementation_status == NO_IMPLEMENTATION


# ---------------------------------------------------------------------------
# SPL spectrum: lattice-agnostic (N/A), implemented + contract-tested
# ---------------------------------------------------------------------------

class TestSPLSpectrumCapability:
    def test_na_implemented_and_contract_tested(self) -> None:
        cap = acoustics_capability_matrix()["spl_spectrum"]["N/A"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint == "tensorlbm.acoustics.compute_spl_spectrum"
        assert cap.test_evidence is not None
        assert "test_acoustics.py" in cap.test_evidence

    def test_lattice_specific_not_implemented(self) -> None:
        for lattice in ("D2Q9", "D3Q19", "D3Q27"):
            cap = acoustics_capability_matrix()["spl_spectrum"][lattice]
            assert cap.implementation_status == NO_IMPLEMENTATION


# ---------------------------------------------------------------------------
# Surface pressure extraction: lattice-specific (D2Q9, D3Q19, D3Q27)
# ---------------------------------------------------------------------------

class TestSurfacePressureExtractionCapability:
    @pytest.mark.parametrize("lattice", ["D2Q9", "D3Q19", "D3Q27"])
    def test_lattice_implemented_and_contract_tested(self, lattice: str) -> None:
        cap = acoustics_capability_matrix()["surface_pressure_extraction"][lattice]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint == "tensorlbm.acoustics.extract_surface_pressure"
        assert cap.test_evidence is not None
        assert "test_acoustics.py" in cap.test_evidence

    def test_na_not_implemented(self) -> None:
        """Surface pressure extraction is lattice-specific; N/A is not applicable."""
        cap = acoustics_capability_matrix()["surface_pressure_extraction"]["N/A"]
        assert cap.implementation_status == NO_IMPLEMENTATION


# ---------------------------------------------------------------------------
# OASPL: lattice-agnostic (N/A), implemented + contract-tested
# ---------------------------------------------------------------------------

class TestOASPLCapability:
    def test_na_implemented_and_contract_tested(self) -> None:
        cap = acoustics_capability_matrix()["oaspl"]["N/A"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint == "tensorlbm.acoustics.oaspl"
        assert cap.test_evidence is not None
        assert "test_acoustics.py" in cap.test_evidence


# ---------------------------------------------------------------------------
# FWH result wrapper: lattice-agnostic (N/A), implemented + contract-tested
# ---------------------------------------------------------------------------

class TestFWHResultWrapperCapability:
    def test_na_implemented_and_contract_tested(self) -> None:
        cap = acoustics_capability_matrix()["fwh_result_wrapper"]["N/A"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint == "tensorlbm.acoustics.compute_fwh_result"
        assert cap.test_evidence is not None
        assert "test_acoustics.py" in cap.test_evidence


# ---------------------------------------------------------------------------
# Fail-closed: no combination is physics-validated
# ---------------------------------------------------------------------------

def test_all_capabilities_fail_closed() -> None:
    matrix = acoustics_capability_matrix()
    for func, lattices in matrix.items():
        for lattice, cap in lattices.items():
            assert not cap.available, (func, lattice)
            assert cap.status.startswith("WITHHELD_"), (func, lattice)


def test_contract_tested_does_not_imply_physics_validation() -> None:
    """Contract tests verify shape/finiteness/causality, not acoustic accuracy."""
    cap = acoustics_capability_matrix()["fwh_far_field"]["N/A"]
    assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
    assert cap.status == WITHHELD_NO_PHYSICS_VALIDATION
    assert not cap.available


# ---------------------------------------------------------------------------
# Post-processing nature: not in timestep hot path
# ---------------------------------------------------------------------------

def test_all_hot_path_notes_are_none() -> None:
    """Acoustics is post-processing; no hot-path sync or allocation issues."""
    matrix = acoustics_capability_matrix()
    for func, lattices in matrix.items():
        for lattice, cap in lattices.items():
            if cap.implementation_status == "IMPLEMENTED":
                assert cap.hot_path_note is None, (func, lattice)


def test_notes_mention_post_processing() -> None:
    matrix = acoustics_capability_matrix()
    for func, lattices in matrix.items():
        for lattice, cap in lattices.items():
            if cap.implementation_status == "IMPLEMENTED":
                note_lower = cap.note.lower()
                assert "post-processing" in note_lower or "post processing" in note_lower, (
                    func, lattice, cap.note
                )


def test_notes_mention_composability() -> None:
    """Acoustics can be composed with any collision/turbulence/boundary."""
    matrix = acoustics_capability_matrix()
    for func, lattices in matrix.items():
        for lattice, cap in lattices.items():
            if cap.implementation_status == "IMPLEMENTED":
                note_lower = cap.note.lower()
                assert "compos" in note_lower or "any collision" in note_lower, (
                    func, lattice, cap.note
                )


# ---------------------------------------------------------------------------
# require_acoustics_capability always raises (fail-closed)
# ---------------------------------------------------------------------------

def test_require_always_raises_for_contract_tested() -> None:
    with pytest.raises(AcousticsWithheldError, match="WITHHELD_NO_PHYSICS_VALIDATION"):
        require_acoustics_capability("fwh_far_field", "N/A")


def test_require_always_raises_for_no_implementation() -> None:
    with pytest.raises(AcousticsWithheldError, match="WITHHELD_NO_IMPLEMENTATION"):
        require_acoustics_capability("fwh_far_field", "D2Q9")


# ---------------------------------------------------------------------------
# Unknown inputs rejected with stable machine-readable codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("function", "lattice", "withheld_code"),
    [
        ("unknown_function", "N/A", WITHHELD_UNKNOWN_FUNCTION),
        ("fwh_far_field", "D4Q99", WITHHELD_UNKNOWN_LATTICE),
    ],
)
def test_unknown_inputs_rejected_before_matrix_lookup(
    function: str, lattice: str, withheld_code: str,
) -> None:
    with pytest.raises(AcousticsWithheldError, match=withheld_code):
        require_acoustics_capability(function, lattice)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Post-processing audit
# ---------------------------------------------------------------------------

def test_post_processing_audit_returns_nonempty() -> None:
    audit = acoustics_post_processing_audit()
    assert len(audit) > 0


def test_post_processing_audit_documents_all_functions() -> None:
    audit = acoustics_post_processing_audit()
    audited_functions = {entry.function for entry in audit}
    assert audited_functions == set(_AUDITED_FUNCTIONS)


def test_post_processing_audit_entries_mark_cold_path() -> None:
    audit = acoustics_post_processing_audit()
    for entry in audit:
        assert entry.path_type == "POST_PROCESSING"
        assert "not in timestep hot path" in entry.note.lower()


def test_post_processing_audit_entries_have_entrypoints() -> None:
    audit = acoustics_post_processing_audit()
    for entry in audit:
        assert entry.entrypoint is not None
        assert entry.entrypoint.startswith("tensorlbm.acoustics.")
