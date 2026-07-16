"""Fail-closed contract tests for audited turbulence model capabilities.

These tests verify the audit boundary, not turbulence physics.  Contract tests
(shape/mass/momentum/identity) that exist for some operators are recorded as
evidence but never promoted to physics validation.
"""
from __future__ import annotations

import pytest

from tensorlbm.turbulence_capability_contract import (
    NO_IMPLEMENTATION,
    VERIFICATION_BENCHMARK_ONLY,
    VERIFICATION_CONTRACT_TESTED,
    VERIFICATION_IMPLEMENTED_ONLY,
    VERIFICATION_NO_IMPLEMENTATION,
    WITHHELD_NO_CONTRACT_TESTS,
    WITHHELD_NO_IMPLEMENTATION,
    WITHHELD_NO_PHYSICS_VALIDATION,
    WITHHELD_UNKNOWN_COLLISION,
    WITHHELD_UNKNOWN_FAMILY,
    WITHHELD_UNKNOWN_LATTICE,
    TurbulenceCapability,
    TurbulenceWithheldError,
    require_turbulence_capability,
    turbulence_capability_matrix,
    turbulence_hot_path_audit,
)


# ---------------------------------------------------------------------------
# Matrix structure
# ---------------------------------------------------------------------------

def test_matrix_covers_all_audited_families_lattices_collisions() -> None:
    matrix = turbulence_capability_matrix()
    expected_families = {
        "smagorinsky", "dynamic_smagorinsky", "wale", "vreman",
        "rans_ke", "rans_sa", "komega_sst", "ddes",
        "wall_function", "wall_distance",
    }
    assert set(matrix) == expected_families
    for family in expected_families:
        assert set(matrix[family]) == {"D2Q9", "D3Q19", "D3Q27"}, family
        for lattice in ("D2Q9", "D3Q19", "D3Q27"):
            assert set(matrix[family][lattice]) == {"BGK", "MRT", "CG", "N/A"}, (family, lattice)


# ---------------------------------------------------------------------------
# Smagorinsky: all 6 lattice/collision combinations implemented + contract-tested
# ---------------------------------------------------------------------------

class TestSmagorinsky:
    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D2Q9", "BGK"), ("D2Q9", "MRT"),
            ("D3Q19", "BGK"), ("D3Q19", "MRT"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_all_six_combinations_implemented_and_contract_tested(
        self, lattice: str, collision: str,
    ) -> None:
        cap = turbulence_capability_matrix()["smagorinsky"][lattice][collision]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint is not None
        assert cap.entrypoint.startswith("tensorlbm.turbulence.collide_smagorinsky")
        assert cap.test_evidence is not None

    def test_d2q9_bgk_has_mass_momentum_identity_tests(self) -> None:
        cap = turbulence_capability_matrix()["smagorinsky"]["D2Q9"]["BGK"]
        assert "test_marine.py" in cap.test_evidence
        assert "mass" in cap.test_evidence.lower()

    def test_d2q9_mrt_has_mass_momentum_tests(self) -> None:
        cap = turbulence_capability_matrix()["smagorinsky"]["D2Q9"]["MRT"]
        assert "test_phase4.py" in cap.test_evidence


# ---------------------------------------------------------------------------
# Dynamic Smagorinsky: only D2Q9 BGK and D3Q19 BGK
# ---------------------------------------------------------------------------

class TestDynamicSmagorinsky:
    @pytest.mark.parametrize(
        ("lattice", "collision", "expected_impl"),
        [
            ("D2Q9", "BGK", "IMPLEMENTED"),
            ("D3Q19", "BGK", "IMPLEMENTED"),
            ("D3Q19", "MRT", "IMPLEMENTED"),
            ("D2Q9", "MRT", NO_IMPLEMENTATION),
            ("D3Q27", "BGK", NO_IMPLEMENTATION),
            ("D3Q27", "MRT", NO_IMPLEMENTATION),
        ],
    )
    def test_combinations(self, lattice: str, collision: str, expected_impl: str) -> None:
        cap = turbulence_capability_matrix()["dynamic_smagorinsky"][lattice][collision]
        assert cap.implementation_status == expected_impl

    def test_implemented_combinations_are_contract_tested(self) -> None:
        matrix = turbulence_capability_matrix()["dynamic_smagorinsky"]
        for lattice in ("D2Q9", "D3Q19"):
            cap = matrix[lattice]["BGK"]
            assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
            assert cap.test_evidence is not None
            assert "test_dynamic_smagorinsky.py" in cap.test_evidence

    def test_d3q19_mrt_is_implemented_and_contract_tested(self) -> None:
        cap = turbulence_capability_matrix()["dynamic_smagorinsky"]["D3Q19"]["MRT"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint == "tensorlbm.turbulence.collide_dynamic_smagorinsky_mrt3d"
        assert cap.test_evidence is not None
        assert "test_dynamic_smagorinsky.py" in cap.test_evidence

    def test_hot_path_note_documents_item_sync(self) -> None:
        cap = turbulence_capability_matrix()["dynamic_smagorinsky"]["D2Q9"]["BGK"]
        assert cap.hot_path_note is not None
        assert "item()" in cap.hot_path_note


# ---------------------------------------------------------------------------
# WALE: BGK only (D2Q9, D3Q19, D3Q27); no MRT
# ---------------------------------------------------------------------------

class TestWALE:
    @pytest.mark.parametrize("lattice", ["D2Q9", "D3Q19", "D3Q27"])
    def test_bgk_implemented_and_contract_tested(self, lattice: str) -> None:
        cap = turbulence_capability_matrix()["wale"][lattice]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint is not None
        assert cap.entrypoint.startswith("tensorlbm.turbulence.collide_wale_bgk")
        assert cap.test_evidence is not None
        assert "test_turbulence_extensions.py" in cap.test_evidence

    @pytest.mark.parametrize("lattice", ["D2Q9", "D3Q19", "D3Q27"])
    def test_mrt_not_implemented(self, lattice: str) -> None:
        cap = turbulence_capability_matrix()["wale"][lattice]["MRT"]
        assert cap.implementation_status == NO_IMPLEMENTATION
        assert cap.verification_level == VERIFICATION_NO_IMPLEMENTATION


# ---------------------------------------------------------------------------
# Vreman: BGK only (D2Q9, D3Q19, D3Q27); no MRT
# ---------------------------------------------------------------------------

class TestVreman:
    @pytest.mark.parametrize("lattice", ["D2Q9", "D3Q19", "D3Q27"])
    def test_bgk_implemented_and_contract_tested(self, lattice: str) -> None:
        cap = turbulence_capability_matrix()["vreman"][lattice]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
        assert cap.entrypoint is not None
        assert cap.entrypoint.startswith("tensorlbm.turbulence.collide_vreman_bgk")
        assert cap.test_evidence is not None
        assert "test_turbulence_extensions.py" in cap.test_evidence

    @pytest.mark.parametrize("lattice", ["D2Q9", "D3Q19", "D3Q27"])
    def test_mrt_not_implemented(self, lattice: str) -> None:
        cap = turbulence_capability_matrix()["vreman"][lattice]["MRT"]
        assert cap.implementation_status == NO_IMPLEMENTATION
        assert cap.verification_level == VERIFICATION_NO_IMPLEMENTATION


# ---------------------------------------------------------------------------
# RANS-KE: only D3Q19 MRT, implemented but no tests
# ---------------------------------------------------------------------------

class TestRansKE:
    def test_d3q19_mrt_implemented_no_tests(self) -> None:
        cap = turbulence_capability_matrix()["rans_ke"]["D3Q19"]["MRT"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_IMPLEMENTED_ONLY
        assert cap.entrypoint is not None
        assert "collide_rans_ke" in cap.entrypoint
        assert "KESolver" in cap.entrypoint
        assert cap.test_evidence is None

    def test_hot_path_note_documents_mask_bool_allocation(self) -> None:
        cap = turbulence_capability_matrix()["rans_ke"]["D3Q19"]["MRT"]
        assert cap.hot_path_note is not None
        assert "mask.bool()" in cap.hot_path_note

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D2Q9", "BGK"), ("D2Q9", "MRT"),
            ("D3Q19", "BGK"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_other_combinations_not_implemented(
        self, lattice: str, collision: str,
    ) -> None:
        cap = turbulence_capability_matrix()["rans_ke"][lattice][collision]
        assert cap.implementation_status == NO_IMPLEMENTATION


# ---------------------------------------------------------------------------
# RANS-SA: only D3Q19 MRT, implemented but no tests
# ---------------------------------------------------------------------------

class TestRansSA:
    def test_d3q19_mrt_implemented_no_tests(self) -> None:
        cap = turbulence_capability_matrix()["rans_sa"]["D3Q19"]["MRT"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_IMPLEMENTED_ONLY
        assert cap.entrypoint is not None
        assert "collide_rans_sa" in cap.entrypoint
        assert "SASolver" in cap.entrypoint
        assert cap.test_evidence is None

    def test_hot_path_note_documents_scalar_averaging(self) -> None:
        cap = turbulence_capability_matrix()["rans_sa"]["D3Q19"]["MRT"]
        assert cap.hot_path_note is not None
        assert "mean().item()" in cap.hot_path_note


# ---------------------------------------------------------------------------
# k-omega SST: only D2Q9 BGK, implemented but no tests
# ---------------------------------------------------------------------------

class TestKOmegaSST:
    def test_d2q9_bgk_implemented_no_tests(self) -> None:
        cap = turbulence_capability_matrix()["komega_sst"]["D2Q9"]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_IMPLEMENTED_ONLY
        assert cap.entrypoint is not None
        assert "komega_sst_collision_d2q9" in cap.entrypoint
        assert cap.test_evidence is None


# ---------------------------------------------------------------------------
# DDES: only D2Q9 BGK, implemented but no tests, 2D only
# ---------------------------------------------------------------------------

class TestDDES:
    def test_d2q9_bgk_implemented_no_tests(self) -> None:
        cap = turbulence_capability_matrix()["ddes"]["D2Q9"]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_IMPLEMENTED_ONLY
        assert cap.entrypoint is not None
        assert "apply_ddes_collision" in cap.entrypoint
        assert cap.test_evidence is None

    def test_note_documents_2d_only(self) -> None:
        cap = turbulence_capability_matrix()["ddes"]["D2Q9"]["BGK"]
        assert "2D" in cap.note or "2-D" in cap.note


# ---------------------------------------------------------------------------
# Wall function: D3Q19 only, benchmark-only (examples, no unit tests)
# ---------------------------------------------------------------------------

class TestWallFunction:
    def test_d3q19_benchmark_only(self) -> None:
        cap = turbulence_capability_matrix()["wall_function"]["D3Q19"]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_BENCHMARK_ONLY
        assert cap.entrypoint is not None
        assert "wall_function_3d" in cap.entrypoint
        assert cap.test_evidence is not None
        assert "example" in cap.test_evidence.lower()

    def test_hot_path_note_documents_bool_sync(self) -> None:
        cap = turbulence_capability_matrix()["wall_function"]["D3Q19"]["BGK"]
        assert cap.hot_path_note is not None
        assert "bool(" in cap.hot_path_note


# ---------------------------------------------------------------------------
# Wall distance: D2Q9 and D3Q19, implemented but no tests
# ---------------------------------------------------------------------------

class TestWallDistance:
    @pytest.mark.parametrize("lattice", ["D2Q9", "D3Q19"])
    def test_implemented_no_tests(self, lattice: str) -> None:
        cap = turbulence_capability_matrix()["wall_distance"][lattice]["N/A"]
        assert cap.implementation_status == "IMPLEMENTED"
        assert cap.verification_level == VERIFICATION_IMPLEMENTED_ONLY
        assert cap.entrypoint is not None
        assert "compute_wall_distance_fmm" in cap.entrypoint
        assert cap.test_evidence is None


# ---------------------------------------------------------------------------
# Fail-closed: no combination is physics-validated
# ---------------------------------------------------------------------------

def test_all_capabilities_fail_closed() -> None:
    matrix = turbulence_capability_matrix()
    for family, lattices in matrix.items():
        for lattice, collisions in lattices.items():
            for collision, cap in collisions.items():
                assert not cap.available, (family, lattice, collision)
                assert cap.status.startswith("WITHHELD_"), (family, lattice, collision)


def test_contract_tested_does_not_imply_physics_validation() -> None:
    """Contract tests verify operator algebra, not turbulence physics."""
    cap = turbulence_capability_matrix()["smagorinsky"]["D2Q9"]["BGK"]
    assert cap.verification_level == VERIFICATION_CONTRACT_TESTED
    assert cap.status == WITHHELD_NO_PHYSICS_VALIDATION
    assert not cap.available


def test_benchmark_only_does_not_imply_physics_validation() -> None:
    cap = turbulence_capability_matrix()["wall_function"]["D3Q19"]["BGK"]
    assert cap.verification_level == VERIFICATION_BENCHMARK_ONLY
    assert cap.status == WITHHELD_NO_PHYSICS_VALIDATION
    assert not cap.available


def test_implemented_only_does_not_imply_physics_validation() -> None:
    cap = turbulence_capability_matrix()["rans_ke"]["D3Q19"]["MRT"]
    assert cap.verification_level == VERIFICATION_IMPLEMENTED_ONLY
    assert cap.status == WITHHELD_NO_CONTRACT_TESTS
    assert not cap.available


# ---------------------------------------------------------------------------
# require_turbulence_capability always raises (fail-closed)
# ---------------------------------------------------------------------------

def test_require_always_raises_for_contract_tested_combination() -> None:
    with pytest.raises(TurbulenceWithheldError, match="WITHHELD_NO_PHYSICS_VALIDATION"):
        require_turbulence_capability("smagorinsky", "D2Q9", "BGK")


def test_require_always_raises_for_implemented_only_combination() -> None:
    with pytest.raises(TurbulenceWithheldError, match="WITHHELD_NO_CONTRACT_TESTS"):
        require_turbulence_capability("rans_ke", "D3Q19", "MRT")


def test_require_always_raises_for_no_implementation() -> None:
    with pytest.raises(TurbulenceWithheldError, match="WITHHELD_NO_IMPLEMENTATION"):
        require_turbulence_capability("wale", "D2Q9", "MRT")


# ---------------------------------------------------------------------------
# Unknown inputs rejected with stable machine-readable codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("family", "lattice", "collision", "withheld_code"),
    [
        ("unknown_family", "D2Q9", "BGK", WITHHELD_UNKNOWN_FAMILY),
        ("smagorinsky", "D9Q99", "BGK", WITHHELD_UNKNOWN_LATTICE),
        ("smagorinsky", "D2Q9", "unknown_collision", WITHHELD_UNKNOWN_COLLISION),
    ],
)
def test_unknown_inputs_rejected_before_matrix_lookup(
    family: str, lattice: str, collision: str, withheld_code: str,
) -> None:
    with pytest.raises(TurbulenceWithheldError, match=withheld_code):
        require_turbulence_capability(family, lattice, collision)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hot-path allocation audit
# ---------------------------------------------------------------------------

def test_hot_path_audit_returns_nonempty_list() -> None:
    audit = turbulence_hot_path_audit()
    assert len(audit) > 0


def test_hot_path_audit_documents_dynamic_smagorinsky_item_sync() -> None:
    audit = turbulence_hot_path_audit()
    dyn_entries = [e for e in audit if "dynamic_smagorinsky" in e.function]
    assert len(dyn_entries) >= 3
    for entry in dyn_entries:
        assert "item()" in entry.pattern
        assert entry.severity == "SYNC"


def test_hot_path_audit_documents_rans_ke_mask_bool_allocation() -> None:
    audit = turbulence_hot_path_audit()
    ke_entries = [e for e in audit if "collide_rans_ke" in e.function]
    assert len(ke_entries) >= 1
    assert any("mask.bool()" in e.pattern for e in ke_entries)
    assert any(e.severity == "ALLOCATION" for e in ke_entries)


def test_hot_path_audit_documents_rans_sa_scalar_averaging() -> None:
    audit = turbulence_hot_path_audit()
    sa_entries = [e for e in audit if "collide_rans_sa" in e.function]
    assert len(sa_entries) >= 1
    assert any("mean().item()" in e.pattern for e in sa_entries)


def test_hot_path_audit_documents_wall_function_bool_sync() -> None:
    audit = turbulence_hot_path_audit()
    wf_entries = [e for e in audit if "wall_function_3d" in e.function]
    assert len(wf_entries) >= 1
    assert any("bool(" in e.pattern for e in wf_entries)
