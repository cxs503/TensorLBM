"""Tests for the physics capability contract (thermal / porous / non-Newtonian / scalar)."""
from __future__ import annotations

import pytest

from tensorlbm.physics_capability_contract import (
    IMPLEMENTED,
    NO_IMPLEMENTATION,
    PhysicsCapability,
    PhysicsWithheldError,
    VERIFICATION_CONTRACT_TESTED,
    WITHHELD_NO_IMPLEMENTATION,
    WITHHELD_NO_PHYSICS_VALIDATION,
    physics_capability_matrix,
    require_physics_capability,
)


class TestCapabilityMatrix:
    def test_matrix_is_non_empty(self) -> None:
        matrix = physics_capability_matrix()
        assert len(matrix) > 0

    def test_all_audited_families_present(self) -> None:
        matrix = physics_capability_matrix()
        families = {cap.family for cap in matrix}
        expected = {"thermal", "conjugate_ht", "porous_media", "non_newtonian", "passive_scalar"}
        assert families == expected

    def test_all_entries_are_withheld(self) -> None:
        """No combination should be certified as physics-validated."""
        matrix = physics_capability_matrix()
        for cap in matrix:
            if cap.implementation_status == IMPLEMENTED:
                assert cap.status == WITHHELD_NO_PHYSICS_VALIDATION, (
                    f"{cap.family}/{cap.lattice}/{cap.collision}: "
                    f"status={cap.status} should be WITHHELD"
                )

    def test_implemented_entries_have_contract_tests(self) -> None:
        """All implemented entries should have CONTRACT_TESTED verification."""
        matrix = physics_capability_matrix()
        for cap in matrix:
            if cap.implementation_status == IMPLEMENTED:
                assert cap.verification_level == VERIFICATION_CONTRACT_TESTED, (
                    f"{cap.family}/{cap.lattice}/{cap.collision}: "
                    f"verification={cap.verification_level}"
                )

    def test_implemented_entries_have_entrypoint(self) -> None:
        matrix = physics_capability_matrix()
        for cap in matrix:
            if cap.implementation_status == IMPLEMENTED:
                assert cap.entrypoint is not None
                assert cap.test_evidence is not None

    def test_unimplemented_combinations_exist(self) -> None:
        """Some combinations should be NO_IMPLEMENTATION (e.g. D2Q9 thermal)."""
        matrix = physics_capability_matrix()
        no_impl = [cap for cap in matrix if cap.implementation_status == NO_IMPLEMENTATION]
        assert len(no_impl) > 0


class TestRequireCapability:
    @pytest.mark.parametrize(
        ("family", "lattice", "collision"),
        [
            ("thermal", "D3Q19", "BGK"),
            ("thermal", "D3Q27", "BGK"),
            ("conjugate_ht", "D3Q19", "N/A"),
            ("porous_media", "D3Q19", "BGK"),
            ("porous_media", "D3Q27", "BGK"),
            ("non_newtonian", "D3Q19", "BGK"),
            ("non_newtonian", "D3Q27", "BGK"),
            ("passive_scalar", "D3Q19", "BGK"),
            ("passive_scalar", "D3Q27", "BGK"),
        ],
    )
    def test_require_implemented(self, family, lattice, collision) -> None:
        cap = require_physics_capability(family, lattice, collision)
        assert cap.implementation_status == IMPLEMENTED
        assert cap.entrypoint is not None

    def test_require_unknown_family_raises(self) -> None:
        with pytest.raises(PhysicsWithheldError, match="family"):
            require_physics_capability("unknown", "D3Q19", "BGK")

    def test_require_unsupported_lattice_raises(self) -> None:
        with pytest.raises(PhysicsWithheldError, match="lattice"):
            require_physics_capability("thermal", "D2Q9", "BGK")

    def test_require_unsupported_collision_raises(self) -> None:
        with pytest.raises(PhysicsWithheldError, match="collision"):
            require_physics_capability("thermal", "D3Q19", "TRT")
