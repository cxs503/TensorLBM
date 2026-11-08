"""Tests for the physics capability contract (thermal / porous / non-Newtonian / scalar)."""

from __future__ import annotations

import pytest

from tensorlbm.physics_capability_contract import (
    IMPLEMENTED,
    NO_IMPLEMENTATION,
    VERIFICATION_CONTRACT_TESTED,
    WITHHELD_NO_PHYSICS_VALIDATION,
    PhysicsWithheldError,
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
            ("non_newtonian", "D3Q19", "BGK"),
            ("passive_scalar", "D3Q19", "BGK"),
            ("passive_scalar", "D3Q27", "BGK"),
        ],
    )
    def test_require_implemented(self, family, lattice, collision) -> None:
        cap = require_physics_capability(family, lattice, collision)
        assert cap.implementation_status == IMPLEMENTED
        assert cap.entrypoint is not None

    @pytest.mark.parametrize(
        ("family", "lattice", "collision"),
        [
            # 2026-11-08 (mission 6598): these four cells used to be declared
            # IMPLEMENTED/CONTRACT_TESTED against
            # ``tensorlbm.porous_media_common.porous_media_step`` and
            # ``tensorlbm.non_newtonian_common.non_newtonian_tau_eff`` — modules
            # and functions that never existed.  The contract now withholds.
            ("porous_media", "D3Q19", "BGK"),
            ("porous_media", "D3Q19", "MRT"),
            ("porous_media", "D3Q27", "BGK"),
            ("porous_media", "D3Q27", "MRT"),
            ("non_newtonian", "D3Q19", "MRT"),
            ("non_newtonian", "D3Q27", "BGK"),
            ("non_newtonian", "D3Q27", "MRT"),
        ],
    )
    def test_require_unimplemented_combinations_withhold(
        self, family, lattice, collision
    ) -> None:
        cap = require_physics_capability(family, lattice, collision)
        assert cap.implementation_status == NO_IMPLEMENTATION
        assert cap.entrypoint is None
        assert cap.status.startswith("WITHHELD")

    def test_require_unknown_family_raises(self) -> None:
        with pytest.raises(PhysicsWithheldError, match="family"):
            require_physics_capability("unknown", "D3Q19", "BGK")

    def test_require_unsupported_lattice_raises(self) -> None:
        with pytest.raises(PhysicsWithheldError, match="lattice"):
            require_physics_capability("thermal", "D2Q9", "BGK")

    def test_require_unsupported_collision_raises(self) -> None:
        with pytest.raises(PhysicsWithheldError, match="collision"):
            require_physics_capability("thermal", "D3Q19", "TRT")


class TestEvidencePointersAreReal:
    """The contract's strings must resolve to code and tests that exist.

    This is what the 2026-11-08 drift audit found broken: 9 of 45 cells named a
    module, a function and a test file that were never in the repository.
    """

    def test_implemented_entrypoints_and_evidence_resolve(self) -> None:
        from tensorlbm.capability_contract_drift import (
            _DOTTED_RE,
            resolve_dotted,
            resolve_test_evidence,
            tests_root,
        )

        root = tests_root()
        assert root is not None
        for cap in physics_capability_matrix():
            if cap.implementation_status != IMPLEMENTED:
                continue
            dotted = _DOTTED_RE.findall(cap.entrypoint or "")
            assert dotted, f"{cap.family}/{cap.lattice}/{cap.collision}: no entrypoint"
            for path in dotted:
                resolve_dotted(path)  # raises EntrypointNotFound if it rotted
            refs, missing = resolve_test_evidence(cap.test_evidence, root)
            assert refs and not missing, (
                f"{cap.family}/{cap.lattice}/{cap.collision}: "
                f"unresolved evidence {missing}"
            )
