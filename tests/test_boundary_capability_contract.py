"""Fail-closed contract tests for audited boundary-condition capabilities."""
from __future__ import annotations

import pytest

from tensorlbm.boundary_capability_contract import (
    BoundaryConditionCapability,
    BoundaryConditionWithheldError,
    WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE,
    WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT,
    WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE,
    WITHHELD_UNKNOWN_BACKEND,
    WITHHELD_UNKNOWN_BOUNDARY,
    WITHHELD_UNKNOWN_COLLISION,
    WITHHELD_UNKNOWN_LATTICE,
    WITHHELD_UNKNOWN_PHYSICS,
    boundary_capability_matrix,
    require_boundary_condition_capability,
)

_AUDITED_BOUNDARIES = (
    "periodic", "zou_he_inlet", "zou_he_outlet", "wall_bounce_back",
    "wall_free_slip", "farfield", "sponge", "nscbc", "bouzidi_interpolated",
)
_AUDITED_LATTICES = ("D2Q9", "D3Q19", "D3Q27")
_AUDITED_COLLISIONS = ("bgk", "mrt", "trt", "smagorinsky", "kbc", "cascaded")
_AUDITED_PHYSICS = ("single_phase", "turbulence", "multiphase", "free_surface", "ibm")
_AUDITED_BACKENDS = ("torch_cpu", "torch_cuda")


# ---------------------------------------------------------------------------
# Matrix structure
# ---------------------------------------------------------------------------

def test_matrix_covers_all_boundary_kinds_and_lattices() -> None:
    matrix = boundary_capability_matrix()

    assert set(matrix.keys()) == set(_AUDITED_BOUNDARIES)
    for kind in _AUDITED_BOUNDARIES:
        assert set(matrix[kind].keys()) == set(_AUDITED_LATTICES)
        for lattice in _AUDITED_LATTICES:
            cap = matrix[kind][lattice]
            assert isinstance(cap, BoundaryConditionCapability)
            assert cap.implementation_status in (
                "NO_IMPLEMENTATION",
                "IMPLEMENTATION_ONLY",
                "MECHANICS_TESTED",
                "PHYSICS_VALIDATED",
            )


def test_matrix_has_27_entries() -> None:
    """9 boundary kinds × 3 lattices = 27 audited cells."""
    matrix = boundary_capability_matrix()
    count = sum(len(matrix[kind]) for kind in matrix)
    assert count == 27


# ---------------------------------------------------------------------------
# Implementation status correctness (audited from source, not docstrings)
# ---------------------------------------------------------------------------

def test_periodic_has_mechanics_tests_but_no_physical_validation() -> None:
    matrix = boundary_capability_matrix()

    for lattice in _AUDITED_LATTICES:
        cap = matrix["periodic"][lattice]
        assert cap.implementation_status == "MECHANICS_TESTED"
        assert cap.entrypoint is not None
        assert "stream" in cap.entrypoint.lower()
        assert "mass conservation" in cap.verification_evidence.lower()


def test_zou_he_inlet_d2q9_is_implementation_only() -> None:
    """D2Q9 Zou-He inlet has no dedicated test; cylinder Cd test uses equilibrium inlet."""
    cap = boundary_capability_matrix()["zou_he_inlet"]["D2Q9"]
    assert cap.implementation_status == "IMPLEMENTATION_ONLY"
    assert cap.entrypoint is not None
    assert "zou_he_inlet_velocity" in cap.entrypoint


def test_zou_he_inlet_d3q27_has_mechanics_tests() -> None:
    """D3Q27 Zou-He inlet has velocity-prescription and finite-output tests."""
    cap = boundary_capability_matrix()["zou_he_inlet"]["D3Q27"]
    assert cap.implementation_status == "MECHANICS_TESTED"
    assert "zou_he_inlet_velocity_27" in cap.entrypoint


def test_zou_he_outlet_is_implementation_only_across_lattices() -> None:
    """No dedicated outlet-pressure unit test for any lattice."""
    matrix = boundary_capability_matrix()
    for lattice in ("D2Q9", "D3Q19", "D3Q27"):
        cap = matrix["zou_he_outlet"][lattice]
        assert cap.implementation_status == "IMPLEMENTATION_ONLY"


def test_wall_bounce_back_d2q9_and_d3q19_are_physics_validated() -> None:
    """Cylinder and sphere Cd tests provide (loose) physical validation."""
    matrix = boundary_capability_matrix()
    assert matrix["wall_bounce_back"]["D2Q9"].implementation_status == "PHYSICS_VALIDATED"
    assert matrix["wall_bounce_back"]["D3Q19"].implementation_status == "PHYSICS_VALIDATED"


def test_wall_bounce_back_d3q27_is_mechanics_tested_only() -> None:
    """D3Q27 bounce-back has shape/ME-force unit tests but no Cd validation."""
    cap = boundary_capability_matrix()["wall_bounce_back"]["D3Q27"]
    assert cap.implementation_status == "MECHANICS_TESTED"


def test_wall_free_slip_only_d3q19_implementation_exists() -> None:
    matrix = boundary_capability_matrix()
    assert matrix["wall_free_slip"]["D2Q9"].implementation_status == "NO_IMPLEMENTATION"
    assert matrix["wall_free_slip"]["D3Q19"].implementation_status == "IMPLEMENTATION_ONLY"
    assert matrix["wall_free_slip"]["D3Q27"].implementation_status == "NO_IMPLEMENTATION"


def test_farfield_is_implementation_only_and_docstring_not_trusted() -> None:
    """Farfield BC has no tests; docstring Cd-error claim is not validation evidence."""
    matrix = boundary_capability_matrix()
    for lattice in ("D2Q9", "D3Q19"):
        cap = matrix["farfield"][lattice]
        assert cap.implementation_status == "IMPLEMENTATION_ONLY"
        assert "not trusted" in cap.verification_evidence.lower() or "no test" in cap.verification_evidence.lower()
    assert matrix["farfield"]["D3Q27"].implementation_status == "NO_IMPLEMENTATION"


def test_sponge_has_mechanics_tests_but_no_physical_validation() -> None:
    matrix = boundary_capability_matrix()
    for lattice in _AUDITED_LATTICES:
        cap = matrix["sponge"][lattice]
        assert cap.implementation_status == "MECHANICS_TESTED"
        assert "no physical validation" in cap.verification_evidence.lower()


def test_nscbc_is_implementation_only_or_missing() -> None:
    matrix = boundary_capability_matrix()
    assert matrix["nscbc"]["D2Q9"].implementation_status == "IMPLEMENTATION_ONLY"
    assert matrix["nscbc"]["D3Q19"].implementation_status == "IMPLEMENTATION_ONLY"
    assert matrix["nscbc"]["D3Q27"].implementation_status == "NO_IMPLEMENTATION"


def test_bouzidi_d2q9_is_mechanics_tested() -> None:
    cap = boundary_capability_matrix()["bouzidi_interpolated"]["D2Q9"]
    assert cap.implementation_status == "MECHANICS_TESTED"
    assert "bouzidi_bounce_back" in cap.entrypoint


def test_bouzidi_d3q19_is_mechanics_tested_not_physics_validated() -> None:
    """sphere_bouzidi.py benchmark is NOT a test; only mechanics tests exist."""
    cap = boundary_capability_matrix()["bouzidi_interpolated"]["D3Q19"]
    assert cap.implementation_status == "MECHANICS_TESTED"
    assert "not a test" in cap.verification_evidence.lower() or "not trusted" in cap.verification_evidence.lower()


def test_bouzidi_d3q27_has_no_implementation() -> None:
    cap = boundary_capability_matrix()["bouzidi_interpolated"]["D3Q27"]
    assert cap.implementation_status == "NO_IMPLEMENTATION"


# ---------------------------------------------------------------------------
# Fail-closed: no combination is AVAILABLE
# ---------------------------------------------------------------------------

def test_no_combination_is_available() -> None:
    """Every (kind, lattice) cell must be withheld — fail-closed."""
    matrix = boundary_capability_matrix()
    for kind in _AUDITED_BOUNDARIES:
        for lattice in _AUDITED_LATTICES:
            cap = matrix[kind][lattice]
            assert not cap.available, f"{kind}/{lattice} is available but should be withheld"
            assert cap.status.startswith("WITHHELD_"), f"{kind}/{lattice} status={cap.status!r}"


def test_no_implementation_cells_carry_specific_withhold_code() -> None:
    matrix = boundary_capability_matrix()
    for kind in _AUDITED_BOUNDARIES:
        for lattice in _AUDITED_LATTICES:
            cap = matrix[kind][lattice]
            if cap.implementation_status == "NO_IMPLEMENTATION":
                assert cap.status == WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE


def test_implementation_cells_carry_composition_withhold_code() -> None:
    matrix = boundary_capability_matrix()
    for kind in _AUDITED_BOUNDARIES:
        for lattice in _AUDITED_LATTICES:
            cap = matrix[kind][lattice]
            if cap.implementation_status != "NO_IMPLEMENTATION":
                assert cap.status == WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE


# ---------------------------------------------------------------------------
# require_boundary_condition_capability — fail-closed dispatcher
# ---------------------------------------------------------------------------

def test_require_raises_for_implemented_single_phase_combination() -> None:
    """Even an implemented, mechanics-tested BC is withheld for complete composition."""
    with pytest.raises(BoundaryConditionWithheldError, match="WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE"):
        require_boundary_condition_capability(
            "wall_bounce_back", "D3Q19", "bgk", "single_phase", "torch_cpu",
        )


def test_require_raises_for_physics_validated_combination() -> None:
    """Physics-validated BCs are still withheld — validation is not a composition contract."""
    with pytest.raises(BoundaryConditionWithheldError, match="WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE"):
        require_boundary_condition_capability(
            "wall_bounce_back", "D2Q9", "bgk", "single_phase", "torch_cpu",
        )


def test_require_raises_for_no_implementation() -> None:
    with pytest.raises(BoundaryConditionWithheldError, match="WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE"):
        require_boundary_condition_capability(
            "bouzidi_interpolated", "D3Q27", "mrt", "single_phase", "torch_cpu",
        )


def test_require_raises_for_coupled_physics() -> None:
    """Non-single-phase physics has no audited BC coupling contract."""
    with pytest.raises(BoundaryConditionWithheldError, match="WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT"):
        require_boundary_condition_capability(
            "zou_he_inlet", "D3Q19", "mrt", "turbulence", "torch_cpu",
        )


@pytest.mark.parametrize(
    ("kind", "lattice", "collision", "physics", "backend", "withheld_code"),
    (
        ("unknown_bc", "D3Q19", "bgk", "single_phase", "torch_cpu", WITHHELD_UNKNOWN_BOUNDARY),
        ("periodic", "D9Q99", "bgk", "single_phase", "torch_cpu", WITHHELD_UNKNOWN_LATTICE),
        ("periodic", "D2Q9", "unknown_collision", "single_phase", "torch_cpu", WITHHELD_UNKNOWN_COLLISION),
        ("periodic", "D2Q9", "bgk", "unknown_physics", "torch_cpu", WITHHELD_UNKNOWN_PHYSICS),
        ("periodic", "D2Q9", "bgk", "single_phase", "unknown_backend", WITHHELD_UNKNOWN_BACKEND),
    ),
)
def test_require_rejects_unknown_inputs_before_matrix_lookup(
    kind: str, lattice: str, collision: str, physics: str, backend: str, withheld_code: str,
) -> None:
    with pytest.raises(BoundaryConditionWithheldError, match=withheld_code):
        require_boundary_condition_capability(kind, lattice, collision, physics, backend)


def test_require_returns_capability_for_no_implementation_before_physics_check() -> None:
    """A missing implementation is reported before the physics-coupling check."""
    with pytest.raises(BoundaryConditionWithheldError, match="WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE"):
        require_boundary_condition_capability(
            "farfield", "D3Q27", "bgk", "turbulence", "torch_cpu",
        )
