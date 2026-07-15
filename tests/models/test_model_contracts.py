from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tensorlbm.models.contracts import (
    CompatibilityGate,
    ComparisonClass,
    ModelComposition,
    PhysicsCapability,
)


def _korner(*, collision: str = "MRT") -> ModelComposition:
    return ModelComposition(
        lattice="D3Q19",
        collision=collision,
        turbulence=None,
        forcing=(),
        boundaries=("halfway_bounce_back",),
        physics_modules={"free_surface": "Körner"},
    )


def test_physics_capability_declares_required_families():
    assert {
        PhysicsCapability.SINGLE_PHASE,
        PhysicsCapability.THERMAL,
        PhysicsCapability.ACOUSTIC,
        PhysicsCapability.FREE_SURFACE,
        PhysicsCapability.PHASE_FIELD,
        PhysicsCapability.FSI,
    } <= set(PhysicsCapability)


def test_composition_is_immutable_and_rejects_empty_or_boolean_values():
    composition = _korner()

    with pytest.raises(FrozenInstanceError):
        composition.lattice = "D3Q27"  # type: ignore[misc]
    with pytest.raises(ValueError, match="lattice"):
        ModelComposition("", "MRT", None, (), (), {})
    with pytest.raises(ValueError, match="collision"):
        ModelComposition("D3Q19", "", None, (), (), {})
    with pytest.raises(ValueError, match="lattice"):
        ModelComposition(True, "MRT", None, (), (), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="turbulence"):
        ModelComposition("D3Q19", "MRT", True, (), (), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forcing"):
        ModelComposition("D3Q19", "MRT", None, (True,), (), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="physics_modules"):
        ModelComposition("D3Q19", "MRT", None, (), (), {"single_phase": True})  # type: ignore[dict-item]


def test_d3q19_korner_and_d3q27_phase_field_are_cross_model():
    korner = _korner()
    phase_field = ModelComposition(
        lattice="D3Q27",
        collision="MRT",
        turbulence=None,
        forcing=(),
        boundaries=("halfway_bounce_back",),
        physics_modules={"phase_field": "free_energy"},
    )

    assert CompatibilityGate.classify(korner, phase_field) is ComparisonClass.CROSS_MODEL


def test_same_korner_with_different_collision_is_same_formulation():
    assert CompatibilityGate.classify(_korner(collision="MRT"), _korner(collision="BGK")) is ComparisonClass.SAME_FORMULATION


def test_identical_all_fields_is_identical_composition():
    left = _korner()
    right = _korner()

    assert CompatibilityGate.classify(left, right) is ComparisonClass.IDENTICAL_COMPOSITION


def test_distinct_phase_formulations_are_cross_model():
    left = ModelComposition("D3Q19", "MRT", None, (), (), {"free_surface": "Körner"})
    right = ModelComposition("D3Q19", "MRT", None, (), (), {"free_surface": "VOF"})

    assert CompatibilityGate.classify(left, right) is ComparisonClass.CROSS_MODEL


def test_any_phase_formulation_difference_is_cross_model():
    left = ModelComposition(
        "D3Q19",
        "MRT",
        None,
        (),
        (),
        {"free_surface": "Körner", "phase_field": "free_energy"},
    )
    right = ModelComposition(
        "D3Q19",
        "MRT",
        None,
        (),
        (),
        {"free_surface": "Körner", "phase_field": "cahn_hilliard"},
    )

    assert CompatibilityGate.classify(left, right) is ComparisonClass.CROSS_MODEL


def test_contract_module_does_not_offer_physical_equivalence_api():
    assert not hasattr(CompatibilityGate, "physical_equivalence")
    assert ComparisonClass.FORBIDDEN.value == "forbidden"
