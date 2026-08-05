"""Cold-path model composition contracts; intentionally backend-independent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class PhysicsCapability(str, Enum):
    """Named physics families available to a model composition."""

    SINGLE_PHASE = "single_phase"
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    FREE_SURFACE = "free_surface"
    PHASE_FIELD = "phase_field"
    FSI = "fsi"


class ComparisonClass(str, Enum):
    """Permitted composition-comparison outcomes."""

    IDENTICAL_COMPOSITION = "identical_composition"
    SAME_FORMULATION = "same_formulation"
    CROSS_MODEL = "cross_model"
    FORBIDDEN = "forbidden"


def _nonempty_string(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of non-empty strings")
    return tuple(_nonempty_string(item, name) for item in value)


def _module_mapping(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("physics_modules must be a mapping of non-empty strings")
    normalized = {
        _nonempty_string(key, "physics_modules"): _nonempty_string(item, "physics_modules")
        for key, item in value.items()
    }
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class ModelComposition:
    """Validated, immutable metadata describing a numerical model composition."""

    lattice: str
    collision: str
    turbulence: str | None
    forcing: tuple[str, ...]
    boundaries: tuple[str, ...]
    physics_modules: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lattice", _nonempty_string(self.lattice, "lattice"))
        object.__setattr__(self, "collision", _nonempty_string(self.collision, "collision"))
        if self.turbulence is not None:
            object.__setattr__(self, "turbulence", _nonempty_string(self.turbulence, "turbulence"))
        object.__setattr__(self, "forcing", _string_tuple(self.forcing, "forcing"))
        object.__setattr__(self, "boundaries", _string_tuple(self.boundaries, "boundaries"))
        object.__setattr__(self, "physics_modules", _module_mapping(self.physics_modules))


class CompatibilityGate:
    """Classify numerical-composition comparisons without physical-validation claims."""

    @staticmethod
    def classify(left: ModelComposition, right: ModelComposition) -> ComparisonClass:
        if _phase_formulations(left) != _phase_formulations(right):
            return ComparisonClass.CROSS_MODEL
        if left == right:
            return ComparisonClass.IDENTICAL_COMPOSITION
        return ComparisonClass.SAME_FORMULATION


def _phase_formulations(composition: ModelComposition) -> tuple[tuple[str, str], ...]:
    return tuple(
        (capability, formulation)
        for capability in (PhysicsCapability.FREE_SURFACE.value, PhysicsCapability.PHASE_FIELD.value)
        if (formulation := composition.physics_modules.get(capability)) is not None
    )
