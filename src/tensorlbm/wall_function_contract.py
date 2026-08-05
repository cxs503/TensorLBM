"""Fail-closed capability contract for existing wall-related implementations.

This is cold-path metadata and admission control only.  It does not alter any
wall numerical operator, and a capability being listed means only the stated
source-level combination is callable; it never infers physical validity.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum
from types import MappingProxyType
from typing import Mapping


class ValidationLevel(IntEnum):
    """Evidence levels, ordered so a requested minimum can fail closed."""

    IMPLEMENTATION_ONLY = 0
    NUMERICAL_REGRESSION = 1
    PHYSICAL_VALIDATION = 2


class WallFunctionCapability(str, Enum):
    """Separately auditable wall-related code paths; they are not interchangeable."""

    DISTANCE_FMM = "distance_fmm"
    LOG_LAW_BODY_FORCE = "log_law_body_force"
    MOVING_BOUNCE_BACK = "moving_bounce_back"
    ROUGH_WALL_SLIP = "rough_wall_slip"
    COMMON_WALL_FUNCTION = "common_wall_function"


WITHHELD_UNVERIFIED_COMBINATION = "WITHHELD_UNVERIFIED_COMBINATION"
WITHHELD_VALIDATION_LEVEL = "WITHHELD_VALIDATION_LEVEL"


class WallFunctionCompatibilityError(NotImplementedError):
    """Raised instead of silently treating an implementation as a general feature."""


@dataclass(frozen=True)
class WallFunctionCapabilityRecord:
    """Exact supported labels for one existing implementation path."""

    entrypoint: str
    lattices: frozenset[str]
    physics: frozenset[str]
    collisions: frozenset[str]
    geometries: frozenset[str]
    backends: frozenset[str]
    validation: ValidationLevel
    note: str


@dataclass(frozen=True)
class WallFunctionRequest:
    """A fully named proposed use; omitted dimensions are intentionally impossible."""

    capability: WallFunctionCapability
    lattice: str
    physics: str
    collision: str
    geometry: str
    backend: str


@dataclass(frozen=True)
class WallFunctionAssessment:
    compatible: bool
    validation: ValidationLevel | None
    status: str
    note: str


def wall_function_capability_matrix() -> Mapping[WallFunctionCapability, WallFunctionCapabilityRecord]:
    """Return the audited matrix without upgrading source claims to evidence.

    The log-law source docstring's ``<1%`` statement has no corresponding
    focused test or checked-in result dataset in this checkout.  It therefore
    remains implementation-only.  D3Q27 moving-wall tests cover a different
    linkwise routine and do not validate these D3Q19 wall-model wrappers.
    """
    return MappingProxyType({
        WallFunctionCapability.DISTANCE_FMM: WallFunctionCapabilityRecord(
            "tensorlbm.wall_model.compute_wall_distance_fmm",
            frozenset({"MASK_2D", "MASK_3D"}), frozenset({"mask_only"}),
            frozenset({"none"}), frozenset({"static_voxel_solid"}), frozenset({"torch"}),
            ValidationLevel.IMPLEMENTATION_ONLY,
            "Iterative 6-neighbour Manhattan-style Eikonal approximation; no lattice, collision, "
            "physics, AMR, or physical-distance validation is admitted.",
        ),
        WallFunctionCapability.LOG_LAW_BODY_FORCE: WallFunctionCapabilityRecord(
            "tensorlbm.wall_model.wall_function_3d",
            frozenset({"D3Q19"}), frozenset({"single_phase_incompressible"}),
            frozenset({"MRT_SMAGORINSKY"}), frozenset({"static_voxel_solid"}), frozenset({"torch"}),
            ValidationLevel.IMPLEMENTATION_ONLY,
            "Observed caller is the D3Q19 MRT+Smagorinsky static-SUBOFF path.  This is not physical "
            "validation: no checked-in artifact supports the source <1% claim; free-surface, D3Q27, "
            "other collisions, moving geometry, AMR, and other backends are withheld.",
        ),
        WallFunctionCapability.MOVING_BOUNCE_BACK: WallFunctionCapabilityRecord(
            "tensorlbm.wall_model.apply_wall_model_bounce_back",
            frozenset({"D3Q19"}), frozenset({"single_phase_incompressible"}),
            frozenset({"BGK", "MRT"}), frozenset({"static_voxel_solid", "moving_voxel_solid"}),
            frozenset({"torch"}), ValidationLevel.IMPLEMENTATION_ONLY,
            "D3Q19 cell-mask wrapper over moving_wall_bounce_back_3d.  It has no focused wrapper "
            "regression or physical validation; D3Q27 linkwise moving-wall tests are not evidence.",
        ),
        WallFunctionCapability.ROUGH_WALL_SLIP: WallFunctionCapabilityRecord(
            "tensorlbm.roughness.apply_rough_wall_bounce_back",
            frozenset({"D3Q19"}), frozenset({"single_phase_incompressible"}),
            frozenset({"BGK", "MRT"}), frozenset({"static_voxel_solid", "moving_voxel_solid"}),
            frozenset({"torch"}), ValidationLevel.IMPLEMENTATION_ONLY,
            "Equivalent-sand-grain slip wrapper shares the D3Q19 moving bounce-back path; existing "
            "tests only exercise imports and roughness-correction monotonicity.",
        ),
        WallFunctionCapability.COMMON_WALL_FUNCTION: WallFunctionCapabilityRecord(
            "tensorlbm.wall_function_common.wall_function",
            frozenset({"D3Q19", "D3Q27"}), frozenset({"single_phase_incompressible"}),
            frozenset({"BGK", "MRT", "CM", "KBC"}), frozenset({"static_voxel_solid"}),
            frozenset({"torch"}), ValidationLevel.IMPLEMENTATION_ONLY,
            "Solver-agnostic common wall-function module: wall_function(f, mask, u_tau, y_plus, ...) "
            "→ f_corrected.  Takes pre-computed u_tau/y_plus so it can combine with any "
            "collision/turbulence.  Supports D3Q19 and D3Q27 via lattice-dispatched body force. "
            "Contract tests verify shape, finiteness, near-wall-only correction, and mass conservation; "
            "no physical validation evidence is admitted.",
        ),
    })


def assess_wall_function(request: WallFunctionRequest) -> WallFunctionAssessment:
    """Assess one exact tuple.  Any unlisted label is withheld rather than guessed."""
    record = wall_function_capability_matrix()[request.capability]
    dimensions = (
        (request.lattice, record.lattices), (request.physics, record.physics),
        (request.collision, record.collisions), (request.geometry, record.geometries),
        (request.backend, record.backends),
    )
    if any(value not in allowed for value, allowed in dimensions):
        return WallFunctionAssessment(False, None, WITHHELD_UNVERIFIED_COMBINATION,
                                      f"{WITHHELD_UNVERIFIED_COMBINATION}: {request!r}")
    return WallFunctionAssessment(True, record.validation, "ADMITTED_IMPLEMENTATION", record.note)


def require_wall_function(
    request: WallFunctionRequest,
    *,
    minimum_validation: ValidationLevel = ValidationLevel.IMPLEMENTATION_ONLY,
) -> WallFunctionCapabilityRecord:
    """Return an admitted record or raise; no execution dispatch is provided."""
    assessment = assess_wall_function(request)
    if not assessment.compatible:
        raise WallFunctionCompatibilityError(assessment.note)
    assert assessment.validation is not None
    if assessment.validation < minimum_validation:
        raise WallFunctionCompatibilityError(
            f"{WITHHELD_VALIDATION_LEVEL}: requested {minimum_validation.name}, "
            f"available {assessment.validation.name}"
        )
    return wall_function_capability_matrix()[request.capability]


__all__ = [
    "WITHHELD_UNVERIFIED_COMBINATION", "WITHHELD_VALIDATION_LEVEL", "ValidationLevel",
    "WallFunctionAssessment", "WallFunctionCapability", "WallFunctionCapabilityRecord",
    "WallFunctionCompatibilityError", "WallFunctionRequest", "assess_wall_function",
    "require_wall_function", "wall_function_capability_matrix",
]
