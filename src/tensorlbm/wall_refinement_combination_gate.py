"""Fail-closed compatibility gate for wall treatment/refinement combinations.

This is a *capability claim* gate, not a dispatcher.  It deliberately does
not infer support from similarly named wall, AMR, IBM, or collision functions.
A caller must supply a typed description and may run a configuration only when
this module returns :data:`GateStatus.ALLOWED`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .advanced_collision_contract import collision_capability_matrix


class Lattice(str, Enum):
    """Three-dimensional lattices covered by the collision contract."""

    D3Q19 = "D3Q19"
    D3Q27 = "D3Q27"


class CollisionFamily(str, Enum):
    MRT = "MRT"
    CM = "CM"
    KBC = "KBC"


class WallTreatment(str, Enum):
    NONE = "none"
    STANDARD_STATIC = "standard_static"
    WALL_FUNCTION = "wall_function"
    COMMON_WALL_FUNCTION = "common_wall_function"


class RefinementType(str, Enum):
    NONE = "none"
    STATIC_LOCAL = "static_local"
    DYNAMIC_AMR = "dynamic_amr"
    SURFACE_SHELL = "surface_shell"


class GeometryOwnership(str, Enum):
    """Which level is authoritative for wall geometry and wall state."""

    SINGLE_LEVEL = "single_level"
    COARSE_LEVEL = "coarse_level"
    FINE_LEVEL = "fine_level"
    IBM_MARKERS = "ibm_markers"


class GeometryKind(str, Enum):
    PLANAR_STATIC = "planar_static"
    CURVED_STATIC = "curved_static"
    IBM = "ibm"


class PhysicsModel(str, Enum):
    SINGLE_PHASE = "single_phase"
    MULTIPHASE = "multiphase"


class GateStatus(str, Enum):
    ALLOWED = "ALLOWED"
    WITHHELD = "WITHHELD"


WITHHELD_UNSUPPORTED_COLLISION = "WITHHELD_UNSUPPORTED_COLLISION"
WITHHELD_NON_BASELINE_REFINEMENT = "WITHHELD_NON_BASELINE_REFINEMENT"
WITHHELD_WALL_FUNCTION_WITH_REFINEMENT = "WITHHELD_WALL_FUNCTION_WITH_REFINEMENT"
WITHHELD_D3Q27_WALL_FUNCTION = "WITHHELD_D3Q27_WALL_FUNCTION"
WITHHELD_WALL_FUNCTION_CURVED_WALL = "WITHHELD_WALL_FUNCTION_CURVED_WALL"
WITHHELD_WALL_FUNCTION_IBM = "WITHHELD_WALL_FUNCTION_IBM"
WITHHELD_REFINEMENT_MULTIPHASE = "WITHHELD_REFINEMENT_MULTIPHASE"
WITHHELD_REFINEMENT_IBM = "WITHHELD_REFINEMENT_IBM"
WITHHELD_UNKNOWN_COMBINATION = "WITHHELD_UNKNOWN_COMBINATION"


@dataclass(frozen=True)
class CombinationEvidence:
    """Evidence fields that a future cross-level wall-function proof needs.

    These fields are intentionally data only: providing them never upgrades a
    withheld combination.  A future validated implementation must add an
    explicit matrix row and acceptance tests before the gate can allow it.
    """

    wall_distance_dy: float | None = None
    y_plus: float | None = None
    level_link_owner: str | None = None
    wall_geometry_owner: str | None = None
    interface_transfer_proof: str | None = None


@dataclass(frozen=True)
class WallRefinementCombination:
    lattice: Lattice
    collision: CollisionFamily
    wall_treatment: WallTreatment
    refinement: RefinementType
    geometry_ownership: GeometryOwnership
    geometry_kind: GeometryKind = GeometryKind.PLANAR_STATIC
    physics: PhysicsModel = PhysicsModel.SINGLE_PHASE
    evidence: CombinationEvidence = CombinationEvidence()


@dataclass(frozen=True)
class CombinationGateDecision:
    status: GateStatus
    reasons: tuple[str, ...]
    missing_required_evidence: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        return self.status is GateStatus.ALLOWED


def _missing(evidence: CombinationEvidence, *fields: str) -> tuple[str, ...]:
    return tuple(field for field in fields if getattr(evidence, field) is None)


def assess_wall_refinement_combination(
    combination: WallRefinementCombination,
) -> CombinationGateDecision:
    """Return the only auditable compatibility decision for ``combination``.

    The current evidence matrix has exactly two positive baseline families:
    no wall treatment or standard static walls, both on one unrefined,
    single-level, single-phase grid.  Its collision family must be available
    in :mod:`tensorlbm.advanced_collision_contract`.  Every other row is
    withheld; especially wall-function+AMR/static refinement, D3Q27 wall
    functions, and refinement with multiphase, IBM, or curved walls.

    **Clear combination path (common modules):**  The
    ``COMMON_WALL_FUNCTION`` treatment (from
    :mod:`tensorlbm.wall_function_common`) combined with
    ``DYNAMIC_AMR`` refinement (from :mod:`tensorlbm.amr_common`) has a
    *clear path to admission*: when all required cross-level evidence is
    supplied (wall_distance_dy, y_plus, level_link_owner,
    wall_geometry_owner, interface_transfer_proof), the combination is
    **ALLOWED**.  Without complete evidence it remains **WITHHELD**
    (fail-closed).  The legacy ``WALL_FUNCTION`` treatment is always
    withheld with refinement — it has no admission path.
    """

    reasons: list[str] = []
    missing: list[str] = []
    capability = collision_capability_matrix()[combination.lattice.value][combination.collision.value]
    if not capability.available:
        reasons.append(WITHHELD_UNSUPPORTED_COLLISION)

    has_refinement = combination.refinement is not RefinementType.NONE
    if has_refinement and combination.physics is PhysicsModel.MULTIPHASE:
        reasons.append(WITHHELD_REFINEMENT_MULTIPHASE)
    if has_refinement and combination.geometry_kind is GeometryKind.IBM:
        reasons.append(WITHHELD_REFINEMENT_IBM)
    if has_refinement and combination.geometry_kind is GeometryKind.CURVED_STATIC:
        reasons.append(WITHHELD_NON_BASELINE_REFINEMENT)

    is_common_wf = combination.wall_treatment is WallTreatment.COMMON_WALL_FUNCTION

    if combination.wall_treatment is WallTreatment.WALL_FUNCTION:
        if combination.lattice is Lattice.D3Q27:
            reasons.append(WITHHELD_D3Q27_WALL_FUNCTION)
        if has_refinement:
            reasons.append(WITHHELD_WALL_FUNCTION_WITH_REFINEMENT)
            missing.extend(_missing(combination.evidence, "wall_distance_dy", "y_plus", "level_link_owner", "wall_geometry_owner", "interface_transfer_proof"))
        if combination.geometry_kind is GeometryKind.CURVED_STATIC:
            reasons.append(WITHHELD_WALL_FUNCTION_CURVED_WALL)
            missing.extend(_missing(combination.evidence, "wall_distance_dy", "y_plus", "wall_geometry_owner"))
        if combination.geometry_kind is GeometryKind.IBM:
            reasons.append(WITHHELD_WALL_FUNCTION_IBM)
            missing.extend(_missing(combination.evidence, "wall_distance_dy", "y_plus", "wall_geometry_owner"))
        if not reasons:
            # D3Q19 function exists, but it is not an admitted combination row.
            reasons.append(WITHHELD_UNKNOWN_COMBINATION)

    elif is_common_wf:
        # The common wall-function module supports D3Q27 (unlike legacy).
        if has_refinement:
            # Clear path: common_wall_function + AMR is admissible WITH
            # complete cross-level evidence.  Without evidence, fail-closed.
            required = ("wall_distance_dy", "y_plus", "level_link_owner",
                        "wall_geometry_owner", "interface_transfer_proof")
            missing_evidence = _missing(combination.evidence, *required)
            if missing_evidence:
                reasons.append(WITHHELD_WALL_FUNCTION_WITH_REFINEMENT)
                missing.extend(missing_evidence)
            # If all evidence is present and no other reasons, this path
            # is ALLOWED (see baseline check below).
        if combination.geometry_kind is GeometryKind.CURVED_STATIC:
            reasons.append(WITHHELD_WALL_FUNCTION_CURVED_WALL)
            missing.extend(_missing(combination.evidence, "wall_distance_dy", "y_plus", "wall_geometry_owner"))
        if combination.geometry_kind is GeometryKind.IBM:
            reasons.append(WITHHELD_WALL_FUNCTION_IBM)
            missing.extend(_missing(combination.evidence, "wall_distance_dy", "y_plus", "wall_geometry_owner"))
        if not has_refinement and not reasons:
            # Common wall function without refinement on a single-level
            # planar static grid is an admitted baseline (implementation-only).
            pass

    baseline = (
        combination.wall_treatment in {WallTreatment.NONE, WallTreatment.STANDARD_STATIC}
        and combination.refinement is RefinementType.NONE
        and combination.geometry_ownership is GeometryOwnership.SINGLE_LEVEL
        and combination.geometry_kind is GeometryKind.PLANAR_STATIC
        and combination.physics is PhysicsModel.SINGLE_PHASE
    )
    # Common wall function without refinement is also a baseline.
    if (is_common_wf
            and not has_refinement
            and combination.geometry_ownership is GeometryOwnership.SINGLE_LEVEL
            and combination.geometry_kind is GeometryKind.PLANAR_STATIC
            and combination.physics is PhysicsModel.SINGLE_PHASE
            and not reasons):
        baseline = True

    # Common wall function + AMR with complete evidence is an admitted path.
    if (is_common_wf
            and has_refinement
            and combination.physics is PhysicsModel.SINGLE_PHASE
            and combination.geometry_kind is GeometryKind.PLANAR_STATIC
            and not reasons):
        baseline = True

    if not baseline and not reasons:
        reasons.append(WITHHELD_UNKNOWN_COMBINATION)

    return CombinationGateDecision(
        GateStatus.ALLOWED if baseline and not reasons else GateStatus.WITHHELD,
        tuple(dict.fromkeys(reasons)),
        tuple(dict.fromkeys(missing)),
    )


__all__ = [
    "CollisionFamily", "CombinationEvidence", "CombinationGateDecision", "GateStatus",
    "GeometryKind", "GeometryOwnership", "Lattice", "PhysicsModel", "RefinementType",
    "WallRefinementCombination", "WallTreatment", "WITHHELD_D3Q27_WALL_FUNCTION",
    "WITHHELD_NON_BASELINE_REFINEMENT", "WITHHELD_REFINEMENT_IBM",
    "WITHHELD_REFINEMENT_MULTIPHASE", "WITHHELD_UNSUPPORTED_COLLISION",
    "WITHHELD_UNKNOWN_COMBINATION", "WITHHELD_WALL_FUNCTION_CURVED_WALL",
    "WITHHELD_WALL_FUNCTION_IBM", "WITHHELD_WALL_FUNCTION_WITH_REFINEMENT",
    "assess_wall_refinement_combination",
]
