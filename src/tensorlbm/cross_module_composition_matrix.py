"""Unified cross-module composition admission matrix.

This module is a cold-path aggregation gate that cross-queries every merged
single-dimension capability contract and aggregates their decisions into a
single fail-closed composition decision.

Aggregation rules:
  - All dimensions ADMITTED (or NOT_APPLICABLE) → SUPPORTED
  - Any dimension WITHHELD (and none NOT_SUPPORTED) → WITHHELD
  - Any dimension NOT_SUPPORTED → NOT_SUPPORTED

This module never modifies any numerical kernel or solver.  It only reads
sub-contract matrices and translates vocabulary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from .advanced_collision_contract import collision_capability_matrix
from .general_capability_matrix import (
    CapabilityRequest as _GMCRequest,
    CapabilityStatus as _GMCStatus,
    assess_capability as _assess_gmc,
)
from .wall_function_contract import (
    WallFunctionCapability,
    WallFunctionCompatibilityError,
    WallFunctionRequest,
    assess_wall_function,
)
from .wall_function_admission import (
    WallFunctionRunRequest,
    require_wall_function_run,
)
from .wall_refinement_combination_gate import (
    CollisionFamily as WRCollisionFamily,
    CombinationEvidence,
    GateStatus,
    GeometryKind,
    GeometryOwnership,
    Lattice as WRLattice,
    PhysicsModel,
    RefinementType,
    WallRefinementCombination,
    WallTreatment,
    assess_wall_refinement_combination,
)
from .amr_capability_contract import (
    REQUIRED_FRONTEND_METADATA,
    local_refinement_capability_matrix,
)
from .boundary_capability_contract import (
    boundary_capability_matrix,
)
from .turbulence_capability_contract import (
    turbulence_capability_matrix,
)
from .accuracy_recommendation import (
    PhysicalAccuracyEvidence,
    recommend_by_physical_accuracy,
)

MATRIX_VERSION = "cross-module-composition-matrix-r1"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompositionStatus(str, Enum):
    """Overall disposition of a complete cross-module composition request."""

    SUPPORTED = "supported"
    WITHHELD = "withheld"
    NOT_SUPPORTED = "not_supported"


class SubContractStatus(str, Enum):
    """Status returned by one sub-contract for one dimension."""

    ADMITTED = "admitted"
    WITHHELD = "withheld"
    NOT_SUPPORTED = "not_supported"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionRequest:
    """Typed cross-module composition candidate.

    Fields mirror :class:`general_capability_matrix.CapabilityRequest` so the
    general matrix can be queried without translation.
    """

    lattice: str
    collision: str
    turbulence: str = "none"
    multiphase: str = "single_phase"
    boundary: str = "static_wall"
    geometry: str = "static_solid_mask"
    wall_treatment: str = "bounce_back"
    refinement: str = "none"
    backend: str = "torch"
    outputs: tuple[str, ...] = ("rho", "velocity")


@dataclass(frozen=True)
class SubContractResult:
    """One sub-contract's assessment of one dimension."""

    contract_name: str
    dimension: str
    status: SubContractStatus
    reason_codes: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class CompositionDecision:
    """Aggregated decision across all sub-contracts."""

    status: CompositionStatus
    sub_contract_results: tuple[SubContractResult, ...]
    missing_dimensions: tuple[str, ...]
    reason_codes: tuple[str, ...]
    normalized_request: CompositionRequest

    @property
    def supported(self) -> bool:
        return self.status is CompositionStatus.SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without framework dependencies."""
        return {
            "status": self.status.value,
            "sub_contract_results": [
                {
                    "contract_name": r.contract_name,
                    "dimension": r.dimension,
                    "status": r.status.value,
                    "reason_codes": list(r.reason_codes),
                    "missing_evidence": list(r.missing_evidence),
                    "note": r.note,
                }
                for r in self.sub_contract_results
            ],
            "missing_dimensions": list(self.missing_dimensions),
            "reason_codes": list(self.reason_codes),
            "normalized_request": {
                **asdict(self.normalized_request),
                "outputs": list(self.normalized_request.outputs),
            },
        }


# ---------------------------------------------------------------------------
# Vocabulary aliases and known values (mirrors general_capability_matrix R1)
# ---------------------------------------------------------------------------

_ALIASES: dict[str, dict[str, str]] = {
    "lattice": {"d3q19": "d3q19", "d3q27": "d3q27"},
    "collision": {"mrt": "mrt", "cm": "cm", "cascaded": "cm", "kbc": "kbc", "entropic_kbc": "kbc"},
    "turbulence": {"none": "none", "laminar": "none", "smagorinsky": "smagorinsky", "les": "smagorinsky"},
    "multiphase": {"single_phase": "single_phase", "single-phase": "single_phase", "free_surface": "free_surface", "phase_field": "phase_field"},
    "boundary": {"static_wall": "static_wall", "static-wall": "static_wall", "velocity_inlet": "velocity_inlet", "pressure_outlet": "pressure_outlet", "periodic": "periodic"},
    "geometry": {"static_solid_mask": "static_solid_mask", "static-mask": "static_solid_mask", "voxel_mask": "static_solid_mask", "immersed_boundary": "immersed_boundary", "ibm": "immersed_boundary", "dynamic_geometry": "dynamic_geometry"},
    "wall_treatment": {"bounce_back": "bounce_back", "bounce-back": "bounce_back", "wall_function": "wall_function", "bouzidi": "bouzidi"},
    "refinement": {"none": "none", "no_amr": "none", "no-amr": "none", "amr": "amr"},
    "backend": {"torch": "torch", "pytorch": "torch", "cuda": "cuda", "cpu": "cpu"},
}

_KNOWN_VALUES: dict[str, frozenset[str]] = {
    "lattice": frozenset(("d3q19", "d3q27")),
    "collision": frozenset(("mrt", "cm", "kbc")),
    "turbulence": frozenset(("none", "smagorinsky")),
    "multiphase": frozenset(("single_phase", "free_surface", "phase_field")),
    "boundary": frozenset(("static_wall", "velocity_inlet", "pressure_outlet", "periodic")),
    "geometry": frozenset(("static_solid_mask", "immersed_boundary", "dynamic_geometry")),
    "wall_treatment": frozenset(("bounce_back", "wall_function", "bouzidi")),
    "refinement": frozenset(("none", "amr")),
    "backend": frozenset(("torch", "cuda", "cpu")),
    "outputs": frozenset(("rho", "velocity", "pressure", "vorticity", "force")),
}

_SCALAR_FIELDS = tuple(_ALIASES)


# ---------------------------------------------------------------------------
# Cross-contract vocabulary mappings
# ---------------------------------------------------------------------------

# Lattice: unified lowercase → sub-contract uppercase
_LATTICE_UPPER = {"d3q19": "D3Q19", "d3q27": "D3Q27"}

# Collision: unified lowercase → collision contract uppercase
_COLLISION_UPPER = {"mrt": "MRT", "cm": "CM", "kbc": "KBC"}

# Collision: unified lowercase → boundary contract lowercase
_COLLISION_TO_BOUNDARY = {"mrt": "mrt", "cm": "cascaded", "kbc": "kbc"}

# Collision: unified lowercase → turbulence contract uppercase
_COLLISION_TO_TURBULENCE = {"mrt": "MRT"}

# Boundary: unified → boundary contract kind
_BOUNDARY_TO_KIND = {
    "static_wall": "wall_bounce_back",
    "velocity_inlet": "zou_he_inlet",
    "pressure_outlet": "zou_he_outlet",
    "periodic": "periodic",
}

# Multiphase: unified → boundary contract physics
_MULTIPHASE_TO_BOUNDARY_PHYSICS = {
    "single_phase": "single_phase",
    "free_surface": "free_surface",
    "phase_field": "multiphase",
}

# Multiphase: unified → AMR contract physics
_MULTIPHASE_TO_AMR_PHYSICS = {
    "single_phase": "single_phase",
    "free_surface": "multiphase",
    "phase_field": "multiphase",
}

# Multiphase: unified → wall function contract physics
_MULTIPHASE_TO_WALL_FN_PHYSICS = {
    "single_phase": "single_phase_incompressible",
    "free_surface": "free_surface",
    "phase_field": "multiphase",
}

# Geometry: unified → wall function contract geometry
_GEOMETRY_TO_WALL_FN = {
    "static_solid_mask": "static_voxel_solid",
    "dynamic_geometry": "moving_voxel_solid",
}

# Geometry: unified → wall refinement gate GeometryKind
_GEOMETRY_TO_WR_KIND = {
    "static_solid_mask": GeometryKind.PLANAR_STATIC,
    "immersed_boundary": GeometryKind.IBM,
    "dynamic_geometry": GeometryKind.CURVED_STATIC,
}

# Wall treatment: unified → wall refinement gate WallTreatment
_WALL_TREATMENT_TO_WR = {
    "bounce_back": WallTreatment.STANDARD_STATIC,
    "wall_function": WallTreatment.WALL_FUNCTION,
}

# Refinement: unified → wall refinement gate RefinementType
_REFINEMENT_TO_WR = {
    "none": RefinementType.NONE,
    "amr": RefinementType.DYNAMIC_AMR,
}

# Refinement: unified → AMR contract path
_REFINEMENT_TO_AMR_PATH = {"amr": "adaptive_dynamic"}

# Multiphase: unified → wall refinement gate PhysicsModel
_MULTIPHASE_TO_WR_PHYSICS = {
    "single_phase": PhysicsModel.SINGLE_PHASE,
    "free_surface": PhysicsModel.MULTIPHASE,
    "phase_field": PhysicsModel.MULTIPHASE,
}

# Backend: unified → boundary contract backend
_BACKEND_TO_BOUNDARY = {
    "torch": "torch_cpu",
    "cuda": "torch_cuda",
    "cpu": "torch_cpu",
}

# Turbulence: unified → turbulence contract family
_TURBULENCE_TO_FAMILY = {"smagorinsky": "smagorinsky"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalise_value(field: str, value: object) -> str:
    if not isinstance(value, str):
        return ""
    compact = value.strip().lower().replace(" ", "_")
    return _ALIASES[field].get(compact, compact)


def _coerce_request(candidate: CompositionRequest | Mapping[str, Any]) -> CompositionRequest:
    if isinstance(candidate, CompositionRequest):
        raw = asdict(candidate)
    elif isinstance(candidate, Mapping):
        allowed = set(CompositionRequest.__dataclass_fields__)
        unknown = set(candidate) - allowed
        if unknown:
            raise ValueError("unknown capability request fields: " + ", ".join(sorted(unknown)))
        raw = dict(candidate)
    else:
        raise TypeError("candidate must be CompositionRequest or a mapping")
    outputs = raw.get("outputs", ("rho", "velocity"))
    if isinstance(outputs, str) or not isinstance(outputs, Sequence) or not all(isinstance(item, str) for item in outputs):
        raise ValueError("outputs must be a sequence of strings, not a single string")
    defaults = asdict(CompositionRequest(lattice="d3q19", collision="mrt"))
    return CompositionRequest(
        **{field: _normalise_value(field, raw.get(field, defaults[field]))
           for field in _SCALAR_FIELDS},
        outputs=tuple(sorted({item.strip().lower() for item in outputs})),
    )


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sub-contract query functions
# ---------------------------------------------------------------------------

def _query_collision_contract(req: CompositionRequest) -> SubContractResult:
    """Query advanced_collision_contract for lattice × collision."""
    lattice = _LATTICE_UPPER.get(req.lattice)
    collision = _COLLISION_UPPER.get(req.collision)
    if lattice is None:
        return SubContractResult("advanced_collision_contract", "collision",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{req.lattice!r} is not in the collision contract lattice vocabulary.")
    if collision is None:
        return SubContractResult("advanced_collision_contract", "collision",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_COLLISION",), (),
            f"{req.collision!r} is not in the collision contract family vocabulary.")
    matrix = collision_capability_matrix()
    if lattice not in matrix or collision not in matrix[lattice]:
        return SubContractResult("advanced_collision_contract", "collision",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE_OR_COLLISION",), (),
            f"{lattice}/{collision} is not in the collision capability matrix.")
    cap = matrix[lattice][collision]
    if cap.available:
        return SubContractResult("advanced_collision_contract", "collision",
            SubContractStatus.ADMITTED, (), (), cap.note)
    return SubContractResult("advanced_collision_contract", "collision",
        SubContractStatus.WITHHELD, (cap.status,), (), cap.note)


def _query_general_capability_matrix(req: CompositionRequest) -> SubContractResult:
    """Query general_capability_matrix for the complete composition."""
    candidate = {
        "lattice": req.lattice, "collision": req.collision,
        "turbulence": req.turbulence, "multiphase": req.multiphase,
        "boundary": req.boundary, "geometry": req.geometry,
        "wall_treatment": req.wall_treatment, "refinement": req.refinement,
        "backend": req.backend, "outputs": list(req.outputs),
    }
    assessment = _assess_gmc(candidate)
    codes = tuple(r.code for r in assessment.reasons)
    note = "; ".join(r.message for r in assessment.reasons) if assessment.reasons else "No reasons."
    if assessment.status is _GMCStatus.SUPPORTED:
        return SubContractResult("general_capability_matrix", "composition",
            SubContractStatus.ADMITTED, codes, (), note)
    if assessment.status is _GMCStatus.WITHHELD:
        return SubContractResult("general_capability_matrix", "composition",
            SubContractStatus.WITHHELD, codes, (), note)
    return SubContractResult("general_capability_matrix", "composition",
        SubContractStatus.NOT_SUPPORTED, codes, (), note)


def _query_wall_function_contract(req: CompositionRequest) -> SubContractResult:
    """Query wall_function_contract (only when wall_treatment == wall_function)."""
    if req.wall_treatment != "wall_function":
        return SubContractResult("wall_function_contract", "wall_treatment",
            SubContractStatus.NOT_APPLICABLE, (), (),
            "Wall function contract is not applicable when wall_treatment != 'wall_function'.")
    lattice = _LATTICE_UPPER.get(req.lattice, req.lattice.upper())
    physics = _MULTIPHASE_TO_WALL_FN_PHYSICS.get(req.multiphase, req.multiphase)
    # LOG_LAW_BODY_FORCE uses MRT_SMAGORINSKY; other capabilities use BGK/MRT
    collision = "MRT_SMAGORINSKY" if req.collision == "mrt" else req.collision.upper()
    geometry = _GEOMETRY_TO_WALL_FN.get(req.geometry, req.geometry)
    backend = "torch"
    wf_req = WallFunctionRequest(
        capability=WallFunctionCapability.LOG_LAW_BODY_FORCE,
        lattice=lattice, physics=physics, collision=collision,
        geometry=geometry, backend=backend,
    )
    assessment = assess_wall_function(wf_req)
    if assessment.compatible:
        return SubContractResult("wall_function_contract", "wall_treatment",
            SubContractStatus.ADMITTED, (), (), assessment.note)
    return SubContractResult("wall_function_contract", "wall_treatment",
        SubContractStatus.WITHHELD, (assessment.status,), (), assessment.note)


def _query_wall_function_admission(req: CompositionRequest) -> SubContractResult:
    """Query wall_function_admission (only when wall_treatment == wall_function)."""
    if req.wall_treatment != "wall_function":
        return SubContractResult("wall_function_admission", "wall_treatment",
            SubContractStatus.NOT_APPLICABLE, (), (),
            "Wall function admission is not applicable when wall_treatment != 'wall_function'.")
    lattice = _LATTICE_UPPER.get(req.lattice, req.lattice.upper())
    physics = _MULTIPHASE_TO_WALL_FN_PHYSICS.get(req.multiphase, req.multiphase)
    collision = "MRT_SMAGORINSKY" if req.collision == "mrt" else req.collision.upper()
    geometry = _GEOMETRY_TO_WALL_FN.get(req.geometry, req.geometry)
    backend = "torch"
    adaptive_mesh = req.refinement != "none"
    free_surface = req.multiphase == "free_surface"
    run_req = WallFunctionRunRequest(
        capability=WallFunctionCapability.LOG_LAW_BODY_FORCE,
        lattice=lattice, physics=physics, collision=collision,
        geometry=geometry, backend=backend,
        adaptive_mesh=adaptive_mesh, free_surface=free_surface,
    )
    try:
        require_wall_function_run(run_req)
        return SubContractResult("wall_function_admission", "wall_treatment",
            SubContractStatus.ADMITTED, (), (),
            "Wall function run admitted (implementation-only validation).")
    except WallFunctionCompatibilityError as exc:
        return SubContractResult("wall_function_admission", "wall_treatment",
            SubContractStatus.WITHHELD, ("WITHHELD_WALL_FUNCTION_RUN",), (),
            str(exc))


def _query_wall_refinement_gate(req: CompositionRequest) -> SubContractResult:
    """Query wall_refinement_combination_gate for wall × refinement × geometry × physics."""
    lattice = _LATTICE_UPPER.get(req.lattice)
    collision = _COLLISION_UPPER.get(req.collision)
    if lattice is None or collision is None:
        return SubContractResult("wall_refinement_combination_gate", "wall×refinement",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE_OR_COLLISION",), (),
            f"{req.lattice!r}/{req.collision!r} not in wall-refinement vocabulary.")
    wr_lattice = WRLattice(lattice)
    wr_collision = WRCollisionFamily(collision)
    wr_wall = _WALL_TREATMENT_TO_WR.get(req.wall_treatment)
    if wr_wall is None:
        return SubContractResult("wall_refinement_combination_gate", "wall×refinement",
            SubContractStatus.NOT_APPLICABLE, (), (),
            f"wall_treatment={req.wall_treatment!r} not in wall-refinement vocabulary.")
    wr_refinement = _REFINEMENT_TO_WR.get(req.refinement, RefinementType.NONE)
    wr_geometry = _GEOMETRY_TO_WR_KIND.get(req.geometry, GeometryKind.PLANAR_STATIC)
    wr_physics = _MULTIPHASE_TO_WR_PHYSICS.get(req.multiphase, PhysicsModel.SINGLE_PHASE)
    combination = WallRefinementCombination(
        lattice=wr_lattice, collision=wr_collision,
        wall_treatment=wr_wall, refinement=wr_refinement,
        geometry_ownership=GeometryOwnership.SINGLE_LEVEL,
        geometry_kind=wr_geometry, physics=wr_physics,
    )
    decision = assess_wall_refinement_combination(combination)
    if decision.status is GateStatus.ALLOWED:
        return SubContractResult("wall_refinement_combination_gate", "wall×refinement",
            SubContractStatus.ADMITTED, (), (), "Baseline wall/refinement combination allowed.")
    return SubContractResult("wall_refinement_combination_gate", "wall×refinement",
        SubContractStatus.WITHHELD, decision.reasons, decision.missing_required_evidence,
        "; ".join(decision.reasons) if decision.reasons else "Combination withheld.")


def _query_amr_contract(req: CompositionRequest) -> SubContractResult:
    """Query amr_capability_contract (only when refinement != none)."""
    if req.refinement == "none":
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.NOT_APPLICABLE, (), (),
            "AMR contract is not applicable when refinement == 'none'.")
    path = _REFINEMENT_TO_AMR_PATH.get(req.refinement)
    if path is None:
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_REFINEMENT_PATH",), (),
            f"{req.refinement!r} is not in the AMR contract path vocabulary.")
    lattice = _LATTICE_UPPER.get(req.lattice)
    if lattice is None:
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{req.lattice!r} is not in the AMR contract lattice vocabulary.")
    physics = _MULTIPHASE_TO_AMR_PHYSICS.get(req.multiphase, req.multiphase)
    matrix = local_refinement_capability_matrix()
    if path not in matrix:
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_PATH",), (),
            f"{path!r} is not in the AMR capability matrix.")
    if lattice not in matrix[path]:
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{lattice!r} is not in the AMR capability matrix for path {path!r}.")
    if physics not in matrix[path][lattice]:
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_PHYSICS",), (),
            f"{physics!r} is not in the AMR capability matrix for {path}/{lattice}.")
    cap = matrix[path][lattice][physics]
    if cap.available:
        return SubContractResult("amr_capability_contract", "refinement",
            SubContractStatus.ADMITTED, (), (), cap.note)
    # Collect missing required metadata
    missing = tuple(REQUIRED_FRONTEND_METADATA)
    return SubContractResult("amr_capability_contract", "refinement",
        SubContractStatus.WITHHELD, (cap.status,), missing, cap.note)


def _query_boundary_contract(req: CompositionRequest) -> SubContractResult:
    """Query boundary_capability_contract for boundary × lattice × physics."""
    kind = _BOUNDARY_TO_KIND.get(req.boundary)
    if kind is None:
        return SubContractResult("boundary_capability_contract", "boundary",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_BOUNDARY",), (),
            f"{req.boundary!r} is not in the boundary contract vocabulary.")
    lattice = _LATTICE_UPPER.get(req.lattice)
    if lattice is None:
        return SubContractResult("boundary_capability_contract", "boundary",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{req.lattice!r} is not in the boundary contract lattice vocabulary.")
    matrix = boundary_capability_matrix()
    if kind not in matrix:
        return SubContractResult("boundary_capability_contract", "boundary",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_BOUNDARY",), (),
            f"{kind!r} is not in the boundary capability matrix.")
    if lattice not in matrix[kind]:
        return SubContractResult("boundary_capability_contract", "boundary",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{lattice!r} is not in the boundary capability matrix for {kind!r}.")
    cap = matrix[kind][lattice]
    if cap.implementation_status == "NO_IMPLEMENTATION":
        return SubContractResult("boundary_capability_contract", "boundary",
            SubContractStatus.NOT_SUPPORTED, (cap.status,), (), cap.note)
    # Check physics coupling
    physics = _MULTIPHASE_TO_BOUNDARY_PHYSICS.get(req.multiphase, req.multiphase)
    if physics != "single_phase":
        return SubContractResult("boundary_capability_contract", "boundary",
            SubContractStatus.WITHHELD, ("WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT",), (),
            f"{physics!r} has no audited boundary-condition coupling contract.")
    # Implementation exists but no complete composition evidence
    return SubContractResult("boundary_capability_contract", "boundary",
        SubContractStatus.WITHHELD, (cap.status,), (), cap.note)


def _query_turbulence_contract(req: CompositionRequest) -> SubContractResult:
    """Query turbulence_capability_contract for turbulence × lattice × collision."""
    if req.turbulence == "none":
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_APPLICABLE, (), (),
            "No turbulence model requested.")
    family = _TURBULENCE_TO_FAMILY.get(req.turbulence)
    if family is None:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_TURBULENCE_FAMILY",), (),
            f"{req.turbulence!r} is not in the turbulence contract family vocabulary.")
    lattice = _LATTICE_UPPER.get(req.lattice)
    if lattice is None:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{req.lattice!r} is not in the turbulence contract lattice vocabulary.")
    collision = _COLLISION_TO_TURBULENCE.get(req.collision)
    if collision is None:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_COLLISION",), (),
            f"{req.collision!r} is not in the turbulence contract collision vocabulary.")
    matrix = turbulence_capability_matrix()
    if family not in matrix:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_FAMILY",), (),
            f"{family!r} is not in the turbulence capability matrix.")
    if lattice not in matrix[family]:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_LATTICE",), (),
            f"{lattice!r} is not in the turbulence capability matrix for {family!r}.")
    if collision not in matrix[family][lattice]:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.NOT_SUPPORTED, ("UNKNOWN_COLLISION",), (),
            f"{collision!r} is not in the turbulence capability matrix for {family}/{lattice}.")
    cap = matrix[family][lattice][collision]
    if cap.available:
        return SubContractResult("turbulence_capability_contract", "turbulence",
            SubContractStatus.ADMITTED, (), (), cap.note)
    return SubContractResult("turbulence_capability_contract", "turbulence",
        SubContractStatus.WITHHELD, (cap.status,), (), cap.note)


def _query_accuracy_recommendation(
    req: CompositionRequest,
    evidence: Sequence[PhysicalAccuracyEvidence] | None,
) -> SubContractResult:
    """Query accuracy_recommendation (only when physical accuracy evidence is provided)."""
    if evidence is None:
        return SubContractResult("accuracy_recommendation", "physical_accuracy",
            SubContractStatus.NOT_APPLICABLE, (), (),
            "No physical accuracy evidence provided; accuracy dimension not assessed.")
    recommendation = recommend_by_physical_accuracy(evidence)
    if recommendation.status == "RECOMMENDED_FROM_PHYSICAL_ACCURACY_EVIDENCE":
        return SubContractResult("accuracy_recommendation", "physical_accuracy",
            SubContractStatus.ADMITTED, (), (),
            f"Recommended candidate: {recommendation.recommended_candidate_id}")
    return SubContractResult("accuracy_recommendation", "physical_accuracy",
        SubContractStatus.WITHHELD, tuple(recommendation.reason_codes),
        tuple(recommendation.missing_requirements),
        "No physical accuracy evidence sufficient for recommendation.")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def assess_composition(
    candidate: CompositionRequest | Mapping[str, Any],
    *,
    physical_accuracy_evidence: Sequence[PhysicalAccuracyEvidence] | None = None,
) -> CompositionDecision:
    """Assess a complete cross-module composition request.

    Cross-queries all merged single-dimension capability contracts and
    aggregates their decisions into a single fail-closed composition decision.

    Args:
        candidate: A :class:`CompositionRequest` or mapping with the same fields.
        physical_accuracy_evidence: Optional sequence of
            :class:`PhysicalAccuracyEvidence` records for the accuracy dimension.

    Returns:
        A :class:`CompositionDecision` with the aggregated status, per-sub-contract
        results, missing dimensions, and reason codes.
    """
    request = _coerce_request(candidate)

    results: list[SubContractResult] = [
        _query_collision_contract(request),
        _query_general_capability_matrix(request),
        _query_wall_function_contract(request),
        _query_wall_function_admission(request),
        _query_wall_refinement_gate(request),
        _query_amr_contract(request),
        _query_boundary_contract(request),
        _query_turbulence_contract(request),
        _query_accuracy_recommendation(request, physical_accuracy_evidence),
    ]

    # Aggregate missing evidence
    missing: list[str] = []
    for r in results:
        missing.extend(r.missing_evidence)

    # Aggregate reason codes (only from non-admitted, non-N/A sub-contracts)
    reason_codes: list[str] = []
    for r in results:
        if r.status in (SubContractStatus.WITHHELD, SubContractStatus.NOT_SUPPORTED):
            reason_codes.extend(r.reason_codes)

    # Determine overall status
    has_not_supported = any(r.status is SubContractStatus.NOT_SUPPORTED for r in results)
    has_withheld = any(r.status is SubContractStatus.WITHHELD for r in results)

    if has_not_supported:
        status = CompositionStatus.NOT_SUPPORTED
    elif has_withheld:
        status = CompositionStatus.WITHHELD
    else:
        status = CompositionStatus.SUPPORTED

    return CompositionDecision(
        status=status,
        sub_contract_results=tuple(results),
        missing_dimensions=tuple(dict.fromkeys(missing)),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        normalized_request=request,
    )


__all__ = [
    "MATRIX_VERSION",
    "CompositionDecision",
    "CompositionRequest",
    "CompositionStatus",
    "SubContractResult",
    "SubContractStatus",
    "assess_composition",
]
