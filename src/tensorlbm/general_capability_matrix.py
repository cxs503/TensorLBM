"""Pure, fail-closed capability assessment for platform integrations (R1).

This is deliberately a declaration and validation layer, not a solver planner.
It does not import or execute a solver, create a web service, or infer that
independently existing legacy functions have been integration-verified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


MATRIX_VERSION = "general-capability-matrix-r1"


class CapabilityStatus(str, Enum):
    """Disposition of one complete requested configuration."""

    SUPPORTED = "supported"
    WITHHELD = "withheld"
    NOT_SUPPORTED = "not_supported"


class EvidenceTier(str, Enum):
    """Strength of evidence for the returned disposition."""

    EXECUTABLE_CONTRACT = "executable_contract"
    COMPONENT_CONTRACT = "component_contract"
    NO_COMPOSITION_EVIDENCE = "no_composition_evidence"
    UNIMPLEMENTED = "unimplemented"
    UNKNOWN_REQUEST = "unknown_request"


@dataclass(frozen=True)
class CapabilityRequest:
    """Typed platform-neutral candidate.

    R1 accepts strings to keep JSON/UI adapters dependency-free.  Values are
    canonicalized by :func:`assess_capability`; callers may pass a mapping to
    that function when they do not need this typed object.
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
class CapabilityReason:
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class CapabilityAssessment:
    """Machine-readable response for a complete candidate, never a promise.

    ``capability_hash`` identifies this immutable R1 registry/evidence set;
    ``config_hash`` identifies the normalized requested candidate.
    """

    status: CapabilityStatus
    reasons: tuple[CapabilityReason, ...]
    evidence_tier: EvidenceTier
    capability_hash: str
    config_hash: str
    normalized_request: CapabilityRequest

    @property
    def supported(self) -> bool:
        return self.status is CapabilityStatus.SUPPORTED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without framework dependencies."""
        return {
            "status": self.status.value,
            "reasons": [asdict(reason) for reason in self.reasons],
            "evidence_tier": self.evidence_tier.value,
            "capability_hash": self.capability_hash,
            "config_hash": self.config_hash,
            "normalized_request": {
                **asdict(self.normalized_request),
                "outputs": list(self.normalized_request.outputs),
            },
        }


# Each component entry is evidence of *that component only*.  The sole complete
# composition admitted by R1 is declared below, so adding a legacy function does
# not accidentally make a new product configuration available.
_COMPONENT_EVIDENCE: dict[str, dict[str, tuple[bool, EvidenceTier, str]]] = {
    "lattice": {
        "d3q19": (True, EvidenceTier.COMPONENT_CONTRACT, "D3Q19 MRT entrypoint is executable: tensorlbm.solver3d.collide_mrt3d."),
        "d3q27": (True, EvidenceTier.COMPONENT_CONTRACT, "D3Q27 MRT entrypoint is executable: tensorlbm.d3q27.collide_mrt27."),
    },
    "collision": {
        "mrt": (True, EvidenceTier.COMPONENT_CONTRACT, "advanced_collision_contract advertises MRT for D3Q19 and D3Q27."),
        "cm": (False, EvidenceTier.UNIMPLEMENTED, "CM is explicitly withheld: no standalone validated central-moment kernel."),
        "kbc": (False, EvidenceTier.UNIMPLEMENTED, "KBC is explicitly withheld: no entropy-solved KBC kernel."),
    },
}

# This is intentionally narrow.  It is a tested common execution path, not a
# catalog of similarly named implementation fragments.
_R1_SUPPORTED = CapabilityRequest(
    lattice="d3q19", collision="mrt", turbulence="none", multiphase="single_phase",
    boundary="static_wall", geometry="static_solid_mask", wall_treatment="bounce_back",
    refinement="none", backend="torch", outputs=("rho", "velocity"),
)

_ALIASES = {
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

# Request-schema membership deliberately differs from implementation evidence:
# known but unverified values are withheld; unknown values are not supported.
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_value(field: str, value: object) -> str:
    if not isinstance(value, str):
        return ""
    compact = value.strip().lower().replace(" ", "_")
    return _ALIASES[field].get(compact, compact)


def _coerce_request(candidate: CapabilityRequest | Mapping[str, Any]) -> CapabilityRequest:
    if isinstance(candidate, CapabilityRequest):
        raw = asdict(candidate)
    elif isinstance(candidate, Mapping):
        allowed = set(CapabilityRequest.__dataclass_fields__)
        unknown = set(candidate) - allowed
        if unknown:
            raise ValueError("unknown capability request fields: " + ", ".join(sorted(unknown)))
        raw = dict(candidate)
    else:
        raise TypeError("candidate must be CapabilityRequest or a mapping")
    outputs = raw.get("outputs", ("rho", "velocity"))
    if isinstance(outputs, str) or not isinstance(outputs, Sequence) or not all(isinstance(item, str) for item in outputs):
        raise ValueError("outputs must be a sequence of strings, not a single string")
    return CapabilityRequest(
        **{field: _normalise_value(field, raw.get(field, getattr(_R1_SUPPORTED, field)))
           for field in _ALIASES},
        outputs=tuple(sorted({item.strip().lower() for item in outputs})),
    )


def capability_hash() -> str:
    """Stable fingerprint of R1's declarations and evidence, not source state."""
    return _hash({
        "version": MATRIX_VERSION,
        "component_evidence": _COMPONENT_EVIDENCE,
        "known_values": {field: sorted(values) for field, values in _KNOWN_VALUES.items()},
        "supported": asdict(_R1_SUPPORTED),
    })


def capability_matrix() -> dict[str, dict[str, dict[str, str | bool]]]:
    """Return the audited component registry as a JSON-ready copy.

    ``available`` is deliberately component-scoped.  Consumers must call
    :func:`assess_capability` before presenting a full candidate as supported.
    """
    return {
        field: {
            value: {"available": available, "evidence_tier": tier.value, "evidence": evidence}
            for value, (available, tier, evidence) in entries.items()
        }
        for field, entries in _COMPONENT_EVIDENCE.items()
    }


def assess_capability(candidate: CapabilityRequest | Mapping[str, Any]) -> CapabilityAssessment:
    """Assess a complete candidate with explicit fail-closed composition rules."""
    request = _coerce_request(candidate)
    reasons: list[CapabilityReason] = []

    # Validate every scalar and each list member before composition evidence.
    # An understood feature may be withheld; an unrecognized input never is.
    for field, known_values in _KNOWN_VALUES.items():
        value = getattr(request, field)
        values = value if field == "outputs" else (value,)
        for item in values:
            if item not in known_values:
                reasons.append(CapabilityReason(
                    "UNKNOWN_VALUE", field,
                    f"{item!r} is not defined by the R1 {field} input schema.",
                ))
    if reasons:
        return CapabilityAssessment(
            CapabilityStatus.NOT_SUPPORTED, tuple(reasons),
            EvidenceTier.UNKNOWN_REQUEST, capability_hash(),
            _hash(asdict(request)), request,
        )

    unsupported = False
    withheld = False

    for field in ("lattice", "collision"):
        value = getattr(request, field)
        entry = _COMPONENT_EVIDENCE[field].get(value)
        if entry is None:
            unsupported = True
            reasons.append(CapabilityReason("UNKNOWN_VALUE", field, f"{value!r} is not defined by R1."))
        elif not entry[0]:
            withheld = True
            code = "WITHHELD_COLLISION_FAMILY"
            reasons.append(CapabilityReason(code, field, entry[2]))

    expected = asdict(_R1_SUPPORTED)
    actual = asdict(request)
    for field in ("turbulence", "multiphase", "boundary", "geometry", "wall_treatment", "refinement", "backend", "outputs"):
        if actual[field] != expected[field]:
            withheld = True
            reasons.append(CapabilityReason(
                "WITHHELD_UNVERIFIED_COMPOSITION", field,
                f"R1 has no complete-composition evidence for {field}={actual[field]!r}; only {expected[field]!r} is admitted.",
            ))

    if request.lattice == "d3q27" and not unsupported:
        withheld = True
        reasons.append(CapabilityReason(
            "WITHHELD_D3Q27_COMPOSITION", "lattice",
            "D3Q27 MRT has component-level executable evidence, but R1 has no verified complete platform composition.",
        ))

    if unsupported:
        status, tier = CapabilityStatus.NOT_SUPPORTED, EvidenceTier.UNKNOWN_REQUEST
    elif withheld:
        status = CapabilityStatus.WITHHELD
        tier = EvidenceTier.UNIMPLEMENTED if any(reason.code == "WITHHELD_COLLISION_FAMILY" for reason in reasons) else EvidenceTier.NO_COMPOSITION_EVIDENCE
    else:
        status, tier = CapabilityStatus.SUPPORTED, EvidenceTier.EXECUTABLE_CONTRACT
        reasons.append(CapabilityReason("R1_VERIFIED_COMPOSITION", "candidate", "D3Q19/MRT single-phase static-wall, bounce-back, no-AMR Torch composition is admitted by R1."))

    return CapabilityAssessment(status, tuple(reasons), tier, capability_hash(), _hash(asdict(request)), request)


__all__ = [
    "MATRIX_VERSION", "CapabilityAssessment", "CapabilityReason", "CapabilityRequest",
    "CapabilityStatus", "EvidenceTier", "assess_capability", "capability_hash", "capability_matrix",
]
