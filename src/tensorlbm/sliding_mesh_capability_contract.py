"""Fail-closed public capability contract for sliding-mesh boundary conditions.

This module is an audit boundary, not a sliding-mesh dispatcher.  It reports
the mechanics actually present in the repository and refuses to certify a
sliding-mesh combination as physics-validated.  In particular, successful
shape/identity tests do not establish physical accuracy, conservation, or
coupled-physics correctness.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/sliding_mesh.py``           – 2-D sliding-mesh (D2Q9)
    - ``tensorlbm/sliding_mesh_common.py``    – 3-D common module (D3Q19/D3Q27)
    - callers: ``rotating_cylinder.py``, ``propeller_ibm.py``
    - tests: ``test_sliding_mesh_common.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SlidingMeshKind = Literal[
    "sliding_mesh_2d",
    "sliding_mesh_3d",
]
LatticeName = Literal["D2Q9", "D3Q19", "D3Q27"]
CollisionFamily = Literal["bgk", "mrt", "trt", "smagorinsky", "kbc", "cascaded"]
PhysicsName = Literal["single_phase", "turbulence", "multiphase", "free_surface", "ibm"]
BackendName = Literal["torch_cpu", "torch_cuda"]

# --------------------------------------------------------------------------- #
# Withholding codes
# --------------------------------------------------------------------------- #

WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE = "WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE"
WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE = "WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE"
WITHHELD_NO_COUPLED_SLIDING_MESH_PHYSICS_CONTRACT = "WITHHELD_NO_COUPLED_SLIDING_MESH_PHYSICS_CONTRACT"
WITHHELD_UNKNOWN_KIND = "WITHHELD_UNKNOWN_KIND"
WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"
WITHHELD_UNKNOWN_COLLISION = "WITHHELD_UNKNOWN_COLLISION"
WITHHELD_UNKNOWN_PHYSICS = "WITHHELD_UNKNOWN_PHYSICS"
WITHHELD_UNKNOWN_BACKEND = "WITHHELD_UNKNOWN_BACKEND"

_AUDITED_KINDS: tuple[SlidingMeshKind, ...] = ("sliding_mesh_2d", "sliding_mesh_3d")
_AUDITED_LATTICES: tuple[LatticeName, ...] = ("D2Q9", "D3Q19", "D3Q27")
_AUDITED_COLLISIONS: tuple[CollisionFamily, ...] = (
    "bgk", "mrt", "trt", "smagorinsky", "kbc", "cascaded",
)
_AUDITED_PHYSICS: tuple[PhysicsName, ...] = (
    "single_phase", "turbulence", "multiphase", "free_surface", "ibm",
)
_AUDITED_BACKENDS: tuple[BackendName, ...] = ("torch_cpu", "torch_cuda")


class SlidingMeshWithheldError(NotImplementedError):
    """Raised when a sliding-mesh request lacks an audited executable contract."""


@dataclass(frozen=True)
class SlidingMeshCapability:
    """Audited state of one sliding-mesh-kind/lattice combination.

    ``implementation_status`` describes the strongest evidence found in the
    repository for this (kind, lattice) pair.  ``status`` is the frontend
    claim and remains withheld for every current combination because none
    has a complete, verified composition contract (collision × physics ×
    backend).
    """

    implementation_status: str
    status: str
    entrypoint: str | None
    verification_evidence: str
    note: str

    @property
    def available(self) -> bool:
        """Whether a frontend may claim this combination is contract-ready."""
        return self.status == "AVAILABLE"


# --------------------------------------------------------------------------- #
# Implementation evidence registry
# --------------------------------------------------------------------------- #

_IMPLEMENTATION_EVIDENCE: dict[tuple[str, str], tuple[str, str | None, str]] = {

    # ---- sliding_mesh_2d (D2Q9) -------------------------------------------
    ("sliding_mesh_2d", "D2Q9"): (
        "MECHANICS_TESTED",
        "tensorlbm.sliding_mesh.apply_sliding_mesh_bc_2d / "
        "rotate_velocity_field_2d / interpolate_interface_2d",
        "test_rotating_cylinder.py: rotor-stator smoke test; "
        "no physical validation of torque/efficiency",
    ),
    ("sliding_mesh_2d", "D3Q19"): (
        "NO_IMPLEMENTATION",
        None,
        "No 2-D sliding-mesh implementation for D3Q19 (use sliding_mesh_3d)",
    ),
    ("sliding_mesh_2d", "D3Q27"): (
        "NO_IMPLEMENTATION",
        None,
        "No 2-D sliding-mesh implementation for D3Q27 (use sliding_mesh_3d)",
    ),

    # ---- sliding_mesh_3d (D3Q19 / D3Q27) ----------------------------------
    ("sliding_mesh_3d", "D2Q9"): (
        "NO_IMPLEMENTATION",
        None,
        "No 3-D sliding-mesh implementation for D2Q9 (use sliding_mesh_2d)",
    ),
    ("sliding_mesh_3d", "D3Q19"): (
        "MECHANICS_TESTED",
        "tensorlbm.sliding_mesh_common.sliding_mesh_step(lattice='D3Q19') / "
        "apply_sliding_mesh_bc_3d / rotate_velocity_field_3d / "
        "interpolate_interface_3d",
        "test_sliding_mesh_common.py: shape, finite, mass conservation, "
        "interface locality, rotation axis parametrisation; "
        "no physical validation of torque/efficiency",
    ),
    ("sliding_mesh_3d", "D3Q27"): (
        "MECHANICS_TESTED",
        "tensorlbm.sliding_mesh_common.sliding_mesh_step(lattice='D3Q27') / "
        "apply_sliding_mesh_bc_3d(lattice='D3Q27')",
        "test_sliding_mesh_common.py: shape, finite, mass conservation, "
        "interface locality, rotation axis parametrisation; "
        "no physical validation of torque/efficiency",
    ),
}


def _capability_for(kind: str, lattice: str) -> SlidingMeshCapability:
    """Return the audited capability for one (sliding_mesh_kind, lattice) pair."""
    detail = _IMPLEMENTATION_EVIDENCE.get((kind, lattice))
    if detail is None:
        return SlidingMeshCapability(
            "NO_IMPLEMENTATION", WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE, None,
            "No implementation evidence registered for this combination.",
            f"{kind}/{lattice} is not in the audited implementation registry.",
        )
    impl_status, entrypoint, evidence = detail

    if impl_status == "NO_IMPLEMENTATION":
        return SlidingMeshCapability(
            impl_status, WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE, entrypoint, evidence,
            f"{kind} has no audited {lattice} implementation.",
        )

    return SlidingMeshCapability(
        impl_status, WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE, entrypoint, evidence,
        "Implementation exists, but no complete collision × physics × backend composition "
        "has been verified as a production-ready contract. Mechanics tests "
        "do not establish composition correctness.",
    )


def sliding_mesh_capability_matrix() -> dict[str, dict[str, SlidingMeshCapability]]:
    """Return the complete audited sliding-mesh-kind × lattice capability matrix."""
    return {
        kind: {
            lattice: _capability_for(kind, lattice)
            for lattice in _AUDITED_LATTICES
        }
        for kind in _AUDITED_KINDS
    }


def require_sliding_mesh_capability(
    kind: str,
    lattice: str,
    collision: str,
    physics: str,
    backend: str,
) -> SlidingMeshCapability:
    """Return only an executable capability; otherwise fail closed.

    Request identities are validated before implementation or matrix lookup,
    so an unsupported public input is always rejected with a stable,
    machine-readable withholding code rather than leaking a ``KeyError``.
    """
    if kind not in _AUDITED_KINDS:
        raise SlidingMeshWithheldError(
            f"{WITHHELD_UNKNOWN_KIND}: {kind!r} is not an audited sliding-mesh kind."
        )
    if lattice not in _AUDITED_LATTICES:
        raise SlidingMeshWithheldError(
            f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited lattice."
        )
    if collision not in _AUDITED_COLLISIONS:
        raise SlidingMeshWithheldError(
            f"{WITHHELD_UNKNOWN_COLLISION}: {collision!r} is not an audited collision family."
        )
    if physics not in _AUDITED_PHYSICS:
        raise SlidingMeshWithheldError(
            f"{WITHHELD_UNKNOWN_PHYSICS}: {physics!r} is not an audited physics selection."
        )
    if backend not in _AUDITED_BACKENDS:
        raise SlidingMeshWithheldError(
            f"{WITHHELD_UNKNOWN_BACKEND}: {backend!r} is not an audited backend."
        )

    capability = sliding_mesh_capability_matrix()[kind][lattice]

    if capability.implementation_status == "NO_IMPLEMENTATION":
        raise SlidingMeshWithheldError(
            f"{capability.status}: {capability.note}"
        )

    # Non-single-phase physics has no audited sliding-mesh coupling contract.
    if physics != "single_phase":
        raise SlidingMeshWithheldError(
            f"{WITHHELD_NO_COUPLED_SLIDING_MESH_PHYSICS_CONTRACT}: {physics!r} has no audited "
            f"sliding-mesh coupling contract for {kind}/{lattice}."
        )

    if not capability.available:
        raise SlidingMeshWithheldError(
            f"{capability.status}: {capability.note}"
        )
    return capability


__all__ = [
    "SlidingMeshCapability",
    "SlidingMeshWithheldError",
    "WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE",
    "WITHHELD_NO_COUPLED_SLIDING_MESH_PHYSICS_CONTRACT",
    "WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE",
    "WITHHELD_UNKNOWN_BACKEND",
    "WITHHELD_UNKNOWN_KIND",
    "WITHHELD_UNKNOWN_LATTICE",
    "WITHHELD_UNKNOWN_COLLISION",
    "WITHHELD_UNKNOWN_PHYSICS",
    "sliding_mesh_capability_matrix",
    "require_sliding_mesh_capability",
]
