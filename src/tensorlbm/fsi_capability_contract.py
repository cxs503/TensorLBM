"""Fail-closed public capability contract for Fluid-Structure Interaction (FSI).

This module is an audit boundary, not an FSI dispatcher.  It reports the FSI
compositions actually present in the repository and their contract-test
evidence, and refuses to certify any FSI combination as physics-validated.
Successful shape/force-conservation/identity contract tests verify operator
algebra (IBM force conservation, rigid-body momentum consistency, reaction-
force sign), **not** physical accuracy of the fluid-structure coupling or
moving-body validation.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/fsi.py``           – one-way / two-way FSI load extraction
    - ``tensorlbm/fsi_common.py``    – solver-agnostic common FSI interface
    - ``tensorlbm/ibm_common.py``    – common IBM direct-forcing
    - ``tensorlbm/sixdof_common.py`` – common 6-DOF rigid-body step
    - tests: ``test_fsi_common.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------- #
# Public type aliases
# --------------------------------------------------------------------------- #

FSILatticeName = Literal["D3Q19", "D3Q27"]
FSICouplingName = Literal["one_way_explicit", "two_way_explicit"]

# --------------------------------------------------------------------------- #
# Machine-readable withheld codes (fail-closed)
# --------------------------------------------------------------------------- #

WITHHELD_NO_PHYSICS_VALIDATION = "WITHHELD_NO_PHYSICS_VALIDATION"
WITHHELD_NO_CONTRACT_TESTS = "WITHHELD_NO_CONTRACT_TESTS"
WITHHELD_NO_IMPLEMENTATION = "WITHHELD_NO_IMPLEMENTATION"
WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"
WITHHELD_UNKNOWN_COUPLING = "WITHHELD_UNKNOWN_COUPLING"

# --------------------------------------------------------------------------- #
# Verification levels
# --------------------------------------------------------------------------- #

VERIFICATION_CONTRACT_TESTED = "CONTRACT_TESTED"
"""Shape/force-conservation/identity unit tests exist.

These verify operator algebra (IBM force conservation, rigid-body momentum
consistency, reaction-force sign), NOT FSI physics correctness or moving-body
validation.
"""

VERIFICATION_IMPLEMENTED_ONLY = "IMPLEMENTED_ONLY"
VERIFICATION_NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

# --------------------------------------------------------------------------- #

IMPLEMENTED = "IMPLEMENTED"
NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

_AUDITED_LATTICES: tuple[str, ...] = ("D3Q19", "D3Q27")
_AUDITED_COUPLINGS: tuple[str, ...] = ("one_way_explicit", "two_way_explicit")


class FSIWithheldError(NotImplementedError):
    """Raised when an FSI capability request lacks a validated composition."""


@dataclass(frozen=True)
class FSICapability:
    """Audited state of one FSI lattice/coupling combination.

    ``verification_level`` records what evidence exists in the repository.
    ``status`` is the fail-closed frontend claim: it is always a
    ``WITHHELD_*`` code because no current FSI combination has physics
    validation evidence.
    """

    lattice: str
    coupling: str
    implementation_status: str
    verification_level: str
    status: str
    entrypoint: str | None
    test_evidence: str | None
    note: str

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE"


# --------------------------------------------------------------------------- #
# Audit registry
# --------------------------------------------------------------------------- #

_RegistryEntry = tuple[str, str, str | None, str | None, str]

_REGISTRY: dict[str, dict[str, _RegistryEntry]] = {
    "D3Q19": {
        "one_way_explicit": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.fsi_common.fsi_step",
            "test_fsi_common.py: shape, zero-flow identity, force-reaction sign, composition consistency",
            "Explicit one-step FSI: IBM direct-forcing (D3Q19) + Symplectic-Euler 6-DOF; reaction force = −Σ IBM fluid force.",
        ),
        "two_way_explicit": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.fsi_common.fsi_step",
            "test_fsi_common.py: shape, finite, two-way re-pass consistency",
            "Explicit two-step FSI: IBM direct-forcing (D3Q19) + 6-DOF advance + second IBM pass with advanced body velocity.",
        ),
    },
    "D3Q27": {
        "one_way_explicit": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.fsi_common.fsi_step",
            "test_fsi_common.py: shape, zero-flow identity, force-reaction sign, composition consistency",
            "Explicit one-step FSI: IBM direct-forcing (D3Q27) + Symplectic-Euler 6-DOF; reaction force = −Σ IBM fluid force.",
        ),
        "two_way_explicit": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.fsi_common.fsi_step",
            "test_fsi_common.py: shape, finite, two-way re-pass consistency",
            "Explicit two-step FSI: IBM direct-forcing (D3Q27) + 6-DOF advance + second IBM pass with advanced body velocity.",
        ),
    },
}


def _status_for(verification: str) -> str:
    if verification == VERIFICATION_CONTRACT_TESTED:
        return WITHHELD_NO_PHYSICS_VALIDATION
    if verification == VERIFICATION_IMPLEMENTED_ONLY:
        return WITHHELD_NO_CONTRACT_TESTS
    return WITHHELD_NO_IMPLEMENTATION


def _capability_for(lattice: str, coupling: str) -> FSICapability:
    lattice_map = _REGISTRY.get(lattice, {})
    entry = lattice_map.get(coupling)
    if entry is None:
        return FSICapability(
            lattice=lattice, coupling=coupling,
            implementation_status=NO_IMPLEMENTATION,
            verification_level=VERIFICATION_NO_IMPLEMENTATION,
            status=WITHHELD_NO_IMPLEMENTATION,
            entrypoint=None, test_evidence=None,
            note=f"No implementation found for FSI {lattice}/{coupling}.",
        )
    impl_status, verif_level, entrypoint, test_ev, note = entry
    return FSICapability(
        lattice=lattice, coupling=coupling,
        implementation_status=impl_status,
        verification_level=verif_level,
        status=_status_for(verif_level),
        entrypoint=entrypoint, test_evidence=test_ev, note=note,
    )


def fsi_capability_matrix() -> dict[str, dict[str, FSICapability]]:
    """Return the complete audited FSI lattice × coupling capability matrix.

    Every entry is fail-closed: no combination has physics validation evidence.
    """
    return {
        lattice: {coupling: _capability_for(lattice, coupling) for coupling in _AUDITED_COUPLINGS}
        for lattice in _AUDITED_LATTICES
    }


def require_fsi_capability(
    lattice: FSILatticeName,
    coupling: FSICouplingName,
) -> FSICapability:
    """Return only a physics-validated FSI capability; otherwise fail closed.

    No current FSI combination has physics validation evidence.  This function
    always raises :class:`FSIWithheldError` for the physics-validated claim.
    """
    if lattice not in _AUDITED_LATTICES:
        raise FSIWithheldError(
            f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited FSI lattice."
        )
    if coupling not in _AUDITED_COUPLINGS:
        raise FSIWithheldError(
            f"{WITHHELD_UNKNOWN_COUPLING}: {coupling!r} is not an audited FSI coupling mode."
        )
    capability = fsi_capability_matrix()[lattice][coupling]
    if not capability.available:
        raise FSIWithheldError(f"{capability.status}: {capability.note}")
    return capability


__all__ = [
    "FSICapability",
    "FSIWithheldError",
    "WITHHELD_NO_PHYSICS_VALIDATION",
    "WITHHELD_NO_CONTRACT_TESTS",
    "WITHHELD_NO_IMPLEMENTATION",
    "WITHHELD_UNKNOWN_LATTICE",
    "WITHHELD_UNKNOWN_COUPLING",
    "VERIFICATION_CONTRACT_TESTED",
    "VERIFICATION_IMPLEMENTED_ONLY",
    "VERIFICATION_NO_IMPLEMENTATION",
    "IMPLEMENTED",
    "NO_IMPLEMENTATION",
    "fsi_capability_matrix",
    "require_fsi_capability",
]
