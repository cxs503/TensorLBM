"""Fail-closed public capability contract for 6-DOF rigid-body dynamics.

This module is an audit boundary, not a 6-DOF dispatcher.  It reports the
rigid-body integrators actually present in the repository and their
contract-test evidence, and refuses to certify any 6-DOF combination as
physics-validated.  Successful shape/momentum/identity contract tests verify
operator algebra (Newton-Euler consistency, DOF constraints, quaternion
normalisation), **not** physical accuracy of the rigid-body motion or
fluid-structure coupling.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/sixdof.py``           – Symplectic-Euler 6-DOF integrator
    - ``tensorlbm/rigid_body_6dof.py``   – Cummins time-domain 6-DOF
    - ``tensorlbm/sixdof_common.py``    – solver-agnostic common 6-DOF interface
    - tests: ``test_sixdof_common.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------- #
# Public type aliases
# --------------------------------------------------------------------------- #

SixDOFIntegratorName = Literal["symplectic_euler", "cummins"]

# --------------------------------------------------------------------------- #
# Machine-readable withheld codes (fail-closed)
# --------------------------------------------------------------------------- #

WITHHELD_NO_PHYSICS_VALIDATION = "WITHHELD_NO_PHYSICS_VALIDATION"
WITHHELD_NO_CONTRACT_TESTS = "WITHHELD_NO_CONTRACT_TESTS"
WITHHELD_NO_IMPLEMENTATION = "WITHHELD_NO_IMPLEMENTATION"
WITHHELD_UNKNOWN_INTEGRATOR = "WITHHELD_UNKNOWN_INTEGRATOR"

# --------------------------------------------------------------------------- #
# Verification levels
# --------------------------------------------------------------------------- #

VERIFICATION_CONTRACT_TESTED = "CONTRACT_TESTED"
"""Shape/momentum/identity unit tests exist.

These verify operator algebra (Newton-Euler consistency, DOF constraint
enforcement, quaternion normalisation), NOT rigid-body physics correctness
or fluid-structure coupling validation.
"""

VERIFICATION_IMPLEMENTED_ONLY = "IMPLEMENTED_ONLY"
VERIFICATION_NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

# --------------------------------------------------------------------------- #

IMPLEMENTED = "IMPLEMENTED"
NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

_AUDITED_INTEGRATORS: tuple[str, ...] = ("symplectic_euler", "cummins")


class SixDOFWithheldError(NotImplementedError):
    """Raised when a 6-DOF capability request lacks a validated integrator."""


@dataclass(frozen=True)
class SixDOFCapability:
    """Audited state of one 6-DOF integrator.

    ``verification_level`` records what evidence exists in the repository.
    ``status`` is the fail-closed frontend claim: it is always a
    ``WITHHELD_*`` code because no current 6-DOF integrator has physics
    validation evidence.
    """

    integrator: str
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

_REGISTRY: dict[str, _RegistryEntry] = {
    "symplectic_euler": (
        IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
        "tensorlbm.sixdof_common.rigid_body_step",
        "test_sixdof_common.py: shape, zero-force identity, constant-force momentum, DOF constraints, quaternion normalisation",
        "Symplectic (semi-implicit) Euler Newton-Euler integrator with quaternion exponential-map update; DOF constraints supported.",
    ),
    "cummins": (
        IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
        "tensorlbm.rigid_body_6dof.cummins_step",
        None,
        "Cummins time-domain equation with retardation-function convolution; requires radiation data and is not exposed through the common rigid_body_step interface.",
    ),
}


def _status_for(verification: str) -> str:
    if verification == VERIFICATION_CONTRACT_TESTED:
        return WITHHELD_NO_PHYSICS_VALIDATION
    if verification == VERIFICATION_IMPLEMENTED_ONLY:
        return WITHHELD_NO_CONTRACT_TESTS
    return WITHHELD_NO_IMPLEMENTATION


def _capability_for(integrator: str) -> SixDOFCapability:
    entry = _REGISTRY.get(integrator)
    if entry is None:
        return SixDOFCapability(
            integrator=integrator,
            implementation_status=NO_IMPLEMENTATION,
            verification_level=VERIFICATION_NO_IMPLEMENTATION,
            status=WITHHELD_NO_IMPLEMENTATION,
            entrypoint=None, test_evidence=None,
            note=f"No implementation found for 6-DOF integrator {integrator!r}.",
        )
    impl_status, verif_level, entrypoint, test_ev, note = entry
    return SixDOFCapability(
        integrator=integrator,
        implementation_status=impl_status,
        verification_level=verif_level,
        status=_status_for(verif_level),
        entrypoint=entrypoint, test_evidence=test_ev, note=note,
    )


def sixdof_capability_matrix() -> dict[str, SixDOFCapability]:
    """Return the complete audited 6-DOF integrator capability matrix.

    Every entry is fail-closed: no integrator has physics validation evidence.
    """
    return {integrator: _capability_for(integrator) for integrator in _AUDITED_INTEGRATORS}


def require_sixdof_capability(integrator: SixDOFIntegratorName) -> SixDOFCapability:
    """Return only a physics-validated 6-DOF capability; otherwise fail closed.

    No current 6-DOF integrator has physics validation evidence.  This function
    always raises :class:`SixDOFWithheldError` for the physics-validated claim.
    """
    if integrator not in _AUDITED_INTEGRATORS:
        raise SixDOFWithheldError(
            f"{WITHHELD_UNKNOWN_INTEGRATOR}: {integrator!r} is not an audited 6-DOF integrator."
        )
    capability = sixdof_capability_matrix()[integrator]
    if not capability.available:
        raise SixDOFWithheldError(f"{capability.status}: {capability.note}")
    return capability


__all__ = [
    "SixDOFCapability",
    "SixDOFWithheldError",
    "WITHHELD_NO_PHYSICS_VALIDATION",
    "WITHHELD_NO_CONTRACT_TESTS",
    "WITHHELD_NO_IMPLEMENTATION",
    "WITHHELD_UNKNOWN_INTEGRATOR",
    "VERIFICATION_CONTRACT_TESTED",
    "VERIFICATION_IMPLEMENTED_ONLY",
    "VERIFICATION_NO_IMPLEMENTATION",
    "IMPLEMENTED",
    "NO_IMPLEMENTATION",
    "sixdof_capability_matrix",
    "require_sixdof_capability",
]
