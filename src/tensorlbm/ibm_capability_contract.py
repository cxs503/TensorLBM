"""Fail-closed public capability contract for the Immersed Boundary Method (IBM).

This module is an audit boundary, not an IBM dispatcher.  It reports the IBM
kernels actually present in the repository and their contract-test evidence,
and refuses to certify any IBM combination as physics-validated.  Successful
shape/force-conservation/identity contract tests verify operator algebra
(force conservation, partition-of-unity of the delta kernel, equilibrium
fixed-point), **not** physical accuracy of the immersed boundary coupling.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/ibm.py``        – direct-forcing IBM kernels (2-D & 3-D)
    - ``tensorlbm/ibm_vec.py``    – vectorized 3-D direct-forcing
    - ``tensorlbm/ibm_common.py`` – solver-agnostic common IBM interface
    - tests: ``test_ibm.py``, ``test_ibm3d.py``, ``test_ibm_common.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------- #
# Public type aliases
# --------------------------------------------------------------------------- #

IBMLatticeName = Literal["D3Q19", "D3Q27"]
IBMKernelName = Literal["hat", "4pt"]

# --------------------------------------------------------------------------- #
# Machine-readable withheld codes (fail-closed)
# --------------------------------------------------------------------------- #

WITHHELD_NO_PHYSICS_VALIDATION = "WITHHELD_NO_PHYSICS_VALIDATION"
"""Implementation exists with contract tests, but no physics validation evidence."""

WITHHELD_NO_CONTRACT_TESTS = "WITHHELD_NO_CONTRACT_TESTS"
"""Implementation exists but no unit tests verify even operator algebra."""

WITHHELD_NO_IMPLEMENTATION = "WITHHELD_NO_IMPLEMENTATION"
"""No implementation found for this lattice/kernel combination."""

WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"
WITHHELD_UNKNOWN_KERNEL = "WITHHELD_UNKNOWN_KERNEL"

# --------------------------------------------------------------------------- #
# Verification levels
# --------------------------------------------------------------------------- #

VERIFICATION_CONTRACT_TESTED = "CONTRACT_TESTED"
"""Shape/force-conservation/equilibrium-identity unit tests exist.

These verify operator algebra (delta kernel partition-of-unity, force
conservation, equilibrium fixed-point), NOT immersed-boundary physics
correctness or moving-body validation.
"""

VERIFICATION_IMPLEMENTED_ONLY = "IMPLEMENTED_ONLY"
"""Implementation exists in source, but no tests exercise it."""

VERIFICATION_NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

# --------------------------------------------------------------------------- #
# Implementation statuses
# --------------------------------------------------------------------------- #

IMPLEMENTED = "IMPLEMENTED"
NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

# --------------------------------------------------------------------------- #
# Audited dimensions
# --------------------------------------------------------------------------- #

_AUDITED_LATTICES: tuple[str, ...] = ("D3Q19", "D3Q27")
_AUDITED_KERNELS: tuple[str, ...] = ("hat", "4pt")


# --------------------------------------------------------------------------- #
# Error type
# --------------------------------------------------------------------------- #


class IBMWithheldError(NotImplementedError):
    """Raised when an IBM capability request lacks a validated kernel."""


# --------------------------------------------------------------------------- #
# Capability dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IBMCapability:
    """Audited state of one IBM lattice/kernel combination.

    ``verification_level`` records what evidence exists in the repository.
    ``status`` is the fail-closed frontend claim: it is always a ``WITHHELD_*``
    code because no current IBM combination has physics validation evidence.
    Contract tests (shape/force-conservation/identity) verify operator algebra
    only.
    """

    lattice: str
    kernel: str
    implementation_status: str
    verification_level: str
    status: str
    entrypoint: str | None
    test_evidence: str | None
    note: str

    @property
    def available(self) -> bool:
        """Whether a frontend may claim this combination is contract-ready."""
        return self.status == "AVAILABLE"


# --------------------------------------------------------------------------- #
# Audit registry
#
# Each entry maps (lattice, kernel) to a tuple of:
#   (implementation_status, verification_level, entrypoint,
#    test_evidence, note)
# --------------------------------------------------------------------------- #

_RegistryEntry = tuple[str, str, str | None, str | None, str]

_REGISTRY: dict[str, dict[str, _RegistryEntry]] = {
    "D3Q19": {
        "hat": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.ibm_common.ibm_direct_forcing_3d_common",
            "test_ibm_common.py: shape, force-conservation, equilibrium fixed-point, zero-force identity",
            "Direct-forcing IBM with 2-point hat kernel; Guo body-force correction via D3Q19 weights.",
        ),
        "4pt": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.ibm_common.ibm_direct_forcing_3d_common",
            "test_ibm_common.py: shape, force-conservation, equilibrium fixed-point",
            "Direct-forcing IBM with 4-point Peskin kernel; Guo body-force correction via D3Q19 weights.",
        ),
    },
    "D3Q27": {
        "hat": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.ibm_common.ibm_direct_forcing_3d_common",
            "test_ibm_common.py: shape, force-conservation, equilibrium fixed-point, zero-force identity",
            "Direct-forcing IBM with 2-point hat kernel; Guo body-force correction via D3Q27 weights.",
        ),
        "4pt": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.ibm_common.ibm_direct_forcing_3d_common",
            "test_ibm_common.py: shape, force-conservation, equilibrium fixed-point",
            "Direct-forcing IBM with 4-point Peskin kernel; Guo body-force correction via D3Q27 weights.",
        ),
    },
}


def _status_for(verification: str) -> str:
    if verification == VERIFICATION_CONTRACT_TESTED:
        return WITHHELD_NO_PHYSICS_VALIDATION
    if verification == VERIFICATION_IMPLEMENTED_ONLY:
        return WITHHELD_NO_CONTRACT_TESTS
    return WITHHELD_NO_IMPLEMENTATION


def _capability_for(lattice: str, kernel: str) -> IBMCapability:
    lattice_map = _REGISTRY.get(lattice, {})
    entry = lattice_map.get(kernel)
    if entry is None:
        return IBMCapability(
            lattice=lattice, kernel=kernel,
            implementation_status=NO_IMPLEMENTATION,
            verification_level=VERIFICATION_NO_IMPLEMENTATION,
            status=WITHHELD_NO_IMPLEMENTATION,
            entrypoint=None, test_evidence=None,
            note=f"No implementation found for IBM {lattice}/{kernel}.",
        )
    impl_status, verif_level, entrypoint, test_ev, note = entry
    return IBMCapability(
        lattice=lattice, kernel=kernel,
        implementation_status=impl_status,
        verification_level=verif_level,
        status=_status_for(verif_level),
        entrypoint=entrypoint, test_evidence=test_ev, note=note,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def ibm_capability_matrix() -> dict[str, dict[str, IBMCapability]]:
    """Return the complete audited IBM lattice × kernel capability matrix.

    Every entry is fail-closed: no combination has physics validation evidence.
    Contract tests (shape/force-conservation/identity) verify operator algebra
    only.
    """
    return {
        lattice: {kernel: _capability_for(lattice, kernel) for kernel in _AUDITED_KERNELS}
        for lattice in _AUDITED_LATTICES
    }


def require_ibm_capability(
    lattice: IBMLatticeName,
    kernel: IBMKernelName,
) -> IBMCapability:
    """Return only a physics-validated IBM capability; otherwise fail closed.

    No current IBM combination has physics validation evidence.  This function
    always raises :class:`IBMWithheldError` for the physics-validated claim,
    but it validates request identities first so unsupported inputs are
    rejected with a stable, machine-readable withholding code.
    """
    if lattice not in _AUDITED_LATTICES:
        raise IBMWithheldError(
            f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited IBM lattice."
        )
    if kernel not in _AUDITED_KERNELS:
        raise IBMWithheldError(
            f"{WITHHELD_UNKNOWN_KERNEL}: {kernel!r} is not an audited IBM delta kernel."
        )
    capability = ibm_capability_matrix()[lattice][kernel]
    if not capability.available:
        raise IBMWithheldError(f"{capability.status}: {capability.note}")
    return capability


__all__ = [
    "IBMCapability",
    "IBMWithheldError",
    "WITHHELD_NO_PHYSICS_VALIDATION",
    "WITHHELD_NO_CONTRACT_TESTS",
    "WITHHELD_NO_IMPLEMENTATION",
    "WITHHELD_UNKNOWN_LATTICE",
    "WITHHELD_UNKNOWN_KERNEL",
    "VERIFICATION_CONTRACT_TESTED",
    "VERIFICATION_IMPLEMENTED_ONLY",
    "VERIFICATION_NO_IMPLEMENTATION",
    "IMPLEMENTED",
    "NO_IMPLEMENTATION",
    "ibm_capability_matrix",
    "require_ibm_capability",
]
