"""Fail-closed public capability contract for physics modules.

This module is an audit boundary for the four composable physics modules:
thermal, porous media, non-Newtonian, and passive scalar.  It reports the
operators actually present in the repository and refuses to certify any
combination as physics-validated without evidence.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/thermal_common.py``       – DDF thermal step (D3Q7 + D3Q19/D3Q27)
    - ``tensorlbm/porous_media_common.py``  – partial bounce-back porous media
    - ``tensorlbm/non_newtonian_common.py`` – power-law / Carreau / Bingham tau_eff
    - ``tensorlbm/passive_scalar_common.py``– D3Q7 passive scalar transport
    - callers: ``thermal.py``, ``thermal3d.py``, ``conjugate_ht.py``,
      ``passive_scalar.py``, ``non_newtonian.py``, ``porous_media.py``,
      ``porous_media3d.py``
    - tests: ``test_thermal_common.py``, ``test_porous_media_common.py``,
      ``test_non_newtonian_common.py``, ``test_passive_scalar_common.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

PhysicsFamily = Literal[
    "thermal",
    "conjugate_ht",
    "porous_media",
    "non_newtonian",
    "passive_scalar",
]
LatticeName = Literal["D2Q9", "D3Q19", "D3Q27"]
CollisionName = Literal["BGK", "MRT", "N/A"]

# ---------------------------------------------------------------------------
# Machine-readable withheld codes (fail-closed)
# ---------------------------------------------------------------------------

WITHHELD_NO_PHYSICS_VALIDATION = "WITHHELD_NO_PHYSICS_VALIDATION"
"""Implementation exists with contract tests, but no physics validation evidence."""

WITHHELD_NO_CONTRACT_TESTS = "WITHHELD_NO_CONTRACT_TESTS"
"""Implementation exists but no unit tests verify even operator algebra."""

WITHHELD_NO_IMPLEMENTATION = "WITHHELD_NO_IMPLEMENTATION"
"""No implementation found for this family/lattice/collision combination."""

# ---------------------------------------------------------------------------
# Verification levels
# ---------------------------------------------------------------------------

VERIFICATION_CONTRACT_TESTED = "CONTRACT_TESTED"
"""Shape/finite/mass/momentum/identity unit tests exist.

These verify operator algebra (conservation, well-formedness), NOT physics
correctness, spectral accuracy, or benchmark validation.
"""

VERIFICATION_BENCHMARK_ONLY = "BENCHMARK_ONLY"
"""Used in examples or benchmarks, but no unit tests assert correctness."""

VERIFICATION_IMPLEMENTED_ONLY = "IMPLEMENTED_ONLY"
"""Implementation exists in source, but no tests or benchmarks exercise it."""

VERIFICATION_NO_IMPLEMENTATION = "NO_IMPLEMENTATION"
"""No implementation found for this combination."""

# ---------------------------------------------------------------------------
# Implementation statuses
# ---------------------------------------------------------------------------

IMPLEMENTED = "IMPLEMENTED"
NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

# ---------------------------------------------------------------------------
# Audited dimensions
# ---------------------------------------------------------------------------

_AUDITED_FAMILIES: tuple[str, ...] = (
    "thermal",
    "conjugate_ht",
    "porous_media",
    "non_newtonian",
    "passive_scalar",
)
_AUDITED_LATTICES: tuple[str, ...] = ("D2Q9", "D3Q19", "D3Q27")
_AUDITED_COLLISIONS: tuple[str, ...] = ("BGK", "MRT", "N/A")


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class PhysicsWithheldError(NotImplementedError):
    """Raised when a physics capability request lacks physics validation."""


# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicsCapability:
    """Audited state of one physics family/lattice/collision combination.

    ``verification_level`` records what evidence exists in the repository.
    ``status`` is the fail-closed frontend claim.
    """

    family: str
    lattice: str
    collision: str
    implementation_status: str
    verification_level: str
    status: str
    entrypoint: str | None
    test_evidence: str | None
    note: str

    @property
    def available(self) -> bool:
        """Whether a frontend may claim this combination is physics-validated."""
        return self.status == "AVAILABLE"


# ---------------------------------------------------------------------------
# Audit registry
#
# Each entry maps (family, lattice, collision) to a tuple of:
#   (implementation_status, verification_level, entrypoint,
#    test_evidence, note)
# ---------------------------------------------------------------------------

_RegistryEntry = tuple[str, str, str | None, str | None, str]

_REGISTRY: dict[str, dict[str, dict[str, _RegistryEntry]]] = {
    # -----------------------------------------------------------------------
    # Thermal — DDF D3Q7 thermal lattice + D3Q19/D3Q27 momentum
    # -----------------------------------------------------------------------
    "thermal": {
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step",
                "test_thermal_common.py: shape, finite, temperature conservation, "
                "equilibrium identity, buoyancy, CHT",
                "DDF thermal LBM: D3Q7 temperature lattice + D3Q19 momentum. "
                "Buoyancy (Boussinesq) coupling via Guo force scheme.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step (momentum MRT is caller's choice)",
                "test_thermal_common.py: thermal_step composes with any collision",
                "Thermal step is collision-agnostic; caller provides MRT-collided f. "
                "Thermal lattice always uses BGK on D3Q7.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step",
                "test_thermal_common.py: shape, finite, temperature conservation, "
                "equilibrium identity, buoyancy",
                "DDF thermal LBM: D3Q7 temperature lattice + D3Q27 momentum. "
                "Buoyancy coupling supports D3Q27 via apply_buoyancy_3d.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step (momentum MRT is caller's choice)",
                "test_thermal_common.py: thermal_step composes with any collision",
                "Thermal step is collision-agnostic; caller provides MRT-collided f.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Conjugate heat transfer — fluid-solid interface
    # -----------------------------------------------------------------------
    "conjugate_ht": {
        "D3Q19": {
            "N/A": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.conjugate_ht_step",
                "test_thermal_common.py: CHT shape, interface coupling, "
                "solid diffusion, heat conservation",
                "Conjugate HT: explicit Euler solid diffusion + harmonic-mean "
                "interface coupling. Works with any momentum lattice.",
            ),
        },
        "D3Q27": {
            "N/A": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.conjugate_ht_step",
                "test_thermal_common.py: CHT shape, interface coupling, "
                "solid diffusion, heat conservation",
                "Conjugate HT: explicit Euler solid diffusion + harmonic-mean "
                "interface coupling. Works with any momentum lattice.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Porous media — partial bounce-back (Dardis & McCloskey)
    # -----------------------------------------------------------------------
    "porous_media": {
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.porous_media_common.porous_media_step",
                "test_porous_media_common.py: shape, mass conservation, "
                "porosity=1 no-op, porosity=0 full bounce-back",
                "Partial bounce-back porous media: f* = ε·f + (1−ε)·f_opp. "
                "Collision-agnostic; call after any collision.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.porous_media_common.porous_media_step",
                "test_porous_media_common.py: shape, mass conservation",
                "Partial bounce-back is collision-agnostic; works with MRT.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.porous_media_common.porous_media_step",
                "test_porous_media_common.py: shape, mass conservation, "
                "porosity=1 no-op, porosity=0 full bounce-back",
                "Partial bounce-back porous media for D3Q27.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.porous_media_common.porous_media_step",
                "test_porous_media_common.py: shape, mass conservation",
                "Partial bounce-back is collision-agnostic; works with MRT.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Non-Newtonian — power-law / Carreau / Bingham tau_eff
    # -----------------------------------------------------------------------
    "non_newtonian": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.non_newtonian_common.non_newtonian_tau_eff",
                "test_non_newtonian_common.py: tau_eff per-cell, >0.5, "
                "power-law, Carreau, Bingham, zero-shear baseline",
                "Non-Newtonian tau_eff via _nu_t_to_tau_eff (same as RANS/LES). "
                "2-D strain rate from velocity gradients.",
            ),
        },
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.non_newtonian_common.non_newtonian_tau_eff",
                "test_non_newtonian_common.py: tau_eff per-cell, >0.5, "
                "power-law, Carreau, Bingham, zero-shear baseline",
                "Non-Newtonian tau_eff via _nu_t_to_tau_eff (same as RANS/LES). "
                "3-D strain rate from velocity gradients.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.non_newtonian_common.non_newtonian_tau_eff",
                "test_non_newtonian_common.py: tau_eff per-cell, >0.5",
                "tau_eff is collision-agnostic; caller uses it in MRT stress modes.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.non_newtonian_common.non_newtonian_tau_eff",
                "test_non_newtonian_common.py: tau_eff per-cell, >0.5, "
                "power-law, Carreau, Bingham, zero-shear baseline",
                "Non-Newtonian tau_eff for D3Q27 via _nu_t_to_tau_eff.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.non_newtonian_common.non_newtonian_tau_eff",
                "test_non_newtonian_common.py: tau_eff per-cell, >0.5",
                "tau_eff is collision-agnostic; caller uses it in MRT stress modes.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Passive scalar — D3Q7 advection-diffusion
    # -----------------------------------------------------------------------
    "passive_scalar": {
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, finite, scalar conservation, "
                "equilibrium identity, source term",
                "D3Q7 passive scalar: BGK collision + periodic streaming. "
                "Velocity from D3Q19 momentum distribution.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, scalar conservation",
                "Scalar step is collision-agnostic; caller provides MRT-collided f.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, finite, scalar conservation, "
                "equilibrium identity, source term",
                "D3Q7 passive scalar with D3Q27 momentum.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, scalar conservation",
                "Scalar step is collision-agnostic; caller provides MRT-collided f.",
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------

def physics_capability_matrix() -> list[PhysicsCapability]:
    """Return the full audited capability matrix as a list of dataclasses."""
    result: list[PhysicsCapability] = []
    for family in _AUDITED_FAMILIES:
        lattice_map = _REGISTRY.get(family, {})
        for lattice in _AUDITED_LATTICES:
            collision_map = lattice_map.get(lattice, {})
            for collision in _AUDITED_COLLISIONS:
                entry = collision_map.get(collision)
                if entry is None:
                    result.append(PhysicsCapability(
                        family=family,
                        lattice=lattice,
                        collision=collision,
                        implementation_status=NO_IMPLEMENTATION,
                        verification_level=VERIFICATION_NO_IMPLEMENTATION,
                        status=WITHHELD_NO_IMPLEMENTATION,
                        entrypoint=None,
                        test_evidence=None,
                        note="No implementation found for this combination.",
                    ))
                else:
                    impl, verif, entrypoint, evidence, note = entry
                    result.append(PhysicsCapability(
                        family=family,
                        lattice=lattice,
                        collision=collision,
                        implementation_status=impl,
                        verification_level=verif,
                        status=WITHHELD_NO_PHYSICS_VALIDATION,
                        entrypoint=entrypoint,
                        test_evidence=evidence,
                        note=note,
                    ))
    return result


def require_physics_capability(
    family: str,
    lattice: str,
    collision: str = "BGK",
) -> PhysicsCapability:
    """Look up a capability entry; raise if not found.

    Args:
        family:    Physics family name.
        lattice:   Lattice name (e.g. ``"D3Q19"``).
        collision: Collision name (e.g. ``"BGK"``).

    Returns:
        The :class:`PhysicsCapability` for the requested combination.

    Raises:
        PhysicsWithheldError: If the combination is not in the registry.
    """
    lattice_map = _REGISTRY.get(family)
    if lattice_map is None:
        raise PhysicsWithheldError(f"Unknown physics family: {family!r}")
    collision_map = lattice_map.get(lattice)
    if collision_map is None:
        raise PhysicsWithheldError(
            f"Family {family!r} has no entry for lattice {lattice!r}"
        )
    entry = collision_map.get(collision)
    if entry is None:
        raise PhysicsWithheldError(
            f"Family {family!r} / lattice {lattice!r} has no entry for "
            f"collision {collision!r}"
        )
    impl, verif, entrypoint, evidence, note = entry
    return PhysicsCapability(
        family=family,
        lattice=lattice,
        collision=collision,
        implementation_status=impl,
        verification_level=verif,
        status=WITHHELD_NO_PHYSICS_VALIDATION,
        entrypoint=entrypoint,
        test_evidence=evidence,
        note=note,
    )
