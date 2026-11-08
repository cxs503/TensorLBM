"""Fail-closed public capability contract for physics modules.

This module is an audit boundary for the four composable physics modules:
thermal, porous media, non-Newtonian, and passive scalar.  It reports the
operators actually present in the repository and refuses to certify any
combination as physics-validated without evidence.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/thermal_common.py``       – DDF thermal step (D3Q7 + D3Q19/D3Q27)
    - ``tensorlbm/non_newtonian.py``        – 2-D power-law ν(γ̇) / BGK collision
    - ``tensorlbm/herschel_bulkley_lbm.py`` – 3-D D3Q19 variable-viscosity BGK
                                              (Papanastasiou-regularised H-B)
    - ``tensorlbm/passive_scalar_common.py``– D3Q7 passive scalar transport
    - callers: ``thermal.py``, ``thermal3d.py``, ``conjugate_ht.py``,
      ``passive_scalar.py``, ``non_newtonian.py``
    - tests: ``test_thermal_common.py``, ``test_non_newtonian.py``,
      ``test_herschel_bulkley.py``, ``test_passive_scalar_common.py``

Repair (2026-11-08, mission 6598): this docstring used to name
``porous_media_common.py`` and ``non_newtonian_common.py``, and the registry
pointed at ``porous_media_step`` / ``non_newtonian_tau_eff`` in those modules
plus ``test_porous_media_common.py`` / ``test_non_newtonian_common.py``.  **None
of those four files ever existed in this repository's history** (``git log
--all -- src/tensorlbm/porous_media_common.py`` is empty), so 9 of the 45 cells
asserted an implementation and a test suite that could not be resolved — the
capability contract was inflating the library.  The gate
(:mod:`tensorlbm.capability_contract_drift`) could not see it because a
list-returning matrix was invisible to its traversal.  The cells now point at
the mechanisms that do exist, and the ones with no mechanism are downgraded to
``NO_IMPLEMENTATION``; see the per-cell notes.
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
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step",
                "test_thermal_common.py: shape, finite, temperature conservation, "
                "equilibrium identity, buoyancy, CHT",
                "DDF thermal LBM: D3Q7 temperature lattice + D3Q19 momentum. "
                "Buoyancy (Boussinesq) coupling via Guo force scheme.",
            ),
            "MRT": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step (momentum MRT is caller's choice)",
                "test_thermal_common.py: thermal_step composes with any collision",
                "Thermal step is collision-agnostic; caller provides MRT-collided f. "
                "Thermal lattice always uses BGK on D3Q7.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.thermal_step",
                "test_thermal_common.py: shape, finite, temperature conservation, "
                "equilibrium identity, buoyancy",
                "DDF thermal LBM: D3Q7 temperature lattice + D3Q27 momentum. "
                "Buoyancy coupling supports D3Q27 via apply_buoyancy_3d.",
            ),
            "MRT": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
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
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.conjugate_ht_step",
                "test_thermal_common.py: CHT shape, interface coupling, "
                "solid diffusion, heat conservation",
                "Conjugate HT: explicit Euler solid diffusion + harmonic-mean "
                "interface coupling. Works with any momentum lattice.",
            ),
        },
        "D3Q27": {
            "N/A": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.thermal_common.conjugate_ht_step",
                "test_thermal_common.py: CHT shape, interface coupling, "
                "solid diffusion, heat conservation",
                "Conjugate HT: explicit Euler solid diffusion + harmonic-mean "
                "interface coupling. Works with any momentum lattice.",
            ),
        },
    },
    # -----------------------------------------------------------------------
    # Porous media — NO partial-bounce-back flow step (see note)
    # -----------------------------------------------------------------------
    # ``porous_media.py`` / ``porous_media3d.py`` implement medium *generation*
    # (random cylinders/spheres, tube arrays) and pore-scale *workflows*
    # (Laplace pressure, capillary invasion, two-phase Poiseuille, drainage)
    # that build on the Shan-Chen multiphase solver; ``porous_weak_inertia.py``
    # implements a Forchheimer-prefix drag closure.  None of them is a generic
    # porosity-weighted (partial bounce-back) flow step, which is the mechanism
    # these four cells used to claim.
    "porous_media": {
        "D3Q19": {
            "BGK": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "No partial bounce-back / porosity-weighted step exists. "
                "Medium generation and pore-scale drainage workflows do exist "
                "(tensorlbm.porous_media, tensorlbm.porous_media3d) but they "
                "are not a reusable porous-flow kernel. Use Shan-Chen with a "
                "solid mask, or implement f* = ε·f + (1−ε)·f_opp.",
            ),
            "MRT": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "No partial bounce-back / porosity-weighted step exists "
                "(collision-agnostic or otherwise).",
            ),
        },
        "D3Q27": {
            "BGK": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "No partial bounce-back / porosity-weighted step exists for "
                "D3Q27; medium generation is 2-D (porous_media) or 3-D "
                "D3Q19 (porous_media3d) only.",
            ),
            "MRT": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "No partial bounce-back / porosity-weighted step exists for "
                "D3Q27.",
            ),
        },
    },
    # -----------------------------------------------------------------------
    # Non-Newtonian — power-law (2-D) and regularised Herschel-Bulkley (3-D)
    # -----------------------------------------------------------------------
    # Two real mechanisms exist, and they are *not* interchangeable across the
    # lattice dimension:
    #   * 2-D: ``non_newtonian.apparent_viscosity_power_law`` (ν = K γ̇^{n-1})
    #     + ``collide_power_law_bgk``, tested by ``test_non_newtonian.py``;
    #   * 3-D D3Q19: ``herschel_bulkley_lbm.collide_hb_bgk_forced_3d`` with
    #     ``shear_rate_3d`` / ``hb_relaxation_time`` (Papanastasiou-regularised
    #     H-B, BGK only), tested by ``test_herschel_bulkley.py``.
    # Both are BGK; no variable-viscosity MRT kernel and no D3Q27 kernel exist.
    "non_newtonian": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.non_newtonian.apparent_viscosity_power_law "
                "/ tensorlbm.non_newtonian.collide_power_law_bgk",
                "test_non_newtonian.py: shear-rate magnitude of a linear shear "
                "field, power-law ν shear-thinning, ν_min/ν_max clamping, "
                "invalid-parameter rejection, collision identity at equilibrium",
                "2-D power-law (Ostwald-de Waele) ν = K·γ̇^(n−1) with optional "
                "clamping, used as the per-cell relaxation time in a "
                "variable-viscosity BGK collision. No Carreau or Bingham branch "
                "in this module (yield stress lives in "
                "tensorlbm.herschel_bulkley_lbm, D3Q19 only).",
            ),
        },
        "D3Q19": {
            "BGK": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.herschel_bulkley_lbm.collide_hb_bgk_forced_3d "
                "/ tensorlbm.herschel_bulkley_lbm.shear_rate_3d",
                "test_herschel_bulkley.py: shear_rate_3d exact on a linear "
                "field, regularised reference profile, plug probe, D3Q19 pipe "
                "smoke, sub-yield channel does not flow, channel matches "
                "Newtonian Poiseuille",
                "3-D D3Q19 variable-viscosity BGK with "
                "Papanastasiou-regularised Herschel-Bulkley viscosity, Guo "
                "forcing with a per-cell relaxation time, half-way bounce-back "
                "walls; end-to-end runners run_hb_pipe / run_hb_channel.",
            ),
            "MRT": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "Every non-Newtonian kernel in this package is BGK "
                "(non_newtonian.collide_power_law_bgk, "
                "herschel_bulkley_lbm.collide_hb_bgk_forced_{2,3}d). There is no "
                "variable-viscosity MRT collision, so 'tau_eff is "
                "collision-agnostic, caller uses it in MRT stress modes' was not "
                "backed by any code.",
            ),
        },
        "D3Q27": {
            "BGK": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "The only 3-D non-Newtonian kernel is D3Q19 "
                "(herschel_bulkley_lbm, opposite-direction table built for 19 "
                "velocities); the 2-D power-law path is D2Q9-only.",
            ),
            "MRT": (
                NO_IMPLEMENTATION,
                VERIFICATION_NO_IMPLEMENTATION,
                None,
                None,
                "No D3Q27 non-Newtonian kernel and no MRT non-Newtonian kernel.",
            ),
        },
    },
    # -----------------------------------------------------------------------
    # Passive scalar — D3Q7 advection-diffusion
    # -----------------------------------------------------------------------
    "passive_scalar": {
        "D3Q19": {
            "BGK": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, finite, scalar conservation, "
                "equilibrium identity, source term",
                "D3Q7 passive scalar: BGK collision + periodic streaming. "
                "Velocity from D3Q19 momentum distribution.",
            ),
            "MRT": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, scalar conservation",
                "Scalar step is collision-agnostic; caller provides MRT-collided f.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.passive_scalar_common.passive_scalar_step",
                "test_passive_scalar_common.py: shape, finite, scalar conservation, "
                "equilibrium identity, source term",
                "D3Q7 passive scalar with D3Q27 momentum.",
            ),
            "MRT": (
                IMPLEMENTED,
                VERIFICATION_CONTRACT_TESTED,
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
                    result.append(
                        PhysicsCapability(
                            family=family,
                            lattice=lattice,
                            collision=collision,
                            implementation_status=NO_IMPLEMENTATION,
                            verification_level=VERIFICATION_NO_IMPLEMENTATION,
                            status=WITHHELD_NO_IMPLEMENTATION,
                            entrypoint=None,
                            test_evidence=None,
                            note="No implementation found for this combination.",
                        )
                    )
                else:
                    impl, verif, entrypoint, evidence, note = entry
                    result.append(
                        PhysicsCapability(
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
                    )
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
        raise PhysicsWithheldError(f"Family {family!r} has no entry for lattice {lattice!r}")
    entry = collision_map.get(collision)
    if entry is None:
        raise PhysicsWithheldError(
            f"Family {family!r} / lattice {lattice!r} has no entry for collision {collision!r}"
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
