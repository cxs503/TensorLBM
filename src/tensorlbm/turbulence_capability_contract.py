"""Fail-closed public capability contract for turbulence models.

This module is an audit boundary, not a turbulence dispatcher.  It reports the
collision operators actually present in the repository and refuses to certify
any turbulence model combination as physics-validated.  In particular,
successful shape/mass/momentum/identity contract tests do not establish
turbulence physics correctness, spectral accuracy, or wall-bounded flow
validation.  Docstring claims of experimental validation (e.g. SUBOFF AFF-8)
are not trusted as physical verification evidence.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/turbulence.py``      – LES closures (Smagorinsky, WALE, Vreman, dynamic)
    - ``tensorlbm/core/turbulence.py`` – identity metadata only (no implementation)
    - ``tensorlbm/rans_ke.py``         – RANS k-epsilon, SA, k-omega SST
    - ``tensorlbm/ddes.py``            – DDES / SAS hybrid (2-D only)
    - ``tensorlbm/wall_model.py``      – wall-function, wall-distance FMM
    - callers: ``suboff_resistance.py``, ``turbulent_channel.py``, examples/
    - tests: ``test_marine.py``, ``test_d3q27.py``, ``test_phase4.py``,
      ``test_dynamic_smagorinsky.py``, ``test_turbulence_extensions.py``,
      ``test_turbulent_channel.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

TurbulenceFamily = Literal[
    "smagorinsky",
    "dynamic_smagorinsky",
    "wale",
    "vreman",
    "rans_ke",
    "rans_sa",
    "komega_sst",
    "ddes",
    "wall_function",
    "wall_distance",
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

WITHHELD_UNKNOWN_FAMILY = "WITHHELD_UNKNOWN_FAMILY"
WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"
WITHHELD_UNKNOWN_COLLISION = "WITHHELD_UNKNOWN_COLLISION"

# ---------------------------------------------------------------------------
# Verification levels
# ---------------------------------------------------------------------------

VERIFICATION_CONTRACT_TESTED = "CONTRACT_TESTED"
"""Shape/finite/mass/momentum/equilibrium-identity unit tests exist.

These verify operator algebra (conservation, well-formedness), NOT turbulence
physics correctness, spectral accuracy, or wall-bounded flow validation.
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
    "smagorinsky",
    "dynamic_smagorinsky",
    "wale",
    "vreman",
    "rans_ke",
    "rans_sa",
    "komega_sst",
    "ddes",
    "wall_function",
    "wall_distance",
)
_AUDITED_LATTICES: tuple[str, ...] = ("D2Q9", "D3Q19", "D3Q27")
_AUDITED_COLLISIONS: tuple[str, ...] = ("BGK", "MRT", "N/A")


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class TurbulenceWithheldError(NotImplementedError):
    """Raised when a turbulence capability request lacks physics validation."""


# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurbulenceCapability:
    """Audited state of one turbulence family/lattice/collision combination.

    ``verification_level`` records what evidence exists in the repository.
    ``status`` is the fail-closed frontend claim: it is always a ``WITHHELD_*``
    code because no current combination has physics validation evidence.
    """

    family: str
    lattice: str
    collision: str
    implementation_status: str
    verification_level: str
    status: str
    entrypoint: str | None
    test_evidence: str | None
    hot_path_note: str | None
    note: str

    @property
    def available(self) -> bool:
        """Whether a frontend may claim this combination is physics-validated."""
        return self.status == "AVAILABLE"


# ---------------------------------------------------------------------------
# Hot-path allocation audit entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HotPathAllocation:
    """Observation of a GPU→CPU sync or per-call allocation in a hot path."""

    function: str
    file: str
    line: int
    pattern: str
    severity: str  # "SYNC" or "ALLOCATION"
    note: str


# ---------------------------------------------------------------------------
# Audit registry
#
# Each entry maps (family, lattice, collision) to a tuple of:
#   (implementation_status, verification_level, entrypoint,
#    test_evidence, hot_path_note, note)
# ---------------------------------------------------------------------------

_RegistryEntry = tuple[str, str, str | None, str | None, str | None, str]

_REGISTRY: dict[str, dict[str, dict[str, _RegistryEntry]]] = {
    # -----------------------------------------------------------------------
    # Smagorinsky — 6 combinations, all CONTRACT_TESTED
    # -----------------------------------------------------------------------
    "smagorinsky": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_smagorinsky_bgk",
                "test_marine.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "Standard Smagorinsky LES; non-equilibrium stress Frobenius norm → per-cell tau_eff.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_smagorinsky_mrt",
                "test_phase4.py: shape, mass, momentum, finite",
                None,
                "D2Q9 MRT with per-cell stress relaxation rate override (modes 7, 8).",
            ),
        },
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_smagorinsky_bgk3d",
                "test_marine.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "D3Q19 BGK + Smagorinsky.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_smagorinsky_mrt3d",
                "test_marine.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "D3Q19 MRT with per-cell stress rate override (modes 9-13). Used in suboff_resistance.py.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_smagorinsky_bgk27",
                "test_d3q27.py: shape, finite, equilibrium identity",
                None,
                "D3Q27 BGK + Smagorinsky.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_smagorinsky_mrt27",
                "test_d3q27.py: mass conservation",
                None,
                "D3Q27 MRT with per-cell stress rate override (modes 5-9).",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Dynamic Smagorinsky — 2 combinations (D2Q9 BGK, D3Q19 BGK), CONTRACT_TESTED
    # -----------------------------------------------------------------------
    "dynamic_smagorinsky": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_dynamic_smagorinsky_bgk",
                "test_dynamic_smagorinsky.py: shape, finite",
                "Global Cs reduction: float(torch.sqrt(...).item()) at turbulence.py:1096 "
                "is a GPU→CPU sync point per collision step.",
                "Dynamic Smagorinsky with Germano identity; test-filter box averaging.",
            ),
        },
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_dynamic_smagorinsky_bgk3d",
                "test_dynamic_smagorinsky.py: shape",
                "Global Cs reduction: float(torch.sqrt(...).item()) at turbulence.py:1160 "
                "is a GPU→CPU sync point per collision step.",
                "D3Q19 dynamic Smagorinsky.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # WALE — BGK (D2Q9, D3Q19, D3Q27) + MRT (D3Q19, D3Q27); all CONTRACT_TESTED
    # -----------------------------------------------------------------------
    "wale": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_wale_bgk",
                "test_turbulence_extensions.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "WALE (Nicoud & Ducros 1999); velocity gradients via periodic central differences.",
            ),
        },
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_wale_bgk3d",
                "test_turbulence_extensions.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "D3Q19 BGK + WALE.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_wale_mrt3d",
                "test_turbulence_extensions.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "D3Q19 MRT with per-cell stress rate override (modes 9-13) from WALE eddy viscosity.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_wale_bgk27",
                "test_turbulence_extensions.py: shape, finite, mass, equilibrium identity",
                None,
                "D3Q27 BGK + WALE; reuses D3Q19 _wale_nu_t_3d helper.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_wale_mrt27",
                "test_turbulence_extensions.py: shape, finite, mass, equilibrium identity",
                None,
                "D3Q27 MRT with per-cell stress rate override (modes 5-9) from WALE eddy viscosity.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Vreman — BGK (D2Q9, D3Q19, D3Q27) + MRT (D3Q19, D3Q27); all CONTRACT_TESTED
    # -----------------------------------------------------------------------
    "vreman": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_vreman_bgk",
                "test_turbulence_extensions.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "Vreman (2004); eddy viscosity from velocity-gradient invariants.",
            ),
        },
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_vreman_bgk3d",
                "test_turbulence_extensions.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "D3Q19 BGK + Vreman.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_vreman_mrt3d",
                "test_turbulence_extensions.py: shape, finite, mass, momentum, equilibrium identity",
                None,
                "D3Q19 MRT with per-cell stress rate override (modes 9-13) from Vreman eddy viscosity.",
            ),
        },
        "D3Q27": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_vreman_bgk27",
                "test_turbulence_extensions.py: shape, finite, mass, equilibrium identity",
                None,
                "D3Q27 BGK + Vreman; reuses D3Q19 _vreman_nu_t_3d helper.",
            ),
            "MRT": (
                IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
                "tensorlbm.turbulence.collide_vreman_mrt27",
                "test_turbulence_extensions.py: shape, finite, mass, equilibrium identity",
                None,
                "D3Q27 MRT with per-cell stress rate override (modes 5-9) from Vreman eddy viscosity.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # RANS k-epsilon — D3Q19 MRT only; IMPLEMENTED_ONLY (no tests)
    # -----------------------------------------------------------------------
    "rans_ke": {
        "D3Q19": {
            "MRT": (
                IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
                "tensorlbm.rans_ke.collide_rans_ke + KESolver",
                None,
                "mask.bool() allocation per call (rans_ke.py:394). "
                "suboff_resistance.py:617 uses scalar nu_t.mean().item() averaging "
                "instead of per-cell collide_rans_ke, losing spatial eddy-viscosity variation.",
                "k-epsilon RANS with Strang splitting; per-cell MRT stress rate override. "
                "No unit tests; suboff_resistance.py uses a scalar-averaging workaround.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # RANS Spalart-Allmaras — D3Q19 MRT only; IMPLEMENTED_ONLY (no tests)
    # -----------------------------------------------------------------------
    "rans_sa": {
        "D3Q19": {
            "MRT": (
                IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
                "tensorlbm.rans_ke.collide_rans_sa + SASolver",
                None,
                "collide_rans_sa uses nu_t.mean().item() scalar averaging (rans_ke.py:821), "
                "losing per-cell eddy viscosity; delegates to collide_smagorinsky_mrt3d(C_s=0.0).",
                "Spalart-Allmaras one-equation RANS; requires wall-distance field. "
                "No unit tests; collision uses scalar averaging, not per-cell nu_t.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # k-omega SST — D2Q9 BGK only; IMPLEMENTED_ONLY (no tests)
    # -----------------------------------------------------------------------
    "komega_sst": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
                "tensorlbm.rans_ke.komega_sst_collision_d2q9 + KOmegaSSTSolver",
                None,
                None,
                "k-omega SST (Menter 1994) with F1/F2 blending; D2Q9 BGK only. No unit tests.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # DDES — D2Q9 BGK only; IMPLEMENTED_ONLY (no tests, no callers)
    # -----------------------------------------------------------------------
    "ddes": {
        "D2Q9": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
                "tensorlbm.ddes.apply_ddes_collision",
                None,
                None,
                "DDES/SAS hybrid (2-D only: ux, uy). All DDES helpers (_strain_rate_magnitude, "
                "_gradient_magnitude, _laplacian, ddes_eddy_viscosity, sas_source_term) are 2-D. "
                "No callers in examples or src; no unit tests.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Wall function — D3Q19 only; BENCHMARK_ONLY (examples, no unit tests)
    # -----------------------------------------------------------------------
    "wall_function": {
        "D3Q19": {
            "BGK": (
                IMPLEMENTED, VERIFICATION_BENCHMARK_ONLY,
                "tensorlbm.wall_model.wall_function_3d",
                "Examples: dg_flatplate_wallfn.py, dg_ship_wallfn.py, dg_suboff_wallfn.py. "
                "Docstring claims SUBOFF AFF-8 Re=2M Ct_total 0.0040 vs experimental 0.004, "
                "but no unit test asserts this.",
                "bool(turb.any()) GPU→CPU sync (wall_model.py:276); "
                ".sum().item() drag sync (wall_model.py:295, 299).",
                "Log-law / Reichardt wall function as Guo body force (decoupled from tau). "
                "D3Q19 only (uses macroscopic3d). No unit tests.",
            ),
        },
    },

    # -----------------------------------------------------------------------
    # Wall distance FMM — D2Q9 and D3Q19; IMPLEMENTED_ONLY (no tests)
    # -----------------------------------------------------------------------
    "wall_distance": {
        "D2Q9": {
            "N/A": (
                IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
                "tensorlbm.wall_model.compute_wall_distance_fmm_2d",
                None,
                None,
                "Iterative Eikonal (FMM-like) wall-distance solver; 2-D. No unit tests.",
            ),
        },
        "D3Q19": {
            "N/A": (
                IMPLEMENTED, VERIFICATION_IMPLEMENTED_ONLY,
                "tensorlbm.wall_model.compute_wall_distance_fmm",
                None,
                None,
                "Iterative Eikonal (FMM-like) wall-distance solver; 3-D. No unit tests.",
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Hot-path allocation audit (item/bool calls in per-step collision operators)
# ---------------------------------------------------------------------------

_HOT_PATH_AUDIT: tuple[HotPathAllocation, ...] = (
    HotPathAllocation(
        function="collide_dynamic_smagorinsky_bgk",
        file="tensorlbm/turbulence.py",
        line=1096,
        pattern="float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())",
        severity="SYNC",
        note="Global Cs reduction to Python float; GPU→CPU sync per collision step. "
             "Architecturally inherent to the dynamic procedure (single global Cs).",
    ),
    HotPathAllocation(
        function="collide_dynamic_smagorinsky_bgk3d",
        file="tensorlbm/turbulence.py",
        line=1160,
        pattern="float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())",
        severity="SYNC",
        note="Global Cs reduction to Python float; GPU→CPU sync per collision step.",
    ),
    HotPathAllocation(
        function="collide_rans_ke",
        file="tensorlbm/rans_ke.py",
        line=394,
        pattern="mask_3d = mask.bool()",
        severity="ALLOCATION",
        note="Allocates a new bool tensor every call when mask is provided. "
             "Should be pre-computed by the caller.",
    ),
    HotPathAllocation(
        function="collide_rans_sa",
        file="tensorlbm/rans_ke.py",
        line=821,
        pattern="nu_eff = nu_lam + nu_t.mean().item()",
        severity="SYNC",
        note="GPU→CPU sync for scalar averaging of per-cell eddy viscosity. "
             "Loses spatial variation; delegates to collide_smagorinsky_mrt3d(C_s=0.0).",
    ),
    HotPathAllocation(
        function="wall_function_3d",
        file="tensorlbm/wall_model.py",
        line=276,
        pattern="if bool(turb.any()):",
        severity="SYNC",
        note="GPU→CPU sync for branch decision on turbulent cell mask.",
    ),
    HotPathAllocation(
        function="wall_function_3d",
        file="tensorlbm/wall_model.py",
        line=295,
        pattern="float((tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item())",
        severity="SYNC",
        note="GPU→CPU sync for friction drag diagnostic (not collision, but per-step).",
    ),
    HotPathAllocation(
        function="wall_function_3d",
        file="tensorlbm/wall_model.py",
        line=299,
        pattern="float((p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item())",
        severity="SYNC",
        note="GPU→CPU sync for pressure drag diagnostic (not collision, but per-step).",
    ),
)


# ---------------------------------------------------------------------------
# Internal: determine fail-closed status from verification level
# ---------------------------------------------------------------------------

def _status_for(verification_level: str) -> str:
    if verification_level == VERIFICATION_NO_IMPLEMENTATION:
        return WITHHELD_NO_IMPLEMENTATION
    if verification_level == VERIFICATION_IMPLEMENTED_ONLY:
        return WITHHELD_NO_CONTRACT_TESTS
    # CONTRACT_TESTED and BENCHMARK_ONLY: implementation exists with some
    # evidence, but no physics validation.
    return WITHHELD_NO_PHYSICS_VALIDATION


# ---------------------------------------------------------------------------
# Internal: look up one capability
# ---------------------------------------------------------------------------

def _capability_for(family: str, lattice: str, collision: str) -> TurbulenceCapability:
    family_map = _REGISTRY.get(family, {})
    lattice_map = family_map.get(lattice, {})
    entry = lattice_map.get(collision)

    if entry is None:
        return TurbulenceCapability(
            family=family,
            lattice=lattice,
            collision=collision,
            implementation_status=NO_IMPLEMENTATION,
            verification_level=VERIFICATION_NO_IMPLEMENTATION,
            status=WITHHELD_NO_IMPLEMENTATION,
            entrypoint=None,
            test_evidence=None,
            hot_path_note=None,
            note=f"No implementation found for {family}/{lattice}/{collision}.",
        )

    impl_status, verif_level, entrypoint, test_ev, hot_note, note = entry
    return TurbulenceCapability(
        family=family,
        lattice=lattice,
        collision=collision,
        implementation_status=impl_status,
        verification_level=verif_level,
        status=_status_for(verif_level),
        entrypoint=entrypoint,
        test_evidence=test_ev,
        hot_path_note=hot_note,
        note=note,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def turbulence_capability_matrix() -> dict[str, dict[str, dict[str, TurbulenceCapability]]]:
    """Return the complete audited turbulence family/lattice/collision matrix.

    Every entry is fail-closed: no combination has physics validation evidence.
    Contract tests (shape/mass/momentum/identity) verify operator algebra only.
    """
    return {
        family: {
            lattice: {
                collision: _capability_for(family, lattice, collision)
                for collision in _AUDITED_COLLISIONS
            }
            for lattice in _AUDITED_LATTICES
        }
        for family in _AUDITED_FAMILIES
    }


def require_turbulence_capability(
    family: TurbulenceFamily,
    lattice: LatticeName,
    collision: CollisionName,
) -> TurbulenceCapability:
    """Return only a physics-validated capability; otherwise fail closed.

    No current turbulence model combination has physics validation evidence.
    This function always raises :class:`TurbulenceWithheldError`.

    Request identities are validated before matrix lookup, so an unsupported
    public input is always rejected with a stable, machine-readable withholding
    code rather than leaking a ``KeyError``.
    """
    if family not in _AUDITED_FAMILIES:
        raise TurbulenceWithheldError(
            f"{WITHHELD_UNKNOWN_FAMILY}: {family!r} is not an audited turbulence family."
        )
    if lattice not in _AUDITED_LATTICES:
        raise TurbulenceWithheldError(
            f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited lattice."
        )
    if collision not in _AUDITED_COLLISIONS:
        raise TurbulenceWithheldError(
            f"{WITHHELD_UNKNOWN_COLLISION}: {collision!r} is not an audited collision."
        )
    capability = turbulence_capability_matrix()[family][lattice][collision]
    if not capability.available:
        raise TurbulenceWithheldError(f"{capability.status}: {capability.note}")
    return capability


def turbulence_hot_path_audit() -> tuple[HotPathAllocation, ...]:
    """Return observations of GPU→CPU syncs and per-call allocations in hot paths.

    This is a cold-path audit of source-level patterns; it does not modify the
    numerical turbulence models.  Observations include ``.item()`` reductions,
    ``bool(...)`` branch syncs, and per-call tensor allocations in collision
    operators and per-step wall-function calls.
    """
    return _HOT_PATH_AUDIT


__all__ = [
    "TurbulenceCapability",
    "TurbulenceWithheldError",
    "HotPathAllocation",
    "WITHHELD_NO_PHYSICS_VALIDATION",
    "WITHHELD_NO_CONTRACT_TESTS",
    "WITHHELD_NO_IMPLEMENTATION",
    "WITHHELD_UNKNOWN_FAMILY",
    "WITHHELD_UNKNOWN_LATTICE",
    "WITHHELD_UNKNOWN_COLLISION",
    "VERIFICATION_CONTRACT_TESTED",
    "VERIFICATION_BENCHMARK_ONLY",
    "VERIFICATION_IMPLEMENTED_ONLY",
    "VERIFICATION_NO_IMPLEMENTATION",
    "IMPLEMENTED",
    "NO_IMPLEMENTATION",
    "turbulence_capability_matrix",
    "require_turbulence_capability",
    "turbulence_hot_path_audit",
]
