"""Fail-closed public capability contract for boundary conditions.

This module is an audit boundary, not a BC dispatcher.  It reports the
mechanics actually present in the repository and refuses to certify a
frontend physics combination until it supplies the required coupling
evidence.  In particular, successful shape/identity tests do not establish
physical accuracy, conservation, or coupled-physics correctness.

Docstring claims of physical validation (e.g. far-field Cd error) are not
trusted as evidence; only executable test evidence is admitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BoundaryKind = Literal[
    "periodic",
    "zou_he_inlet",
    "zou_he_outlet",
    "wall_bounce_back",
    "wall_free_slip",
    "farfield",
    "sponge",
    "nscbc",
    "bouzidi_interpolated",
]
LatticeName = Literal["D2Q9", "D3Q19", "D3Q27"]
CollisionFamily = Literal["bgk", "mrt", "trt", "smagorinsky", "kbc", "cascaded"]
PhysicsName = Literal["single_phase", "turbulence", "multiphase", "free_surface", "ibm"]
BackendName = Literal["torch_cpu", "torch_cuda"]

# ---------------------------------------------------------------------------
# Withholding codes
# ---------------------------------------------------------------------------

WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE = "WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE"
WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE = "WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE"
WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT = "WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT"
WITHHELD_UNKNOWN_BOUNDARY = "WITHHELD_UNKNOWN_BOUNDARY"
WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"
WITHHELD_UNKNOWN_COLLISION = "WITHHELD_UNKNOWN_COLLISION"
WITHHELD_UNKNOWN_PHYSICS = "WITHHELD_UNKNOWN_PHYSICS"
WITHHELD_UNKNOWN_BACKEND = "WITHHELD_UNKNOWN_BACKEND"

_AUDITED_BOUNDARIES: tuple[BoundaryKind, ...] = (
    "periodic", "zou_he_inlet", "zou_he_outlet", "wall_bounce_back",
    "wall_free_slip", "farfield", "sponge", "nscbc", "bouzidi_interpolated",
)
_AUDITED_LATTICES: tuple[LatticeName, ...] = ("D2Q9", "D3Q19", "D3Q27")
_AUDITED_COLLISIONS: tuple[CollisionFamily, ...] = (
    "bgk", "mrt", "trt", "smagorinsky", "kbc", "cascaded",
)
_AUDITED_PHYSICS: tuple[PhysicsName, ...] = (
    "single_phase", "turbulence", "multiphase", "free_surface", "ibm",
)
_AUDITED_BACKENDS: tuple[BackendName, ...] = ("torch_cpu", "torch_cuda")


class BoundaryConditionWithheldError(NotImplementedError):
    """Raised when a boundary-condition request lacks an audited executable contract."""


@dataclass(frozen=True)
class BoundaryConditionCapability:
    """Audited state of one boundary-kind/lattice combination.

    ``implementation_status`` describes the strongest evidence found in the
    repository for this (kind, lattice) pair, ranging from no implementation
    to executable physical validation.  ``status`` is the frontend claim and
    remains withheld for every current combination because none has a
    complete, verified composition contract (collision × physics × backend).

    Docstring assertions of physical accuracy are not admitted as
    ``PHYSICS_VALIDATED`` evidence; only executable test evidence counts.
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


# ---------------------------------------------------------------------------
# Implementation evidence registry
#
# Each entry is (implementation_status, entrypoint, verification_evidence).
# implementation_status is the strongest evidence level found by source audit:
#   NO_IMPLEMENTATION    – no code found for this (kind, lattice)
#   IMPLEMENTATION_ONLY  – code exists but no test of any kind
#   MECHANICS_TESTED     – shape/finite/unit tests exist; no physical validation
#   PHYSICS_VALIDATED    – executable physical validation test exists (e.g. Cd)
#
# Boundary conditions are applied post-streaming (or in-stream for periodic),
# so they are collision-agnostic at the implementation level.  Verification
# evidence is collision-specific but the implementation_status reflects the
# strongest evidence for any collision.
# ---------------------------------------------------------------------------

_IMPLEMENTATION_EVIDENCE: dict[tuple[str, str], tuple[str, str | None, str]] = {

    # ---- periodic ----------------------------------------------------------
    ("periodic", "D2Q9"): (
        "MECHANICS_TESTED",
        "tensorlbm.lattice.stream / tensorlbm.solver.stream (torch.roll)",
        "test_lattice.py, test_solver.py: mass conservation under periodic streaming; "
        "no periodic-flow physical validation",
    ),
    ("periodic", "D3Q19"): (
        "MECHANICS_TESTED",
        "tensorlbm.solver3d.stream3d (torch.roll)",
        "test_solver3d.py: mass conservation under periodic streaming; "
        "no periodic-flow physical validation",
    ),
    ("periodic", "D3Q27"): (
        "MECHANICS_TESTED",
        "tensorlbm.d3q27.stream27 (gather)",
        "test_d3q27.py: mass conservation under periodic streaming; "
        "no periodic-flow physical validation",
    ),

    # ---- zou_he_inlet ------------------------------------------------------
    ("zou_he_inlet", "D2Q9"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries.zou_he_inlet_velocity",
        "No dedicated unit test; test_cylinder_cd.py uses apply_simple_channel_boundaries "
        "(equilibrium inlet), not Zou-He",
    ),
    ("zou_he_inlet", "D3Q19"): (
        "MECHANICS_TESTED",
        "tensorlbm.boundaries3d.zou_he_inlet_velocity_3d / "
        "tensorlbm.wave_bc.zou_he_inlet_velocity_profile_3d",
        "test_full_wet.py (regression), test_marine.py (profile shape/finite); "
        "no physical validation of Zou-He accuracy",
    ),
    ("zou_he_inlet", "D3Q27"): (
        "MECHANICS_TESTED",
        "tensorlbm.boundaries_d3q27.zou_he_inlet_velocity_27",
        "test_d3q27.py: velocity prescription (ux matches prescribed), finite output; "
        "no physical validation",
    ),

    # ---- zou_he_outlet -----------------------------------------------------
    ("zou_he_outlet", "D2Q9"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries.zou_he_outlet_pressure",
        "No dedicated unit test",
    ),
    ("zou_he_outlet", "D3Q19"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries3d.zou_he_outlet_pressure_3d / zou_he_outlet_pressure_z",
        "Used in test_full_wet.py and test_marine.py via apply_zou_he_channel_boundaries_3d / "
        "apply_wave_inlet_3d; no dedicated outlet-pressure test",
    ),
    ("zou_he_outlet", "D3Q27"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries_d3q27.zou_he_outlet_pressure_27",
        "Used in test_d3q27.py via apply_zou_he_channel_boundaries_27; "
        "no dedicated outlet-pressure test",
    ),

    # ---- wall_bounce_back --------------------------------------------------
    ("wall_bounce_back", "D2Q9"): (
        "PHYSICS_VALIDATED",
        "tensorlbm.boundaries.bounce_back_cells",
        "test_cylinder_cd.py: Cd validation (2.0 < Cd < 8.0 at Re=100, "
        "1.0 < Cd < 4.0 at Re=200); very loose tolerance, BGK only",
    ),
    ("wall_bounce_back", "D3Q19"): (
        "PHYSICS_VALIDATED",
        "tensorlbm.boundaries3d.bounce_back_cells_3d",
        "test_sphere_cd.py: Cd validation (err < 120–150%); "
        "very loose tolerance, BGK only",
    ),
    ("wall_bounce_back", "D3Q27"): (
        "MECHANICS_TESTED",
        "tensorlbm.boundaries_d3q27.bounce_back_cells_27 / "
        "tensorlbm.d3q27.moving_wall_linkwise_me_force_torque",
        "test_d3q27.py: shape preservation; "
        "test_d3q27_moving_wall_momentum_exchange.py: ME force unit test; "
        "no physical validation",
    ),

    # ---- wall_free_slip ----------------------------------------------------
    ("wall_free_slip", "D2Q9"): (
        "NO_IMPLEMENTATION",
        None,
        "No free-slip implementation for D2Q9",
    ),
    ("wall_free_slip", "D3Q19"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries3d.free_slip_cells_3d / free_slip_y_walls_3d / "
        "free_slip_x_walls_3d / free_slip_z_walls_3d",
        "No tests; docstring references waLBerla FreeSlip pattern",
    ),
    ("wall_free_slip", "D3Q27"): (
        "NO_IMPLEMENTATION",
        None,
        "No free-slip implementation for D3Q27",
    ),

    # ---- farfield ----------------------------------------------------------
    ("farfield", "D2Q9"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries.far_field_bc_2d",
        "No tests; docstring claims ~9% Cd error but this is not a test and "
        "is not trusted as validation evidence",
    ),
    ("farfield", "D3Q19"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries3d.far_field_bc_3d",
        "No tests; docstring claims ~9% Cd error (channel ~65% → far-field ~9%) "
        "but this is not a test and is not trusted as validation evidence",
    ),
    ("farfield", "D3Q27"): (
        "NO_IMPLEMENTATION",
        None,
        "No far-field BC implementation for D3Q27",
    ),

    # ---- sponge ------------------------------------------------------------
    ("sponge", "D2Q9"): (
        "MECHANICS_TESTED",
        "tensorlbm.sponge_bc.apply_viscous_sponge_2d / apply_target_sponge_2d / sponge_profile",
        "test_gap_improvements.py: profile shape, damping behavior, shape conservation; "
        "no physical validation of wave absorption",
    ),
    ("sponge", "D3Q19"): (
        "MECHANICS_TESTED",
        "tensorlbm.sponge_bc.apply_viscous_sponge_3d(lattice='D3Q19') / apply_target_sponge_3d",
        "test_gap_improvements.py: 3D target sponge no-damping test; "
        "no physical validation of wave absorption",
    ),
    ("sponge", "D3Q27"): (
        "MECHANICS_TESTED",
        "tensorlbm.sponge_bc.apply_viscous_sponge_3d(lattice='D3Q27') / apply_target_sponge_3d",
        "test_gap_improvements.py: 3D target sponge no-damping test (lattice-agnostic); "
        "no physical validation of wave absorption",
    ),

    # ---- nscbc -------------------------------------------------------------
    ("nscbc", "D2Q9"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries.nscbc_outlet_2d",
        "No tests; simplified single-characteristic relaxation, not full NSCBC",
    ),
    ("nscbc", "D3Q19"): (
        "IMPLEMENTATION_ONLY",
        "tensorlbm.boundaries3d.nscbc_outlet_3d",
        "No tests; simplified single-characteristic relaxation, not full NSCBC",
    ),
    ("nscbc", "D3Q27"): (
        "NO_IMPLEMENTATION",
        None,
        "No NSCBC implementation for D3Q27",
    ),

    # ---- bouzidi_interpolated ----------------------------------------------
    ("bouzidi_interpolated", "D2Q9"): (
        "MECHANICS_TESTED",
        "tensorlbm.interpolated_bc.bouzidi_bounce_back / compute_q_circle",
        "test_interpolated_bc.py: shape, finite, halfway-q=standard BB, "
        "linear/quad branch; no physical validation",
    ),
    ("bouzidi_interpolated", "D3Q19"): (
        "MECHANICS_TESTED",
        "tensorlbm.interpolated_bc.bouzidi_bounce_back_3d / compute_q_sphere / "
        "tensorlbm.interpolated_bc_ellipsoid.compute_q_ellipsoid",
        "test_interpolated_bc.py: shape, finite, halfway-q, compute_q_sphere; "
        "sphere_bouzidi.py benchmark reports ~13% Cd error but is NOT a test; "
        "no executable physical validation",
    ),
    ("bouzidi_interpolated", "D3Q27"): (
        "NO_IMPLEMENTATION",
        None,
        "No Bouzidi interpolated bounce-back implementation for D3Q27",
    ),
}


def _capability_for(kind: str, lattice: str) -> BoundaryConditionCapability:
    """Return the audited capability for one (boundary_kind, lattice) pair."""
    detail = _IMPLEMENTATION_EVIDENCE.get((kind, lattice))
    if detail is None:
        return BoundaryConditionCapability(
            "NO_IMPLEMENTATION", WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE, None,
            "No implementation evidence registered for this combination.",
            f"{kind}/{lattice} is not in the audited implementation registry.",
        )
    impl_status, entrypoint, evidence = detail

    if impl_status == "NO_IMPLEMENTATION":
        return BoundaryConditionCapability(
            impl_status, WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE, entrypoint, evidence,
            f"{kind} has no audited {lattice} implementation.",
        )

    # Implementation exists (at any evidence level) but no complete composition
    # contract has been verified for any collision × physics × backend.
    return BoundaryConditionCapability(
        impl_status, WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE, entrypoint, evidence,
        "Implementation exists, but no complete collision × physics × backend composition "
        "has been verified as a production-ready contract. Mechanics or physics tests "
        "do not establish composition correctness.",
    )


def boundary_capability_matrix() -> dict[str, dict[str, BoundaryConditionCapability]]:
    """Return the complete audited boundary-kind × lattice capability matrix."""
    return {
        kind: {
            lattice: _capability_for(kind, lattice)
            for lattice in _AUDITED_LATTICES
        }
        for kind in _AUDITED_BOUNDARIES
    }


def require_boundary_condition_capability(
    kind: str,
    lattice: str,
    collision: str,
    physics: str,
    backend: str,
) -> BoundaryConditionCapability:
    """Return only an executable capability; otherwise fail closed.

    Request identities are validated before implementation or matrix lookup,
    so an unsupported public input is always rejected with a stable,
    machine-readable withholding code rather than leaking a ``KeyError``.
    Even a PHYSICS_VALIDATED implementation cannot be claimed as a complete
    composition contract: the caller must establish collision × physics ×
    backend evidence that is emitted by the selected runtime.
    """
    if kind not in _AUDITED_BOUNDARIES:
        raise BoundaryConditionWithheldError(
            f"{WITHHELD_UNKNOWN_BOUNDARY}: {kind!r} is not an audited boundary kind."
        )
    if lattice not in _AUDITED_LATTICES:
        raise BoundaryConditionWithheldError(
            f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited lattice."
        )
    if collision not in _AUDITED_COLLISIONS:
        raise BoundaryConditionWithheldError(
            f"{WITHHELD_UNKNOWN_COLLISION}: {collision!r} is not an audited collision family."
        )
    if physics not in _AUDITED_PHYSICS:
        raise BoundaryConditionWithheldError(
            f"{WITHHELD_UNKNOWN_PHYSICS}: {physics!r} is not an audited physics selection."
        )
    if backend not in _AUDITED_BACKENDS:
        raise BoundaryConditionWithheldError(
            f"{WITHHELD_UNKNOWN_BACKEND}: {backend!r} is not an audited backend."
        )

    capability = boundary_capability_matrix()[kind][lattice]

    # A missing implementation is reported before the physics-coupling check.
    if capability.implementation_status == "NO_IMPLEMENTATION":
        raise BoundaryConditionWithheldError(
            f"{capability.status}: {capability.note}"
        )

    # Non-single-phase physics has no audited BC coupling contract.
    if physics != "single_phase":
        raise BoundaryConditionWithheldError(
            f"{WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT}: {physics!r} has no audited "
            f"boundary-condition coupling contract for {kind}/{lattice}."
        )

    if not capability.available:
        raise BoundaryConditionWithheldError(
            f"{capability.status}: {capability.note}"
        )
    return capability


__all__ = [
    "BoundaryConditionCapability",
    "BoundaryConditionWithheldError",
    "WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE",
    "WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT",
    "WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE",
    "WITHHELD_UNKNOWN_BACKEND",
    "WITHHELD_UNKNOWN_BOUNDARY",
    "WITHHELD_UNKNOWN_COLLISION",
    "WITHHELD_UNKNOWN_LATTICE",
    "WITHHELD_UNKNOWN_PHYSICS",
    "boundary_capability_matrix",
    "require_boundary_condition_capability",
]
