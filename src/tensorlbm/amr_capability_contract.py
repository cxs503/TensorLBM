"""Fail-closed public capability contract for local refinement / AMR.

This module is an audit boundary, not an AMR dispatcher.  It reports the
mechanics actually present in the repository and refuses to certify a frontend
physics combination until it supplies the required coupling evidence.  In
particular, successful shape/identity tests do not establish conservation,
interface accuracy, or coupled-physics correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

LatticeName = Literal["D2Q9", "D3Q19", "D3Q27"]
PhysicsName = Literal["single_phase", "turbulence", "multiphase", "ibm", "curved_wall"]
RefinementPath = Literal["adaptive_dynamic", "multigrid_static", "surface_shell", "multipatch_static"]

REQUIRED_FRONTEND_METADATA = (
    "subcycling",
    "ratio",
    "exchange_scheme",
    "geometry_remesh_provenance",
    "flux_inventory_ledger",
    "refinement_decision_evidence",
)

WITHHELD_REQUIRED_METADATA_NOT_EMITTED = "WITHHELD_REQUIRED_METADATA_NOT_EMITTED"
WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT = "WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT"
WITHHELD_NO_D3Q27_LOCAL_REFINEMENT = "WITHHELD_NO_D3Q27_LOCAL_REFINEMENT"
WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE = "WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE"
WITHHELD_UNKNOWN_PATH = "WITHHELD_UNKNOWN_PATH"
WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"
WITHHELD_UNKNOWN_PHYSICS = "WITHHELD_UNKNOWN_PHYSICS"

_AUDITED_PATHS: tuple[RefinementPath, ...] = (
    "adaptive_dynamic", "multigrid_static", "surface_shell", "multipatch_static",
)
_AUDITED_LATTICES: tuple[LatticeName, ...] = ("D2Q9", "D3Q19", "D3Q27")
_AUDITED_PHYSICS: tuple[PhysicsName, ...] = (
    "single_phase", "turbulence", "multiphase", "ibm", "curved_wall",
)


class LocalRefinementWithheldError(NotImplementedError):
    """Raised when a local-refinement request lacks an audited executable contract."""


@dataclass(frozen=True)
class LocalRefinementCapability:
    """Audited state of one refinement-path/lattice/physics combination.

    ``mechanics_status`` only describes source-level patch mechanics.  ``status``
    is the frontend claim and remains withheld for every current path because
    none emit the complete metadata/evidence contract.
    """

    mechanics_status: str
    status: str
    entrypoint: str | None
    exchange_scheme: str | None
    note: str

    @property
    def available(self) -> bool:
        """Whether a frontend may claim this combination is contract-ready."""
        return self.status == "AVAILABLE"


def _capability_for(path: RefinementPath, lattice: LatticeName, physics: PhysicsName) -> LocalRefinementCapability:
    if lattice == "D3Q27":
        return LocalRefinementCapability(
            "NO_IMPLEMENTATION", WITHHELD_NO_D3Q27_LOCAL_REFINEMENT, None, None,
            "No D3Q27 local-refinement/AMR solver or exchange implementation was found.",
        )

    mechanics: dict[tuple[RefinementPath, LatticeName], tuple[str, str, str]] = {
        ("adaptive_dynamic", "D2Q9"): (
            "AVAILABLE_MECHANICS_ONLY", "tensorlbm.adaptive_refinement.AdaptiveSolver2D",
            "FH helper (specific adaptive path; otherwise bilinear/block-average)",
        ),
        ("adaptive_dynamic", "D3Q19"): (
            "AVAILABLE_MECHANICS_ONLY", "tensorlbm.adaptive_refinement.AdaptiveSolver3D",
            "FH helper (specific adaptive path; otherwise trilinear/block-average)",
        ),
        ("multigrid_static", "D3Q19"): (
            "AVAILABLE_MECHANICS_ONLY", "tensorlbm.refinement.MultiGridSolver",
            "plain trilinear interpolation/block-average restriction",
        ),
        ("surface_shell", "D3Q19"): (
            "AVAILABLE_MECHANICS_ONLY", "tensorlbm.surface_refinement.SurfaceRefinementSolver",
            "plain interpolation/block-average via refinement/multipatch helpers",
        ),
        ("multipatch_static", "D3Q19"): (
            "AVAILABLE_MECHANICS_ONLY", "tensorlbm.multipatch.MultiPatchSolver",
            "plain trilinear interpolation/block-average restriction",
        ),
    }
    detail = mechanics.get((path, lattice))
    if detail is None:
        return LocalRefinementCapability(
            "NO_IMPLEMENTATION", WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE, None, None,
            f"{path} has no audited {lattice} implementation.",
        )
    mechanics_status, entrypoint, exchange_scheme = detail
    if physics != "single_phase":
        return LocalRefinementCapability(
            mechanics_status, WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT, entrypoint, exchange_scheme,
            "The named physics exists elsewhere in the package, but this refinement path has no audited "
            "coupling, geometry-update, exchange, or conservation/evidence contract for it.",
        )
    return LocalRefinementCapability(
        mechanics_status, WITHHELD_REQUIRED_METADATA_NOT_EMITTED, entrypoint, exchange_scheme,
        "Patch mechanics exist, but current paths do not emit all required frontend metadata or a "
        "flux/inventory ledger; they are not precision/physics validation claims.",
    )


def local_refinement_capability_matrix() -> dict[RefinementPath, dict[LatticeName, dict[PhysicsName, LocalRefinementCapability]]]:
    """Return the complete audited local-refinement lattice/physics matrix."""
    return {
        path: {
            lattice: {physics: _capability_for(path, lattice, physics) for physics in _AUDITED_PHYSICS}
            for lattice in _AUDITED_LATTICES
        }
        for path in _AUDITED_PATHS
    }


def require_local_refinement_capability(
    path: RefinementPath,
    lattice: LatticeName,
    physics: PhysicsName,
    *,
    metadata: Mapping[str, object] | None = None,
) -> LocalRefinementCapability:
    """Return only an executable capability; otherwise fail closed.

    Request identities are validated before metadata or matrix lookup, so an
    unsupported public input is always rejected with a stable, machine-readable
    withholding code rather than leaking a ``KeyError``. Even complete
    caller-supplied metadata cannot upgrade a current implementation: it must
    be emitted/proven by the selected runtime.
    """
    if path not in _AUDITED_PATHS:
        raise LocalRefinementWithheldError(f"{WITHHELD_UNKNOWN_PATH}: {path!r} is not an audited refinement path.")
    if lattice not in _AUDITED_LATTICES:
        raise LocalRefinementWithheldError(f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited lattice.")
    if physics not in _AUDITED_PHYSICS:
        raise LocalRefinementWithheldError(f"{WITHHELD_UNKNOWN_PHYSICS}: {physics!r} is not an audited physics selection.")
    if metadata is not None:
        missing = [key for key in REQUIRED_FRONTEND_METADATA if key not in metadata]
        if missing:
            raise ValueError(f"metadata missing required keys: {', '.join(missing)}")
    capability = local_refinement_capability_matrix()[path][lattice][physics]
    if not capability.available:
        raise LocalRefinementWithheldError(f"{capability.status}: {capability.note}")
    return capability


__all__ = [
    "LocalRefinementCapability", "LocalRefinementWithheldError", "REQUIRED_FRONTEND_METADATA",
    "WITHHELD_REQUIRED_METADATA_NOT_EMITTED", "WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT",
    "WITHHELD_NO_D3Q27_LOCAL_REFINEMENT", "WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE",
    "WITHHELD_UNKNOWN_PATH", "WITHHELD_UNKNOWN_LATTICE", "WITHHELD_UNKNOWN_PHYSICS",
    "local_refinement_capability_matrix", "require_local_refinement_capability",
]
