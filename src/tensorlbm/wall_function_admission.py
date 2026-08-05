"""Public cold-path admission boundary for wall-function-enabled runs.

Legacy numerical helpers in :mod:`wall_model` remain directly importable for
reproducibility.  They are not a generic public feature declaration.  New
configuration/run entry points must use this module before entering a solver
loop; it deliberately has no tensor arguments and must never be called per
cell or per time step.
"""
from __future__ import annotations

from dataclasses import dataclass

from .wall_function_contract import (
    ValidationLevel,
    WallFunctionCapability,
    WallFunctionCapabilityRecord,
    WallFunctionRequest,
    require_wall_function,
)


@dataclass(frozen=True)
class WallFunctionRunRequest:
    """Fully specified cold-path request from a public run configuration.

    ``adaptive_mesh`` and ``free_surface`` are explicit because neither can be
    inferred safely from the legacy wall-operator call signature.
    """

    capability: WallFunctionCapability
    lattice: str
    physics: str
    collision: str
    geometry: str
    backend: str
    adaptive_mesh: bool = False
    free_surface: bool = False
    minimum_validation: ValidationLevel = ValidationLevel.IMPLEMENTATION_ONLY


def require_wall_function_run(request: WallFunctionRunRequest) -> WallFunctionCapabilityRecord:
    """Admit one public run request or fail closed before solver execution.

    AMR and free-surface modify the declared dimensions to deliberately
    unlisted labels.  Thus a caller cannot accidentally inherit the static,
    single-phase D3Q19 implementation-only evidence by enabling either mode.
    """
    physics = "free_surface" if request.free_surface else request.physics
    geometry = (
        f"amr_{request.geometry}" if request.adaptive_mesh else request.geometry
    )
    return require_wall_function(
        WallFunctionRequest(
            capability=request.capability,
            lattice=request.lattice,
            physics=physics,
            collision=request.collision,
            geometry=geometry,
            backend=request.backend,
        ),
        minimum_validation=request.minimum_validation,
    )


__all__ = ["WallFunctionRunRequest", "require_wall_function_run"]
