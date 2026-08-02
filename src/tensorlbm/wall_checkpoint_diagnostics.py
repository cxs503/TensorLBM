"""Read-only wall-model diagnostics for saved D3Q19 population states.

The production wall boundary owns population updates and force application.
Checkpoint audits need the same exchange-location sampling and wall-law solve,
but must never advance or modify a saved state.  This module provides that
small, geometry-agnostic bridge so benchmark-specific inspection tools do not
duplicate wall-model physics.
"""

from __future__ import annotations

import torch

from .wall_model import WallStressDiagnostics, bfl_wall_function_3d


def diagnose_bfl_wall_exchange_state(
    populations: torch.Tensor,
    solid: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    nu: float,
    *,
    wall_law: str = "musker",
    near_mask: torch.Tensor | None = None,
    stress_exchange_distance: float = 1.0,
    wall_normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    area_weight: torch.Tensor | None = None,
    use_low_memory_macroscopic: bool = True,
    y_plus_lower_bound: float = 30.0,
    y_plus_upper_bound: float = 1000.0,
    minimum_y_plus_in_range_fraction: float = 0.9,
) -> WallStressDiagnostics:
    """Diagnose a frozen wall state without BFL, forcing, or time advance.

    ``populations`` is passed as both streamed and post-collision state only
    because the common wall routine requires both inputs.  With BFL and wall
    forcing disabled neither state is written, and all returned quantities are
    observations of the supplied population field.
    """
    if populations.ndim != 4 or populations.shape[0] != 19:
        raise ValueError("populations must have shape (19,nz,ny,nx)")
    if solid.shape != populations.shape[1:] or solid.dtype is not torch.bool:
        raise ValueError("solid must be bool with the population spatial shape")
    if fluid_boundary_mask.shape != populations.shape:
        raise ValueError("fluid_boundary_mask must match populations")
    if fluid_boundary_mask.dtype is not torch.bool:
        raise ValueError("fluid_boundary_mask must be boolean")
    if q_field.shape != populations.shape:
        raise ValueError("q_field must match populations")
    if not populations.is_floating_point():
        raise ValueError("populations must be floating point")
    if not torch.isfinite(populations).all():
        raise FloatingPointError("populations contain non-finite values")
    devices = {
        populations.device,
        solid.device,
        fluid_boundary_mask.device,
        q_field.device,
    }
    if wall_normals is not None:
        devices.update(component.device for component in wall_normals)
    if area_weight is not None:
        devices.add(area_weight.device)
    if len(devices) != 1:
        raise ValueError("population and wall geometry tensors must share a device")

    _, _, _, diagnostics = bfl_wall_function_3d(
        populations,
        populations,
        solid,
        nu,
        fluid_boundary_mask,
        q_field,
        wall_law=wall_law,
        near_mask=near_mask,
        apply_bfl=False,
        bfl_wall_mode="wall_model_slip",
        wall_activation=1.0,
        stress_exchange_distance=stress_exchange_distance,
        wall_normals=wall_normals,
        area_weight=area_weight,
        apply_wall_stress=False,
        use_low_memory_macroscopic=use_low_memory_macroscopic,
        return_wall_diagnostics=True,
        y_plus_lower_bound=y_plus_lower_bound,
        y_plus_upper_bound=y_plus_upper_bound,
        minimum_y_plus_in_range_fraction=(minimum_y_plus_in_range_fraction),
    )
    return diagnostics


__all__ = ["diagnose_bfl_wall_exchange_state"]
