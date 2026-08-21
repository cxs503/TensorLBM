"""Discrete kinetic control-volume force observation for LBM.

The observer uses an interior Cartesian control volume and the populations
that actually cross its outer faces during streaming.  It is independent of
the particular immersed/voxel wall rule inside the volume and is therefore a
useful cross-check for link momentum exchange, wall functions and IBM.

For a fixed body contained by the control volume,

``force_on_body = net_momentum_import - change_of_fluid_momentum``.

Both terms are evaluated in lattice units over one time step.  The control
volume must be strictly interior and its outer shell must contain fluid only;
physical boundary conditions and sponge forcing must remain outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def _lattice_velocities(q: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if q == 19:
        from .d3q19 import C
    elif q == 27:
        from .d3q27 import C
    else:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    return C.to(device=device, dtype=dtype)


def _validate(
    f: torch.Tensor,
    control_volume: torch.Tensor,
    periodic_axes: tuple[str, ...] = (),
) -> None:
    if f.ndim != 4 or f.shape[0] not in (19, 27):
        raise ValueError("populations must have shape (19|27,nz,ny,nx)")
    if control_volume.shape != f.shape[1:] or control_volume.dtype is not torch.bool:
        raise ValueError("control_volume must be bool with the spatial grid shape")
    if control_volume.device != f.device:
        raise ValueError("populations and control_volume must share a device")
    if not set(periodic_axes) <= {"x", "y", "z"}:
        raise ValueError("periodic_axes may contain only x, y, z")
    touches_nonperiodic = (
        (
            "z" not in periodic_axes
            and (bool(control_volume[0].any()) or bool(control_volume[-1].any()))
        )
        or (
            "y" not in periodic_axes
            and (bool(control_volume[:, 0].any()) or bool(control_volume[:, -1].any()))
        )
        or (
            "x" not in periodic_axes
            and (bool(control_volume[:, :, 0].any()) or bool(control_volume[:, :, -1].any()))
        )
    )
    if touches_nonperiodic:
        raise ValueError("control volume must be strictly interior")


def fluid_momentum(
    f: torch.Tensor,
    control_volume: torch.Tensor,
    *,
    solid: torch.Tensor | None = None,
    periodic_axes: tuple[str, ...] = (),
) -> torch.Tensor:
    """Return the total fluid momentum inside a control volume."""
    _validate(f, control_volume, periodic_axes)
    owned = control_volume
    if solid is not None:
        if solid.shape != owned.shape or solid.dtype is not torch.bool:
            raise ValueError("solid must be bool with the spatial grid shape")
        if solid.device != f.device:
            raise ValueError("solid and populations must share a device")
        owned = owned & ~solid
    # Force is a small residual of large momentum inventories.  Accumulate
    # float32 populations in float64 to avoid cancellation at production
    # control-volume sizes.
    accumulator_dtype = torch.float64 if f.dtype == torch.float32 else f.dtype
    c = _lattice_velocities(f.shape[0], f.device, accumulator_dtype)
    momentum = torch.zeros(3, device=f.device, dtype=accumulator_dtype)
    for direction in range(1, f.shape[0]):
        inventory = f[direction][owned].sum(dtype=accumulator_dtype)
        momentum = momentum + inventory * c[direction]
    return momentum


def fluid_momentum_change(
    f_old: torch.Tensor,
    f_new: torch.Tensor,
    control_volume: torch.Tensor,
    *,
    solid: torch.Tensor | None = None,
    periodic_axes: tuple[str, ...] = (),
) -> torch.Tensor:
    """Accumulate local population changes without subtracting large totals."""
    if f_old.shape != f_new.shape:
        raise ValueError("old and new populations must share shape")
    _validate(f_old, control_volume, periodic_axes)
    owned = control_volume
    if solid is not None:
        if solid.shape != owned.shape or solid.dtype is not torch.bool:
            raise ValueError("solid must be bool with the spatial grid shape")
        if solid.device != f_old.device:
            raise ValueError("solid and populations must share a device")
        owned = owned & ~solid
    accumulator_dtype = torch.float64 if f_old.dtype == torch.float32 else f_old.dtype
    # Subtract locally before reduction.  Summing old/new inventories first
    # and then subtracting loses a small force beneath O(N-cell) float32 totals.
    c = _lattice_velocities(f_old.shape[0], f_old.device, accumulator_dtype)
    momentum_change = torch.zeros(
        3,
        device=f_old.device,
        dtype=accumulator_dtype,
    )
    for direction in range(1, f_old.shape[0]):
        population_change = (f_new[direction][owned] - f_old[direction][owned]).sum(
            dtype=accumulator_dtype
        )
        momentum_change = momentum_change + population_change * c[direction]
    return momentum_change


def streaming_momentum_import(
    f_post_collision: torch.Tensor,
    control_volume: torch.Tensor,
    *,
    periodic_axes: tuple[str, ...] = (),
) -> torch.Tensor:
    """Net momentum imported through the outer control-volume faces.

    Positive values enter the control volume.  Populations are sampled from
    their source cells in the post-collision, pre-stream state.
    """
    _validate(f_post_collision, control_volume, periodic_axes)
    accumulator_dtype = (
        torch.float64 if f_post_collision.dtype == torch.float32 else f_post_collision.dtype
    )
    c = _lattice_velocities(
        f_post_collision.shape[0],
        f_post_collision.device,
        accumulator_dtype,
    )
    net = torch.zeros(3, device=f_post_collision.device, dtype=accumulator_dtype)
    for direction in range(1, f_post_collision.shape[0]):
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        # At source x, this is CV(x+c_q).
        destination_inside = torch.roll(
            control_volume,
            shifts=(-cz, -cy, -cx),
            dims=(0, 1, 2),
        )
        incoming = ~control_volume & destination_inside
        outgoing = control_volume & ~destination_inside
        scalar_flux = f_post_collision[direction][incoming].sum(
            dtype=accumulator_dtype
        ) - f_post_collision[direction][outgoing].sum(dtype=accumulator_dtype)
        net = net + scalar_flux * c[direction]
    return net


@dataclass(frozen=True)
class ControlVolumeForceResult:
    force_on_body: torch.Tensor
    momentum_import: torch.Tensor
    fluid_momentum_change: torch.Tensor

    @property
    def force_tuple(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.force_on_body.tolist())


@dataclass(frozen=True)
class NestedControlVolumeAssessment:
    """Invariance evidence for one primary and multiple enclosing CVs."""

    auxiliary_count: int
    differences_pct: tuple[float, ...]
    maximum_difference_pct: float
    finite: bool

    def meets(self, target_pct: float, *, minimum_auxiliary_count: int = 2) -> bool:
        if target_pct < 0.0 or minimum_auxiliary_count < 1:
            raise ValueError("targets must be non-negative with at least one auxiliary CV")
        return (
            self.finite
            and self.auxiliary_count >= minimum_auxiliary_count
            and self.maximum_difference_pct <= target_pct
        )


def assess_nested_control_volume_invariance(
    primary_force: float,
    auxiliary_forces: Sequence[float],
) -> NestedControlVolumeAssessment:
    """Compare independently enclosed force balances without selecting one."""
    auxiliary = tuple(float(value) for value in auxiliary_forces)
    finite = math.isfinite(primary_force) and all(math.isfinite(value) for value in auxiliary)
    if finite:
        denominator = max(abs(primary_force), 1e-30)
        differences = tuple(abs(value - primary_force) / denominator * 100.0 for value in auxiliary)
        maximum = max(differences, default=math.inf)
    else:
        differences = tuple(math.inf for _ in auxiliary)
        maximum = math.inf
    return NestedControlVolumeAssessment(
        auxiliary_count=len(auxiliary),
        differences_pct=differences,
        maximum_difference_pct=maximum,
        finite=finite,
    )


def observe_control_volume_force(
    f_old: torch.Tensor,
    f_new: torch.Tensor,
    f_post_collision: torch.Tensor,
    control_volume: torch.Tensor,
    *,
    solid: torch.Tensor | None = None,
    periodic_axes: tuple[str, ...] = (),
) -> ControlVolumeForceResult:
    """Observe force on an enclosed body over one complete LBM step."""
    if f_old.shape != f_new.shape or f_old.shape != f_post_collision.shape:
        raise ValueError("all population tensors must have the same shape")
    change = fluid_momentum_change(
        f_old,
        f_new,
        control_volume,
        solid=solid,
        periodic_axes=periodic_axes,
    )
    imported = streaming_momentum_import(
        f_post_collision,
        control_volume,
        periodic_axes=periodic_axes,
    )
    return ControlVolumeForceResult(imported - change, imported, change)


def box_control_volume(
    shape: tuple[int, int, int],
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    z0: int,
    z1: int,
    device: torch.device | str = "cpu",
    periodic_axes: tuple[str, ...] = (),
) -> torch.Tensor:
    """Create a strictly interior Cartesian control-volume mask."""
    nz, ny, nx = shape
    valid_x = 0 <= x0 < x1 <= nx if "x" in periodic_axes else 0 < x0 < x1 < nx - 1
    valid_y = 0 <= y0 < y1 <= ny if "y" in periodic_axes else 0 < y0 < y1 < ny - 1
    valid_z = 0 <= z0 < z1 <= nz if "z" in periodic_axes else 0 < z0 < z1 < nz - 1
    if not set(periodic_axes) <= {"x", "y", "z"}:
        raise ValueError("periodic_axes may contain only x, y, z")
    if not (valid_x and valid_y and valid_z):
        raise ValueError("control-volume bounds must be strictly interior")
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[z0:z1, y0:y1, x0:x1] = True
    return mask


__all__ = [
    "ControlVolumeForceResult",
    "NestedControlVolumeAssessment",
    "assess_nested_control_volume_invariance",
    "box_control_volume",
    "fluid_momentum",
    "fluid_momentum_change",
    "observe_control_volume_force",
    "streaming_momentum_import",
]
