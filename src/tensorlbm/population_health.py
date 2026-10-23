"""Low-frequency, solver-independent health diagnostics for LBM populations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class PopulationHealth:
    """Scalar health summary that can be persisted without retaining fields."""

    finite: bool
    minimum_population: float
    minimum_population_direction: int | None
    minimum_population_index_zyx: tuple[int, int, int] | None
    maximum_population: float
    minimum_density: float | None
    maximum_density: float | None
    maximum_speed: float | None
    maximum_speed_index_zyx: tuple[int, int, int] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_population_health(f: torch.Tensor) -> PopulationHealth:
    """Inspect finiteness, population range, density range and peak speed.

    This routine is intended for a configurable diagnostic cadence, not the
    hot path.  It deliberately returns only host scalars so a long run cannot
    retain large diagnostic tensors.
    """
    if not isinstance(f, torch.Tensor) or f.ndim != 4 or f.shape[0] not in (19, 27):
        raise ValueError("f must have shape (19|27,nz,ny,nx)")
    if not f.is_floating_point():
        raise TypeError("f must be floating point")

    minimum, maximum = torch.aminmax(f)
    minimum_value = float(minimum.item())
    maximum_value = float(maximum.item())
    finite = math.isfinite(minimum_value) and math.isfinite(maximum_value)
    if not finite:
        return PopulationHealth(
            False,
            minimum_value,
            None,
            None,
            maximum_value,
            None,
            None,
            None,
            None,
        )

    minimum_flat_index = int(f.argmin().item())
    nz, ny, nx = f.shape[1:]
    spatial_cell_count = nz * ny * nx
    minimum_population_direction = minimum_flat_index // spatial_cell_count
    minimum_spatial_flat_index = minimum_flat_index % spatial_cell_count
    minimum_population_index = (
        minimum_spatial_flat_index // (ny * nx),
        (minimum_spatial_flat_index % (ny * nx)) // nx,
        minimum_spatial_flat_index % nx,
    )

    if f.shape[0] == 19:
        from .d3q19 import C
    else:
        from .d3q27 import C
    c = C.to(device=f.device, dtype=f.dtype)
    density = f.sum(dim=0)
    density_minimum, density_maximum = torch.aminmax(density)
    momentum_x = (f * c[:, 0, None, None, None]).sum(dim=0)
    momentum_y = (f * c[:, 1, None, None, None]).sum(dim=0)
    momentum_z = (f * c[:, 2, None, None, None]).sum(dim=0)
    speed = torch.sqrt(
        momentum_x.square() + momentum_y.square() + momentum_z.square(),
    ) / density.abs().clamp_min(torch.finfo(f.dtype).tiny)
    maximum_speed = float(speed.max().item())
    maximum_speed_flat_index = int(speed.argmax().item())
    ny, nx = speed.shape[1:]
    maximum_speed_index = (
        maximum_speed_flat_index // (ny * nx),
        (maximum_speed_flat_index % (ny * nx)) // nx,
        maximum_speed_flat_index % nx,
    )
    density_minimum_value = float(density_minimum.item())
    density_maximum_value = float(density_maximum.item())
    finite = all(
        math.isfinite(value)
        for value in (
            density_minimum_value,
            density_maximum_value,
            maximum_speed,
        )
    )
    return PopulationHealth(
        finite,
        minimum_value,
        minimum_population_direction,
        minimum_population_index,
        maximum_value,
        density_minimum_value,
        density_maximum_value,
        maximum_speed,
        maximum_speed_index,
    )


__all__ = ["PopulationHealth", "inspect_population_health"]
