"""SUBOFF planning helpers for the generic static block-AMR runtime."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .refinement import BoxRegion
from .suboff_cad import SuboffConfig, build_suboff_mask


@dataclass(frozen=True)
class SuboffStaticAMRPlan:
    coarse_shape: tuple[int, int, int]
    coarse_hull_length: float
    ratio: int
    box: BoxRegion
    fine_physical_shape: tuple[int, int, int]
    coarse_cells: int
    fine_allocated_cells: int
    total_allocated_cells: int
    uniform_fine_cells: int
    cell_saving_fraction: float
    effective_hull_length_cells: float
    effective_diameter_cells: float

    def estimated_peak_gib(self, bytes_per_cell: float = 943.0) -> float:
        """Empirical peak estimate; 943 B/cell matches the RTX3090 probes."""
        return self.total_allocated_cells * bytes_per_cell / 2**30


def plan_suboff_static_amr(
    solid_coarse: torch.Tensor,
    *,
    coarse_hull_length: float,
    ratio: int = 2,
    wall_margin: int = 6,
    wake_cells: int = 40,
    ghost: int = 1,
) -> SuboffStaticAMRPlan:
    """Build a tight hull+wake refinement box on a coarse SUBOFF domain."""
    if solid_coarse.ndim != 3 or solid_coarse.dtype is not torch.bool:
        raise ValueError("solid_coarse must be a 3-D boolean tensor")
    if ratio != 2 or ghost != 1:
        raise ValueError("the production runtime currently supports ratio=2, ghost=1")
    if wall_margin < 2 or wake_cells < 0:
        raise ValueError("wall_margin must be >=2 and wake_cells non-negative")
    indices = solid_coarse.nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError("solid_coarse contains no SUBOFF cells")
    nz, ny, nx = solid_coarse.shape
    z_min, y_min, x_min = (int(indices[:, axis].min().item()) for axis in range(3))
    z_max, y_max, x_max = (int(indices[:, axis].max().item()) + 1 for axis in range(3))
    x0 = max(1, x_min - wall_margin)
    x1 = min(nx - 1, x_max + wall_margin + wake_cells)
    y0 = max(1, y_min - wall_margin)
    y1 = min(ny - 1, y_max + wall_margin)
    z0 = max(1, z_min - wall_margin)
    z1 = min(nz - 1, z_max + wall_margin)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError("refinement box is too small")
    box = BoxRegion(x0, x1, y0, y1, z0, z1)
    fine_shape = (
        (z1 - z0) * ratio,
        (y1 - y0) * ratio,
        (x1 - x0) * ratio,
    )
    allocated_fine_shape = tuple(size + 2 * ghost for size in fine_shape)
    coarse_cells = nz * ny * nx
    fine_cells = math.prod(allocated_fine_shape)
    total = coarse_cells + fine_cells
    uniform = coarse_cells * ratio**3
    effective_length = coarse_hull_length * ratio
    return SuboffStaticAMRPlan(
        coarse_shape=(nz, ny, nx),
        coarse_hull_length=coarse_hull_length,
        ratio=ratio,
        box=box,
        fine_physical_shape=fine_shape,
        coarse_cells=coarse_cells,
        fine_allocated_cells=fine_cells,
        total_allocated_cells=total,
        uniform_fine_cells=uniform,
        cell_saving_fraction=1.0 - total / uniform,
        effective_hull_length_cells=effective_length,
        effective_diameter_cells=effective_length / 8.57,
    )


def build_fine_suboff_mask(
    plan: SuboffStaticAMRPlan,
    *,
    hull_type: str,
    coarse_center: tuple[float, float, float],
    config: SuboffConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Generate exact CAD geometry on the physical fine block (no voxel repeat)."""
    if config is None:
        config = SuboffConfig()
    r, box = plan.ratio, plan.box
    cx, cy, cz = coarse_center
    local_cx = cx * r - box.x0 * r
    local_cy = cy * r - box.y0 * r
    local_cz = cz * r - box.z0 * r
    nz_f, ny_f, nx_f = plan.fine_physical_shape
    return build_suboff_mask(
        hull_type,
        nx_f, ny_f, nz_f,
        cx=local_cx, cy=local_cy, cz=local_cz,
        length=plan.coarse_hull_length * r,
        radius=config.r_over_l * plan.coarse_hull_length * r,
        config=config,
        device=device,
    )


__all__ = [
    "SuboffStaticAMRPlan",
    "build_fine_suboff_mask",
    "plan_suboff_static_amr",
]
