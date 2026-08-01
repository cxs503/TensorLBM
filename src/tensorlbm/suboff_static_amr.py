"""SUBOFF planning helpers for the generic static block-AMR runtime."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .refinement import BoxRegion
from .suboff_cad import SuboffConfig, build_suboff_mask

# These are evidence gates, not stability limits.  A coarse member of a
# Richardson sequence may be useful below the absolute-reference threshold,
# but it still needs enough curvature/appendage thickness to represent the
# same geometry family as the finer members.
MIN_CONVERGENCE_DIAMETER_CELLS = 16.0
MIN_ABSOLUTE_REFERENCE_DIAMETER_CELLS = 24.0
MIN_CONVERGENCE_APPENDAGE_THICKNESS_CELLS = 3
MIN_ABSOLUTE_REFERENCE_APPENDAGE_THICKNESS_CELLS = 4


@dataclass(frozen=True)
class SuboffGeometryResolution:
    """Measured voxel-resolution evidence for an AFF-1/AFF-8 geometry."""

    hull_type: str
    fine_hull_length_cells: float
    diameter_cells: float
    solid_cells: int
    bare_hull_cells: int
    sail_only_cells: int
    fin_only_cells: int
    sail_max_thickness_cells: int | None
    vertical_fin_max_thickness_cells: int | None
    horizontal_fin_max_thickness_cells: int | None
    appendage_halfway_links: int
    convergence_member_resolved: bool
    absolute_reference_resolved: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "hull_type": self.hull_type,
            "fine_hull_length_cells": self.fine_hull_length_cells,
            "diameter_cells": self.diameter_cells,
            "solid_cells": self.solid_cells,
            "bare_hull_cells": self.bare_hull_cells,
            "sail_only_cells": self.sail_only_cells,
            "fin_only_cells": self.fin_only_cells,
            "sail_max_thickness_cells": self.sail_max_thickness_cells,
            "vertical_fin_max_thickness_cells": (
                self.vertical_fin_max_thickness_cells
            ),
            "horizontal_fin_max_thickness_cells": (
                self.horizontal_fin_max_thickness_cells
            ),
            "appendage_halfway_links": self.appendage_halfway_links,
            "minimum_convergence_diameter_cells": (
                MIN_CONVERGENCE_DIAMETER_CELLS
            ),
            "minimum_absolute_reference_diameter_cells": (
                MIN_ABSOLUTE_REFERENCE_DIAMETER_CELLS
            ),
            "minimum_convergence_appendage_thickness_cells": (
                MIN_CONVERGENCE_APPENDAGE_THICKNESS_CELLS
            ),
            "minimum_absolute_reference_appendage_thickness_cells": (
                MIN_ABSOLUTE_REFERENCE_APPENDAGE_THICKNESS_CELLS
            ),
            "convergence_member_resolved": self.convergence_member_resolved,
            "absolute_reference_resolved": self.absolute_reference_resolved,
        }


def assess_suboff_geometry_resolution(
    solid: torch.Tensor,
    *,
    hull_type: str,
    fine_hull_length_cells: float,
    center_yz: tuple[float, float] | None = None,
    bare_hull: torch.Tensor | None = None,
    with_sail: torch.Tensor | None = None,
    appendage_halfway_links: int = 0,
) -> SuboffGeometryResolution:
    """Measure whether a voxelized SUBOFF is fit for convergence/reference use.

    AFF-8 checks use *actual rasterized masks*.  In particular, the sail and
    both cruciform-fin thicknesses must occupy several lattice cells; an
    analytical nominal thickness alone cannot detect a vanished appendage.
    """
    if solid.ndim != 3 or solid.dtype is not torch.bool:
        raise ValueError("solid must be a 3-D boolean tensor")
    if fine_hull_length_cells <= 0.0:
        raise ValueError("fine_hull_length_cells must be positive")
    if hull_type not in {"bare_hull", "full"}:
        raise ValueError("hull_type must be 'bare_hull' or 'full'")
    if appendage_halfway_links < 0:
        raise ValueError("appendage_halfway_links must be non-negative")

    diameter = float(fine_hull_length_cells) / 8.57
    solid_cells = int(solid.sum().item())
    bare_cells = solid_cells
    sail_cells = fin_cells = 0
    sail_thickness = vertical_thickness = horizontal_thickness = None
    appendages_resolved_for_convergence = True
    appendages_resolved_for_reference = True

    if hull_type == "full":
        if bare_hull is None or with_sail is None or center_yz is None:
            raise ValueError(
                "full geometry needs bare_hull, with_sail, and center_yz",
            )
        if (
            bare_hull.shape != solid.shape
            or with_sail.shape != solid.shape
            or bare_hull.dtype is not torch.bool
            or with_sail.dtype is not torch.bool
        ):
            raise ValueError("component masks must be boolean and match solid")
        if bool((bare_hull & ~with_sail).any()) or bool((with_sail & ~solid).any()):
            raise ValueError("component masks must be nested bare <= sail <= full")

        bare_cells = int(bare_hull.sum().item())
        sail_only = with_sail & ~bare_hull
        fin_only = solid & ~with_sail
        sail_cells = int(sail_only.sum().item())
        fin_cells = int(fin_only.sum().item())
        sail_thickness = int(sail_only.sum(dim=1).max().item())

        cy, cz = center_yz
        nz, ny, _ = solid.shape
        y_distance = (
            torch.arange(ny, device=solid.device, dtype=torch.float32)
            .view(1, ny, 1)
            .sub(float(cy))
            .abs()
        )
        z_distance = (
            torch.arange(nz, device=solid.device, dtype=torch.float32)
            .view(nz, 1, 1)
            .sub(float(cz))
            .abs()
        )
        vertical_fins = fin_only & (y_distance >= z_distance)
        horizontal_fins = fin_only & (z_distance > y_distance)
        vertical_thickness = int(vertical_fins.sum(dim=0).max().item())
        horizontal_thickness = int(horizontal_fins.sum(dim=1).max().item())
        measured_thicknesses = (
            sail_thickness, vertical_thickness, horizontal_thickness,
        )
        appendages_present = (
            sail_cells > 0 and fin_cells > 0 and appendage_halfway_links > 0
        )
        appendages_resolved_for_convergence = appendages_present and min(
            measured_thicknesses,
        ) >= MIN_CONVERGENCE_APPENDAGE_THICKNESS_CELLS
        appendages_resolved_for_reference = appendages_present and min(
            measured_thicknesses,
        ) >= MIN_ABSOLUTE_REFERENCE_APPENDAGE_THICKNESS_CELLS

    return SuboffGeometryResolution(
        hull_type=hull_type,
        fine_hull_length_cells=float(fine_hull_length_cells),
        diameter_cells=diameter,
        solid_cells=solid_cells,
        bare_hull_cells=bare_cells,
        sail_only_cells=sail_cells,
        fin_only_cells=fin_cells,
        sail_max_thickness_cells=sail_thickness,
        vertical_fin_max_thickness_cells=vertical_thickness,
        horizontal_fin_max_thickness_cells=horizontal_thickness,
        appendage_halfway_links=appendage_halfway_links,
        convergence_member_resolved=(
            diameter >= MIN_CONVERGENCE_DIAMETER_CELLS
            and appendages_resolved_for_convergence
        ),
        absolute_reference_resolved=(
            diameter >= MIN_ABSOLUTE_REFERENCE_DIAMETER_CELLS
            and appendages_resolved_for_reference
        ),
    )


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
    "MIN_ABSOLUTE_REFERENCE_APPENDAGE_THICKNESS_CELLS",
    "MIN_ABSOLUTE_REFERENCE_DIAMETER_CELLS",
    "MIN_CONVERGENCE_APPENDAGE_THICKNESS_CELLS",
    "MIN_CONVERGENCE_DIAMETER_CELLS",
    "SuboffGeometryResolution",
    "SuboffStaticAMRPlan",
    "assess_suboff_geometry_resolution",
    "build_fine_suboff_mask",
    "plan_suboff_static_amr",
]
