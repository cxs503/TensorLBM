from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_static_amr import (
    build_fine_suboff_mask,
    build_nested_fine_suboff_mask,
    plan_nested_suboff_static_amr,
    plan_suboff_static_amr,
)


def _nested_case():
    nx, ny, nz, length = 200, 80, 80, 80.0
    center = (nx * 0.35, ny / 2.0, nz / 2.0)
    coarse, _ = build_suboff_mask(
        "bare_hull",
        nx,
        ny,
        nz,
        cx=center[0],
        cy=center[1],
        cz=center[2],
        length=length,
    )
    outer = plan_suboff_static_amr(
        coarse,
        coarse_hull_length=length,
        wall_margin=6,
        wake_cells=30,
    )
    outer_solid, _ = build_fine_suboff_mask(
        outer,
        hull_type="bare_hull",
        coarse_center=center,
    )
    nested = plan_nested_suboff_static_amr(
        outer,
        outer_solid,
        wall_margin=3,
        wake_cells=4,
    )
    return center, outer_solid, nested


def test_nested_plan_uses_outer_allocated_coordinates_and_saves_cells() -> None:
    _, outer_solid, plan = _nested_case()
    box = plan.box_in_outer_allocated_coordinates
    parent_shape = tuple(size + 2 for size in outer_solid.shape)

    assert 0 < box.x0 < box.x1 < parent_shape[2] - 1
    assert 0 < box.y0 < box.y1 < parent_shape[1] - 1
    assert 0 < box.z0 < box.z1 < parent_shape[0] - 1
    assert plan.effective_hull_length_cells == 320.0
    assert plan.effective_diameter_cells == 320.0 / 8.57
    assert plan.total_allocated_cells < plan.uniform_finest_cells
    assert plan.cell_saving_fraction > 0.9
    assert plan.wall_buffer_parent_cells == 3
    assert plan.wall_buffer_finest_cells == 6
    assert plan.downstream_buffer_parent_cells == 7
    assert plan.downstream_buffer_finest_cells == 14


def test_second_level_regenerates_exact_cad_and_contains_complete_hull() -> None:
    center, outer_solid, plan = _nested_case()
    nested_solid, geometry = build_nested_fine_suboff_mask(
        plan,
        hull_type="bare_hull",
        coarse_center=center,
    )

    assert nested_solid.shape == plan.fine_physical_shape
    assert geometry["length"] == 320.0
    assert geometry["solid_cells"] == int(nested_solid.sum())
    indices = nested_solid.nonzero(as_tuple=False)
    assert int(indices[:, 0].min()) >= 2 * 3
    assert int(indices[:, 1].min()) >= 2 * 3
    assert int(indices[:, 2].min()) >= 2 * 3
    assert int(indices[:, 0].max()) < nested_solid.shape[0] - 2 * 3
    assert int(indices[:, 1].max()) < nested_solid.shape[1] - 2 * 3
    assert int(indices[:, 2].max()) < nested_solid.shape[2] - 2 * (3 + 4)

    # Exact level-2 CAD is not a simple replication of outer voxels.
    box = plan.box_in_outer_allocated_coordinates
    outer_with_ghost = torch.zeros(
        tuple(size + 2 for size in outer_solid.shape),
        dtype=torch.bool,
    )
    outer_with_ghost[1:-1, 1:-1, 1:-1] = outer_solid
    parent_patch = outer_with_ghost[
        box.z0 : box.z1,
        box.y0 : box.y1,
        box.x0 : box.x1,
    ]
    repeated = parent_patch.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
    assert torch.count_nonzero(nested_solid ^ repeated).item() > 0


def test_third_refinement_block_is_recursive_and_reports_exact_memory() -> None:
    center, _, parent = _nested_case()
    parent_solid, _ = build_nested_fine_suboff_mask(
        parent,
        hull_type="bare_hull",
        coarse_center=center,
    )
    deepest = plan_nested_suboff_static_amr(
        parent,
        parent_solid,
        wall_margin=2,
        wake_cells=2,
    )
    deepest_solid, geometry = build_nested_fine_suboff_mask(
        deepest,
        hull_type="bare_hull",
        coarse_center=center,
    )

    assert deepest.refinement_depth == 3
    assert deepest.cumulative_ratio == 8
    assert deepest.effective_hull_length_cells == 640.0
    assert geometry["length"] == 640.0
    assert deepest_solid.shape == deepest.fine_physical_shape
    assert len(deepest.allocated_cells_by_level) == 4
    assert sum(deepest.allocated_cells_by_level) == deepest.total_allocated_cells
    assert deepest.uniform_finest_cells == math.prod(parent.outer.coarse_shape) * 8**3

    indices = deepest_solid.nonzero(as_tuple=False)
    # Two parent cells become a four-cell wall buffer on the child grid.
    assert deepest.wall_buffer_finest_cells == 4
    # Cell-centred rasterization may occupy the cell immediately below the
    # continuous four-cell geometric clearance.
    assert min(int(indices[:, axis].min()) for axis in range(3)) >= 3
    assert deepest.estimated_peak_gib(742.0) == pytest.approx(
        deepest.total_allocated_cells * 742.0 / 2**30,
    )
