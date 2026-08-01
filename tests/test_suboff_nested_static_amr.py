from __future__ import annotations

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
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )
    outer = plan_suboff_static_amr(
        coarse, coarse_hull_length=length, wall_margin=6, wake_cells=30,
    )
    outer_solid, _ = build_fine_suboff_mask(
        outer, hull_type="bare_hull", coarse_center=center,
    )
    nested = plan_nested_suboff_static_amr(
        outer, outer_solid, wall_margin=3, wake_cells=4,
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
        plan, hull_type="bare_hull", coarse_center=center,
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
        tuple(size + 2 for size in outer_solid.shape), dtype=torch.bool,
    )
    outer_with_ghost[1:-1, 1:-1, 1:-1] = outer_solid
    parent_patch = outer_with_ghost[
        box.z0:box.z1, box.y0:box.y1, box.x0:box.x1,
    ]
    repeated = (
        parent_patch.repeat_interleave(2, 0)
        .repeat_interleave(2, 1)
        .repeat_interleave(2, 2)
    )
    assert torch.count_nonzero(nested_solid ^ repeated).item() > 0
