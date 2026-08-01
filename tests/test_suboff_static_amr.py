from __future__ import annotations

import torch

from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_static_amr import (
    build_fine_suboff_mask,
    plan_suboff_static_amr,
)


def _coarse_case():
    nx, ny, nz, length = 200, 80, 80, 80.0
    center = (nx * 0.35, ny / 2.0, nz / 2.0)
    solid, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2],
        length=length, device="cpu",
    )
    return solid, center, length


def test_suboff_plan_refines_hull_and_wake_with_large_cell_saving() -> None:
    solid, _, length = _coarse_case()
    plan = plan_suboff_static_amr(
        solid, coarse_hull_length=length, wall_margin=6, wake_cells=30,
    )
    indices = solid.nonzero()
    assert plan.box.x0 < int(indices[:, 2].min())
    assert plan.box.x1 > int(indices[:, 2].max()) + 1
    assert plan.effective_hull_length_cells == 160.0
    assert plan.effective_diameter_cells == 160.0 / 8.57
    assert plan.cell_saving_fraction > 0.75
    assert plan.total_allocated_cells < plan.uniform_fine_cells


def test_fine_mask_is_regenerated_from_cad_not_voxel_repeated() -> None:
    solid, center, length = _coarse_case()
    plan = plan_suboff_static_amr(
        solid, coarse_hull_length=length, wall_margin=6, wake_cells=30,
    )
    fine, geometry = build_fine_suboff_mask(
        plan, hull_type="bare_hull", coarse_center=center,
        device=torch.device("cpu"),
    )
    assert fine.shape == plan.fine_physical_shape
    assert geometry["length"] == 2.0 * length
    assert geometry["solid_cells"] == int(fine.sum())
    # A fresh curved CAD raster is not an exact 8x replication of coarse voxels.
    coarse_patch = solid[
        plan.box.z0:plan.box.z1,
        plan.box.y0:plan.box.y1,
        plan.box.x0:plan.box.x1,
    ]
    repeated = (
        coarse_patch.repeat_interleave(2, 0)
        .repeat_interleave(2, 1)
        .repeat_interleave(2, 2)
    )
    assert torch.count_nonzero(fine ^ repeated).item() > 0


def test_l120_plan_reaches_28_cells_across_diameter_with_small_memory() -> None:
    # Geometry extent scales linearly, so build an L=80 mask and evaluate the
    # exact L=120 production dimensions analytically through a scaled mask.
    nx, ny, nz, length = 300, 120, 120, 120.0
    center = (nx * 0.35, ny / 2.0, nz / 2.0)
    solid, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2],
        length=length, device="cpu",
    )
    plan = plan_suboff_static_amr(
        solid, coarse_hull_length=length, wall_margin=8, wake_cells=50,
    )
    assert plan.effective_diameter_cells > 28.0 - 0.01
    assert plan.cell_saving_fraction > 0.75
    assert plan.estimated_peak_gib() < 8.0
