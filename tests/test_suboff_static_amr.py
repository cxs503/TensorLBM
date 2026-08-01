from __future__ import annotations

import torch

from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_static_amr import (
    apply_suboff_appendage_halfway_links,
    assess_suboff_geometry_resolution,
    build_fine_suboff_mask,
    count_suboff_appendage_boundary_links,
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


def _full_component_resolution(length: float):
    nx = int(length + 40)
    ny = nz = int(length * 0.28) + 10
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    masks = {
        hull_type: build_suboff_mask(
            hull_type, nx, ny, nz,
            cx=center[0], cy=center[1], cz=center[2], length=length,
        )[0]
        for hull_type in ("bare_hull", "with_sail", "full")
    }
    return assess_suboff_geometry_resolution(
        masks["full"], hull_type="full", fine_hull_length_cells=length,
        center_yz=(center[1], center[2]),
        bare_hull=masks["bare_hull"], with_sail=masks["with_sail"],
        appendage_halfway_links=1,
    )


def test_aff8_resolution_measures_rasterized_sail_and_cruciform_fins() -> None:
    coarse = _full_component_resolution(180.0)
    assert coarse.diameter_cells == 180.0 / 8.57
    assert coarse.sail_only_cells > 0
    assert coarse.fin_only_cells > 0
    assert coarse.sail_max_thickness_cells == 3
    assert coarse.vertical_fin_max_thickness_cells == 3
    assert coarse.horizontal_fin_max_thickness_cells == 3
    assert coarse.convergence_member_resolved is True
    assert coarse.absolute_reference_resolved is False

    reference = _full_component_resolution(240.0)
    assert reference.sail_max_thickness_cells >= 4
    assert reference.vertical_fin_max_thickness_cells >= 4
    assert reference.horizontal_fin_max_thickness_cells >= 4
    assert reference.absolute_reference_resolved is True


def test_aff8_resolution_fails_closed_without_boundary_links() -> None:
    assessment = _full_component_resolution(240.0)
    masks = torch.zeros((8, 8, 8), dtype=torch.bool)
    # Component masks are intentionally empty: nominal hull length alone is
    # insufficient evidence that AFF-8 appendages survived rasterization.
    missing = assess_suboff_geometry_resolution(
        masks, hull_type="full", fine_hull_length_cells=240.0,
        center_yz=(4.0, 4.0), bare_hull=masks, with_sail=masks,
        appendage_halfway_links=0,
    )
    assert assessment.absolute_reference_resolved is True
    assert missing.convergence_member_resolved is False
    assert missing.absolute_reference_resolved is False


def test_aff8_geometry_only_boundary_link_count_is_positive() -> None:
    length = 240.0
    nx = int(length + 40)
    ny = nz = int(length * 0.28) + 10
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    masks = {
        hull_type: build_suboff_mask(
            hull_type, nx, ny, nz,
            cx=center[0], cy=center[1], cz=center[2], length=length,
        )[0]
        for hull_type in ("bare_hull", "full")
    }

    count = count_suboff_appendage_boundary_links(
        masks["full"], masks["bare_hull"],
    )

    assert count > 0


def test_geometry_only_appendage_link_count_matches_runtime_treatment() -> None:
    nx, ny, nz, length = 120, 40, 40, 80.0
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    bare = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    full = build_suboff_mask(
        "full", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    link_mask, q = compute_q_suboff(
        nx, ny, nz, *center, length,
        hull_type="full", solid_mask=full,
    )

    preflight_count = count_suboff_appendage_boundary_links(full, bare)
    runtime_count = apply_suboff_appendage_halfway_links(
        full, link_mask, q, center=center, length=length,
    )

    assert preflight_count > 0
    assert runtime_count == preflight_count
