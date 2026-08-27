from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.surface_area_weights import bfl_surface_area_weights


def test_axis_aligned_plane_has_unit_area_per_crossing() -> None:
    shape = (3, 4, 5)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    mask[4, :, 1, :] = True
    zero = torch.zeros(shape)
    ny = torch.zeros(shape)
    ny[:, 1, :] = 1.0
    weights, diagnostics = bfl_surface_area_weights(
        mask,
        (zero, ny, zero),
    )
    assert weights[:, 1, :].sum().item() == pytest.approx(15.0)
    assert torch.all(weights[:, 1, :] == 1.0)
    assert diagnostics.raw_area == pytest.approx(15.0)
    assert diagnostics.unweighted_nodes == 0


def test_diagonal_patch_uses_inverse_l1_projection() -> None:
    shape = (3, 3, 3)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    cell = (1, 1, 1)
    mask[(1,) + cell] = True
    mask[(3,) + cell] = True
    zero = torch.zeros(shape)
    nx = zero.clone()
    ny = zero.clone()
    nx[cell] = ny[cell] = 1.0 / math.sqrt(2.0)
    weights, _ = bfl_surface_area_weights(mask, (nx, ny, zero))
    assert weights[cell].item() == pytest.approx(math.sqrt(2.0), rel=1e-6)


def test_reference_area_calibrates_total_without_flattening_distribution() -> None:
    shape = (3, 3, 4)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    first, second = (1, 1, 1), (1, 1, 2)
    mask[(1,) + first] = True
    mask[(1,) + second] = True
    mask[(3,) + second] = True
    zero = torch.zeros(shape)
    nx = zero.clone()
    ny = zero.clone()
    nx[first] = 1.0
    nx[second] = ny[second] = 1.0 / math.sqrt(2.0)
    weights, diagnostics = bfl_surface_area_weights(
        mask,
        (nx, ny, zero),
        reference_area=10.0,
    )
    assert weights.sum().item() == pytest.approx(10.0)
    assert weights[second] > weights[first]
    assert diagnostics.calibrated_area == pytest.approx(10.0)
    with pytest.raises(ValueError, match="not both"):
        bfl_surface_area_weights(
            mask,
            (nx, ny, zero),
            reference_area=10.0,
            calibration_factor=1.0,
        )


def test_optional_boundary_mask_excludes_diagonal_only_nodes() -> None:
    shape = (3, 3, 4)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    active, diagonal = (1, 1, 1), (1, 1, 2)
    mask[(1,) + active] = True
    mask[(7,) + diagonal] = True
    boundary = torch.zeros(shape, dtype=torch.bool)
    boundary[active] = True
    nx = torch.zeros(shape)
    ny = torch.zeros(shape)
    nz = torch.zeros(shape)
    nx[active] = 1.0
    nx[diagonal] = ny[diagonal] = 1.0 / math.sqrt(2.0)
    weights, diagnostics = bfl_surface_area_weights(
        mask,
        (nx, ny, nz),
        boundary_mask=boundary,
    )
    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[diagonal].item() == pytest.approx(0.0)
    assert diagnostics.boundary_nodes == 1
    assert diagnostics.unweighted_nodes == 0


def test_surface_area_weights_are_public() -> None:
    import tensorlbm

    assert tensorlbm.bfl_surface_area_weights is bfl_surface_area_weights


def test_full_suboff_area_adds_appendage_surface_with_bare_calibration() -> None:
    from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d
    from tensorlbm.interpolated_bc_suboff import compute_q_suboff
    from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

    nx, ny, nz, length = 80, 40, 40, 24.0
    cx, cy, cz = 28.0, 20.0, 20.0
    radius = length / (2.0 * 8.57)
    config = SuboffConfig()
    bare, geometry = build_suboff_mask(
        "bare_hull",
        nx,
        ny,
        nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=length,
        radius=radius,
        config=config,
    )
    full, _ = build_suboff_mask(
        "full",
        nx,
        ny,
        nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=length,
        radius=radius,
        config=config,
    )
    bare_near = get_near_wall_3d(bare)
    full_near = get_near_wall_3d(full)
    bare_surface = SurfaceMesh.from_gradient(bare, bare_near)
    full_surface = SurfaceMesh.from_gradient(full, full_near)
    bare_links, _ = compute_q_suboff(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        length,
        hull_type="bare_hull",
        config=config,
    )
    full_links, _ = compute_q_suboff(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        length,
        hull_type="full",
        config=config,
    )
    _, bare_diagnostics = bfl_surface_area_weights(
        bare_links,
        (bare_surface.nx_n, bare_surface.ny_n, bare_surface.nz_n),
        reference_area=float(geometry["wetted_area_lu2"]),
        boundary_mask=bare_near,
    )
    _, full_diagnostics = bfl_surface_area_weights(
        full_links,
        (full_surface.nx_n, full_surface.ny_n, full_surface.nz_n),
        calibration_factor=bare_diagnostics.calibration_factor,
        boundary_mask=full_near,
    )
    assert full_diagnostics.unweighted_nodes == 0
    assert full_diagnostics.calibrated_area > float(geometry["wetted_area_lu2"])
