from __future__ import annotations

import torch

from tensorlbm.d3q19 import C
from tensorlbm.interpolated_bc_suboff import (
    SUBOFF_APPENDAGE_LINK_SCHEME,
    compute_q_suboff,
    refine_q_suboff_appendages,
)
from tensorlbm.suboff_cad import (
    SuboffConfig,
    build_suboff_mask,
    suboff_appendages_contain_points,
)
from tensorlbm.suboff_static_amr import (
    apply_suboff_appendage_halfway_links,
    count_suboff_appendage_boundary_links,
)


def test_full_suboff_appendage_links_receive_audited_halfway_fallback() -> None:
    nx, ny, nz = 180, 70, 70
    center = (60.0, 35.0, 35.0)
    length = 120.0
    config = SuboffConfig()
    solid, _ = build_suboff_mask(
        "full", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2],
        length=length, config=config,
    )
    mask, q = compute_q_suboff(
        nx, ny, nz, *center, length,
        hull_type="full", config=config, solid_mask=solid,
    )

    count = apply_suboff_appendage_halfway_links(
        solid, mask, q, center=center, length=length, config=config,
    )

    assert count > 0
    assert int((q == 0.5).sum()) >= count


def test_appendage_link_contract_validates_shapes() -> None:
    solid = torch.zeros((8, 8, 8), dtype=torch.bool)
    mask = torch.zeros((19, 8, 8, 8), dtype=torch.bool)
    q = torch.zeros((19, 8, 8, 7))
    try:
        apply_suboff_appendage_halfway_links(
            solid, mask, q, center=(4.0, 4.0, 4.0), length=4.0,
        )
    except ValueError as error:
        assert "must match" in str(error)
    else:
        raise AssertionError("shape mismatch must fail closed")


def test_continuous_appendage_predicate_reproduces_voxel_components() -> None:
    nx, ny, nz, length = 180, 70, 70, 120.0
    center = (60.0, 35.0, 35.0)
    bare = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    full = build_suboff_mask(
        "full", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    z, y, x = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )

    appendages = suboff_appendages_contain_points(
        x, y, z, center=center, length=length,
    )

    assert torch.equal(appendages & ~bare, full & ~bare)


def test_full_suboff_appendage_links_use_continuous_bisection_q() -> None:
    nx, ny, nz, length = 180, 70, 70, 120.0
    center = (70.0, 35.0, 35.0)
    full = build_suboff_mask(
        "full", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    bare = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    mask, halfway_q = compute_q_suboff(
        nx, ny, nz, *center, length,
        hull_type="full", solid_mask=full,
    )

    exact_q, diagnostics = refine_q_suboff_appendages(
        mask, halfway_q, full, bare, center=center, length=length,
    )

    assert diagnostics.scheme == SUBOFF_APPENDAGE_LINK_SCHEME
    assert diagnostics.target_links == count_suboff_appendage_boundary_links(
        full, bare,
    )
    assert diagnostics.minimum_q is not None and diagnostics.minimum_q > 0.0
    assert diagnostics.maximum_q is not None and diagnostics.maximum_q < 1.0
    selected = torch.zeros_like(mask)
    appendage_only = full & ~bare
    for direction in range(1, 19):
        cx, cy, cz = (int(value) for value in C[direction].tolist())
        selected[direction] = mask[direction] & torch.roll(
            appendage_only, shifts=(-cz, -cy, -cx), dims=(0, 1, 2),
        )
    assert int(selected.sum()) == diagnostics.target_links
    assert torch.any(exact_q[selected] != 0.5)
    assert torch.equal(exact_q[~selected], halfway_q[~selected])


def test_appendage_q_can_update_a_fresh_production_field_in_place() -> None:
    nx, ny, nz, length = 120, 50, 50, 80.0
    center = (60.0, 25.0, 25.0)
    full = build_suboff_mask(
        "full", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    bare = build_suboff_mask(
        "bare_hull", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2], length=length,
    )[0]
    mask, q = compute_q_suboff(
        nx, ny, nz, *center, length,
        hull_type="full", solid_mask=full,
    )

    refined, diagnostics = refine_q_suboff_appendages(
        mask, q, full, bare, center=center, length=length, inplace=True,
    )

    assert refined.data_ptr() == q.data_ptr()
    assert diagnostics.target_links > 0
