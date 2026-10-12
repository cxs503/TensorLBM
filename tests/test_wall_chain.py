"""Unit tests for the shared per-level BFL wall-chain module.

Covers the pieces factored out of the uniform/AMR runners by the owner
directive 2026-10-11: halfway BFL links, surface-mesh source dispatch and
the one-call level chain, plus the geometric-vacuousness counter.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.wall_chain import (
    bfl_halfway_links_3d,
    build_level_wall_chain,
    count_boundary_links_outside_box,
    suboff_surface_mesh,
)

_NZ, _NY, _NX = 12, 12, 14
_SLICE_LO, _SLICE_HI = 2, 6  # slab interior: away from the roll-wrapped faces


def _slab_solid() -> torch.Tensor:
    """Fluid-bounded solid slab (torch.roll wraps at the domain faces)."""
    solid = torch.zeros((_NZ, _NY, _NX), dtype=torch.bool)
    solid[_SLICE_LO:_SLICE_HI] = True
    return solid


def _down_directions() -> list[int]:
    """D3Q19 directions with cz < 0 (no body diagonals in Q19: 5 of them)."""
    return [q for q in range(19) if float(C[q][2]) < 0.0]


def _up_directions() -> list[int]:
    return [q for q in range(19) if float(C[q][2]) > 0.0]


def test_halfway_links_flat_slab_count_and_location() -> None:
    solid = _slab_solid()
    mask, q_field = bfl_halfway_links_3d(solid)
    assert mask.shape == (19, _NZ, _NY, _NX)
    assert mask.dtype == torch.bool
    # q = 0.5 everywhere on the mask (halfway convention)
    assert torch.all(q_field[mask] == 0.5)
    # links live exactly on the two boundary fluid planes of the slab
    planes = mask.any(dim=0)
    assert planes[_SLICE_LO - 1].all() and planes[_SLICE_HI].all()
    others = planes.clone()
    others[_SLICE_LO - 1] = False
    others[_SLICE_HI] = False
    assert not bool(others.any())
    # a direction-q link lives on the fluid cell whose q-neighbour is solid
    # (the halfway-BFL streaming target): cz>0 below the slab, cz<0 above,
    # in-plane directions carry no links for the infinite slab
    down = _down_directions()
    up = _up_directions()
    assert len(down) == 5 and len(up) == 5  # D3Q19 has no body diagonals
    for direction in range(1, 19):
        if direction in down:
            assert bool(mask[direction][_SLICE_HI].all())
            assert not bool(mask[direction][_SLICE_LO - 1].any())
        elif direction in up:
            assert bool(mask[direction][_SLICE_LO - 1].all())
            assert not bool(mask[direction][_SLICE_HI].any())
        else:
            assert not bool(mask[direction].any())
    assert int(mask.sum().item()) == 10 * _NY * _NX


def test_halfway_links_singleton_block_all_directions() -> None:
    solid = torch.zeros((14, 14, 14), dtype=torch.bool)
    solid[6:8, 6:8, 6:8] = True
    mask, _ = bfl_halfway_links_3d(solid)
    # every non-rest direction has a fluid cell bordering the block
    directions = [q for q in range(1, 19) if bool(mask[q].any())]
    assert len(directions) == 18
    assert int(mask.sum().item()) > 0


def test_count_boundary_links_outside_box() -> None:
    solid = _slab_solid()
    expected = 10 * _NY * _NX  # both boundary fluid planes, Q19 directions
    everything = torch.ones_like(solid)
    assert count_boundary_links_outside_box(solid, everything) == 0
    far_field = torch.zeros_like(solid)
    far_field[_SLICE_HI + 2 :] = True  # child owns the far field only
    assert count_boundary_links_outside_box(solid, far_field) == expected
    covering = torch.zeros_like(solid)
    covering[_SLICE_LO - 1 : _SLICE_HI + 1] = True  # owns both fluid planes
    assert count_boundary_links_outside_box(solid, covering) == 0


def _small_suboff(hull_type: SuboffHullType) -> tuple[torch.Tensor, tuple[float, ...]]:
    nx, ny, nz, length = 96, 48, 48, 48.0
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    solid, _ = build_suboff_mask(
        hull_type,
        nx,
        ny,
        nz,
        cx=center[0],
        cy=center[1],
        cz=center[2],
        length=length,
        device="cpu",
    )
    return solid, center


def test_surface_mesh_source_dispatch() -> None:
    solid, (cx, cy, cz) = _small_suboff(SuboffHullType.WITH_SAIL)
    near = get_near_wall_3d(solid)
    assert not bool((near & solid).any())
    analytic = suboff_surface_mesh(solid, near, cx, cy, cz, 48.0, "suboff")
    gradient = suboff_surface_mesh(solid, near, cx, cy, cz, 48.0, "gradient")
    hybrid = suboff_surface_mesh(solid, near, cx, cy, cz, 48.0, "hybrid")
    for mesh in (analytic, gradient, hybrid):
        for component in (mesh.nx_n, mesh.ny_n, mesh.nz_n):
            assert torch.isfinite(component[near]).all()
    # hybrid is a genuine blend: it differs from both pure sources
    assert not torch.equal(hybrid.nx_n, analytic.nx_n)
    assert not torch.equal(hybrid.nx_n, gradient.nx_n)


def test_surface_mesh_rejects_unknown_source() -> None:
    solid, (cx, cy, cz) = _small_suboff(SuboffHullType.BARE_HULL)
    near = get_near_wall_3d(solid)
    with pytest.raises(ValueError, match="mesh_source"):
        suboff_surface_mesh(solid, near, cx, cy, cz, 48.0, "bogus")


def test_build_level_wall_chain_fields() -> None:
    solid, (cx, cy, cz) = _small_suboff(SuboffHullType.BARE_HULL)
    chain = build_level_wall_chain(
        solid,
        wall_nu=1.3e-4,
        cx=cx,
        cy=cy,
        cz=cz,
        length=48.0,
        mesh_source="hybrid",
    )
    assert chain.wall_nu == pytest.approx(1.3e-4)
    assert chain.near.shape == solid.shape
    assert chain.bfl_mask.shape == (19, *solid.shape)
    assert chain.bfl_q.dtype == torch.float32
    assert torch.all(chain.bfl_q[chain.bfl_mask] == 0.5)
    for component in chain.normals:
        assert component.shape == solid.shape
    # links live on fluid cells only
    assert not bool((chain.bfl_mask.any(dim=0) & solid).any())
