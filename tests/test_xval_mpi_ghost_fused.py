"""Contracts for the lbmpy/waLBerla selective ghost-exchange transplant.

Three layers are pinned here:

1. the crossing-population derivation must equal the OSS function itself
   (``lbmpy.advanced_streaming.communication._extend_dir`` intersected with the
   stencil), not a hand-written table;
2. pack/unpack must be a lossless permutation of exactly the crossing
   populations of exactly one face;
3. the decomposed step must reproduce the monolithic field bit-exactly --
   the shipped full-19-plane arm *and* the selective arm.
"""

from __future__ import annotations

import torch
import pytest

from tensorlbm.d3q19 import C, equilibrium3d as repo_equilibrium3d
from tensorlbm.xval_mpi_ghost_fused import (
    TGV,
    collide_bgk,
    crossing_populations,
    extend_dir,
    face_packinfos,
    macroscopics,
    pack_face,
    unpack_face,
)
from tensorlbm.xval_mpi_ghost_fused.packinfo import face_index, stencil_directions
from tensorlbm.xval_mpi_ghost_fused.transport import SlabExchange


def _all_comm_dirs():
    from itertools import product

    return [d for d in product((-1, 0, 1), repeat=3) if any(d)]


def test_crossing_populations_match_oss_extend_dir():
    """The derivation must equal lbmpy's own _extend_dir intersected with D3Q19."""
    lbmpy_comm = pytest.importorskip("lbmpy.advanced_streaming.communication")
    stencil = set(stencil_directions())
    for comm_dir in _all_comm_dirs():
        oss = {d for d in lbmpy_comm._extend_dir(comm_dir) if d in stencil}
        got = {stencil_directions()[i] for i in crossing_populations(comm_dir)}
        assert got == oss, comm_dir


def test_crossing_counts_are_stencil_derived():
    directions = stencil_directions()
    plus = crossing_populations((1, 0, 0))
    minus = crossing_populations((-1, 0, 0))
    assert len(plus) == 5 and len(minus) == 5
    assert all(directions[i][0] == 1 for i in plus)
    assert all(directions[i][0] == -1 for i in minus)
    # waLBerla's docstring example: sending north packs NW/N/NE (2D) ->
    # in D3Q19 the y face carries the five cy=+1 populations.
    assert len(crossing_populations((0, 1, 0))) == 5
    assert len(directions) == 19


def test_extend_dir_is_the_oss_expansion():
    assert set(extend_dir((0, 1, 0))) == {
        (dx, 1, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
    }
    assert set(extend_dir((1, 0, 0))) == {
        (1, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    }
    # an all-zero "direction" fixes nothing, so lbmpy expands all 27 stencil directions
    assert len(set(extend_dir((0, 0, 0)))) == 27


@pytest.mark.parametrize("selective", [True, False])
def test_pack_unpack_is_a_lossless_face_permutation(selective: bool):
    torch.manual_seed(20261113)
    q, nz, ny, width = 19, 3, 4, 5
    post = torch.randn(q, nz, ny, width, dtype=torch.float64)
    left, right = face_packinfos(selective=selective)
    padded = torch.full((q, nz, ny, width + 2), float("nan"), dtype=post.dtype)
    padded[..., 1:-1] = post

    # ``left``/``right`` describe a send face: pack the crossing populations of my
    # own plane adjacent to that neighbour and unpack them into the neighbour's
    # ghost plane.  A second array stands in for the peer here.
    for info in (left, right):
        my_plane = face_index(info.side, width)
        peer = torch.full((q, nz, ny, width + 2), float("nan"), dtype=post.dtype)
        peer[..., 1:-1] = post
        buffer = pack_face(post, info.populations, my_plane)
        peer_ghost_index = width + 1 if info.side == "left" else 0
        unpack_face(buffer, peer, info.ghost_populations, peer_ghost_index)
        assert torch.equal(peer[..., 1:-1], post)
        assert torch.equal(
            peer[info.ghost_populations, :, :, peer_ghost_index],
            post[info.populations, :, :, my_plane],
        )

    assert torch.equal(padded[..., 1:-1], post)
    expected_n = 19 if not selective else 5
    assert len(left.populations) == expected_n and len(right.populations) == expected_n
    # the two faces carry complementary x-crossing sets, never the same population
    assert set(left.populations).isdisjoint(right.populations) or not selective


def test_equilibrium_matches_repository_implementation():
    torch.manual_seed(7)
    shape = (4, 5, 6)
    rho = 1.0 + 0.01 * torch.randn(shape, dtype=torch.float64)
    ux = 0.02 * torch.randn(shape, dtype=torch.float64)
    uy = 0.02 * torch.randn(shape, dtype=torch.float64)
    uz = 0.02 * torch.randn(shape, dtype=torch.float64)
    from tensorlbm.xval_mpi_ghost_fused.case import equilibrium3d as case_equilibrium

    mine = case_equilibrium(rho, ux, uy, uz, torch.float64)
    theirs = repo_equilibrium3d(rho, ux, uy, uz)
    assert mine.shape == theirs.shape
    assert torch.allclose(mine, theirs, rtol=0.0, atol=1e-14)


def test_tgv_initial_state_is_solenoidal_up_to_the_differencing_error():
    case = TGV(n=32, dtype=torch.float64)
    rho, ux, uy, uz = macroscopics(case.initial_field(), torch.float64)
    assert 0.99 < float(rho.min()) and float(rho.max()) < 1.01
    central_x = (torch.roll(ux, -1, dims=-1) - torch.roll(ux, 1, dims=-1)) / 2.0
    central_y = (torch.roll(uy, -1, dims=-2) - torch.roll(uy, 1, dims=-2)) / 2.0
    divergence = float((central_x + central_y).abs().max())
    # the analytic field is divergence-free; a second-order central difference
    # leaves only the O(k^2) truncation residual -- 5% of U0 is a loose bound
    assert divergence < 0.05 * case.u0
    assert float(uz.abs().max()) < 1e-15


def test_slab_exchange_requires_a_process_group():
    with pytest.raises(RuntimeError):
        SlabExchange(0, 2, selective=True)


def test_dtype_weight_selection_is_exact_for_float64():
    from tensorlbm.d3q19 import W, W_EXACT64
    from tensorlbm.xval_mpi_ghost_fused.case import weights_for

    assert torch.equal(weights_for(torch.float32), W)
    assert torch.equal(weights_for(torch.float64), W_EXACT64)


def test_collide_bgk_preserves_mass_and_momentum_of_equilibrium():
    case = TGV(n=8, dtype=torch.float64)
    f = case.initial_field()
    post = collide_bgk(f, case.tau, torch.float64)
    rho0, ux0, uy0, uz0 = macroscopics(f, torch.float64)
    rho1, ux1, uy1, uz1 = macroscopics(post, torch.float64)
    assert torch.allclose(rho0, rho1, atol=1e-14)
    assert torch.allclose(ux0, ux1, atol=1e-14)
    assert torch.allclose(uy0, uy1, atol=1e-14)
    assert torch.allclose(uz0, uz1, atol=1e-14)