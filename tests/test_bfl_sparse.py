"""Tests for the sparse BFL kernels (bfl_boundary_link_indices and friends).

The sparse kernels must be bit-identical to the dense vectorised
implementations (same expressions, same evaluation order), intercept
upstream wraparound by default, and keep their index sets consistent with
the dense mask.  All comparisons use ``torch.equal`` (exact) on CPU; the
elementwise parity argument is device-independent.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.bfl_common import (
    bfl_bounce_back_common,
    bfl_bounce_back_sparse,
    bfl_boundary_link_indices,
    bfl_force_ledger_sparse,
    bfl_moving_wall_correction,
    bfl_moving_wall_correction_sparse,
    compute_q_sphere_common,
)
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19


def _sphere_case(n=32, radius=6.0, lattice="D3Q19"):
    cx = cy = cz = n / 2.0
    mask, q_field = compute_q_sphere_common(
        n, n, n, cx, cy, cz, radius, torch.device("cpu"), lattice=lattice
    )
    links = bfl_boundary_link_indices(mask, q_field, lattice=lattice)
    return mask, q_field, links


def _random_pair(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = (torch.randn(shape, generator=g) * 0.03).contiguous()
    f_prev = (torch.randn(shape, generator=g) * 0.05).contiguous()
    return f, f_prev


class TestLinkIndices:
    def test_link_set_matches_dense_mask(self) -> None:
        mask, q_field, links = _sphere_case()
        assert links.n_links == int(mask.sum().item())
        for d in range(19):
            idx_d = mask[d].reshape(-1).nonzero(as_tuple=True)[0]
            sl = links.direction_slice(d)
            assert sl.stop - sl.start == int(idx_d.numel())
            assert torch.equal(idx_d, links.link_idx[sl])
            if idx_d.numel():
                assert torch.equal(q_field[d].reshape(-1)[idx_d], links.link_q[sl])

    def test_link_indices_ascending_per_direction(self) -> None:
        _, _, links = _sphere_case()
        for d in range(19):
            lk = links.link_idx[links.direction_slice(d)]
            if lk.numel() > 1:
                assert bool((lk[1:] > lk[:-1]).all().item())

    def test_scatter_pairs_unique(self) -> None:
        _, _, links = _sphere_case()
        n = links.grid_shape[0] * links.grid_shape[1] * links.grid_shape[2]
        key = links.link_out_dir * n + links.link_idx
        assert key.numel() == torch.unique(key).numel()

    def test_no_wrap_for_interior_sphere(self) -> None:
        _, _, links = _sphere_case()
        assert links.n_wrapped == 0


class TestBitwiseParity:
    def test_populations_no_correction(self) -> None:
        mask, q_field, links = _sphere_case()
        f, f_prev = _random_pair((19, 32, 32, 32))
        out_dense = bfl_bounce_back_common(f, f_prev, mask, q_field)
        out_sparse = bfl_bounce_back_sparse(f, f_prev, links)
        assert torch.equal(out_dense, out_sparse)

    def test_populations_wall_correction_dense_tensor(self) -> None:
        mask, q_field, links = _sphere_case()
        f, f_prev = _random_pair((19, 32, 32, 32), seed=1)
        moving = torch.zeros((32, 32, 32), dtype=torch.bool)
        moving[10:16, 10:16, 10:16] = True
        corr = bfl_moving_wall_correction(mask, moving, (0.02, -0.01, 0.0))
        out_dense = bfl_bounce_back_common(f, f_prev, mask, q_field, wall_correction=corr)
        out_sparse = bfl_bounce_back_sparse(f, f_prev, links, wall_correction=corr)
        assert torch.equal(out_dense, out_sparse)

    def test_populations_wall_correction_sparse_builder(self) -> None:
        mask, q_field, links = _sphere_case()
        f, f_prev = _random_pair((19, 32, 32, 32), seed=2)
        moving = torch.zeros((32, 32, 32), dtype=torch.bool)
        moving[10:16, 10:16, 10:16] = True
        corr_full = bfl_moving_wall_correction(mask, moving, (0.02, -0.01, 0.0))
        corr_link = bfl_moving_wall_correction_sparse(links, moving, (0.02, -0.01, 0.0))
        out_dense = bfl_bounce_back_common(f, f_prev, mask, q_field, wall_correction=corr_full)
        out_sparse = bfl_bounce_back_sparse(f, f_prev, links, wall_correction=corr_link)
        assert torch.equal(out_dense, out_sparse)

    def test_force_ledger_default_path(self) -> None:
        mask, q_field, links = _sphere_case()
        f, f_prev = _random_pair((19, 32, 32, 32), seed=3)
        _, force_dense = bouzidi_bounce_back_d3q19(f, f_prev, mask, q_field, return_force=True)
        force_sparse, n_active = bfl_force_ledger_sparse(f, f_prev, links)
        assert n_active == links.n_links
        for a, b in zip(force_dense, force_sparse):
            assert torch.equal(a, b)

    def test_force_ledger_fraction_and_frames(self) -> None:
        mask, q_field, links = _sphere_case()
        f, f_prev = _random_pair((19, 32, 32, 32), seed=4)
        for kw in ({"boundary_fraction": 0.7}, {"force_frame": "wall"}):
            _, force_dense = bouzidi_bounce_back_d3q19(
                f, f_prev, mask, q_field, return_force=True, **kw
            )
            skw = {}
            if "boundary_fraction" in kw:
                skw["fraction"] = kw["boundary_fraction"]
            if "force_frame" in kw:
                skw["force_frame"] = kw["force_frame"]
            force_sparse, _ = bfl_force_ledger_sparse(f, f_prev, links, **skw)
            for a, b in zip(force_dense, force_sparse):
                assert torch.equal(a, b)

    def test_d3q27_populations(self) -> None:
        mask, q_field, links = _sphere_case(n=32, radius=6.0, lattice="D3Q27")
        f, f_prev = _random_pair((27, 32, 32, 32), seed=5)
        out_dense = bfl_bounce_back_common(f, f_prev, mask, q_field, lattice="D3Q27")
        out_sparse = bfl_bounce_back_sparse(f, f_prev, links)
        assert torch.equal(out_dense, out_sparse)

    def test_empty_link_set_returns_copy(self) -> None:
        mask = torch.zeros((19, 8, 8, 8), dtype=torch.bool)
        links = bfl_boundary_link_indices(mask)
        f, f_prev = _random_pair((19, 8, 8, 8), seed=6)
        out = bfl_bounce_back_sparse(f, f_prev, links)
        assert torch.equal(out, f)
        assert out is not f
        force, n_active = bfl_force_ledger_sparse(f, f_prev, links)
        assert n_active == 0
        assert all(float(v.item()) == 0.0 for v in force)


class TestUpstreamWraparound:
    def test_out_of_domain_upstream_intercepted(self) -> None:
        mask = torch.zeros((19, 9, 9, 9), dtype=torch.bool)
        mask[1, 4, 4, 0] = True  # direction +x at x=0 -> upstream x=-1
        with pytest.raises(ValueError, match="upstream"):
            bfl_boundary_link_indices(mask)

    def test_allow_wrap_reproduces_roll_semantics(self) -> None:
        mask = torch.zeros((19, 9, 9, 9), dtype=torch.bool)
        mask[1, 4, 4, 0] = True
        q_field = torch.full(mask.shape, 0.3)
        links = bfl_boundary_link_indices(mask, q_field, allow_wrap=True)
        assert links.n_wrapped == 1
        f, f_prev = _random_pair((19, 9, 9, 9), seed=7)
        out_dense = bfl_bounce_back_common(f, f_prev, mask, q_field)
        out_sparse = bfl_bounce_back_sparse(f, f_prev, links)
        assert torch.equal(out_dense, out_sparse)


class TestInputValidation:
    def test_noncontiguous_rejected(self) -> None:
        _, _, links = _sphere_case(n=24, radius=4.0)
        f = torch.empty(19, 24, 24, 48)[:, :, :, ::2]  # right shape, strided
        assert f.shape == (19, 24, 24, 24) and not f.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            bfl_bounce_back_sparse(f, f.clone(), links)

    def test_shape_mismatch_rejected(self) -> None:
        _, _, links = _sphere_case(n=24, radius=4.0)
        f, f_prev = _random_pair((19, 20, 20, 20), seed=8)
        with pytest.raises(ValueError, match="match the links"):
            bfl_bounce_back_sparse(f, f_prev, links)

    def test_wall_correction_bad_shape_rejected(self) -> None:
        _, _, links = _sphere_case(n=24, radius=4.0)
        f, f_prev = _random_pair((19, 24, 24, 24), seed=9)
        with pytest.raises(ValueError, match="wall_correction"):
            bfl_bounce_back_sparse(f, f_prev, links, wall_correction=torch.zeros(3))
