"""Regression tests for sparse-leaf LES direction and shard context."""

from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.geometry import SOLID
from tensorlbm.octree_boundary.les import _gradient, leaf_les_collide


def test_sparse_gradient_uses_spatial_axis_not_rest_direction() -> None:
    # Three leaves along z.  Direction 5 is +z and 6 is -z; direction 0 is
    # rest and must never be used as the z derivative stencil.
    n = 3
    nt = torch.full((19, n), SOLID, dtype=torch.int64)
    nt[5, 0], nt[5, 1] = 1, 2
    nt[6, 1], nt[6, 2] = 0, 1
    u = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    f = torch.zeros((19, n), dtype=torch.float64)
    grad_z = _gradient(f, 2, u, nt, n, 1.0)
    torch.testing.assert_close(grad_z, torch.ones_like(u), atol=0.0, rtol=0.0)


def test_leaf_les_supports_d3q19() -> None:
    n = 8
    f = equilibrium3d(
        torch.ones((1, 1, n), dtype=torch.float64),
        torch.zeros((1, 1, n), dtype=torch.float64),
        torch.zeros((1, 1, n), dtype=torch.float64),
        torch.zeros((1, 1, n), dtype=torch.float64),
    )
    nt = torch.full((19, n), SOLID, dtype=torch.int64)
    centers = torch.zeros((n, 3), dtype=torch.float32)
    out = leaf_les_collide(
        f,
        0.51,
        nt,
        leaf_level=torch.ones(n, dtype=torch.int64),
        leaf_center=centers,
    )
    assert out.shape == f.shape
    assert bool(torch.isfinite(out).all())
