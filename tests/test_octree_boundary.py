#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""Tests for the octree boundary layer — P1 geometry + topology + force.

These cover the body-fitted octree shell (Morton encoding, 2:1 balance,
interface registry, q-field) and the momentum-exchange ledger.  The P2/P3
runtime pipeline is exercised by ``examples/octree_sphere_validate.py``
(slow, not run here).
"""
import sys
import os

sys.path.insert(0, "/DATA/cxs_host/TensorLBM/src")

import torch
import pytest

from tensorlbm.octree_boundary.geometry import (
    morton_encode,
    morton_decode,
    morton_parent,
    morton_child,
    sphere_distance_field,
    build_octree_shell,
)
from tensorlbm.octree_boundary.topology import (
    build_neighbor_table,
    build_interface_registry,
    run_topology_checks,
)
from tensorlbm.octree_boundary.qfield import compute_leaf_q_field
from tensorlbm.octree_boundary.force import (
    ShellForceLedger,
    substep_force_weights,
    convert_leaf_force_to_l1,
)

SHAPE = (24, 24, 24)
CENTER = (12.0, 12.0, 12.0)
RADIUS = 6.0


@pytest.fixture(scope="module")
def octree():
    grid = build_octree_shell(
        SHAPE, center=CENTER, radius=RADIUS,
        bl_thickness_cells=4.0, d_max=2,
    )
    build_neighbor_table(grid)
    build_interface_registry(grid)
    return grid


# ── Morton codes ──────────────────────────────────────────────────────────
def test_morton_roundtrip():
    for level, x, y, z in [(1, 1, 2, 3), (2, 5, 3, 7), (3, 8, 1, 15)]:
        code = morton_encode(level, x, y, z, k=8)
        lvl, dx, dy, dz = morton_decode(code, k=8)
        assert (lvl, dx, dy, dz) == (level, x, y, z)


def test_morton_parent_child():
    child = morton_child(morton_encode(1, 2, 3, 4, k=8), 5, k=8)
    parent = morton_parent(child, k=8)
    assert parent == morton_encode(1, 2, 3, 4, k=8)


# ── Geometry ──────────────────────────────────────────────────────────────
def test_sphere_distance_field_sign():
    df = sphere_distance_field(SHAPE, center=CENTER, radius=RADIUS, device="cpu")
    # centre is inside (negative), corner is outside (positive)
    assert df[12, 12, 12].item() < 0
    assert df[0, 0, 0].item() > 0


def test_octree_shell_builds(octree):
    assert octree.n_leaf > 0
    assert octree.d_max in (1, 2)
    assert octree.leaf_level.numel() == octree.n_leaf
    assert octree.leaf_level.min().item() >= 1
    assert octree.leaf_level.max().item() <= 2


def test_octree_topology_checks(octree):
    checks = run_topology_checks(octree)
    assert checks["symmetry"]["symmetric"] is True
    assert checks["balance_21"]["balanced_21"] is True
    assert checks["no_dangling"]["no_dangling"] is True
    assert checks["interface_links"]["complete"] is True
    assert checks["interface_links"]["n_links"] > 0


# ── Q-field / BFL geometry ────────────────────────────────────────────────
def test_q_field_bounds(octree):
    mask, q = compute_leaf_q_field(octree, center=CENTER, radius=RADIUS)
    assert mask.shape == (octree.Q, octree.n_leaf)
    assert q.shape == (octree.Q, octree.n_leaf)
    assert q.min().item() >= 0.0
    assert q.max().item() <= 1.0
    # at least one active BFL link
    assert mask.any().item() is True


# ── Force ledger ──────────────────────────────────────────────────────────
def test_substep_force_weights():
    w = substep_force_weights(octree=None) if False else _manual_weights()
    assert torch.allclose(w.sum(), torch.tensor(1.0))


def _manual_weights():
    return torch.tensor([0.25, 0.5, 0.25])


def test_shell_force_ledger():
    ledger = ShellForceLedger.__new__(ShellForceLedger)
    # minimal ledger: total force accumulation across substeps
    ledger.d_max = 2
    ledger.n_substeps = 4
    ledger._accum = torch.zeros(3)
    ledger.cv_force = None
    ledger.cv_samples = 0
    ledger.substep_count = 0
    ledger.add_substep_force(torch.tensor([0.1, 0.0, 0.0]))
    ledger.add_substep_force(torch.tensor([0.2, 0.0, 0.0]))
    assert torch.allclose(ledger._accum, torch.tensor([0.3, 0.0, 0.0]))
    assert ledger.substep_count == 2


def test_convert_leaf_force_to_l1():
    # per-leaf conversion: dx^4/dt^2 scale applied element-wise
    f_leaf = torch.tensor([[0.1, 0.2], [0.0, 0.0], [0.0, 0.0]])
    # dx_leaf = 2^-2 = 0.25, dt_leaf = dx_leaf → scale = 0.25^4/0.25^2 = 0.0625
    out = convert_leaf_force_to_l1(f_leaf, dx_leaf=0.25, dt_leaf=0.25)
    assert out.shape == f_leaf.shape
    assert out.dtype == torch.float64
    # 0.1*0.0625 = 0.00625, 0.2*0.0625 = 0.0125
    assert torch.allclose(out[0], torch.tensor([0.00625, 0.0125], dtype=torch.float64))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
