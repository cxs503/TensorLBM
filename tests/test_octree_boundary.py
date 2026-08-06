"""P1 octree boundary geometry/topology tests."""
import pytest
import torch

from tensorlbm.octree_boundary.geometry import (
    build_octree_shell,
    morton_encode,
    morton_decode,
    morton_parent,
    morton_child,
)
from tensorlbm.octree_boundary.topology import (
    build_neighbor_table,
    build_interface_registry,
    run_topology_checks,
)


def _sphere_shell(**kw):
    params = dict(
        shape=(40, 40, 40), center=(20, 20, 20), radius=10,
        bl_thickness_cells=3, d_max=2, device="cpu",
    )
    params.update(kw)
    return build_octree_shell(**params)


def test_morton_roundtrip():
    """Morton encode/decode round-trips for level-1 and level-2 cells."""
    k = 5
    for level, x, y, z in ((1, 3, 5, 7), (2, 11, 13, 17), (1, 0, 0, 0)):
        bits = morton_encode(level, x, y, z, k)
        lvl, rx, ry, rz = morton_decode(bits, k)
        assert lvl == level
        assert (rx, ry, rz) == (x, y, z)


def test_morton_parent_child():
    """Child morton derives from parent + child index; decode matches."""
    k = 5
    parent_bits = morton_encode(1, 3, 5, 7, k)
    for child in range(8):
        cb = morton_child(parent_bits, child, k)
        lvl, x, y, z = morton_decode(cb, k)
        assert lvl == 2
        # parent cell must contain the child cell
        assert (x >> 1, y >> 1, z >> 1) == (3, 5, 7)


def test_shell_builds_and_saves():
    """Shell builds; cell saving far exceeds the 20% acceptance line."""
    shell = _sphere_shell()
    assert shell.n_leaf > 0
    assert shell.stats["saving_fraction"] > 0.20
    assert shell.stats["saving_fraction"] > 0.5  # typical ~94%


def test_topology_checks_pass():
    """Neighbor symmetry, 2:1 balance, no dangling, complete interface links."""
    shell = _sphere_shell()
    build_neighbor_table(shell)
    build_interface_registry(shell)
    checks = run_topology_checks(shell)
    assert checks["symmetry"]["symmetric"] is True
    assert checks["symmetry"]["viol_same_level"] == 0
    assert checks["balance_21"]["balanced_21"] is True
    assert checks["no_dangling"]["no_dangling"] is True
    assert checks["interface_links"]["complete"] is True
    assert checks["interface_links"]["n_links"] > 0


def test_level_distribution():
    """Both level-1 and level-2 leaves exist for d_max=2."""
    shell = _sphere_shell()
    assert shell.stats["n_leaf_l1"] > 0
    assert shell.stats["n_leaf_l2"] > 0
    assert shell.stats["n_leaf"] == shell.stats["n_leaf_l1"] + shell.stats["n_leaf_l2"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
