"""P1 octree boundary geometry/topology tests."""
import pytest
import torch

from tensorlbm.octree_boundary.geometry import (
    SHELL_OUTSIDE,
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


def test_parent_solid_contains_only_fully_embedded_cells():
    """Cut cells are owned by the octree, never frozen on the parent L1."""
    shape = (40, 42, 44)
    center = (21.0, 20.0, 19.0)
    radius = 9.0
    shell = build_octree_shell(
        shape=shape, center=center, radius=radius,
        bl_thickness_cells=3, d_max=1, device="cpu",
    )
    zz, yy, xx = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float64) + 0.5,
        torch.arange(shape[1], dtype=torch.float64) + 0.5,
        torch.arange(shape[2], dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    offset = torch.stack((xx - center[0], yy - center[1], zz - center[2]), dim=-1).abs()
    max_dist = ((offset + 0.5).square().sum(dim=-1)).sqrt()
    expected = max_dist <= radius
    assert shell._solid is not None
    assert torch.equal(shell._solid, expected)
    assert shell._shell_mask is not None
    assert not bool((shell._solid & shell._shell_mask).any())


def test_bfl_upstream_donors_are_real_shell_leaves():
    """Cut parent cells must retain the BFL q<0.5 upstream fluid donor."""
    shell = _sphere_shell(d_max=1)
    opp = shell._opp
    linear = shell.bfl_mask & (shell.q_field < 0.5)
    # For the Bouzidi linear branch the donor is x-c_d, i.e. the opposite
    # neighbour.  It must not be replaced by a shell-interface ghost value.
    for direction in range(1, shell.Q):
        active = linear[direction]
        if bool(active.any()):
            upstream = shell.neighbor_table[int(opp[direction]), active]
            assert not bool((upstream == SHELL_OUTSIDE).any())


@pytest.mark.parametrize("d_max", (1, 2))
def test_terminal_leaves_are_all_fluid_centres(d_max: int):
    """No centre-inside leaf may be advanced as a fluid LBM node."""
    shell = _sphere_shell(d_max=d_max)
    centre = torch.tensor(shell.meta["center"], dtype=torch.float32)
    r2 = shell.meta["radius"] ** 2
    dist2 = ((shell.leaf_center - centre) ** 2).sum(dim=1)
    assert bool((dist2 > r2).all())


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
