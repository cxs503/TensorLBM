"""Runner/sharding facade forwarding of the force cadence frame (kappa API).

The runner facade (octree_shell_runner.make_shell_bfl_fn) and the sharding
facade must both forward host_steps_per_shell_call into the leaf force
weights.  This test rides with the runner slice because the kwarg lives in
octree_shell_runner.
"""

import torch

from tensorlbm.octree_boundary.bfl import leaf_force_spatial_weights
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.sharding import shard_octree_shell


def _shell(d_max: int):
    return build_octree_shell(
        (32, 32, 32),
        center=(16.0, 16.0, 16.0),
        radius=9.0,
        bl_thickness_cells=4.0,
        d_max=d_max,
        transition=1,
        lattice="D3Q19",
        device="cpu",
    )


def _legacy_weights(octree, *, include_spatial: bool) -> torch.Tensor:
    levels = octree.leaf_level.to(torch.float64)
    w = 2.0 ** (-(float(octree.d_max) - levels))
    if include_spatial:
        w = w * leaf_force_spatial_weights(octree, reference_level=1)
    return w


def test_runner_and_shard_facade_forward_the_frame():
    from tensorlbm.octree_shell_runner import make_shell_bfl_fn

    octree = _shell(2)
    legacy = _legacy_weights(octree, include_spatial=False)
    bfl_fn = make_shell_bfl_fn(octree, None, torch.device("cpu"), host_steps_per_shell_call=2.0)
    assert torch.equal(bfl_fn.leaf_weights, legacy * 2.0)
    bfl_default = make_shell_bfl_fn(octree, None, torch.device("cpu"))
    assert torch.equal(bfl_default.leaf_weights, legacy)

    shards = shard_octree_shell(
        octree,
        [torch.device("cpu"), torch.device("cpu")],
        host_steps_per_shell_call=2.0,
    )
    for shard in shards:
        expected = (legacy * 2.0)[slice(shard.lo, shard.hi)]
        assert torch.equal(shard.force_weights, expected)
