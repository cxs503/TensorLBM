"""Octree boundary layer of the hybrid AMR architecture.

P1 geometry + P2 stepping + P3 body-fitted physics:

* P1 (``geometry.py`` / ``topology.py`` / ``qfield.py``): shell cell mask,
  Morton-encoded body-fitted octree leaves (depth 1-2, 2:1 balanced),
  neighbour table with explicit cross-level donor/fanout registry,
  interface-link registry, per-leaf BFL q-field and leaf statistics.
* P2 (``stepping.py``): shell advance, time-lerped ghost fill, gather
  streaming, restriction + kinetic-flux reflux ledger.
* P3 (``bfl.py`` / ``force.py``): gather-based Bouzidi BFL on the leaves,
  momentum-exchange force with per-leaf substep weights, and the
  control-volume cross-validation with a fail-closed clearance gate.

See ``docs/octree-boundary-design.md`` for the contract.
"""
from tensorlbm.octree_boundary.distributed_stepping import (
    split_leaf_bounds,
    step_octree_shell_distributed,
    stream_gather_distributed,
)
from tensorlbm.octree_boundary.geometry import (
    DOMAIN_OUT,
    FANOUT,
    SHELL_OUTSIDE,
    SOLID,
    OctreeGrid,
    analytic_shell_volume,
    build_octree_shell,
    build_shell_cell_mask,
    cell_saving_report,
    morton_child,
    morton_decode,
    morton_decode_batch,
    morton_encode,
    morton_encode_batch,
    morton_parent,
    sphere_distance_field,
)
from tensorlbm.octree_boundary.geometry_adapters import (
    solid_mask_inside_fn,
    solid_mask_shell_fn,
    sphere_inside_fn,
)
from tensorlbm.octree_boundary.qfield import (
    compute_leaf_q_field,
    compute_q_sphere_at_points,
)
from tensorlbm.octree_boundary.topology import (
    build_interface_registry,
    build_neighbor_table,
    check_balance_21,
    check_interface_links,
    check_neighbor_symmetry,
    check_no_dangling,
    run_topology_checks,
)
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather,
    bfl_ramp_wall_velocity,
    leaf_force_weights,
    leaf_macroscopic,
    upstream_donor_table,
)
from tensorlbm.octree_boundary.force import (
    ShellForceLedger,
    build_shell_control_volume,
    convert_leaf_force_to_l1,
    substep_force_weights,
)
from tensorlbm.octree_boundary.sharding import (
    OctreeLeafShard,
    refresh_octree_f_leaf,
    shard_octree_shell,
    shards_all_finite,
    shards_f_leaf,
    split_leaf_bounds,
)
from tensorlbm.octree_boundary.stepping import (
    ShellGhostPlan,
    build_ghost_plan,
    build_plane_shell,
    build_shell_coarse_links,
    fill_ghost,
    observe_shell_interface_transfer,
    restrict_shell_to_block,
    step_octree_shell,
    step_octree_shell_sharded,
    stream_gather,
)

__all__ = [
    "OctreeGrid",
    "build_octree_shell",
    "build_shell_cell_mask",
    "sphere_distance_field",
    "split_leaf_bounds",
    "stream_gather_distributed",
    "step_octree_shell_distributed",
    "analytic_shell_volume",
    "cell_saving_report",
    "morton_encode",
    "morton_decode",
    "morton_encode_batch",
    "morton_decode_batch",
    "morton_parent",
    "morton_child",
    "compute_q_sphere_at_points",
    "compute_leaf_q_field",
    "sphere_inside_fn",
    "solid_mask_inside_fn",
    "solid_mask_shell_fn",
    "build_neighbor_table",
    "build_interface_registry",
    "check_neighbor_symmetry",
    "check_balance_21",
    "check_no_dangling",
    "check_interface_links",
    "run_topology_checks",
    "bfl_apply_gather",
    "bfl_ramp_wall_velocity",
    "leaf_force_weights",
    "leaf_macroscopic",
    "upstream_donor_table",
    "ShellForceLedger",
    "build_shell_control_volume",
    "convert_leaf_force_to_l1",
    "substep_force_weights",
    "OctreeLeafShard",
    "refresh_octree_f_leaf",
    "shard_octree_shell",
    "shards_all_finite",
    "shards_f_leaf",
    "split_leaf_bounds",
    "ShellGhostPlan",
    "build_ghost_plan",
    "build_plane_shell",
    "build_shell_coarse_links",
    "fill_ghost",
    "observe_shell_interface_transfer",
    "restrict_shell_to_block",
    "step_octree_shell",
    "step_octree_shell_sharded",
    "stream_gather",
    "SHELL_OUTSIDE",
    "SOLID",
    "DOMAIN_OUT",
    "FANOUT",
    "REMOTE",
]

