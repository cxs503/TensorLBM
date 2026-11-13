"""TensorLBM 的跨 shard 并行支撑件（XVAL 移植面）。

* :mod:`tensorlbm.parallel.ppermute_halo` —— vivsim ``lax.ppermute``
  周期 halo（wrap-then-permute，零 staging）的 torch 移植。
* :mod:`tensorlbm.parallel.sharded_executor` —— 用真求解器
  ``LBMStepExecutor`` 驱动多 shard 跑法。
"""

from __future__ import annotations

from .ppermute_halo import (
    JAX_AVAILABLE,
    OssBranchFinding,
    PpermutePeriodicHalo,
    axis_pull_groups,
    tensor_component_index,
    device_pairs,
    device_pairs_reverse,
    oss_branch_findings,
    oss_stream_cross_devices,
    oss_stream_cross_devices_jax,
    ppermute,
    sender_index,
)
from .sharded_executor import ShardedPpermuteRunner, ShardedStats

__all__ = [
    "JAX_AVAILABLE",
    "OssBranchFinding",
    "PpermutePeriodicHalo",
    "ShardedPpermuteRunner",
    "ShardedStats",
    "axis_pull_groups",
    "tensor_component_index",
    "device_pairs",
    "device_pairs_reverse",
    "oss_branch_findings",
    "oss_stream_cross_devices",
    "oss_stream_cross_devices_jax",
    "ppermute",
    "sender_index",
]