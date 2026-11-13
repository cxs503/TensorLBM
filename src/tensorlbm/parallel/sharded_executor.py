"""Sharded runner: 用真求解器（``LBMStepExecutor``）+ ppermute 周期 halo 跑多 rank。

这是 ``ppermute_halo`` 的**集成调用面**（L3）：``tensorlbm.lbm_step.LBMStepExecutor``
是仓库里 53 个求解器共用的逐步执行器；本模块把同一条求解路径切成 N 个 shard，
每个 shard 用同一个 ``LBMStepExecutor``（同一套 collide/stream 公式），
shard 之间只多一次 :class:`~tensorlbm.parallel.ppermute_halo.PpermutePeriodicHalo`
集合交换。

放置约定（与 OSS 一致）
----------------------
``LBMStepExecutor`` 的 streaming 是 ``torch.roll``（拉式 + 块内自回卷），
所以 halo 必须在 **streaming 之后**修边界；交换的仅是沿分块轴穿越的
``|c_axis| == 1`` 方向平面（D3Q19 z-slab = 5 + 5 = 10 个平面），
而不是全部 Q 个平面。

时序等价性
----------
``run()`` 把一次交换放在 ``step()`` 返回之后（下一步 collide 之前）。
BGK collide 是逐点局部的，不跨邻居读，所以"post-stream 交换"与
"post-step 交换"在周期域上数值等价 —— 由 ``tests/test_parallel_ppermute_halo.py``
对 NP=1 单域解逐位比对证明（max|Δf| = 0）。
若 shard 内部面上有 post-streaming 边界条件（如 Zou-He），交换必须挪到
BC 之前；本模块用 ``boundary_fn`` 时会拒绝（见 ``allow_boundary``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from ..lbm_step import LBMStepExecutor
from .ppermute_halo import PpermutePeriodicHalo

__all__ = ["ShardedPpermuteRunner"]


def _as_tensor(f: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(f, torch.Tensor):
        return f.to(device=device, dtype=dtype)
    return torch.as_tensor(f, device=device, dtype=dtype)


@dataclass
class ShardedStats:
    """每个 shard 的逐步记账（用于报告里的通信成本表）。"""

    n_ranks: int
    steps: int
    plane_shape: tuple[int, ...]
    planes_per_exchange: int
    bytes_per_exchange: int
    tensor_ops_per_exchange: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_ranks": self.n_ranks,
            "steps": self.steps,
            "plane_shape": list(self.plane_shape),
            "planes_per_exchange": self.planes_per_exchange,
            "bytes_per_exchange_per_rank": self.bytes_per_exchange,
            "total_bytes_per_step": self.bytes_per_exchange * self.n_ranks,
            "tensor_ops_per_exchange_per_rank": self.tensor_ops_per_exchange,
        }


class ShardedPpermuteRunner:
    """沿 ``axis`` 把周期域切成 ``n_ranks`` 块，用 ppermute halo 串起来。

    Args:
        f_global: 初始分布 ``(Q, nz, ny, nx)``（torch 或 numpy）。
        n_ranks: shard 数（= 模拟 device 数）。
        lattice: ``"D3Q19"`` / ``"D3Q27"``。
        axis: 分块的空间轴（0 → tensor dim 1）。
        tau, dtype, device: 传给 ``LBMStepExecutor``。
        collide_fn: ``"bgk"`` 或外置碰撞算子（与单域跑法完全一致）。
        boundary_fn: 可选；内部面上有 BC 时与 post-step 交换不等价，
            故本 runner 要求 ``None``（fail-closed）。
    """

    def __init__(
        self,
        f_global: Any,
        *,
        n_ranks: int,
        lattice: str = "D3Q19",
        axis: int = 0,
        tau: float = 0.6,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        collide_fn: str | Callable[..., torch.Tensor] = "bgk",
        boundary_fn: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        if boundary_fn is not None:
            raise ValueError(
                "ShardedPpermuteRunner applies the periodic halo after step(); "
                "post-streaming boundary conditions would see uncorrected ghost "
                "planes. Pass boundary_fn=None or use HaloExchange."
            )
        self.device = torch.device(device)
        self.dtype = dtype
        self.lattice = lattice.upper()
        self.axis = axis
        self.n_ranks = int(n_ranks)
        self.tau = tau
        self.n_steps = 0

        f = _as_tensor(f_global, dtype, self.device)
        if f.dim() != 4:
            raise ValueError(f"f_global must be (Q, n0, n1, n2), got {tuple(f.shape)}")
        self.shape = tuple(f.shape)
        self.Q = self.shape[0]
        self.spatial = self.shape[1:]
        if not (0 <= axis < 3):
            raise ValueError(f"axis must be 0..2, got {axis}")
        n_axis = self.spatial[axis]
        if n_axis % self.n_ranks != 0:
            raise ValueError(
                f"axis {axis} length {n_axis} not divisible by n_ranks {self.n_ranks}"
            )
        self.n_local = n_axis // self.n_ranks

        # 分块：沿 axis 直接切
        self.shards: list[torch.Tensor] = [
            f.narrow(axis + 1, r * self.n_local, self.n_local).clone().contiguous()
            for r in range(self.n_ranks)
        ]

        self.halo = PpermutePeriodicHalo.from_lattice(
            lattice=self.lattice, n_ranks=self.n_ranks, axis=axis
        )

        self.executors: list[LBMStepExecutor] = []
        for shard in self.shards:
            local = tuple(shard.shape[1:])
            dims = {"x": local[-1], "y": local[-2], "z": local[-3]}
            self.executors.append(
                LBMStepExecutor(
                    self.lattice,
                    collide_fn=collide_fn,
                    device=self.device,
                    nx=dims["x"],
                    ny=dims["y"],
                    nz=dims["z"],
                    tau=tau,
                    dtype=dtype,
                )
            )

    # -- 主循环 -------------------------------------------------------
    def step(self, n_steps: int = 1) -> None:
        """推进 ``n_steps`` 步：每步 = N×（collide+stream）+ 1 次 ppermute 交换。"""
        for _ in range(int(n_steps)):
            new_shards: list[torch.Tensor] = []
            for r, ex in enumerate(self.executors):
                f_out, _diag = ex.step(self.shards[r])
                new_shards.append(f_out)
            self.shards = new_shards
            self.halo.apply(self.shards)  # post-stream（此处 = post-step）周期修正
        self.n_steps += int(n_steps)

    # -- 取回 ---------------------------------------------------------
    def gather(self) -> torch.Tensor:
        """拼回全局分布 ``(Q, nz, ny, nx)``。"""
        return torch.cat(self.shards, dim=self.axis + 1)

    def gather_macro(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """拼回宏观量 ``(rho, ux, uy, uz)``，逐 shard 用仓库自带 macroscopic。"""
        if self.lattice == "D3Q19":
            from ..d3q19 import macroscopic3d as macro
        elif self.lattice == "D3Q27":
            from ..d3q27 import macroscopic27 as macro
        else:  # pragma: no cover
            raise ValueError(self.lattice)
        outs = [macro(s) for s in self.shards]
        rho = torch.cat([o[0] for o in outs], dim=self.axis)
        ux = torch.cat([o[1] for o in outs], dim=self.axis)
        uy = torch.cat([o[2] for o in outs], dim=self.axis)
        uz = torch.cat([o[3] for o in outs], dim=self.axis)
        return rho, ux, uy, uz

    # -- 台账 ---------------------------------------------------------
    @property
    def plane_shape(self) -> tuple[int, ...]:
        """单个通信平面的尺寸（去掉 Q 维与分块轴）。"""
        return tuple(d for i, d in enumerate(self.spatial) if i != self.axis)

    def stats(self) -> ShardedStats:
        itemsize = torch.tensor([], dtype=self.dtype).element_size()
        # 每个 rank 都搬它自己那一份 10 个平面（permutation 是全体同时交换）
        per_rank = self.halo.wire_bytes_per_step(self.plane_shape, itemsize)
        return ShardedStats(
            n_ranks=self.n_ranks,
            steps=self.n_steps,
            plane_shape=self.plane_shape,
            planes_per_exchange=self.halo.planes_per_exchange,
            bytes_per_exchange=per_rank,
            tensor_ops_per_exchange=self.halo.tensor_ops // max(self.halo.exchanges, 1),
        )

    @property
    def direction_groups(self) -> dict[str, Sequence[int]]:
        return {"pos": self.halo.pos_group, "neg": self.halo.neg_group}