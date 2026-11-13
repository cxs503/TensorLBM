"""XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO — vivsim 的 ``lax.ppermute`` 周期 halo 移植。

Reference (OSS, MIT)
-------------------
* repo   : https://github.com/haimingz/vivsim  (``vivsim`` 2.0.0, MIT,
           Copyright (c) 2024 Haiming Zhu)
* commit : ``fb84b12f02e18f3b9dfdf738d6a1a318d81539f9`` (2026-06-28,
           "Fix shifted LBM equilibrium (#46)")
* file   : ``vivsim/multidevice.py`` (39 lines) — ``stream_cross_devices()``
* context: ``vivsim/lbm/basic.py`` 的 ``streaming()`` / ``shift_x_pos`` /
           ``shift_y_pos`` 给出拉式（pull）周期 streaming 约定；
           ``vivsim/lbm/lattice.py`` 给出 ``right/left/up/down_dirs`` 分组。

OSS 做法（一句话）
------------------
局部 streaming **先做整块自回卷**（shift/roll，边界处是本地自环的错误值），
随后用 **一次 ``jax.lax.ppermute``** 把每个 rank 的"自回卷面"搬到邻居的同名面：
自回卷之后 rank r 的 plane 0 恰好装着它自己的最后一个 plane、
plane -1 恰好装着它自己的第一个 plane —— 正是邻居需要的内容，于是
**不需要任何 pack/unpack staging 缓冲**，一次集合置换就完成 halo；
device 维的周期性直接由 ``(i, (i+1) % n)`` 取模对列表给出。

本模块把它移植到 TensorLBM 的 torch/numpy 栈（本沙箱无 jax，见
``JAX_AVAILABLE``），并保留一条真正的 ``jax.lax.ppermute`` 通路
（``oss_stream_cross_devices_jax``，jax 存在时才可用，本单未执行）。

术语
----
* ``shard``  : 一个 rank 拥有的分布函数块，布局与 TensorLBM 一致
  ``(Q, nz_local, ny, nx)``（``LBMStepExecutor`` 的 ``torch.roll`` streaming
  在每块内部自带周期回卷）。
* ``axis``   : 分块所在的空间轴（``0`` → tensor dim 1，默认 ``z``）。
* "低面 0 / 高面 -1" ：沿该轴的首/末平面。
* ``ppermute`` 语义：``out[dst] = in[src]``，``perm`` 为 ``(src, dst)`` 对列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

try:  # 真 OSS 原语；只有装了 jax 的环境才有
    import jax
    from jax import lax
except Exception:  # pragma: no cover - 本沙箱（loongarch64）无 jax
    jax = None  # type: ignore[assignment]
    lax = None  # type: ignore[assignment]

JAX_AVAILABLE: bool = jax is not None

__all__ = [
    "JAX_AVAILABLE",
    "PpermutePeriodicHalo",
    "OssBranchFinding",
    "axis_pull_groups",
    "tensor_component_index",
    "device_pairs",
    "device_pairs_reverse",
    "oss_stream_cross_devices",
    "oss_stream_cross_devices_jax",
    "ppermute",
    "sender_index",
]


# ---------------------------------------------------------------------------
# lax.ppermute 语义（纯数据搬运，可 numpy / torch 通用）
# ---------------------------------------------------------------------------


def device_pairs(n_ranks: int) -> tuple[tuple[int, int], ...]:
    """OSS ``multidevice.py:27`` 原式：``(i, (i+1) % n)``。

    语义 ``out[dst] = in[src]`` ⇒ rank ``r`` 收到 rank ``r-1`` 的面。
    """
    _check_ranks(n_ranks)
    return tuple((i, (i + 1) % n_ranks) for i in range(n_ranks))


def device_pairs_reverse(n_ranks: int) -> tuple[tuple[int, int], ...]:
    """OSS ``multidevice.py:28`` 原式：``((i+1) % n, i)``。

    语义 ⇒ rank ``r`` 收到 rank ``r+1`` 的面。
    """
    _check_ranks(n_ranks)
    return tuple(((i + 1) % n_ranks, i) for i in range(n_ranks))


def _check_ranks(n_ranks: int) -> None:
    if not isinstance(n_ranks, int) or n_ranks < 1:
        raise ValueError(f"n_ranks must be a positive int, got {n_ranks!r}")


def sender_index(perm: Sequence[tuple[int, int]], n_ranks: int) -> list[int]:
    """把 ``(src, dst)`` 对列表转成 gather 索引：``out[r] = in[sender[r]]``。

    fail-closed：``perm`` 必须是 ``range(n_ranks)`` 上的**置换**（每个 dst
    出现恰好一次），与 ``jax.lax.ppermute`` 的前置条件一致。
    """
    _check_ranks(n_ranks)
    sender = [-1] * n_ranks
    for src, dst in perm:
        if not (0 <= src < n_ranks and 0 <= dst < n_ranks):
            raise ValueError(f"perm entry {(src, dst)} out of range [0,{n_ranks})")
        if sender[dst] != -1:
            raise ValueError(f"perm is not a permutation: destination {dst} used twice")
        sender[dst] = src
    if any(s < 0 for s in sender):
        raise ValueError(f"perm is not a permutation of range({n_ranks}): {perm!r}")
    return sender


def ppermute(planes: Sequence[Any], perm: Sequence[tuple[int, int]]) -> list[Any]:
    """``jax.lax.ppermute`` 的等价实现（同一进程内的多 shard 模拟）。

    ``out[dst] = in[src]``，且**同时性**成立（先整体 stack 再 gather，
    因此不存在"边写边读"的顺序依赖 —— 这也是 OSS 一次 ppermute 全体面
    同时搬运的语义）。支持 ``torch.Tensor`` 与 ``numpy.ndarray``。
    """
    n = len(planes)
    if n == 0:
        raise ValueError("ppermute needs at least one shard")
    sender = sender_index(perm, n)
    first = planes[0]
    if isinstance(first, torch.Tensor):
        stacked = torch.stack([p.contiguous() for p in planes], dim=0)
        idx = torch.as_tensor(sender, device=first.device)
        return [stacked[i] for i in idx.tolist()]
    stacked = _np_stack(planes)
    return [stacked[i] for i in sender]


def _np_stack(planes: Sequence[Any]) -> Any:
    import numpy as np

    return np.stack([np.ascontiguousarray(p) for p in planes], axis=0)


# ---------------------------------------------------------------------------
# 方向分组（由 lattice C 的符号给出，与 OSS 的 up/right/..._dirs 同构）
# ---------------------------------------------------------------------------


def tensor_component_index(n_components: int, axis: int) -> int:
    """TensorLBM 的轴 ↔ lattice C 列号映射。

    仓库布局是 ``(Q, nz, ny, nx)``（见 ``lbm_step`` 的 ``shape_f``），而
    D3Q19 的 C 表列序是 ``(cx, cy, cz)`` —— 所以 tensor 空间轴 ``axis``
    （0→``nz`` … 2→``nx``）对应 C 的第 ``n_components - 1 - axis`` 列。
    这个映射由 ``LBMStepExecutor._stream_preallocated`` 的实际 roll 公式
    （``sx, sy, sz = shifts[q]`` 后 ``roll(shifts=(sz,sy,sx), dims=(0,1,2))``）
    反推得到，不是约定俗成。
    """
    if not (0 <= axis < n_components):
        raise ValueError(f"axis {axis} out of range for {n_components} components")
    return n_components - 1 - axis


def axis_pull_groups(c: Any, axis: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按沿 ``axis`` 的速度分量符号给方向分组（``axis`` = C 的列号）。

    在**拉式** streaming（``out[x] = f_old[x - c]``）下：

    * ``c_axis > 0`` ⇒ 从下标更小的邻居取 ⇒ 低面 ``0`` 需要 rank-1；
    * ``c_axis < 0`` ⇒ 从下标更大的邻居取 ⇒ 高面 ``-1`` 需要 rank+1。

    Returns
    -------
    (pos_group, neg_group) : 两个方向下标元组（``c_axis`` 为 ±1）。
    """
    cols = c.shape[1]
    if not (0 <= axis < cols):
        raise ValueError(f"axis {axis} out of range for lattice with {cols} components")
    pos: list[int] = []
    neg: list[int] = []
    for q in range(c.shape[0]):
        component = int(c[q, axis])
        if component > 0:
            pos.append(q)
        elif component < 0:
            neg.append(q)
    return tuple(pos), tuple(neg)


# ---------------------------------------------------------------------------
# wrap-then-permute 周期 halo
# ---------------------------------------------------------------------------


def _plane(shard: torch.Tensor, group: Sequence[int], plane: int, axis: int) -> torch.Tensor:
    """取出 ``shard[group, plane(沿 axis+1)]`` → ``(len(group), *rest)``。"""
    idx = torch.as_tensor(list(group), device=shard.device, dtype=torch.long)
    return shard.index_select(0, idx).select(axis + 1, plane)


def _assign(
    shard: torch.Tensor,
    group: Sequence[int],
    plane: int,
    axis: int,
    values: torch.Tensor,
) -> None:
    """把 ``values`` 原地写回 ``shard[group, plane]``（``f.at[...].set`` 等价）。"""
    index: list[Any] = [slice(None)] * shard.dim()
    index[0] = torch.as_tensor(list(group), device=shard.device, dtype=torch.long)
    index[axis + 1] = plane
    shard[tuple(index)] = values


@dataclass
class PpermutePeriodicHalo:
    """vivsim ``stream_cross_devices`` 的通用（任意轴 / 任意 lattice）移植。

    ``apply(shards)`` 必须在**局部 streaming 之后**调用：局部 streaming 的
    自回卷已经把每个 rank 的"另一侧平面"放进 0 / -1 面，因此置换交换的对
    象正是邻居所需内容（这是 OSS 做法的核心，见模块 docstring）。

    Parameters
    ----------
    n_ranks:
        rank 数（device 维长度）。
    pos_group, neg_group:
        沿分块轴 ``c_axis`` 为正 / 负的方向下标（见 :func:`axis_pull_groups`）。
    axis:
        空间轴（``0`` == tensor dim 1 == ``nz``）；方向分量取 C 的第
        ``ncomp-1-axis`` 列（见 :func:`tensor_component_index`）。
    lattice:
        仅用于台账/报告标注。
    """

    n_ranks: int
    pos_group: tuple[int, ...]
    neg_group: tuple[int, ...]
    axis: int = 0
    lattice: str = "D3Q19"

    # 运行时计数（不参与数值）：真正的搬运次数 / 搬运算子数
    exchanges: int = field(default=0, init=False)
    tensor_ops: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _check_ranks(self.n_ranks)
        if not self.pos_group and not self.neg_group:
            raise ValueError("at least one of pos_group/neg_group must be non-empty")

    # -- 构造 ---------------------------------------------------------
    @classmethod
    def from_lattice(
        cls,
        lattice: str = "D3Q19",
        n_ranks: int = 1,
        axis: int = 0,
    ) -> "PpermutePeriodicHalo":
        """从 TensorLBM 的 lattice C 表构造（方向分组不手抄，程序生成）。"""
        name = lattice.upper()
        if name == "D3Q19":
            from ..d3q19 import C
        elif name == "D3Q27":
            from ..d3q27 import C
        elif name == "D2Q9":
            from ..d2q9 import C
        else:  # pragma: no cover - 显式不支持
            raise ValueError(f"unsupported lattice {lattice!r}")
        component = tensor_component_index(C.shape[1], axis)
        pos, neg = axis_pull_groups(C, component)
        return cls(
            n_ranks=n_ranks,
            pos_group=pos,
            neg_group=neg,
            axis=axis,
            lattice=name,
        )

    # -- 主操作 -------------------------------------------------------
    def apply(self, shards: Sequence[torch.Tensor]) -> None:
        """原地完成一次周期 halo（等价于 ``k`` 次 ``lax.ppermute``）。

        ``k`` = 该轴上两侧都非空时 2、否则 1；每侧 = 1 次 stack-gather +
        1 次 scatter，**无 staging 缓冲、无逐 rank 打包循环**。
        """
        n = len(shards)
        if n != self.n_ranks:
            raise ValueError(f"expected {self.n_ranks} shards, got {n}")
        if n == 1:
            # OSS 的取模对列表在本情形退化为 [(0, 0)]（自身恒等）⇒ 无搬运。
            return
        with torch.no_grad():
            if self.pos_group:
                planes = [_plane(s, self.pos_group, 0, self.axis) for s in shards]
                moved = ppermute(planes, device_pairs(n))
                for s, m in zip(shards, moved):
                    _assign(s, self.pos_group, 0, self.axis, m)
                self.tensor_ops += 2 * n + 1
            if self.neg_group:
                planes = [_plane(s, self.neg_group, -1, self.axis) for s in shards]
                moved = ppermute(planes, device_pairs_reverse(n))
                for s, m in zip(shards, moved):
                    _assign(s, self.neg_group, -1, self.axis, m)
                self.tensor_ops += 2 * n + 1
            self.exchanges += 1

    def apply_functional(self, shards: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """函数式版本（``f = f.at[...].set(...)`` 风格，不改入参）。"""
        clones = [s.clone() for s in shards]
        self.apply(clones)
        return clones

    # -- 记账 ---------------------------------------------------------
    @property
    def planes_per_exchange(self) -> int:
        """每次集合交换搬运的平面数（两个面各 len(group) 个）。"""
        return len(self.pos_group) + len(self.neg_group)

    def wire_bytes_per_step(self, plane_shape: Sequence[int], itemsize: int = 4) -> int:
        """每次交换的搬运字节数（物理面数 × 面尺寸 × 字节宽）。"""
        n_cells = 1
        for d in plane_shape:
            n_cells *= int(d)
        return self.planes_per_exchange * n_cells * itemsize


# ---------------------------------------------------------------------------
# OSS 原式（2D，逐行对应 vivsim/multidevice.py::stream_cross_devices）
# ---------------------------------------------------------------------------


@dataclass
class OssBranchFinding:
    """OSS ``x`` / ``y`` 分支与拉式 streaming 约定的一致性判据结果。"""

    branch: str
    matches_pull_convention: bool
    sender_expected: list[int]
    sender_got: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "matches_pull_convention": self.matches_pull_convention,
            "sender_expected": self.sender_expected,
            "sender_got": self.sender_got,
        }


def oss_stream_cross_devices(
    shards: Sequence[Any],
    dir: str,  # noqa: A002 - 与 OSS 形参同名
    n_ranks: int,
    *,
    swap_x_branch: bool = False,
) -> tuple[list[Any], list[OssBranchFinding]]:
    """vivsim ``stream_cross_devices`` 的逐行移植（2D D2Q9，MIT）。

    输入 / 输出均为**已做过局部自回卷 streaming** 的 shard 列表，
    每个 shard 形状 ``(9, nx_local, ny)``（与 ``vivsim/lbm/basic.py`` 的
    ``(9, NX, NY)`` 一致）。方向分组用 ``vivsim/lbm/lattice.py`` 的
    ``D2Q9`` 表（``right=[1,5,8]``, ``left=[3,7,6]``, ``up=[2,5,6]``,
    ``down=[4,7,8]``）。

    ``swap_x_branch=True`` 时把 OSS ``dir == 'x'`` 分支的两个 perm 互换
    ——用于把"OSS 原样"与"修正后"两臂分开测量（见
    :func:`oss_branch_findings` 与 docs/XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO.md）。
    """
    if dir not in ("x", "y"):
        raise ValueError("dir must be 'x' or 'y'")
    if len(shards) != n_ranks:
        raise ValueError(f"expected {n_ranks} shards, got {len(shards)}")

    right = (1, 5, 8)
    left = (3, 7, 6)
    up = (2, 5, 6)
    down = (4, 7, 8)

    pairs = list(device_pairs(n_ranks))
    pairs_rev = list(device_pairs_reverse(n_ranks))

    work = list(shards)
    findings: list[OssBranchFinding] = []

    def _exchange(group: Sequence[int], plane: int, perm, axis: int) -> None:
        planes = _np_take_plane(work, group, plane, axis)
        moved = ppermute(planes, perm)
        for r in range(n_ranks):
            _np_put_plane(work[r], group, plane, axis, moved[r])

    if dir == "y":
        # OSS line 31/32：up 在 y=0 面用 pairs；down 在 y=-1 面用 pairs_rev
        _exchange(up, 0, pairs, axis=1)
        _exchange(down, -1, pairs_rev, axis=1)
        findings.append(
            OssBranchFinding(
                "y (up@0 pairs, down@-1 reverse)",
                True,
                sender_index(pairs, n_ranks),
                sender_index(pairs, n_ranks),
            )
        )
    else:
        # OSS line 35/36：right 在 x=0 面用 pairs_rev；left 在 x=-1 面用 pairs
        right_perm, left_perm = (pairs, pairs_rev) if swap_x_branch else (pairs_rev, pairs)
        _exchange(right, 0, right_perm, axis=0)
        _exchange(left, -1, left_perm, axis=0)
        findings.append(
            OssBranchFinding(
                "x (right@0 pairs_rev, left@-1 pairs)" if not swap_x_branch else "x (swapped)",
                swap_x_branch,
                # 拉式约定下 right(+x)@面0 应从 rank-1 取 = pairs
                sender_index(pairs, n_ranks),
                sender_index(right_perm, n_ranks),
            )
        )
    return work, findings


def oss_branch_findings(n_ranks: int = 4) -> list[OssBranchFinding]:
    """把 OSS 两个分支的 perm 方向与拉式 streaming 约定对齐核账。"""
    pairs = device_pairs(n_ranks)
    rev = device_pairs_reverse(n_ranks)
    exp = sender_index(pairs, n_ranks)  # 低面取 rank-1
    exp_hi = sender_index(rev, n_ranks)  # 高面取 rank+1
    return [
        OssBranchFinding("y: up@plane0 device_pairs", True, exp, sender_index(pairs, n_ranks)),
        OssBranchFinding(
            "y: down@plane-1 device_pairs_reverse", True, exp_hi, sender_index(rev, n_ranks)
        ),
        OssBranchFinding(
            "x: right@plane0 device_pairs_reverse", False, exp, sender_index(rev, n_ranks)
        ),
        OssBranchFinding("x: left@plane-1 device_pairs", False, exp_hi, sender_index(pairs, n_ranks)),
    ]


def _np_take_plane(shards: Sequence[Any], group: Sequence[int], plane: int, axis: int) -> list[Any]:
    return [_take_one(s, group, plane, axis) for s in shards]


def _take_one(x: Any, group: Sequence[int], plane: int, axis: int) -> Any:
    if isinstance(x, torch.Tensor):
        return _plane(x, group, plane, axis)
    import numpy as np

    sub = x[list(group)]  # (G, ...) — 高级索引在最前
    index = tuple((plane if i == axis + 1 else slice(None)) for i in range(sub.ndim))
    return np.ascontiguousarray(sub[index])


def _np_put_plane(x: Any, group: Sequence[int], plane: int, axis: int, values: Any) -> None:
    if isinstance(x, torch.Tensor):
        _assign(x, group, plane, axis, values)
        return
    index: list[Any] = [(plane if i == axis + 1 else slice(None)) for i in range(x.ndim)]
    index[0] = list(group)
    x[tuple(index)] = values


# ---------------------------------------------------------------------------
# 真 · jax.lax.ppermute 通路（jax 存在才可用；本沙箱未安装 ⇒ 未执行）
# ---------------------------------------------------------------------------


def oss_stream_cross_devices_jax(f: Any, dir: str, device_axis: str, n_devices: int) -> Any:  # noqa: A002
    """vivsim ``stream_cross_devices`` 原文语义（``jax.lax.ppermute`` 真通路）。

    与 OSS 的差别：（1）不吞掉 x 分支的 perm 方向问题——本函数按拉式约定
    修正为 ``right: device_pairs`` / ``left: device_pairs_reverse``；
    （2）仅在 jax 可导入时可调用。本沙箱（loongarch64，无 jax wheel）未执行，
    见 docs 的 WITHHELD 声明。
    """
    if not JAX_AVAILABLE:  # pragma: no cover - 本沙箱走这里
        raise RuntimeError(
            "jax is not installed in this environment; the genuine "
            "jax.lax.ppermute path is withheld (see docs). Use the torch "
            "PpermutePeriodicHalo emulation instead."
        )
    import jax.numpy as jnp

    if dir not in ("x", "y"):
        raise ValueError("dir must be 'x' or 'y'")
    pairs = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    pairs_rev = [((i + 1) % n_devices, i) for i in range(n_devices)]
    right = jnp.asarray([1, 5, 8])
    left = jnp.asarray([3, 7, 6])
    up = jnp.asarray([2, 5, 6])
    down = jnp.asarray([4, 7, 8])
    if dir == "y":
        f = f.at[up, :, 0].set(lax.ppermute(f[up, :, 0], device_axis, pairs))
        f = f.at[down, :, -1].set(lax.ppermute(f[down, :, -1], device_axis, pairs_rev))
    else:
        f = f.at[right, 0].set(lax.ppermute(f[right, 0], device_axis, pairs))
        f = f.at[left, -1].set(lax.ppermute(f[left, -1], device_axis, pairs_rev))
    return f


# ---------------------------------------------------------------------------
# CLI 自检入口：python -m tensorlbm.parallel.ppermute_halo
# ---------------------------------------------------------------------------


def _main() -> int:  # pragma: no cover - 手工自检
    import numpy as np

    print(f"JAX_AVAILABLE={JAX_AVAILABLE}")
    for n in (1, 2, 4):
        print(f"device_pairs({n})       = {device_pairs(n)}")
        print(f"device_pairs_reverse({n}) = {device_pairs_reverse(n)}")
        print(f"  sender(pairs)   = {sender_index(device_pairs(n), n)}")
        print(f"  sender(reverse) = {sender_index(device_pairs_reverse(n), n)}")

    halo = PpermutePeriodicHalo.from_lattice("D3Q19", n_ranks=4, axis=0)
    print(f"D3Q19 axis0 pos={halo.pos_group} neg={halo.neg_group} "
          f"planes/exchange={halo.planes_per_exchange}")

    # 单调性冒烟：4 个 shard 填 rank 号，交换后 z=0 面应为 rank-1 的号
    shards = [torch.full((19, 4, 2, 2), float(r)) for r in range(4)]
    halo.apply(shards)
    print("plane0 value per rank after exchange:", [float(s[halo.pos_group[0], 0, 0, 0]) for s in shards])
    print("expected (rank-1):", [(r - 1) % 4 for r in range(4)])
    np.testing.assert_allclose(
        [float(s[halo.pos_group[0], 0, 0, 0]) for s in shards],
        [(r - 1) % 4 for r in range(4)],
    )
    for f in oss_branch_findings(4):
        print(f.as_dict())
    print("self-check OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())