"""XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO 的验收测试。

断言的是**不变量**而不是"代码跑过"：
1. ``lax.ppermute`` 语义（``out[dst] = in[src]``）与置换前置条件（fail-closed）；
2. wrap-then-permute 后 shard 上的拉式周期 streaming **逐位等于**单域 streaming
   （三根轴 × N=2/3/4）；
3. 用仓库真求解器 ``LBMStepExecutor`` 跑多步，sharded 解与单域解**逐位一致**；
4. OSS 2D 分支的方向核账：y 分支与拉式约定一致；x 分支原样在 N≥3 时发散，
   perm 互换后恢复一致（这条把"OSS 代码的既有缺陷"钉成可复现判据）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.parallel import (
    JAX_AVAILABLE,
    PpermutePeriodicHalo,
    ShardedPpermuteRunner,
    device_pairs,
    device_pairs_reverse,
    oss_stream_cross_devices,
    oss_stream_cross_devices_jax,
    ppermute,
    sender_index,
)
from tensorlbm.parallel.ppermute_halo import tensor_component_index

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# 1. ppermute 语义
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_device_pairs_are_permutations(n: int) -> None:
    for perm in (device_pairs(n), device_pairs_reverse(n)):
        assert sorted(dst for _src, dst in perm) == list(range(n))
        assert sorted(src for src, _dst in perm) == list(range(n))


def test_ppermute_moves_dst_to_src_torch_and_numpy() -> None:
    planes = [torch.full((2, 3), float(r)) for r in range(4)]
    moved = ppermute(planes, device_pairs(4))
    # device_pairs: (i, i+1) ⇒ out[r] = in[r-1]
    for r in range(4):
        assert float(moved[r][0, 0]) == float((r - 1) % 4)
    moved_rev = ppermute(planes, device_pairs_reverse(4))
    for r in range(4):
        assert float(moved_rev[r][0, 0]) == float((r + 1) % 4)

    np_planes = [np.full((2, 3), float(r)) for r in range(3)]
    np_moved = ppermute(np_planes, device_pairs(3))
    for r in range(3):
        assert float(np_moved[r][0, 0]) == float((r - 1) % 3)


def test_sender_index_is_fail_closed() -> None:
    assert sender_index(device_pairs(3), 3) == [2, 0, 1]
    with pytest.raises(ValueError):
        sender_index([(0, 0), (1, 0), (2, 1)], 3)  # dst 0 用了两次
    with pytest.raises(ValueError):
        sender_index([(0, 0), (1, 1)], 3)  # 缺 dst 2
    with pytest.raises(ValueError):
        sender_index([(0, 3)], 3)  # 越界


def test_axis_component_mapping_matches_executor_roll() -> None:
    """tensor 轴 ↔ C 列号映射必须与 ``LBMStepExecutor`` 的 roll 公式一致。"""
    from tensorlbm.lbm_step import LBMStepExecutor

    ex = LBMStepExecutor(
        "D3Q19", collide_fn="bgk", device=torch.device("cpu"), nx=4, ny=4, nz=4, tau=0.6
    )
    torch.manual_seed(0)
    f = torch.rand(19, 4, 4, 4)
    streamed = ex._stream_preallocated(f).clone()
    manual = torch.stack(
        [
            torch.roll(f[q], shifts=(int(c[2]), int(c[1]), int(c[0])), dims=(0, 1, 2))
            for q, c in enumerate(torch.as_tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]).tolist())
        ]
        + [f[q] for q in range(3, 19)]
    )
    # 只核对 C 表前三个方向（其余方向用原值占位，避免手抄全表）
    for q in range(3):
        assert torch.equal(streamed[q], manual[q])
    assert tensor_component_index(3, 0) == 2
    assert tensor_component_index(3, 2) == 0


# ---------------------------------------------------------------------------
# 2. wrap-then-permute == 单域 streaming（三根轴）
# ---------------------------------------------------------------------------


def _stream_d3q19(f: torch.Tensor) -> torch.Tensor:
    from tensorlbm.d3q19 import C

    return torch.stack(
        [
            torch.roll(f[q], shifts=(int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), dims=(0, 1, 2))
            for q in range(f.shape[0])
        ]
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("n_ranks", [2, 3, 4])
def test_wrap_then_permute_equals_monolithic_stream(axis: int, n_ranks: int) -> None:
    torch.manual_seed(0)
    n = 12
    f = torch.rand(19, n, n, n)
    ref = _stream_d3q19(f)

    n_local = n // n_ranks
    shards = [
        f.narrow(axis + 1, r * n_local, n_local).clone().contiguous() for r in range(n_ranks)
    ]
    local = [_stream_d3q19(s) for s in shards]
    halo = PpermutePeriodicHalo.from_lattice("D3Q19", n_ranks=n_ranks, axis=axis)
    halo.apply(local)
    got = torch.cat(local, dim=axis + 1)
    assert torch.equal(got, ref), f"axis={axis} n_ranks={n_ranks} max diff " \
                                 f"{(got - ref).abs().max().item():.3e}"


def test_single_rank_is_identity() -> None:
    halo = PpermutePeriodicHalo.from_lattice("D3Q19", n_ranks=1, axis=0)
    shard = torch.rand(19, 5, 4, 4)
    before = shard.clone()
    halo.apply([shard])
    assert torch.equal(shard, before)
    assert halo.exchanges == 0


def test_halo_rejects_wrong_shard_count() -> None:
    halo = PpermutePeriodicHalo.from_lattice("D3Q19", n_ranks=4, axis=0)
    with pytest.raises(ValueError):
        halo.apply([torch.zeros(19, 4, 2, 2)] * 3)


# ---------------------------------------------------------------------------
# 3. 真求解器集成：sharded == 单域（逐位）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_sharded_executor_bit_exact_against_monolithic(n_ranks: int) -> None:
    n, steps = 8, 5
    z = torch.arange(n, dtype=torch.float32).view(n, 1, 1)
    uy = (0.01 * torch.sin(2 * math.pi * z / n)).expand(n, n, n).contiguous()
    zeros = torch.zeros_like(uy)
    f0 = equilibrium3d(torch.ones_like(uy), zeros, uy, zeros)

    mono = ShardedPpermuteRunner(f0, n_ranks=1, tau=0.6)
    sharded = ShardedPpermuteRunner(f0, n_ranks=n_ranks, tau=0.6)
    mono.step(steps)
    sharded.step(steps)

    ref, got = mono.gather(), sharded.gather()
    assert torch.equal(ref, got), f"max|Δf| = {(ref - got).abs().max().item():.3e}"

    rho_m, uy_m = mono.gather_macro()[0], mono.gather_macro()[2]
    rho_s, uy_s = sharded.gather_macro()[0], sharded.gather_macro()[2]
    assert torch.equal(rho_m, rho_s)
    assert torch.equal(uy_m, uy_s)


def test_runner_rejects_boundary_fn() -> None:
    f0 = torch.ones(19, 4, 4, 4) / 19.0
    with pytest.raises(ValueError):
        ShardedPpermuteRunner(f0, n_ranks=2, boundary_fn=lambda f, **kw: f)


def test_runner_rejects_indivisible_axis() -> None:
    f0 = torch.ones(19, 7, 4, 4) / 19.0
    with pytest.raises(ValueError):
        ShardedPpermuteRunner(f0, n_ranks=4)


def test_comm_accounting_selective() -> None:
    n = 8
    f0 = torch.ones(19, n, n, n) / 19.0
    runner = ShardedPpermuteRunner(f0, n_ranks=4)
    st = runner.stats().as_dict()
    # D3Q19 沿 z 穿越面只有 |cz|=1 的 5 个方向 ⇒ 10 平面/步，非 2Q=38
    assert st["planes_per_exchange"] == 10
    assert st["planes_per_exchange"] < 2 * 19
    assert st["bytes_per_exchange_per_rank"] == 10 * n * n * 4


# ---------------------------------------------------------------------------
# 4. OSS 2D 分支核账
# ---------------------------------------------------------------------------


def _d2q9_stream(f: torch.Tensor) -> torch.Tensor:
    """vivsim ``lbm/basic.py::streaming`` 的移植（本测试独立实现一份用于 ref）。"""
    cat = torch.cat
    return cat(
        (
            f[0:1],
            cat((f[1:2, -1:, :], f[1:2, :-1, :]), dim=1),
            cat((f[2:3, :, -1:], f[2:3, :, :-1]), dim=2),
            cat((f[3:4, 1:, :], f[3:4, :1, :]), dim=1),
            cat((f[4:5, :, 1:], f[4:5, :, :1]), dim=2),
            cat((cat((f[5:6, -1:, :], f[5:6, :-1, :]), dim=1)[:, :, -1:],), dim=2),
        ),
        dim=0,
    )


def _oss_stream_full(f: torch.Tensor) -> torch.Tensor:
    """完整 OSS streaming（9 方向，逐行对应 basic.py::streaming）。"""
    from tensorlbm.parallel.sharded_executor import LBMStepExecutor  # noqa: F401  (import 检查)

    xp = lambda x: torch.cat((x[:, -1:, :], x[:, :-1, :]), dim=1)  # noqa: E731
    xn = lambda x: torch.cat((x[:, 1:, :], x[:, :1, :]), dim=1)  # noqa: E731
    yp = lambda x: torch.cat((x[:, :, -1:], x[:, :, :-1]), dim=2)  # noqa: E731
    yn = lambda x: torch.cat((x[:, :, 1:], x[:, :, :1]), dim=2)  # noqa: E731
    return torch.cat(
        (
            f[0:1],
            xp(f[1:2]),
            yp(f[2:3]),
            xn(f[3:4]),
            yn(f[4:5]),
            xp(yp(f[5:6])),
            xn(yp(f[6:7])),
            xn(yn(f[7:8])),
            xp(yn(f[8:9])),
        ),
        dim=0,
    )


def test_oss_y_branch_matches_pull_convention() -> None:
    torch.manual_seed(3)
    nx, ny, n = 12, 12, 3
    f0 = torch.rand(9, nx, ny)
    ref = _oss_stream_full(f0)
    nyl = ny // n
    shards = [f0[:, :, r * nyl : (r + 1) * nyl].clone().contiguous() for r in range(n)]
    local = [_oss_stream_full(s) for s in shards]
    got, findings = oss_stream_cross_devices(local, "y", n)
    assert torch.equal(torch.cat(got, dim=2), ref)
    assert all(f.matches_pull_convention for f in findings)


def test_oss_x_branch_as_written_diverges_and_swap_fixes_it() -> None:
    """OSS x 分支的 perm 与拉式约定相反 ⇒ N≥3 时发散；互换后逐位一致。"""
    torch.manual_seed(4)
    nx, ny, n = 12, 8, 3
    f0 = torch.rand(9, nx, ny)
    ref = _oss_stream_full(f0)
    nxl = nx // n

    shards = [f0[:, r * nxl : (r + 1) * nxl, :].clone().contiguous() for r in range(n)]
    local = [_oss_stream_full(s) for s in shards]
    as_written, findings = oss_stream_cross_devices(local, "x", n, swap_x_branch=False)
    assert not torch.equal(torch.cat(as_written, dim=1), ref)
    assert not findings[0].matches_pull_convention

    shards = [f0[:, r * nxl : (r + 1) * nxl, :].clone().contiguous() for r in range(n)]
    local = [_oss_stream_full(s) for s in shards]
    swapped, _ = oss_stream_cross_devices(local, "x", n, swap_x_branch=True)
    assert torch.equal(torch.cat(swapped, dim=1), ref)


@pytest.mark.skipif(JAX_AVAILABLE, reason="jax installed: the guard path is not exercised")
def test_jax_path_is_withheld_without_jax() -> None:
    with pytest.raises(RuntimeError, match="jax is not installed"):
        oss_stream_cross_devices_jax(torch.zeros(9, 4, 4), "y", "dev", 2)