#!/usr/bin/env python3
"""D3Q19 自由滑移（镜面反射）壁的解析解验证 — 把 ``IMPLEMENTATION_ONLY`` 清偿为证据面。

背景
----
``tensorlbm.boundaries3d.free_slip_cells_3d`` 及其三个便捷包装（x/y/z 面）在
``docs/boundary_capability_contract_r1.md`` 里长期记为 ``IMPLEMENTATION_ONLY``：
docstring 引用 waLBerla ``FreeSlip.h``，但**没有任何测试**，也没有任何解析判据。
自由滑移壁是溃坝/水箱/对称面这几类算例的默认壁面（waLBerla DamBreak 全部用 FreeSlip），
没有验证就等于"默认壁面"处在无证据状态。

本文件用三段可裁决的解析事实把它钉住（全部计算在
``tensorlbm.free_slip_verification``，本文件只断言，不重算）：

1. **代数**：镜面反射必须把 ``c_q`` 映射到"只翻法向分量、保切向分量"的方向，
   并且是 19 方向上的对合双射。
2. **不变量**：算子对单格点是置换 ⇒ 质量、切向动量精确守恒、法向动量精确取反
   ⇒ 两侧自由滑移壁的通道里**总 x 动量精确守恒**（无壁面动量汇）。无滑移壁是负控。
3. **谱**：均匀流（plug flow）是精确不动点；剪切驻波按 ν=(τ−½)/3 衰减。

判据与被测出的非平凡结论
------------------------
离散剪切本征模态是 ``cos(k_eff (y − δ))``，表观自由滑移面位于壁格点中心**内侧**
δ ≈ 0.33 lu。因此：
* 按拟合出的表观壁面算 ν，误差 <1%（实测 0.056% / 0.093%）→ **判据通过**；
* 按"壁面=格点中心"的朴素做法算 ν，误差是**一阶 O(k)**（H=32: 4.30%，H=64: 2.12%，
  比值 2.03）→ 这条也写成断言，把该偏置固化进证据面，避免以后被当成"跑得不够久"。

负控两枚（证明判据有鉴别力，而不是恒绿）：
* 同初始场换成无滑移壁 ⇒ 表观 ν 偏差 252%、剖面配准残差 0.71（余弦驻波在无滑移壁上不成立）；
* 自由滑移通道动量守恒 vs 无滑移通道丢 81% 动量。

运行：
    PYTHONPATH=src python3 -m pytest tests/test_free_slip_d3q19_analytic.py -q
全套约 80 s（单线程 CPU，float64）。
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.free_slip_verification import (
    EXACT_TOL,
    INSET_RANGE,
    REL_TOL,
    build_results,
)

pytestmark = [pytest.mark.physics]

_AXES = (0, 1, 2)


@pytest.fixture(scope="module")
def results() -> dict:
    """全套案例只跑一次（约 80 s），所有断言共用同一份数字。"""
    torch.set_num_threads(1)
    return build_results()


# ---------------------------------------------------------------------------
# 1. 代数层：镜面反射 = 只翻法向的解析映射
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", _AXES)
def test_mirror_table_is_exact_specular_map(results: dict, axis: int) -> None:
    """镜像表逐方向等于解析镜面映射，且为对合双射（axis=0/1/2 全查）。"""
    case = results["cases"]["algebra"][f"axis{axis}"]
    assert case["mismatched_directions"] == 0
    assert case["involution"] is True
    assert case["bijective"] is True


# ---------------------------------------------------------------------------
# 2. 不变量层：置换算子的精确守恒 / 精确取反
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", _AXES)
def test_operator_invariants_are_exact(results: dict, axis: int) -> None:
    """质量与切向动量逐格点精确守恒，法向动量精确取反（double 下 ≈1e-16）。"""
    case = results["cases"]["invariants"][f"axis{axis}"]
    for key, value in case.items():
        if key.endswith("_rel"):
            assert value < EXACT_TOL, f"{key} = {value:g} 超出精确判据 {EXACT_TOL:g}"


def test_free_slip_channel_conserves_streamwise_momentum(results: dict) -> None:
    """两侧自由滑移壁之间总 x 动量精确守恒（无壁面动量汇）。"""
    case = results["cases"]["momentum_free_slip"]
    assert case["momentum_rel_change"] < EXACT_TOL
    # 均匀分量存活率≈1（自由滑移不吸走任何流向动量）
    assert case["mean_profile_ratio"] == pytest.approx(1.0, abs=1e-10)


def test_no_slip_control_destroys_momentum(results: dict) -> None:
    """负控：同初始场、同壁位、仅换半路反弹 ⇒ 无滑移壁吸走大部分流向动量。

    若这条不红，说明上面那条动量守恒判据对"反射算子写错"没有鉴别力。
    """
    case = results["cases"]["momentum_no_slip_control"]
    assert case["momentum_rel_change"] > 0.5  # 实测 0.81
    assert case["mean_profile_ratio"] < 0.3  # 实测 0.146


# ---------------------------------------------------------------------------
# 3. 不动点：均匀流无寄生阻力
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ("gap32", "z_axis"))
def test_plug_flow_is_exact_fixed_point(results: dict, key: str) -> None:
    """均匀来流在自由滑移壁下 2000 步后仍精确不变（deviation ≈ 1e-15）。"""
    case = results["cases"]["plug_flow"][key]
    assert case["max_relative_deviation"] < EXACT_TOL
    normal = case["max_normal_velocity"]
    if normal is not None:
        assert normal < EXACT_TOL


# ---------------------------------------------------------------------------
# 4. 谱层：剪切驻波 → 表观壁面位置 + 黏度恢复
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ("gap32", "gap64", "z_axis"))
def test_shear_wave_viscosity_recovery(results: dict, key: str) -> None:
    """按拟合表观壁面算出的 ν 与 (τ−½)/3 相对误差 < 1%（实测 0.06% / 0.09%）。"""
    case = results["cases"]["shear_wave"][key]
    assert case["err_fitted"] < REL_TOL
    # 剖面必须是单一余弦本征模态（配准残差 = 高阶模态污染的量度）
    assert case["shape_residual"] < 1e-3


@pytest.mark.parametrize("key", ("gap32", "gap64", "z_axis"))
def test_apparent_wall_plane_is_inside_node_center(results: dict, key: str) -> None:
    """表观自由滑移面在壁格点中心内侧 δ≈0.33 lu（既非 0 也非半路 0.5），
    且 ``有效间距 = π/k_eff`` 与 ``H − 2δ`` 自洽到 <0.05 lu。"""
    case = results["cases"]["shear_wave"][key]
    low, high = INSET_RANGE
    assert low < case["delta"] < high, f"delta = {case['delta']:.4f} 不在 {INSET_RANGE}"
    assert case["effective_gap"] == pytest.approx(case["gap_minus_twice_delta"], abs=0.05)


def test_apparent_wall_plane_is_resolution_stable(results: dict) -> None:
    """δ 与分辨率无关（H=32: 0.331 vs H=64: 0.349，差 0.018 lu），
    即它是算子的系统性亚格子壁位，而非网格噪声。"""
    small = results["cases"]["shear_wave"]["gap32"]["delta"]
    large = results["cases"]["shear_wave"]["gap64"]["delta"]
    assert abs(large - small) < 0.05


def test_nominal_wall_center_assumption_shows_first_order_bias(results: dict) -> None:
    """把壁面当成"格点中心"（δ=0）算衰减率的偏置是一阶 O(k)：

    H=32 → 4.30%，H=64 → 2.12%，比值 ≈2（一阶收敛）；且都远大于按拟合壁面的误差。
    这条把"自由滑移壁的亚格子壁位偏置"固化成事实，防止后续被误判为分辨率不足。
    """
    small = results["cases"]["shear_wave"]["gap32"]
    large = results["cases"]["shear_wave"]["gap64"]
    assert small["err_nominal"] > 0.03
    ratio = small["err_nominal"] / large["err_nominal"]
    assert 1.8 < ratio < 2.2, f"一阶收敛比 = {ratio:.3f}"
    # 拟合壁面后误差必须至少降一个数量级以上
    assert small["err_fitted"] < 0.1 * small["err_nominal"]


def test_free_slip_is_rotation_isotropic(results: dict) -> None:
    """y 面壁与 z 面壁的结果必须逐位一致（D3Q19 各向同性 ⇒ 镜像表三套应等价）。"""
    y_case = results["cases"]["shear_wave"]["gap32"]
    z_case = results["cases"]["shear_wave"]["z_axis"]
    assert y_case["delta"] == pytest.approx(z_case["delta"], abs=1e-9)
    assert y_case["nu_fitted"] == pytest.approx(z_case["nu_fitted"], rel=1e-9)


# ---------------------------------------------------------------------------
# 5. 负控：同一套测量流程换掉反射算子必须变红
# ---------------------------------------------------------------------------


def test_no_slip_control_rejects_the_cosine_standing_wave(results: dict) -> None:
    """余弦驻波在无滑移壁上不满足零切应力 ⇒ 表观 ν 偏差 >50%、剖面残差 >0.1。

    若这条不红，说明剪切波判据（ν 恢复 <1%）没有鉴别力。
    """
    case = results["cases"]["noslip_control"]
    assert case["err_apparent"] > 0.5  # 实测 2.52
    assert case["shape_residual"] > 0.1  # 实测 0.709


def test_criteria_are_stricter_than_platform_threshold(results: dict) -> None:
    """判据 1% 比平台 15% 精度门槛严格一个数量级（防止判据被悄悄放宽）。"""
    assert results["tolerances"]["relative"] == REL_TOL
    assert REL_TOL <= 0.01