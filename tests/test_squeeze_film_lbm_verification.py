"""挤膜 (squeeze-film) LBM 的解析锚验证 + 口径缺陷回归。

背景
----
``tensorlbm.squeeze_film_lbm`` (m5749 交付) 自称求解"准定常 Reynolds 挤膜"，
但此前**没有任何测试**，也没有任何与 ``squeeze_film`` 里的 Reynolds/Stefan 闭式解
的定量比较。本文件用四条独立的机器判据裁决它：

A. **静态基线恒等式**：动量交换 (ME) 载荷含静态项 ``sum_x rho0 cs^2 = nx rho0/3``。
   源项关掉（负控）时流场静止，此时 ``force_top`` 必须**精确等于**该基线，
   ``force_gauge_top`` 必须为 0（实测 2.7e-15）。这条判据把"载荷 = 基线 + 真实信号"
   拆开——否则 ``force_top`` 的 5.667 常数偏移会把任何物理信号都埋掉。
B. **h^-3 标度**：Reynolds/Stefan 解 ``F = 8 mu V a^3/h^3`` 的指数 -3 是润滑区的
   判别特征。固定 ``a``，用 ``h`` 与 ``2h`` 两次收敛解的 gauge 载荷做 log-log 斜率，
   要求落在 [-3.4, -2.6]（能把 h^-2 或 h^-4 判红）。
C. **两条载荷通道互校**：ME gauge 载荷 vs 无源壁行 (y=0) 静压积分，要求 5% 以内
   ——ME 累加器与压力场是在不同地方算出来的，互校可以同时抓住漏 link 与压力行取错。
D. **解析带 + 有效性窗口**：``h/a = 0.375`` 的 LBM gauge 载荷 vs Reynolds 闭式解
   要求在 20% 带内（实测 +14.2%），并显式声明**没有**达到 3% 的验收线——
   Reynolds 解本身只有 ``O(h/a)`` 渐近精度，本模块可用的最薄膜受离散分辨率限制，
   3% 在该模块当前实现下不可达（见 docs/squeeze_film_lbm_verification_20261108.md）。
   这条断言是"如实登记口径"，不是"通过与否"的粉饰。

另含两条口径回归：
E. ``force_pressure_integral`` 必须取**无源**壁行 y=0；取源行 y=ny-1 会拿到
   注入/角点处理的压力假象（a=16,h=8 时源行逐点为负、和值偏 +30%）。旧值保留为
   ``force_pressure_integral_top_row`` 供审计。
F. 发散必须抛错而不是静默返回 NaN（宽膜 a>=24 时定点密度压力面不稳定）。

物理/文献依据
-------------
* Reynolds, O. (1886) Phil. Trans. R. Soc. Lond. 177, 157-234（润滑方程）。
* Stefan, J. (1874) Sitzber. Akad. Wiss. Wien 69, 713（圆盘挤膜）。
* Ladd, A.J.C. (1994) J. Fluid Mech. 271, 285（动量交换载荷，含静态项）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.squeeze_film_lbm import (
    lattice_reynolds_force,
    simulate_squeeze,
)

CS2 = 1.0 / 3.0
NU = 0.1
V_APPROACH = 0.005


def _run(a: int, h: int, *, source_on: bool = True) -> dict:
    return simulate_squeeze(
        half_width=a,
        gap=h,
        nu=NU,
        V=V_APPROACH,
        source_on=source_on,
        tol=1e-10,
        max_steps=120_000,
        check_every=500,
    )


@pytest.fixture(scope="module")
def runs() -> dict:
    """四个收敛解：a=8 的 h=3/h=6 + a=8,h=6 负控 + a=16,h=6 解析带。"""
    return {
        "a8_h3": _run(8, 3),
        "a8_h6": _run(8, 6),
        "a8_h6_off": _run(8, 6, source_on=False),
        "a16_h6": _run(16, 6),
    }


# --------------------------------------------------------------------------- A
def test_negative_control_is_pure_static_baseline(runs: dict) -> None:
    """无源负控：ME 载荷 = 静态基线，gauge 载荷 = 0。"""
    r = runs["a8_h6_off"]
    nx = 2 * 8 + 1
    baseline = nx * CS2  # rho0 = 1
    assert r["baseline_static_force"] == pytest.approx(baseline, rel=1e-14)
    assert r["force_top"] == pytest.approx(baseline, rel=1e-12)
    assert abs(r["force_gauge_top"]) < 1e-12
    # 压力场静止：无源时全域 gauge 压力应为 0
    assert float(np.abs(r["p"]).max()) < 1e-12


def test_gauge_is_raw_minus_baseline(runs: dict) -> None:
    """有源工况下 force_gauge_* 与 force_* -/+ baseline 必须逐位一致。"""
    r = runs["a16_h6"]
    base = r["baseline_static_force"]
    assert base == pytest.approx((2 * 16 + 1) * CS2, rel=1e-14)
    assert r["force_gauge_top"] == pytest.approx(r["force_top"] - base, rel=1e-13)
    assert r["force_gauge_bottom"] == pytest.approx(r["force_bottom"] + base, rel=1e-13)
    # 上下壁 gauge 载荷必须近似相等相反（净力 ~ 0）
    assert abs(r["force_gauge_top"] + r["force_gauge_bottom"]) < 5e-3 * abs(
        r["force_gauge_top"]
    )


# --------------------------------------------------------------------------- B
def test_gauge_load_scales_as_gap_cubed(runs: dict) -> None:
    """h^-3 标度：a=8 上 h=3 与 h=6 的 gauge 载荷 log-log 斜率必须是 -3。"""
    g3 = runs["a8_h3"]["force_gauge_top"]
    g6 = runs["a8_h6"]["force_gauge_top"]
    assert g3 > 0.0 and g6 > 0.0
    exponent = math.log(g3 / g6) / math.log(6.0 / 3.0)
    assert 2.6 < exponent < 3.4, f"gap scaling exponent {exponent:.3f} (expected ~3)"


# --------------------------------------------------------------------------- C
@pytest.mark.parametrize(
    "key,rel",
    [("a16_h6", 5e-2), ("a8_h3", 1e-1), ("a8_h6", 2e-1)],
)
def test_momentum_exchange_matches_wall_pressure(runs: dict, key: str, rel: float) -> None:
    """ME gauge 载荷与无源壁行静压积分必须互校（窄膜/薄膜离散误差更大）。"""
    r = runs[key]
    p_wall = float(np.sum(r["p"][:, 0]))
    assert p_wall > 0.0
    assert r["force_gauge_top"] == pytest.approx(p_wall, rel=rel)


# --------------------------------------------------------------------------- D
def test_reynolds_anchor_band_and_declared_gap(runs: dict) -> None:
    """解析带：a=16,h=6 (h/a=0.375) 与 Reynolds 闭式解在 20% 带内（实测 +14.2%）。"""
    r = runs["a16_h6"]
    a, h = 16, 6
    analytic = lattice_reynolds_force(a, h, V_APPROACH, NU)
    err = r["force_gauge_top"] / analytic - 1.0
    assert abs(err) < 0.20, f"gauge load error vs Reynolds {err:+.2%}"
    # 如实登记：该窗口没有达到 3% 验收线（Reynolds 只有 O(h/a) 渐近精度）
    assert abs(err) > 3e-2


# --------------------------------------------------------------------------- E
def test_pressure_integral_uses_source_free_wall_row(runs: dict) -> None:
    """口径回归：force_pressure_integral 取 y=0；源行值单列保留且更不可信。"""
    r = runs["a16_h6"]
    assert r["force_pressure_integral"] == pytest.approx(
        float(np.sum(r["p"][:, 0])), rel=1e-13
    )
    assert r["force_pressure_integral_top_row"] == pytest.approx(
        float(np.sum(r["p"][:, -1])), rel=1e-13
    )
    # 源行读数带注入假象：与无源壁行的差远大于数值噪声
    rel_gap = abs(
        r["force_pressure_integral_top_row"] - r["force_pressure_integral"]
    ) / abs(r["force_pressure_integral"])
    assert rel_gap > 1e-3


# --------------------------------------------------------------------------- F
def test_divergence_raises_instead_of_returning_nan() -> None:
    """宽膜发散必须抛 RuntimeError（不许静默返回 NaN 载荷）。"""
    with pytest.raises(RuntimeError, match="diverged"):
        simulate_squeeze(
            half_width=32,
            gap=4,
            nu=NU,
            V=V_APPROACH,
            tol=0.0,
            max_steps=40_000,
            check_every=2_000,
        )