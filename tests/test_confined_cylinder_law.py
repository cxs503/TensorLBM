"""P187-W1-N08 受限域圆柱双参数偏差定律：单元/集成测试。

覆盖点：
* 有效几何量（D_eff / H_eff / H/D）口径；
* 迁移算子等价性（roll 版 vs 仓库 gather 版）；
* Churchill-Bernstein 参考值；
* 窗均值抽取；
* 解析场检验：``T = 1 + A ln(r/R)``（等温圆柱二维 Laplace 解）下
  ``-oint dT/dn ds`` 应等于 ``2 pi A``——直接量化表面热流估算器的精度；
* 乘法律合成数据可恢复性（已知 a, m, b, beta）+ AIC 判据 + 交互项残差；
* 负控（因子置常数 -> 对应参数不显著）；
* 小型真实算例冒烟（有限、正、量级合理）。

本文件只依赖 torch/numpy/scipy + 本仓 ``tensorlbm`` 包。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm import confined_cylinder_law as law
from tensorlbm.solver import stream as solver_stream
from tensorlbm.thermal import temperature_stream


# ============================================================================
# 几何口径
# ============================================================================

def test_effective_geometry_measures():
    cfg = law.ConfinedCylinderConfig(h_over_d=8.0, diameter=16)
    geo = law.build_geometry(cfg)
    assert geo.D_eff == pytest.approx(17.0)          # 2R+1：半程壁
    assert geo.H_eff == pytest.approx(geo.ny - 2.0)  # 上下壁中点距
    assert geo.h_over_d == pytest.approx(8.0, rel=1e-9)
    # 圆柱不触壁
    assert geo.cy - geo.R > 0 and geo.cy + geo.R < geo.ny - 1
    # 阻塞比越大，域越高
    hi = law.build_geometry(law.ConfinedCylinderConfig(h_over_d=24.0, diameter=16))
    assert hi.ny > geo.ny
    assert hi.h_over_d == pytest.approx(24.0, rel=1e-2)


# ============================================================================
# 迁移算子等价性
# ============================================================================

def test_stream_roll_matches_solver_stream():
    torch.manual_seed(0)
    ny, nx = 11, 17
    f = torch.rand(9, ny, nx, dtype=torch.float64)
    assert torch.allclose(law._stream9(f), solver_stream(f), atol=1e-14)
    g = torch.rand(5, ny, nx, dtype=torch.float64)
    assert torch.allclose(law._stream5(g), temperature_stream(g), atol=1e-14)


# ============================================================================
# 经验式基准
# ============================================================================

def test_churchill_bernstein_reference_values():
    """Churchill-Bernstein (1977) 公式值（与本仓 heat_convection_benchmarks 同式）。

    Re=100, Pr=0.7 -> 5.156（文献常用值 ~5.16）；Re=200, Pr=0.7 -> 7.188；
    Re=100, Pr=7 -> 11.821。注意：旧报告里 Re=200 写成 6.84 是笔误，公式值为
    7.19（见 docs/P187_W1_N08_... 注意点一节）。
    """
    assert law.churchill_bernstein(100.0, 0.7) == pytest.approx(5.156, abs=0.01)
    assert law.churchill_bernstein(200.0, 0.7) == pytest.approx(7.188, abs=0.02)
    assert law.churchill_bernstein(100.0, 7.0) == pytest.approx(11.821, abs=0.01)
    # 单调性
    assert law.churchill_bernstein(10.0, 0.7) < law.churchill_bernstein(100.0, 0.7)


# ============================================================================
# 窗均值
# ============================================================================

def test_window_means_basic():
    times = np.arange(0.0, 10.0, 0.5)
    series = np.ones_like(times)
    series[times >= 5.0] = 2.0
    out = law.window_means(times, series, [(0.0, 5.0), (5.0, 10.0)])
    assert out[(0.0, 5.0)] == pytest.approx(1.0)
    assert out[(5.0, 10.0)] == pytest.approx(2.0)
    # 无样本窗返回 nan
    assert math.isnan(law.window_means(times, series, [(50.0, 60.0)])[(50.0, 60.0)])


# ============================================================================
# 表面热流估算器：解析 Laplace 场
# ============================================================================

@pytest.mark.parametrize("diameter", [8, 16, 24, 32])
def test_surface_flux_characterisation_on_laplace_field(diameter):
    """标定（非"通过性"）测试：阶梯壁面三点单侧差分积分器在解析 Laplace 场上的误差。

    ``T = 1 + A ln(r/R)`` 是等温圆柱二维纯导热解析解，``-oint dT/dn ds = 2 pi A``
    精确已知，故该测试给出积分器的**离散误差画像**（与流场无关）。

    实测（A=0.37, H/D=6，见报告第 4/6 节）::

        D_eff= 9  误差 14.8%
        D_eff=17  误差 24.1%
        D_eff=25  误差 50.3%
        D_eff=33  误差 21.0%

    -> 误差 15-50% 且**不随网格单调收敛**（阶梯外凸角处法向定义不唯一）。
    因此该积分器只作诊断量，绝对口径必须用等温反反弹边界自身的计量
    (:func:`tensorlbm.confined_cylinder_law._bc_heat_flux`)。
    本测试锁定"误差不超过 60%"这一画像上界，防止静默恶化。
    """
    cfg = law.ConfinedCylinderConfig(h_over_d=6.0, diameter=diameter)
    geo = law.build_geometry(cfg)
    R = geo.R
    A = 0.37
    yy, xx = torch.meshgrid(
        torch.arange(geo.ny, dtype=torch.float64),
        torch.arange(geo.nx, dtype=torch.float64),
        indexing="ij",
    )
    r = torch.sqrt((xx - geo.cx) ** 2 + (yy - geo.cy) ** 2).clamp(min=0.5)
    T = torch.where(geo.solid, torch.ones_like(r), 1.0 + A * torch.log(r / R))
    q = law._surface_flux(T, geo, 1.0)
    exact = 2.0 * math.pi * A
    err = abs(q / (-exact) - 1.0)
    assert err < 0.60, f"staircase flux integrator error {err*100:.1f}% exceeds characterised 60% bound"
    print(f"[calibration] D_eff={geo.D_eff:.0f} staircase surface-flux error={err*100:.1f}%")


# ============================================================================
# 乘法律拟合 / 判据
# ============================================================================

TRUE = {"nu_inf": 5.05, "a": 0.42, "m": 1.15, "b": 0.055, "beta": 0.85}


def _synthetic_rows(windows, hd_values=(4.0, 6.0, 8.0, 12.0, 16.0, 24.0), noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for hd in hd_values:
        for w in windows:
            tau = w[1] - w[0]
            nu = (
                TRUE["nu_inf"]
                * (1.0 + TRUE["a"] * hd ** (-TRUE["m"]))
                * (1.0 + TRUE["b"] * tau ** (-TRUE["beta"]))
            )
            if noise:
                nu *= 1.0 + noise * rng.standard_normal()
            rows.append({"h_over_d": hd, "tau": tau, "nu": nu, "window": tuple(w)})
    return rows


def test_fit_multiplicative_recovers_synthetic_law():
    rows = _synthetic_rows(law.STATIONARY_WINDOWS)
    hd = [r["h_over_d"] for r in rows]
    tau = [r["tau"] for r in rows]
    nu = [r["nu"] for r in rows]
    fit = law.fit_multiplicative_law(hd, tau, nu)
    for key, ref in TRUE.items():
        assert fit["params"][key] == pytest.approx(ref, rel=0.02), f"{key} not recovered"
    assert fit["rmse_log_pct"] < 0.5


def test_multiplicative_beats_additive_on_multiplicative_data():
    rows = _synthetic_rows(law.STATIONARY_WINDOWS, noise=0.004, seed=7)
    hd = [r["h_over_d"] for r in rows]
    tau = [r["tau"] for r in rows]
    nu = [r["nu"] for r in rows]
    inter = law.interaction_residuals(hd, tau, nu)
    add = law.fit_additive_law(hd, tau, nu)
    assert inter["law_max_abs_pct"] < 1.0
    assert inter["anova_interaction_max_abs_pct"] < 1.0
    assert inter["fit"]["aic"] < add["aic"], "multiplicative should win on multiplicative data"


def test_negative_control_flags_no_spurious_effect():
    rows = _synthetic_rows(law.STATIONARY_WINDOWS, noise=0.004, seed=11)
    hd = [r["h_over_d"] for r in rows]
    tau = [r["tau"] for r in rows]
    nu = [r["nu"] for r in rows]
    nc_hd = law.negative_control_ftest(hd, tau, nu, factor="hd")
    nc_tau = law.negative_control_ftest(hd, tau, nu, factor="tau")
    assert nc_hd["p_value"] > 0.05, "constant H/D must render blockage term insignificant"
    assert nc_tau["p_value"] > 0.05, "constant tau must render window term insignificant"


def test_evaluate_criteria_on_synthetic_passes_c1_c3_c5():
    # 小数据集（4 堵塞比 x 3 窗）：判据机器全链路覆盖，同时把多起点拟合成本压到
    # 单次 <40 s（本机 loong64 上 least_squares 较慢）。
    hds = (4.0, 8.0, 12.0, 24.0)
    stat = law.STATIONARY_WINDOWS[:3]
    miss = law.MISSION_WINDOWS  # 需要含 C3 的两个测试窗，拟合集才够秩
    primary = _synthetic_rows(stat, hd_values=hds, noise=0.006, seed=3)
    mission = _synthetic_rows(miss, hd_values=hds, noise=0.006, seed=4)
    coarse = _synthetic_rows(miss, hd_values=hds, noise=0.006, seed=5)
    cb = law.churchill_bernstein(100.0, 0.7)
    res = law.evaluate_criteria(primary, mission, coarse, cb)
    assert res["C1"]["pass"]
    assert res["C5"]["pass"]
    # C3 需要留出窗拟合集可辨（本小数据 4x4 仅 8 点喂 5 参数），此处只锁定
    # "C3 机器跑通且出数"，真假由真实扫掠数据决定（报告第 3.3 节）。
    assert "observed_diff_pct" in res["C3"] or res["C3"]["pass"] is None
    assert set(("C1", "C2", "C3", "C4", "C5")) <= set(res)
    assert isinstance(res["all_pass"], bool)


# ============================================================================
# 小型真实算例冒烟
# ============================================================================

def test_simulation_smoke_small():
    cfg = law.ConfinedCylinderConfig(
        h_over_d=4.0, diameter=8, re=100.0, t_end_d=3.0,
        dt_sample_d=0.5, threads=1, dtype="float32",
    )
    res = law.simulate_confined_cylinder(cfg)
    assert res["stable"]
    assert res["geometry"]["D_eff"] == 9.0
    nu = np.array(res["nu_local"], dtype=float)
    assert np.all(np.isfinite(nu))
    assert 2.0 < nu.mean() < 20.0, nu.mean()
    assert len(res["times"]) == len(nu) == len(res["nu_global"])
    # 表面热流与进出口焓通量在早期必然不同（热尾流未出流），只要求都为有限正数
    assert res["final"]["u_max"] > 0.05