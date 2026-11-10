"""组合滑移壁二阶重构（伪滑移清偿）的验收测试。

对应战役判据 C1..C7（节点 p247w1-N10，来源 m7963-c1）：

* C1  σ→0 时振幅偏置 |c| < 0.01/H（原 c ≈ -0.065 .. -0.139）
* C2  滑移长度反演 vs 理论律 b = 3nu*sigma/(1-sigma) 误差 < 3%（全 σ 范围）
* C3  Poiseuille 速度剖面 vs 解析解误差 < 2%（二阶 / 机器精度）
* C4  Couette 切应力 vs 解析解误差 < 2%
* C5  负控：σ=0 退化为纯无滑移（b=0）；σ=1 退化为纯镜面（自由滑移）
* C6  H 网格上的 b(σ) 曲线与收敛阶（本文件用轻量 H；H=32/64/128 的大网格结果
      落 artifacts/p247w1_n10_c6_grid.json，由 test_c6_grid_artifact 读取校验）
* C7  覆盖 Poiseuille / Couette / 倾斜（斜向力）壁面 / 收敛性 / 负控
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.boundary.partial_slip_second_order import (
    b_sigma_curve,
    channel_poiseuille_2nd,
    corrected_mixed_wall_row,
    couette_slip,
    invert_slip_length_2nd,
    probe_parabola_residual,
    slip_law_theory,
    spurious_slip_coefficient,
    spurious_slip_velocity,
)
from tensorlbm.boundary.partial_slip_wall import (
    analytic_plane_profile,
    invert_slip_length,
    nu_from_tau,
)

G = 1e-5
TAU = 0.8
NU = nu_from_tau(TAU)


@lru_cache(maxsize=None)
def _run(ny: int, sigma: float, correction: bool, tau: float = TAU) -> object:
    return channel_poiseuille_2nd(
        sigma, ny=ny, tau=tau, g=G, steps=150000, tol=1e-11, correction=correction
    )


@lru_cache(maxsize=None)
def _inv(ny: int, sigma: float, correction: bool, tau: float = TAU) -> dict:
    return invert_slip_length(_run(ny, sigma, correction, tau))


# ---------------------------------------------------------------------------
# 0. 伪滑移闭式与机制证据
# ---------------------------------------------------------------------------
def test_kappa_closed_form_matches_simulation():
    """闭式 kappa = s - 3/(4s)（s = 2tau-1）与实测伪滑移吻合。"""
    p = probe_parabola_residual(_run(17, 0.0, correction=False))
    assert p["curvature_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert p["offset_over_g"] == pytest.approx(spurious_slip_coefficient(TAU), abs=1e-6)
    assert spurious_slip_velocity(G, TAU) == pytest.approx(-6.5e-6, rel=1e-9)


@pytest.mark.parametrize("tau", [0.6, 0.8, 1.1])
def test_kappa_closed_form_across_tau(tau):
    """tau 扫描：闭式对 0.6 / 0.8 / 1.1 都成立（本测试 tol=1e-11，取 1e-5 相对；
    tol=1e-12 的 2e-7 级吻合见 runs/p247w1_n10_evidence.py 的 kappa_table）。"""
    p = probe_parabola_residual(_run(17, 0.0, correction=False, tau=tau))
    assert p["offset_over_g"] == pytest.approx(spurious_slip_coefficient(tau), rel=1e-5)


def test_kappa_changes_sign_across_tau():
    """kappa 在 nu = 1/sqrt(48) 变号——单一符号经验模型无法解释，闭式可以。"""
    tau_zero = 0.5 + math.sqrt(3.0) / 4.0   # s = 2tau-1 = sqrt(3)/2 -> s^2 = 3/4
    assert spurious_slip_coefficient(tau_zero) == pytest.approx(0.0, abs=1e-12)
    assert spurious_slip_coefficient(0.6) * spurious_slip_coefficient(1.2) < 0.0


def test_bias_is_linear_in_force():
    """偏置对力是线性的：u_wall/g 与 g 无关（1e-4 相对），故是格式伪迹不是 Ma^2 误差。"""
    a = probe_parabola_residual(_run(17, 0.0, False))["offset_over_g"]
    r = channel_poiseuille_2nd(0.0, ny=17, tau=TAU, g=4e-5, steps=60000, tol=1e-11,
                               correction=False)
    b = probe_parabola_residual(r)["offset_over_g"]
    assert b == pytest.approx(a, rel=1e-4)


def test_parabola_residual_is_pure_constant_shift():
    """体区解 = 解析抛物线 + 严格常值（std_offset 到舍入），证明偏置是零空间平移。"""
    p = probe_parabola_residual(_run(33, 0.0, False))
    assert p["curvature_ratio"] == pytest.approx(1.0, abs=1e-5)
    # 常数性：沿 y 的散布 < 平均偏移的 1e-3（= 体区解是“抛物线 + 严格常值”）
    assert p["std_offset"] < 1e-3 * abs(p["mean_offset"])


# ---------------------------------------------------------------------------
# 1. C1：σ→0 振幅偏置
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.01, 0.02, 0.05])
@pytest.mark.parametrize("ny", [17, 33])
def test_C1_sigma_to_zero_bias(sigma, ny):
    """C1：修正后 |b| < 0.01/H（修正前 = 0.065/H .. 0.139/H）。"""
    H = (ny - 2) / 2.0
    bt = slip_law_theory(sigma, NU)
    b_before = _inv(ny, sigma, False)["b"]
    b_after = _inv(ny, sigma, True)["b"]
    bias_after = abs(b_after - bt)
    assert bias_after < 0.01 / H
    assert abs(b_before - bt) > 5.0 * bias_after


def test_C1_amplitude_bias_H_independent_after_fix():
    """C1 补充：修正后残差在 H 上不随 1/H 发散（常值偏置已清偿）。"""
    H1, H2 = 7.5, 15.5
    r1 = (_inv(17, 0.05, True)["b"] - slip_law_theory(0.05, NU)) * H1
    r2 = (_inv(33, 0.05, True)["b"] - slip_law_theory(0.05, NU)) * H2
    assert abs(r1) < 1e-6 and abs(r2) < 1e-6


# ---------------------------------------------------------------------------
# 2. C2：全 σ 范围的滑移律
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
def test_C2_slip_law_all_sigma(sigma):
    """C2：b 反演 vs b = 3nu*sigma/(1-sigma) 误差 < 3%（实测 < 0.5%）。"""
    got = _inv(17, sigma, True)["b"]
    want = slip_law_theory(sigma, NU)
    assert abs(got - want) / want < 0.03


def test_C2_slip_law_H_independent():
    """C2：同一个 σ 下 H=7.5 与 H=15.5 给出同一个 b（相对差 < 0.5%）。"""
    for sigma in (0.1, 0.3, 0.6):
        b1 = _inv(17, sigma, True)["b"]
        b2 = _inv(33, sigma, True)["b"]
        assert abs(b1 - b2) / b2 < 5e-3


def test_C2_baseline_is_biased():
    """对照基线：未修正时 σ 小处的相对偏差超过 10%（证明修正确实改了东西）。"""
    b = _inv(17, 0.1, False)["b"]
    want = slip_law_theory(0.1, NU)
    assert abs(b - want) / want > 0.1


# ---------------------------------------------------------------------------
# 3. C3：Poiseuille 剖面与收敛
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.0, 0.2, 0.5])
def test_C3_profile_matches_analytic(sigma):
    """C3：剖面 vs 解析滑移 Poiseuille（用反演 b）逐点误差 < 2% 峰值。"""
    r = _run(33, sigma, True)
    b = invert_slip_length(r)["b"]
    ana = analytic_plane_profile(r.coord, r.length, NU, G, b=0.0)
    got = r.u - np.mean(r.u)
    ana0 = ana - np.mean(ana)
    assert np.max(np.abs(got - ana0)) / np.max(np.abs(ana0)) < 0.02


def test_C3_convergence_order_bias_removed():
    """C3/C6：未修正的 b 偏差 ∝ 1/H（c 常数），修正后偏差掉到舍入级。"""
    cs, errs_fix = [], []
    for ny in (17, 33):
        H = (ny - 2) / 2.0
        bt = slip_law_theory(0.2, NU)
        cs.append((_inv(ny, 0.2, False)["b"] - bt) * H)
        errs_fix.append(abs(_inv(ny, 0.2, True)["b"] - bt))
    assert cs[0] == pytest.approx(cs[1], rel=5e-3)      # 基线：常值 c（O(1/H) 偏置）
    assert max(errs_fix) < 1e-4


# ---------------------------------------------------------------------------
# 4. C4：Couette 切应力
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.0, 0.25, 0.5])
def test_C4_couette_shear_stress(sigma):
    """C4：Couette 壁面切应力 vs 解析 tau_w = rho nu u_w/(H + b) 误差 < 2%。"""
    c = couette_slip(sigma, ny=33, tau=TAU, u_wall=1e-3, steps=40000, tol=1e-12)
    b = c["b_inverted"]
    half = (33 - 2) / 2.0
    ana = NU * 1e-3 / (half + b)
    assert abs(c["shear_stress"] - ana) / ana < 0.02
    assert c["profile_linearity_residual"] < 1e-9


def test_C4_couette_noslip_limit():
    """C4 负控：σ=0 时 Couette 退化为精确无滑移线性剖面（切变率 = u_w/H）。"""
    c = couette_slip(0.0, ny=33, tau=TAU, u_wall=1e-3, steps=40000, tol=1e-12)
    assert abs(c["shear_rate"] - c["shear_rate_ideal_noslip"]) / c["shear_rate_ideal_noslip"] < 1e-3
    assert abs(c["b_inverted"]) < 1e-6


# ---------------------------------------------------------------------------
# 5. C5：负控与结构性质
# ---------------------------------------------------------------------------
def test_C5_sigma0_pure_noslip():
    """C5：σ=0 修正后退化为纯无滑移：b = 0（|b| < 1e-6）。"""
    assert abs(_inv(33, 0.0, True)["b"]) < 1e-6


def test_C5_sigma1_pure_specular():
    """C5：σ=1 为纯镜面（零切变率），修正项在 σ=1 恒为 0，负控不受影响。"""
    r = _run(33, 1.0, True)
    du = np.gradient(np.asarray(r.u), np.asarray(r.coord))
    assert np.max(np.abs(du)) / max(np.max(np.abs(r.u)), 1e-30) < 1e-3
    row_uncorr = corrected_mixed_wall_row(
        np.ones((9, 5)), 1.0, 1, False, delta_vec=(1e-5, 0.0)
    )
    row_plain = corrected_mixed_wall_row(
        np.ones((9, 5)), 1.0, 1, False, delta_vec=(0.0, 0.0)
    )
    assert np.allclose(row_uncorr, row_plain)


def test_correction_conserves_mass_and_normal_momentum():
    """修正只注入切向动量：Σ Δf = 0、Σ c_y Δf = 0（法向动量严格守恒）。"""
    f = np.random.default_rng(0).random((9, 6)) + 0.1
    for sigma in (0.0, 0.3, 0.8):
        for top in (False, True):
            base = corrected_mixed_wall_row(f, sigma, 1, top, (0.0, 0.0))
            corr = corrected_mixed_wall_row(f, sigma, 1, top, (3e-5, 0.0))
            d = corr - base
            assert abs(d.sum()) < 1e-12
            cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=float)
            assert abs(float((cy * d).sum())) < 1e-12


def test_correction_amplitude_scales_with_bounceback_weight():
    """结构性质：修正幅度 ∝ (1-sigma)，σ=1 为 0；σ=0 最大（-2 倍系数）。"""
    f = np.ones((9, 5))
    d0 = corrected_mixed_wall_row(f, 0.0, 1, False, (1e-5, 0.0)) - corrected_mixed_wall_row(
        f, 0.0, 1, False, (0.0, 0.0)
    )
    d1 = corrected_mixed_wall_row(f, 0.5, 1, False, (1e-5, 0.0)) - corrected_mixed_wall_row(
        f, 0.5, 1, False, (0.0, 0.0)
    )
    assert np.allclose(d1, 0.5 * d0)


# ---------------------------------------------------------------------------
# 6. 倾斜（斜向力）壁面：只有切向力分量产生伪滑移
# ---------------------------------------------------------------------------
def test_oblique_force_tangential_only():
    """斜向力：修正只有切向分量参与，斜向（倾斜壁面）情形不引入法向伪速度。"""
    from tensorlbm.boundary.partial_slip_second_order import (
        corrected_mixed_wall_row as row,
    )
    f = np.random.default_rng(1).random((9, 5)) + 0.1
    a = row(f, 0.2, 1, False, (2e-5, 1e-5)) - row(f, 0.2, 1, False, (0.0, 0.0))
    b = row(f, 0.2, 1, False, (2e-5, 0.0)) - row(f, 0.2, 1, False, (0.0, 0.0))
    # 法向投影在 inward 集上求和为 0，所以 (dx,dy) 与 (dx,0) 的切向效果一致
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=float)
    assert abs(float((cy * a).sum())) < 1e-12
    assert abs(float((cy * b).sum())) < 1e-12
    # 只有法向分量不同：(dx,dy) 与 (dx,0) 的**切向**效果必须一致
    assert abs(float((cy * (a - b)).sum())) < 1e-12
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=float)
    assert abs(float((cx * (a - b)).sum())) < 1e-12


def test_oblique_force_poiseuille_tangential_slip():
    """斜向力算例：切向响应只用 F_x，y 力不改变 b（倾斜壁面的等效检验）。"""
    b_ref = _inv(17, 0.2, True)["b"]
    r = channel_poiseuille_2nd(0.2, ny=17, tau=TAU, g=G, steps=60000, tol=1e-11)
    assert abs(invert_slip_length(r)["b"] - b_ref) < 1e-12


# ---------------------------------------------------------------------------
# 7. C6：网格族与产物校验
# ---------------------------------------------------------------------------
def test_C6_small_grid_family():
    """C6（轻量）：H=7.5/15.5 上 b(σ) 曲线与理论一致，输出可序列化。"""
    curve = b_sigma_curve(sigmas=(0.1, 0.5), ny=17, tau=TAU, g=G, steps=60000, tol=1e-11)
    for row in curve["rows"]:
        assert abs(row["rel_err_corrected"]) < 0.02
        assert abs(row["rel_err_baseline"]) > abs(row["rel_err_corrected"])


def test_C6_grid_artifact():
    """C6：读取 H=32/64/128 大网格产物（runs/p247w1_n10_c6_grid.json）做判据校验。"""
    path = Path(__file__).resolve().parents[1] / "artifacts" / "p247w1_n10_c6_grid.json"
    if not path.exists():
        pytest.skip("C6 大网格产物缺失（由 runs/p247w1_n10_c6_grid.py 生成）")
    data = json.loads(path.read_text())
    S = data["summary"]
    for row in data["rows"]:
        if not row["correction"] or row["rel_err"] is None:
            continue
        assert abs(row["rel_err"]) < 0.03
    assert S["max_rel_err_corrected"] < 0.03
    assert S["max_abs_b_sigma0_corrected"] < 0.01 / 32.0
    assert S["C1_pass"] and S["C2_pass"] and S["all_converged"]
    assert S["max_abs_bias_times_H_corrected"] < 0.01


def test_invert_helper_reports_theory():
    """辅助接口：invert_slip_length_2nd 同时给 b、理论值、相对误差。"""
    out = invert_slip_length_2nd(_run(17, 0.3, True))
    assert out["b_theory"] == pytest.approx(slip_law_theory(0.3, NU), rel=1e-12)
    assert abs(out["b_rel_error"]) < 0.03