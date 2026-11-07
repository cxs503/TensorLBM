"""Schnerr–Sauer 质量传递源项的解析基准验证（Rayleigh–Plesset 锚定）。

背景
----
``tensorlbm.cavitation.schnerr_sauer_source`` 是库内唯一的蒸汽/液体质量传递
模型（Schnerr & Sauer 2001），此前**没有任何直接测试**：``tests/test_sc_eos_coexistence.py``
只覆盖 ``psi_cavitation`` / ``apply_cavitation_force`` 的耦合与符号约定，而源项本身
（气泡半径映射、蒸发/凝结分支、饱和点行为）从未被裁决过。

本文件用三条彼此独立的解析锚：

A. **Rayleigh–Plesset 壁面速度**：惯性主导下气泡壁速度 ``dR/dt = sqrt(2 Δp / (3 ρ_l))``
   ——源项里 ``sqrt(2|p-p_sat|/(3 ρ_l))`` 正是这个量。用 RK4 直接积分
   ``R R̈ + 1.5 Ṙ² = ±Δp/ρ`` 独立复现该前因子（不读实现里的常数）。
B. **Rayleigh 溃灭时间**：``t_c = 0.914681 R₀ sqrt(ρ/Δp)``（Rayleigh 1917 闭式解），
   验证 A 里同一个 ODE 的积分路径与解析常数一致（< 1%）。
C. **Schnerr–Sauer 闭式解**：``ṁ = C (3 ρ_l ρ_v/ρ) α_l α_v / R_b sqrt(2|Δp|/(3 ρ_l))``
   的逐点数值、饱和点零值、单相极限、``sqrt(|Δp|)`` 标度、``R_b(α_v)`` 反演。

另含**鉴别力负控**（§D）：把历史上合并进 ``cavitation.py`` 的两个缺陷形态
（只带 α_l / α_v 之一；R_b 被 clamp 到常数 0.5）重建成 ``_legacy_*`` 变体，
证明上面的性质断言确实会把它们判红——否则这些断言只是摆设。

物理/文献依据
-------------
* Schnerr, G. H. & Sauer, J. (2001), 4th Int. Conf. Multiphase Flow, New Orleans.
* Rayleigh, Lord (1917), Phil. Mag. 34, 94–98（空泡溃灭时间 0.914681 R₀√(ρ/Δp)）。
* Brennen, C. E. (1995), *Cavitation and Bubble Dynamics*, §2–3（RP 方程一阶积分）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.cavitation import (
    N_NUC_LATTICE_DEFAULT,
    schnerr_sauer_bubble_radius,
    schnerr_sauer_source,
)

RHO_L = 2.65
RHO_V = 0.038
RHO_MID = 0.5 * (RHO_L + RHO_V)  # alpha_v = 0.5


# ---------------------------------------------------------------------------
# A/B. Rayleigh–Plesset 独立积分（不引用实现中的任何常数）
# ---------------------------------------------------------------------------
def rayleigh_plesset_collapse(
    r0: float = 1.0,
    dp: float = 1.0,
    rho_l: float = RHO_L,
    stop_fraction: float = 0.02,
    n_steps: int = 40_000,
) -> dict:
    """RK4 积分空泡溃灭 ``R R̈ + 1.5 Ṙ² = -Δp/ρ_l``，返回溃灭时间。

    ``Δp = p_∞ - p_b > 0``（外压高于泡内压 → 溃灭）。积分到 ``R/R₀ = stop_fraction``
    后剩余时间 ≤ 6e-6 t_c（``∫₀^0.02 x^{3/2}dx / 0.7468``），故直接沿用测量值。
    解析基准 ``t_c = 0.914681 R₀ sqrt(ρ_l/Δp)``（Rayleigh 1917）。
    """
    t = 0.0
    r = r0
    v = 0.0
    dt = (0.914681 * r0 * math.sqrt(rho_l / dp)) / n_steps

    def accel(r_: float, v_: float) -> float:
        return (-dp / rho_l - 1.5 * v_ * v_) / r_

    for _ in range(n_steps):
        k1r, k1v = v, accel(r, v)
        k2r, k2v = v + 0.5 * dt * k1v, accel(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v)
        k3r, k3v = v + 0.5 * dt * k2v, accel(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v)
        k4r, k4v = v + dt * k3v, accel(r + dt * k3r, v + dt * k3v)
        r += dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v += dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        if r <= stop_fraction * r0:
            break
    return {"t_c": t, "r_end": r, "t_c_exact": 0.914681 * r0 * math.sqrt(rho_l / dp)}


def rayleigh_plesset_growth(
    r0: float = 0.1,
    dp: float = 0.01,
    rho_l: float = RHO_L,
    r_end: float = 1.0,
    dt: float = 1.0e-4,
) -> dict:
    """RK4 积分气泡惯性生长 ``R R̈ + 1.5 Ṙ² = +Δp/ρ_l``，返回远端壁速。

    ``Δp = p_b - p_∞ > 0``（泡内压高于外压 → 生长）。一阶积分给出
    ``Ṙ² = (2Δp/(3ρ_l))(1 - R₀³/R³)``，故 ``R ≫ R₀`` 时 ``Ṙ → sqrt(2Δp/(3ρ_l))``。
    """
    r, v, t = r0, 0.0, 0.0
    while r < r_end:
        k1r, k1v = v, (dp / rho_l - 1.5 * v * v) / r
        k2r, k2v = v + 0.5 * dt * k1v, (dp / rho_l - 1.5 * (v + 0.5 * dt * k1v) ** 2) / (
            r + 0.5 * dt * k1r
        )
        k3r, k3v = v + 0.5 * dt * k2v, (dp / rho_l - 1.5 * (v + 0.5 * dt * k2v) ** 2) / (
            r + 0.5 * dt * k2r
        )
        k4r, k4v = v + dt * k3v, (dp / rho_l - 1.5 * (v + dt * k3v) ** 2) / (r + dt * k3r)
        r += dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v += dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
    exact = math.sqrt(2.0 * dp / (3.0 * rho_l)) * math.sqrt(1.0 - (r0 / r_end) ** 3)
    return {"wall_speed": v, "wall_speed_exact": exact, "r_end": r}


@pytest.mark.unit
def test_rp_collapse_time_reproduces_rayleigh_constant() -> None:
    """RP 方程积分给出 Rayleigh 溃灭时间 t_c = 0.914681 R₀ sqrt(ρ/Δp)，误差 < 1%。"""
    for dp in (0.5, 1.0, 4.0):
        res = rayleigh_plesset_collapse(dp=dp, n_steps=40_000)
        rel = abs(res["t_c"] - res["t_c_exact"]) / res["t_c_exact"]
        assert rel < 0.01, f"Δp={dp}: t_c={res['t_c']:.6f} vs {res['t_c_exact']:.6f} ({rel:.3%})"


@pytest.mark.unit
def test_rp_growth_wall_speed_is_the_source_prefactor() -> None:
    """惯性生长远端壁速 = sqrt(2Δp/(3ρ_l))——源项前因子的物理出处。"""
    for dp in (0.004, 0.01, 0.04):
        res = rayleigh_plesset_growth(dp=dp)
        rel = abs(res["wall_speed"] - res["wall_speed_exact"]) / res["wall_speed_exact"]
        assert rel < 0.01, (
            f"Δp={dp}: Ṙ={res['wall_speed']:.6f} vs {res['wall_speed_exact']:.6f} ({rel:.3%})"
        )


@pytest.mark.unit
def test_source_driving_term_equals_rp_wall_speed() -> None:
    """源项里的 sqrt(2|Δp|/(3ρ_l)) 与 RP 壁速逐点一致（反解实现，误差 < 1e-12）。"""
    n = 7
    alpha_v = torch.full((n,), 0.5, dtype=torch.float64)
    rho = RHO_L - alpha_v * (RHO_L - RHO_V)
    n_nuc = 1.0e-4
    for dp in (1.0e-4, 1.0e-3, 1.0e-2):
        m_dot = schnerr_sauer_source(
            rho, torch.full((n,), -dp, dtype=torch.float64), p_sat=0.0,
            n_nuc=n_nuc, c_vap=1.0, c_cond=1.0,
        )
        r_b = schnerr_sauer_bubble_radius(alpha_v, n_nuc)
        pref = (RHO_L * RHO_V / rho) * (1.0 - alpha_v) * alpha_v * (3.0 / r_b)
        recovered = m_dot / pref
        expected = math.sqrt(2.0 * dp / (3.0 * RHO_L))
        assert torch.allclose(recovered, torch.full_like(recovered, expected), rtol=1e-12), (
            f"Δp={dp}: 反解驱动项 {recovered[0].item():.12e} vs RP 壁速 {expected:.12e}"
        )


# ---------------------------------------------------------------------------
# C. Schnerr–Sauer 闭式的性质断言
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_source_is_exactly_zero_at_saturation() -> None:
    """p = p_sat 时净传递必须恒等于 0（无压差 = 无相变）。"""
    rho = torch.linspace(RHO_V + 1e-6, RHO_L - 1e-6, 32, dtype=torch.float64)
    p = torch.full_like(rho, 0.123)
    m_dot = schnerr_sauer_source(rho, p, p_sat=0.123, n_nuc=1e-4)
    assert torch.all(m_dot == 0.0), f"饱和点非零传质：max|ṁ|={m_dot.abs().max().item():.3e}"


@pytest.mark.unit
def test_source_vanishes_in_single_phase_limits() -> None:
    """纯液（α_v→0）蒸发与纯汽（α_l→0）凝结都必须趋于零（无气泡即无传质）。"""
    for dp_sign in (-1.0, +1.0):
        for alpha_v in (1e-2, 1e-4, 1e-6, 1e-8):
            rho = RHO_L - alpha_v * (RHO_L - RHO_V)
            m_dot = schnerr_sauer_source(
                torch.tensor([rho], dtype=torch.float64),
                torch.tensor([dp_sign * 1e-2], dtype=torch.float64),
                p_sat=0.0, n_nuc=1e-4,
            )
            # α_l α_v / R_b ∝ α_v^{2/3} → 1e-8 档应 ≲ 1e-5 量级的基准值
            assert abs(m_dot.item()) < 1e-3, (
                f"α_v={alpha_v:g}, dp={dp_sign:+.0e}: |ṁ|={abs(m_dot.item()):.3e} 未趋于零"
            )
    # 严格的单相端点：α_v = 0（ρ = ρ_l）与 α_v = 1（ρ = ρ_v）必须精确为零
    for rho_edge in (RHO_L, RHO_V):
        m_dot = schnerr_sauer_source(
            torch.tensor([rho_edge], dtype=torch.float64),
            torch.tensor([-1e-2], dtype=torch.float64),
            p_sat=0.0, n_nuc=1e-4,
        )
        assert m_dot.item() == 0.0, f"ρ={rho_edge} 端点 ṁ={m_dot.item():.3e}"


@pytest.mark.unit
def test_source_follows_sqrt_dp_scaling() -> None:
    """|ṁ| ∝ sqrt(|p−p_sat|)：四点最小二乘斜率 = 0.5（误差 < 1e-9）。"""
    dp = torch.tensor([1e-5, 1e-4, 1e-3, 1e-2], dtype=torch.float64)
    rho = torch.full_like(dp, RHO_MID)
    m_dot = schnerr_sauer_source(rho, -dp, p_sat=0.0, n_nuc=1e-4).abs()
    slope = float(np.polyfit(np.log(dp.numpy()), np.log(m_dot.numpy()), 1)[0])
    assert abs(slope - 0.5) < 1e-9, f"sqrt 标度斜率 {slope:.12f} ≠ 0.5"


@pytest.mark.unit
def test_bubble_radius_mapping_matches_schnerr_sauer() -> None:
    """R_b 反演：源项里的 3/R_b 必须等于 (α_l/α_v · 4π n /3)^{1/3}（不得被 clamp 掉）。"""
    n_nuc = 1.0e-4
    for alpha_v in (0.01, 0.05, 0.2, 0.5, 0.8, 0.95):
        av = torch.tensor([alpha_v], dtype=torch.float64)
        rho = torch.tensor([RHO_L - alpha_v * (RHO_L - RHO_V)], dtype=torch.float64)
        m_dot = schnerr_sauer_source(rho, torch.tensor([-1e-2], dtype=torch.float64),
                                     p_sat=0.0, n_nuc=n_nuc)
        pref_no_rb = float((RHO_L * RHO_V / rho) * (1.0 - alpha_v) * alpha_v)
        driving = math.sqrt(2.0 * 1e-2 / (3.0 * RHO_L))
        r_b_measured = 3.0 * pref_no_rb * driving / float(m_dot)
        r_b_exact = float(schnerr_sauer_bubble_radius(av, n_nuc))
        # 解析闭式（独立于实现写一遍）
        r_b_closed = ((alpha_v / (1.0 - alpha_v)) * 3.0 / (4.0 * math.pi * n_nuc)) ** (1.0 / 3.0)
        assert abs(r_b_measured - r_b_closed) / r_b_closed < 1e-9, (
            f"α_v={alpha_v}: R_b 反演 {r_b_measured:.6e} vs 闭式 {r_b_closed:.6e}"
        )
        assert abs(r_b_exact - r_b_closed) / r_b_closed < 1e-9


@pytest.mark.unit
def test_source_matches_hand_computed_value() -> None:
    """逐点数值 vs 手算闭式（α_v=0.5, n=1e-4, Δp=1e-2）：相对误差 < 1e-12。"""
    alpha_v = 0.5
    rho = RHO_L - alpha_v * (RHO_L - RHO_V)
    n_nuc = 1e-4
    dp = 1e-2
    r_b = ((alpha_v / (1.0 - alpha_v)) * 3.0 / (4.0 * math.pi * n_nuc)) ** (1.0 / 3.0)
    hand = (RHO_L * RHO_V / rho) * (1.0 - alpha_v) * alpha_v * (3.0 / r_b) * math.sqrt(
        2.0 * dp / (3.0 * RHO_L)
    )
    got = schnerr_sauer_source(
        torch.tensor([rho], dtype=torch.float64),
        torch.tensor([-dp], dtype=torch.float64),
        p_sat=0.0, n_nuc=n_nuc, c_vap=1.0, c_cond=1.0,
    )
    assert abs(got.item() - hand) / hand < 1e-12, f"{got.item():.12e} vs 手算 {hand:.12e}"


@pytest.mark.unit
def test_evaporation_and_condensation_branches_are_antisymmetric() -> None:
    """c_vap = c_cond 时 ṁ(p_sat−Δ) = −ṁ(p_sat+Δ)（同一 ρ、逐点）。"""
    dp = torch.tensor([1e-4, 1e-3, 1e-2], dtype=torch.float64)
    rho = torch.full_like(dp, RHO_MID)
    evap = schnerr_sauer_source(rho, -dp, p_sat=0.0, n_nuc=1e-4, c_vap=1.0, c_cond=1.0)
    cond = schnerr_sauer_source(rho, +dp, p_sat=0.0, n_nuc=1e-4, c_vap=1.0, c_cond=1.0)
    assert torch.allclose(evap, -cond, rtol=1e-12, atol=0.0)
    # 默认系数（c_cond=0.2）只是等比缩放，符号与比例必须保持
    cond_def = schnerr_sauer_source(rho, +dp, p_sat=0.0, n_nuc=1e-4)
    assert torch.allclose(cond_def, 0.2 * cond, rtol=1e-12, atol=0.0)


@pytest.mark.unit
def test_si_to_lattice_nucleation_conversion() -> None:
    """n_nuc_si + lattice_length_m 与直接给格点 n_nuc 等价；非法输入报错。"""
    rho = torch.tensor([RHO_MID], dtype=torch.float64)
    p = torch.tensor([-1e-2], dtype=torch.float64)
    a = schnerr_sauer_source(rho, p, p_sat=0.0, n_nuc=1e-4)
    b = schnerr_sauer_source(
        rho, p, p_sat=0.0, n_nuc_si=1.0e8, lattice_length_m=1.0e-4
    )
    assert torch.allclose(a, b, rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="lattice_length_m"):
        schnerr_sauer_source(rho, p, p_sat=0.0, n_nuc_si=1.0e8)
    with pytest.raises(ValueError, match="n_nuc must be > 0"):
        schnerr_sauer_source(rho, p, p_sat=0.0, n_nuc=0.0)
    # 默认值必须是格点量级（1 核 / 10^4 单元），不是 SI 的 1.6e13
    assert 0.0 < N_NUC_LATTICE_DEFAULT < 1.0


# ---------------------------------------------------------------------------
# D. 鉴别力负控：历史上合并过的两个缺陷形态必须被判红
# ---------------------------------------------------------------------------
def _legacy_source(  # noqa: PLR0913
    rho: torch.Tensor,
    p: torch.Tensor,
    p_sat: float,
    *,
    single_alpha: bool = False,
    r_b_clamp: float | None = None,
    n_nuc: float = 1.6e-17,
) -> torch.Tensor:
    """修复前 ``cavitation.py`` 的源项形态（用于负控，不进库）。

    ``single_alpha=True``：只乘 α_l（蒸发）/ α_v（凝结）之一——缺一个相因子。
    ``r_b_clamp``：与历史实现一致把 R_b 硬 clamp 到上限（默认 0.5 lu）。
    """
    rho_l, rho_v = RHO_L, RHO_V
    alpha_v = ((rho_l - rho) / (rho_l - rho_v + 1e-12)).clamp(0.0, 1.0)
    alpha_l = 1.0 - alpha_v
    r_b = (alpha_v / alpha_l.clamp(min=1e-6) * (3.0 / (4.0 * math.pi * n_nuc + 1e-40))) ** (
        1.0 / 3.0
    )
    r_b = r_b.clamp(min=1e-6, max=0.5 if r_b_clamp is None else r_b_clamp)
    dp = p - p_sat
    dp_mag = dp.abs().clamp(min=1e-12)
    base = (rho_l * rho_v / rho.clamp(min=1e-4)) * (3.0 / r_b) * torch.sqrt(
        2.0 / 3.0 * dp_mag / rho_l
    )
    alpha_factor = alpha_l if single_alpha else alpha_l * alpha_v
    return torch.where(dp < 0, 1.0 * alpha_factor * base, -0.2 * alpha_v * base)


@pytest.mark.unit
def test_negative_control_missing_phase_factor_is_detected() -> None:
    """缺 α 因子 → 单相极限不为零：负控在单相性质上必须报出 ≫1 的误差。"""
    rho = torch.tensor([RHO_L - 1e-8 * (RHO_L - RHO_V)], dtype=torch.float64)  # α_v≈1e-8
    p = torch.tensor([-1e-2], dtype=torch.float64)
    legacy = _legacy_source(rho, p, 0.0, single_alpha=True, r_b_clamp=1.0)
    fixed = schnerr_sauer_source(rho, p, p_sat=0.0, n_nuc=1e-4)
    assert abs(float(legacy)) > 1e3 * abs(float(fixed)) + 1e-6, (
        f"负控未被判红：legacy|ṁ|={abs(float(legacy)):.3e} vs fixed={abs(float(fixed)):.3e}"
    )


@pytest.mark.unit
def test_negative_control_clamped_radius_is_detected() -> None:
    """R_b 被 clamp 成常数 → R_b 反演误差 ≫ 1e-12（本文件的 R_b 断言会判红）。"""
    av = torch.tensor([0.5], dtype=torch.float64)
    alpha_v = 0.5
    rho = torch.tensor([RHO_L - alpha_v * (RHO_L - RHO_V)], dtype=torch.float64)
    p = torch.tensor([-1e-2], dtype=torch.float64)
    legacy = _legacy_source(rho, p, 0.0, r_b_clamp=0.5)
    pref_no_rb = float((RHO_L * RHO_V / rho) * (1.0 - alpha_v) * alpha_v)
    driving = math.sqrt(2.0 * 1e-2 / (3.0 * RHO_L))
    r_b_measured = 3.0 * pref_no_rb * driving / float(legacy)
    assert abs(r_b_measured - 0.5) < 1e-9
    r_b_exact = float(schnerr_sauer_bubble_radius(av, 1e-4))
    assert abs(r_b_measured - r_b_exact) / r_b_exact > 0.9, (
        f"clamp 负控未暴露：R_b 反演 {r_b_measured:.3e} vs 解析 {r_b_exact:.3e}"
    )


@pytest.mark.unit
def test_negative_control_saturation_leak_is_detected() -> None:
    """修复前 p=p_sat 时走凝结分支留下非零伪传质：负控必须被判红。"""
    rho = torch.tensor([RHO_MID], dtype=torch.float64)
    p = torch.tensor([0.0], dtype=torch.float64)
    legacy = _legacy_source(rho, p, 0.0, r_b_clamp=0.5)
    fixed = schnerr_sauer_source(rho, p, p_sat=0.0, n_nuc=1e-4)
    assert fixed.item() == 0.0
    assert abs(float(legacy)) > 0.0, "负控应当复现饱和点的非零伪传质"


# ---------------------------------------------------------------------------
# E. 与库内调用方的一致性（源项量级不得把 apply_cavitation_force 打成非有限值）
# ---------------------------------------------------------------------------
@pytest.mark.regression
def test_apply_cavitation_force_stays_finite_with_new_source() -> None:
    from tensorlbm.cavitation import apply_cavitation_force
    from tensorlbm.d2q9 import equilibrium

    n = 16
    x = torch.arange(n, dtype=torch.float32) / n
    rho = 1.2 + 0.6 * x.reshape(1, n).repeat(n, 1)
    zeros = torch.zeros(n, n)
    f = equilibrium(rho, zeros, zeros)
    out = apply_cavitation_force(f, rho, zeros, zeros, tau=1.0)
    assert bool(torch.isfinite(out).all())
    # 质量源量级必须远小于当地密度（否则单步强行注入会立刻失稳）
    m_dot = schnerr_sauer_source(rho, rho / 3.0, p_sat=float(rho.mean()) / 3.0)
    assert float(m_dot.abs().max()) < 0.5 * float(rho.mean()), (
        f"源项量级过大：max|ṁ|={float(m_dot.abs().max()):.3e}"
    )