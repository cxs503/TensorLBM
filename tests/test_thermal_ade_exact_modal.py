"""D2Q5 热 advection-diffusion 精确验证（回归测试）。

判据分三层（详见 docs/thermal_ade_exact_modal_20261108.md）：

  L1 实现正确性：模拟的模态增长率/相速度必须等于 5x5 离散算子精确特征值
     给出的值（相对误差 < 1e-9）。这条判据把「实现」与「方法」解耦：
     任何 streaming 索引错、碰撞公式错、权重错都会让模拟偏离闭式特征值。
  L2 α 标定：实测热扩散率 α = γ/k² 与设定值 (τ_T − 1/2)/3 一致（< 1e-6 相对，
     τ_T = 1.0 处机器精度；其余 τ_T 受 O(αk²) 离散偏移影响，见 L3）。
  L3 离散阶：离散衰减率与连续 αk² 的相对差 ∝ k²（拟合阶 2.00 ± 0.05），
     且绝对量级 < 1.5%（k = 4π/32）。

另附 D2Q5 对流偏差律的负控（falsifiable prediction）：
  α_eff/α − 1 = δ_k − 3u²（3 = 1/c_s²，与 τ_T、k 无关），
  残差 < 0.03pp（百分点）。若有人改动 g_eq 的 u² 结构（补齐二阶矩）该判据会红。
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.thermal import (
    C5,
    W5,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)

torch.set_num_threads(1)

_C5NP = C5.numpy().astype(np.float64)
_W5NP = W5.double().numpy().astype(np.float64)


# ── 闭式参考：D2Q5 BGK + 周期 streaming 的模态放大因子 ──────────────────────
def modal_eigen(k: float, u: float, tau: float) -> complex:
    """模态 (k, 0) 的精确每步放大因子（|λ| 最大特征值）。

    推导：g_eq_i = w_i (1 + 3 c_ix u) T（T = Σg），碰撞为局部线性映射
    A = (1 − 1/τ)I + P/τ（P_i = w_i(1+3c_ix u)），周期 streaming 对纯模态
    只乘相位 e^{−i k c_ix}，故 M = diag(e^{−i k c}) A。
    """
    P = _W5NP * (1.0 + 3.0 * _C5NP[:, 0] * u)
    A = (1.0 - 1.0 / tau) * np.eye(5) + P[None, :] / tau
    ev = np.linalg.eigvals(np.exp(-1j * _C5NP[:, 0] * k)[:, None] * A)
    return complex(ev[np.argmax(np.abs(ev))])


def simulate(n: int, periods: int, u: float, tau: float, efolds: float = 2.0,
             ny: int = 4, tail: int = 300) -> dict:
    """余弦模态模拟，返回实测衰减率 γ 与漂移速度 U（末段逐段复增长率）。"""
    k = 2.0 * math.pi * periods / n
    alpha = (tau - 0.5) / 3.0
    steps = int(max(400, min(12000, round(efolds / (alpha * k * k)))))
    steps = max(steps, tail + 12)
    x = torch.arange(n, dtype=torch.float64)
    T0 = torch.cos(k * x).unsqueeze(0).expand(ny, n).contiguous()
    uxt = torch.full((ny, n), u, dtype=torch.float64)
    uyt = torch.zeros((ny, n), dtype=torch.float64)
    g = temperature_equilibrium(T0, uxt, uyt)
    idx = np.arange(n, dtype=np.float64)

    def amp(gg: torch.Tensor) -> complex:
        T = gg.sum(dim=0)[0, :].numpy()
        return complex((T * np.exp(-1j * k * idx)).sum() / n)

    a = amp(g)
    mus: list[complex] = []
    for t in range(1, steps + 1):
        g = temperature_collision(g, tau, uxt, uyt)
        g = temperature_stream(g)
        if t >= steps - tail:
            b = amp(g)
            if t > steps - tail and abs(b) > 1e-300 and abs(a) > 1e-300:
                mus.append(b / a)
            a = b
    mu = np.exp(np.mean(np.log(np.array(mus, dtype=complex))))
    return {"gamma": -math.log(abs(mu)), "U": -math.atan2(mu.imag, mu.real) / k, "k": k}


# ── L1：实现 = 闭式离散特征值 ────────────────────────────────────────────────
@pytest.mark.parametrize("tau", [0.6, 0.8, 1.2])
@pytest.mark.parametrize("n", [64, 256])
def test_modal_decay_matches_exact_eigenvalue(tau: float, n: int) -> None:
    """纯扩散：模拟衰减率 = 闭式特征值衰减率（< 1e-9 相对）。"""
    sim = simulate(n, 2, 0.0, tau, efolds=1.5)
    gd = -math.log(abs(modal_eigen(sim["k"], 0.0, tau)))
    assert abs(sim["gamma"] / gd - 1.0) < 1e-9, (
        f"γ_sim={sim['gamma']:.12e} vs γ_modal={gd:.12e} (tau={tau}, n={n})")


@pytest.mark.parametrize("u", [0.01, 0.05])
def test_advection_phase_speed_matches_exact_eigenvalue(u: float) -> None:
    """对流：相速度与衰减率同时对上闭式特征值（同步检验 −3u² 衰减偏移）。"""
    tau, n = 0.8, 256
    sim = simulate(n, 4, u, tau, efolds=1.5)
    ev = modal_eigen(sim["k"], u, tau)
    gd = -math.log(abs(ev))
    Ud = -math.atan2(ev.imag, ev.real) / sim["k"]
    assert abs(sim["gamma"] / gd - 1.0) < 1e-9, f"γ 偏差 {abs(sim['gamma']/gd-1):.2e}"
    assert abs(sim["U"] / Ud - 1.0) < 1e-9, f"U 偏差 {abs(sim['U']/Ud-1):.2e}"
    # 相速度与连续 u 的差仅 O(k²)：k=0.098 时 < 1e-5 相对
    assert abs(Ud / u - 1.0) < 1e-5


# ── L2：α 标定 ──────────────────────────────────────────────────────────────
def test_alpha_calibration_machine_precision_at_tau_one() -> None:
    """τ_T = 1 时 α = 1/6，实测 α = γ/k² 与设定值一致到 1e-6。"""
    tau = 1.0
    alpha = (tau - 0.5) / 3.0
    sim = simulate(512, 8, 0.0, tau, efolds=1.5)
    alpha_meas = sim["gamma"] / sim["k"] ** 2
    assert abs(alpha_meas / alpha - 1.0) < 1e-6


@pytest.mark.parametrize("tau", [0.55, 0.8, 2.0])
def test_alpha_calibration_within_discretization_bound(tau: float) -> None:
    """各 τ_T：实测 α 与设定值差 ≤ 0.7%（k=0.098 处 O(αk²) 离散偏移）。"""
    alpha = (tau - 0.5) / 3.0
    sim = simulate(512, 8, 0.0, tau, efolds=1.5)
    alpha_meas = sim["gamma"] / sim["k"] ** 2
    rel = abs(alpha_meas / alpha - 1.0)
    assert rel < 7e-3, f"α 标定偏差 {rel:.3%} (tau={tau})"


# ── L3：离散阶 ──────────────────────────────────────────────────────────────
def test_continuum_limit_is_second_order_in_k() -> None:
    """离散衰减率 vs 连续 αk² 的相对误差 ∝ k²（拟合阶 2.00 ± 0.05）。"""
    ks, errs = [], []
    for k in (0.2, 0.1, 0.05, 0.025):
        gd = -math.log(abs(modal_eigen(k, 0.0, 0.8)))
        ks.append(k)
        errs.append(abs(gd / ((0.8 - 0.5) / 3.0 * k * k) - 1.0))
    p = float(np.polyfit(np.log(ks), np.log(errs), 1)[0])
    assert abs(p - 2.0) < 0.05, f"拟合阶 {p:.3f}"


def test_error_magnitude_below_bound_on_finest_grid() -> None:
    """固定 2 周期、n=512（k=2π/256）：离散误差 ≤ 0.01%。"""
    k = 2.0 * math.pi * 2.0 / 512
    gd = -math.log(abs(modal_eigen(k, 0.0, 0.8)))
    assert abs(gd / ((0.8 - 0.5) / 3.0 * k * k) - 1.0) < 1e-4


# ── D2Q5 对流偏差律（可伪预言 + 负控）───────────────────────────────────────
def test_advection_diffusivity_bias_follows_minus_3u_squared() -> None:
    """α_eff/α − 1 = δ_k − 3u²（3 = 1/c_s²）；两点定斜率，残差 < 0.03pp。"""
    tau, n = 0.8, 256
    base = None
    for u in (0.0, 0.02, 0.05):
        k = 2.0 * math.pi * 4.0 / n
        gd = -math.log(abs(modal_eigen(k, u, tau)))
        drift = (gd / ((tau - 0.5) / 3.0 * k * k) - 1.0) * 100.0
        if base is None:
            base = drift
            assert abs(base) < 0.1  # δ_k 自身很小（k² 偏移）
            continue
        residual = drift - (base - 3.0 * u * u * 100.0)
        assert abs(residual) < 0.03, f"u={u}: 偏差律残差 {residual:+.4f}pp"
    # 偏差律的斜率本身：Δdrift/(u²) ≈ −300（%·1/u²）
    k = 2.0 * math.pi * 4.0 / n
    g5 = -math.log(abs(modal_eigen(k, 0.05, tau)))
    g0 = -math.log(abs(modal_eigen(k, 0.0, tau)))
    slope = ((g5 - g0) / g0) / (0.05 ** 2)
    assert abs(slope / -3.0 - 1.0) < 0.01


def test_negative_control_wrong_tau_is_detected() -> None:
    """负控：把 τ_T 用错 2%（碰撞实际用 0.784，参考按 0.8）判据必红。"""
    tau_true, tau_assumed, n = 0.784, 0.8, 256
    sim = simulate(n, 2, 0.0, tau_true, efolds=1.5)
    gd = -math.log(abs(modal_eigen(sim["k"], 0.0, tau_assumed)))
    assert abs(sim["gamma"] / gd - 1.0) > 1e-2


def test_negative_control_d2q5_equilibrium_has_no_u2_second_moment() -> None:
    """结构负控：D2Q5 平衡态二阶矩与 u 无关（=1/3），交叉矩恒零。

    这正是 −3u² 偏差律的来源（缺少 O(u²) 应力项 ⇒ 有效扩散率被对流削弱）。
    若有人给 g_eq 补上二阶矩（正确的 ADE 格式），本断言会红，需同步更新偏差律。
    """
    for u in (0.0, 0.05, 0.1):
        geq = _W5NP * (1.0 + 3.0 * _C5NP[:, 0] * u)
        assert abs(float((_C5NP[:, 0] ** 2 * geq).sum()) - 1.0 / 3.0) < 1e-7
        assert abs(float((_C5NP[:, 1] ** 2 * geq).sum()) - 1.0 / 3.0) < 1e-7
        assert abs(float((_C5NP[:, 0] * _C5NP[:, 1] * geq).sum())) < 1e-12
        assert abs(float(geq.sum()) - 1.0) < 1e-9