"""XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION / R2 的回归锁。

锁三件事（全部无求解器在环，秒级）：

1. **模板四阶矩比**：D2Q9 与 D3Q19 的 ``κ3 = a3/a1`` 由库里的速度集/权重算出，
   必须精确等于 1/6 且两者相等 —— 这是 R2 跨维预言 ``σ_3D = σ_2D`` 的唯一输入。
2. **Maxwell 等面积构造**（解析基准）：SC EOS 上的共存密度必须复现首轮/R1 的
   参考值（ρ_v≈0.1529, ρ_l≈1.9289, p_c≈0.03421）到 1% 以内，防止分支跳变回归。
3. **Korteweg 锚 σ_K = |G_eff κ3| ∫(∂_n ψ)² dn** 的 1D 实现：在解析 tanh 剖面上
   必须复现闭式 ``|G_eff κ3| Δρ²/(3w)``（ψ_ρ→1 的低密度极限），误差 < 2%。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d2q9 import C as C2  # noqa: E402
from tensorlbm.d2q9 import W as W2  # noqa: E402

R2 = REPO_ROOT / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION" / "verify_r2"
sys.path.insert(0, str(R2))

from tensorlbm.multiphase import psi_exp  # noqa: E402
from run_r2 import G_EFF, kappa3, maxwell_coexistence, p_eos, sigma_korteweg_1d  # noqa: E402


def test_kappa3_equals_one_sixth_in_both_dims():
    k2, k3 = kappa3(2), kappa3(3)
    assert k2["kappa3"] == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert k3["kappa3"] == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert k2["kappa3"] == pytest.approx(k3["kappa3"], rel=1e-12)


def test_kappa3_uses_library_velocity_sets():
    assert kappa3(2)["n_vel"] == 9
    assert kappa3(3)["n_vel"] == 19
    C, W = torch.as_tensor(C2, dtype=torch.float64), torch.as_tensor(W2, dtype=torch.float64)
    a1 = float((W * C[:, 0] ** 2).sum())
    assert a1 == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_maxwell_coexistence_reproduces_round1_reference():
    mw = maxwell_coexistence()
    assert mw["rho_v"] == pytest.approx(0.152895, rel=0.01)
    assert mw["rho_l"] == pytest.approx(1.928866, rel=0.01)
    assert mw["p_c"] == pytest.approx(0.034214, rel=0.01)
    assert 0.0 < mw["rho_v"] < mw["rho_l"]
    assert abs(mw["mu_residual"]) < 1e-3


def test_sigma_korteweg_matches_closed_form_on_tanh():
    """低密度极限（ψ_ρ≈1）：σ_K = |G_eff κ3| Δρ²/(3w)，因 ∫sech⁴ = 4w/3。"""
    n, half, w = 192, 48.0, 3.0
    x = torch.arange(n, dtype=torch.float64)
    shape = 0.5 * (torch.tanh((x - (n / 2 - half)) / w) - torch.tanh((x - (n / 2 + half)) / w))
    rho_v, drho = 0.02, 0.02  # Δρ→0 使 ψ_ρ≈const，闭式成立
    prof = (rho_v + drho * shape).float()
    k3 = kappa3(2)["kappa3"]
    out = sigma_korteweg_1d(prof, k3, half)
    # 两个界面，各贡献 Δψ²/(3w)
    dpsi = float(psi_exp(torch.tensor([rho_v + drho])) - psi_exp(torch.tensor([rho_v])))
    closed = 2.0 * abs(G_EFF * k3) * dpsi**2 / (3.0 * w)
    assert out["sigma_K"] == pytest.approx(closed, rel=0.03)


def test_pressure_closure_has_spinodal():
    rho = torch.linspace(0.05, 3.0, 500, dtype=torch.float64)
    p = p_eos(rho)
    dp = torch.gradient(p, spacing=(rho,))[0]
    assert bool((dp < 0).any()), "EOS 应有负斜率段（相分离前提）"
    assert math.isfinite(float(p[0])) and float(p[-1]) > 0.0
