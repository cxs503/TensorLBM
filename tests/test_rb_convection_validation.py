"""P96W3-N04: 热对流定量验证面的 pytest 用例（快档）。

覆盖：(几何/边界口径自洽) · (纯导热极限 Nu=1) · (绝热壁焓守恒) ·
(正弦波衰减反演 α 与精度阶) · (线性起稳 σ 符号 + 时间推进 Ra_c 读数) ·
(无量纲自洽 ΔT 缩放不变性) · (老缺陷回归：H 口径不一致必须被本模块挡住)。

重档（多分辨率 GCI、Ra=1e5/1e6、k_c 扫描）由
``scripts/p96w3_n04_campaign.py`` 在集群/后台跑，结果落
``artifacts/p96w3_n04/*.json``；这里只保留能在单机分钟级复现的判据。
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.rb_convection_validation import (
    A_CRIT,
    RA_CRIT,
    Convection2D,
    adiabatic_enthalpy_drift,
    diffusion_decay_alpha,
    diffusion_decay_order,
    dimensionless_selfconsistency,
    make_convection,
    measure_growth_rate,
    no_buoyancy_control,
    wall_flux_nusselt,
)


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# ── 口径自洽（几何唯一真源）────────────────────────────────────────────────


class TestGeometryConvention:
    def test_height_is_ny(self):
        """H 必须等于 ny（半程壁 −1/2 … ny−1/2）：P96W2 的 ny−1/ny 混用回归。"""
        sim = make_convection(ny=32, nx=64, ra=1e4, alpha=1 / 6)
        assert sim.H == 32.0
        assert sim.info()["H"] == 32.0
        # 流动间隙（pre-半程反弹实测 d_v = ny−2）与温度间隙（ABB 实测 d_T ≈ ny）
        assert sim.gap_flow == 30.0
        # 基态必须恰好铺满 ΔT（半程到半程）
        Tb = sim.base_profile()
        assert abs(float(Tb[0, 0].item()) - (1.0 - 0.5 / 32.0)) < 1e-12
        assert abs(float(Tb[-1, 0].item()) - 0.5 / 32.0) < 1e-12

    def test_linear_profile_gives_unit_nusselt(self):
        """线性剖面上二阶壁面差分必须逐位给 Nu=1（Nu 口径正确性）。"""
        ny, nx = 40, 8
        H = float(ny)
        T = torch.empty((ny, nx), dtype=torch.float64)
        for j in range(ny):
            T[j, :] = 1.0 + (0.0 - 1.0) * (j + 0.5) / H
        nu_bot = wall_flux_nusselt(T, axis=0, side="lo", t_wall=1.0, dT=1.0, H=H)
        nu_top = wall_flux_nusselt(T, axis=0, side="hi", t_wall=0.0, dT=1.0, H=H)
        assert torch.allclose(nu_bot, torch.ones_like(nu_bot), atol=1e-14)
        assert torch.allclose(nu_top, torch.ones_like(nu_top), atol=1e-14)

    def test_dhc_base_profile_orientation(self):
        """DHC 基态：左热右冷（x=0 侧为 t_hot）。"""
        sim = make_convection(ny=16, nx=16, ra=1e3, case="dhc", alpha=1 / 6)
        Tb = sim.base_profile()
        assert float(Tb[0, 0].item()) > float(Tb[0, -1].item())


# ── C5 负控：纯导热极限 ────────────────────────────────────────────────────


class TestNoBuoyancyLimit:
    def test_rb_nu_equals_one(self):
        out = no_buoyancy_control(ny=32, steps=5000)
        assert out["nu_error"] < 1e-5, out
        assert out["u_max"] < 1e-12
        assert out["profile_linf_over_dT"] < 1e-6

    def test_dhc_nu_equals_one(self):
        out = no_buoyancy_control(ny=32, steps=5000, case="dhc")
        assert out["nu_error"] < 1e-5, out
        assert out["u_max"] < 1e-12

    def test_adiabatic_enthalpy_conserved(self):
        out = adiabatic_enthalpy_drift(ny=32, nx=64, steps=10000)
        assert out["drift_rel"] <= 1e-6, out


# ── C6 热扩散率反演与精度阶 ────────────────────────────────────────────────


class TestDiffusionDecay:
    def test_alpha_recovered(self):
        out = diffusion_decay_alpha(n=32)
        assert out["alpha_rel_error_abs"] < 0.02, out
        assert out["tau_T"] == pytest.approx(1.0)

    def test_second_order_or_better(self):
        """网格加倍误差比 ≥ 4（二阶及以上）：本格式实测约 4 阶。"""
        out = diffusion_decay_order(ns=(16, 32, 64))
        ratios = [r["ratio"] for r in out["ratios"]]
        assert all(r >= 4.0 for r in ratios), ratios


# ── C1 线性起稳 ────────────────────────────────────────────────────────────


class TestLinearStability:
    def test_growth_sign_around_critical(self):
        """亚临界必须衰减、超临界必须增长，且拟合质量 R²>0.999。

        幅值自适应窗（1e-10 … 0.05）保证强衰减臂不会掉进 float64 舍入底、
        强增长臂不会进入非线性饱和——两者都会让斜率变成垃圾（历史缺陷）。
        """
        below = measure_growth_rate(ny=32, nx=64, ra=0.7 * RA_CRIT, alpha=1 / 6,
                                    steps=6000)
        above = measure_growth_rate(ny=32, nx=64, ra=1.5 * RA_CRIT, alpha=1 / 6,
                                    steps=6000)
        assert below["sigma_star"] < 0.0, below
        assert above["sigma_star"] > 0.0, above
        assert below["fit_r2"] > 0.999 and above["fit_r2"] > 0.999, (below, above)
        assert abs(above["sigma_star"]) > 1.0 and abs(below["sigma_star"]) > 1.0

    def test_zero_crossing_near_chandrasekhar(self):
        """时间推进测得的 Ra_c 必须落在 1707.76 的 ±10% 内（快档粗网格）。

        正式判据（≤5%）在战役网格 ny=48 上由
        ``scripts/p96w3_n04_campaign.py --task c1-rac`` 给出（artifacts JSON）；
        这里只保证机制在最小算例上就能复现（粗格 ny=32 的 Ma≈0.38，压缩性
        会把零交点推偏几个百分点，属已知系统误差）。
        """
        from tensorlbm.rb_convection_validation import critical_rayleigh_measured

        out = critical_rayleigh_measured(
            ny=32, nx=64, ratios=(0.9, 1.0, 1.1), steps=6000, alpha=1 / 6)
        assert out["ra_c_error_pct"] <= 10.0, out
        assert math.isfinite(out["kH"])


# ── C4 无量纲自洽 ──────────────────────────────────────────────────────────


class TestDimensionalSelfConsistency:
    def test_dt_scaling_invariance(self):
        """ΔT→sΔT 且 gβ→gβ/s（Ra 固定）时 Nu 必须不变：散布 ≤1%。"""
        out = dimensionless_selfconsistency(ny=24, ra=1e4, dTs=(0.5, 1.0, 2.0),
                                            steps=8000)
        assert out["spread_pct"] <= 1.0, out
        assert out["spread_pct"] < 1e-6, out  # 实际只应受浮点结合律影响

    def test_ra_definition_roundtrip(self):
        """info() 里 Ma 估计与 Ra 定义自洽（g_beta 复算 Ra）。"""
        sim: Convection2D = make_convection(ny=32, nx=64, ra=1e5, alpha=1 / 6)
        ra_back = sim.g_beta * sim.dT * sim.gap_flow**3 / (sim.nu_lat * sim.alpha_lat)
        assert ra_back == pytest.approx(1e5, rel=1e-12)
        assert sim.info()["ma_estimate"] == pytest.approx(
            sim.alpha_lat * math.sqrt(1e5 * 0.71) / (sim.H * math.sqrt(1 / 3)), rel=1e-12)