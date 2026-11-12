"""CASE-LBM-KOVASZNAY-EXACT 独立复核工装的自检（m16861）。

目的：把这次审计里**最容易腐坏的三件事**钉成测试，而不是只留一份 Markdown：

1. 参考解自检：λ 恒等式、λ(Re=40) 的数值、以及"λ=−0.5 对应 Re≈79.5 而非 40"这一口径纠正；
   解析场代入离散算子后的 N-S 残差必须是截断量级（否则后续所有误差判定都不成立）。
2. 独立实现的**可用性判别力**：小网格算例必须仍是稳定的、有限的，且与解析解的 L2 误差
   在合理量级（<2%）；同时**反例**：把入口 Zou-He 的对角量配对写错（f5↔f6 而非 f5↔f7）
   必须发散——证明本测试不是"跑通即通过"。
3. 已归档证据（artifacts/m16861/independent_audit.json）与锚点判据的一致性：
   ny>=128 的 v L2 相对误差 <3%；且出口误差占比 >50%（审计的核心结论）。

运行：python3 -m pytest tests/test_kovasznay_independent_audit.py -q
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
MOD = REPO / "benchmarks" / "verified" / "kovasznay_2d" / "independent_audit.py"
ART = REPO / "artifacts" / "m16861" / "independent_audit.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("kov_audit", MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


A = _load_module()


# ---------------------------------------------------------------- 1. 参考解
def test_lambda_identity_and_value():
    lam = A.kovasznay_lambda(40.0)
    assert abs(lam * lam - 40.0 * lam - 4 * math.pi ** 2) < 1e-10
    assert lam == pytest.approx(-0.9637405441957689, abs=1e-12)


def test_lambda_task_statement_is_wrong():
    """任务书写 "Re=40 时 λ≈−0.5"；λ=−0.5 实际对应 Re≈79.46（不是 40）。"""
    re_at_half = -(4 * math.pi ** 2 + 0.25) / (-0.5)
    assert re_at_half == pytest.approx(79.4568, abs=1e-3)
    assert abs(A.kovasznay_lambda(40.0) + 0.5) > 0.4


def test_analytic_field_satisfies_discrete_navier_stokes():
    """解析场代入同一离散算子：残差应是差分量级，不是物理量级。"""
    sc = A.cross_check(re=40.0)
    assert sc["lambda_identity_residual"] < 1e-10
    assert sc["ns_res_max_analytic_u"] < 5e-3
    assert sc["ns_res_max_analytic_v"] < 5e-3
    assert sc["div_max_analytic"] < 5e-3


# ---------------------------------------------------------------- 2. 独立实现
def test_independent_solver_small_grid_is_accurate_and_finite():
    # Re=20 而非 40：ny=16 时 Re=40 会把 τ 压到 0.536（本实现在该 τ 上入口 Zou-He 不稳健，
    # 见 docs ⑤ 注意点 7）；Re=20 给出 τ=0.572，与算例主配置同一稳定区。
    m = A.run(16, 20.0, 0.03, min_steps=1200, max_steps=2400)
    assert m["finite"] is True
    assert m["u_l2_rel"] < 0.02, m["u_l2_rel"]
    assert len(m["err_profile_u_over_u0"]) == m["nx"]
    # 误差集中在出口侧这一结论在 ny>=32 的归档证据里断言（ny=16 太粗，入口区噪声可占优）
    assert m["u_err_last10pct_over_u0"] > 0.0


def _wrong_pairing_inlet(f, ux_row, uy_row):
    """反例：把 f5 与 f6 配对（错），正解是 f5↔f7。"""
    f0, f2, f3, f4 = f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0]
    f6, f7 = f[6, :, 0], f[7, :, 0]
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_row)
    f[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux_row
    f[5, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux_row + 0.5 * rho * uy_row
    f[8, :, 0] = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux_row - 0.5 * rho * uy_row


def test_wrong_zouhe_pairing_is_detected_by_divergence(monkeypatch):
    ny = 16
    monkeypatch.setattr(A, "zou_he_inlet", _wrong_pairing_inlet)
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        m = A.run(ny, 40.0, 0.03, min_steps=600, max_steps=600)
        # run() 自身不抛错时，用有限性强行失败（反例必须被抓到）
        assert m["finite"] is True


# ---------------------------------------------------------------- 3. 归档证据
@pytest.mark.skipif(not ART.exists(), reason="审计工件未生成")
def test_archived_evidence_supports_audit_conclusions():
    d = json.loads(ART.read_text())
    grids = d.get("grids", [])
    g128 = [g for g in grids if g["ny"] >= 128]
    assert g128, "工件缺 ny>=128 的臂"
    vmax = max(g["v_l2_rel"] for g in g128)
    assert vmax < 0.03, f"锚点判据 v L2 {vmax:.4f} 不满足 <3%"

    zg = next(g for g in grids if g["ny"] == 64)
    di = next(g for g in d.get("grids_dirichlet", []) if g["ny"] == 64)
    share = 1.0 - di["max_abs_err_u_over_u0"] / zg["max_abs_err_u_over_u0"]
    assert share > 0.5, f"出口 BC 贡献 {share:.2f} 应为主项"

    fit = d["ma_budget"]["fit"]
    ma2 = next(a["ma2"] for a in d["ma_budget"]["arms"] if abs(a["u0"] - 0.03) < 1e-9)
    frac = fit["slope"] * ma2 / (fit["intercept"] + fit["slope"] * ma2)
    assert frac < 0.25, f"Ma² 项占比 {frac:.2f} 应是小头（证伪 README 归因）"