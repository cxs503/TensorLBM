"""NSCBC 出口反射系数验证（D2Q9 声脉冲，解析锚标定的仪器）。

被测：``tensorlbm.boundaries.nscbc_outlet_2d``。该能力在 2026-11-12 前登记为
``IMPLEMENTATION_ONLY / No tests``；本次把它重写为"外向声学特征 + 缓慢压力松弛"
并建立本验证面。

仪器：``benchmarks/verified/nscbc_outlet_reflection/run.py`` —— 1-D 声脉冲打击
出口，探针处作右行/左行特征分解 A± = ½(p' ± ρ0·c·u')，反射系数
R = A−/(A+·衰减)。仪器自身由两个**解析极限**标定：

* 压力 Dirichlet（压力节点）→ |R| = 1、符号负；
* 刚性壁（全反射）→ |R| = 1、符号正。

两个锚都过不了 ⇒ 任何 R 读数不成立（先判仪器，再判被测）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_BENCH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "nscbc_outlet_reflection"
    / "run.py"
)

# 测试档：nx=900, n_steps=2600, width=48（与 result.json 归档配置同参数）（入射窗与反射窗已分离；单档 ~8 s）
CFG = dict(nx=900, ny=4, n_steps=2600, width=48.0, x0=200, nu=0.1, amp=1e-3)


@pytest.fixture(scope="module")
def run_case():
    spec = importlib.util.spec_from_file_location("nscbc_reflection_bench", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.run_case


@pytest.fixture(scope="module")
def results(run_case):
    out = {}
    for bc, sigma in (
        ("mirror", 0.0),
        ("pin", 0.0),
        ("zerograd", 0.0),
        ("nscbc", 0.25),
        ("nscbc", 1.0),
    ):
        out[f"{bc}@{sigma}"] = run_case(bc, sigma, **CFG)
    return out


def test_instrument_anchors_are_exact_reflectors(results):
    """仪器标定：两个解析极限 |R|=1（含符号），否则后续读数无意义。"""
    mir = results["mirror@0.0"]
    pin = results["pin@0.0"]
    assert mir["reflection_sign"] > 0
    assert pin["reflection_sign"] < 0
    assert abs(mir["reflection_coeff_atten_corrected"]) == pytest.approx(1.0, abs=0.10)
    assert abs(pin["reflection_coeff_atten_corrected"]) == pytest.approx(1.0, abs=0.10)
    # 入射段自带左行底噪必须远小于 1（否则 R 无从分辨）
    assert results["pin@0.0"]["floor_over_incident"] < 0.05


def test_dalembert_travel_time_and_shape(results):
    """入射段解析锚：d'Alembert 平移解——波速误差 <1%、波形误差 <6%。"""
    pin = results["pin@0.0"]
    assert abs(pin["wave_speed_err_pct"]) < 1.0
    assert pin["dalembert_shape_err_pct"] < 6.0


def test_nscbc_default_is_non_reflecting(results):
    """库默认参数（sigma=0.25）必须是"非反射"：|R| ≤ 5%。"""
    r = results["nscbc@0.25"]["reflection_coeff_atten_corrected"]
    assert abs(r) <= 0.05, f"|R|={abs(r):.4f} > 5%"


def test_nscbc_beats_zero_gradient_outlet(results):
    """对照档：零梯度出口在本仪器上 ~28% 反射；NSCBC 至少好 5 倍。"""
    r_ns = abs(results["nscbc@0.25"]["reflection_coeff_atten_corrected"])
    r_zg = abs(results["zerograd@0.0"]["reflection_coeff_atten_corrected"])
    assert r_zg > 0.10, f"零梯度参照异常（{r_zg:.3f}）——仪器或配置改了"
    assert r_zg >= 5.0 * r_ns, f"nscbc {r_ns:.4f} vs zerograd {r_zg:.4f}"


def test_nscbc_not_worse_at_large_sigma(results):
    """σ 有真实时间尺度：σ=1 不得退化成近乎全反射（旧实现 14.8%）。"""
    r1 = results["nscbc@1.0"]["reflection_coeff_atten_corrected"]
    assert abs(r1) <= 0.05, f"sigma=1 时 |R|={abs(r1):.4f}"


def test_archived_result_json_agrees_with_implementation():
    """存档 result.json 与本实现一致（防报告数与代码脱节）。"""
    import json

    res = json.loads((_BENCH.parent / "result.json").read_text())
    v = res["verdict"]
    assert v["instrument"]["pass"] is True
    assert v["claims_passed"] == v["claims_total"]
    c3 = v["library_nscbc_claims"]["claim3_default_sigma025_non_reflecting_|R|<=5pct"]
    assert c3["pass"] is True and c3["measured"] <= 0.05
