#!/usr/bin/env python3
"""LBM 100% 解析解验证（Step A–D）—— 绑定判据版。

历史（m3579 验证欠账）：本文件四个用例原先只 ``print`` 误差后 ``return passed``，
pytest 从不校验返回值（``PytestReturnNotNoneWarning``），因此**任何误差都会"通过"**；
再加上四个用例被 ``tests/ci_quarantine.txt`` 整体摘出 CI 闸门、设备硬编码
``sdaa:0``（CPU-only CI 直接 collect 失败），验证面等于失效。

现在：计算集中在 ``tests/validation/lbm_analytic_chain.py``（测试与文档注册表
共用的单一真源），本文件只做**绑定断言**，默认 CPU（可用
``TENSORLBM_VERIFY_DEVICE=sdaa:0`` 切加速卡）。文档数字由
``tests/test_lbm_verification_registry.py`` 做漂移判定。

判定阈值与教学模块声明一致：相对误差 < 1% 为 PASS。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))

import lbm_analytic_chain as chain  # noqa: E402

THRESHOLD_PCT = chain.THRESHOLD_PCT


# ---------------------------------------------------------------------------
# Step A: 剪切波衰减 → 粘度 ν
# ---------------------------------------------------------------------------
def test_step_a_shear_wave(device="auto"):
    """ν_measured vs ν=(τ-0.5)/3，相对误差 < 1%。"""
    m = chain.run_shear_wave(device)
    print(f"\n  Step A: ν_num={m.measured:.8f} ν_exact={m.expected:.8f} "
          f"err={m.rel_err_pct:.4f}% (device={m.settings['device']})")
    assert m.rel_err_pct < THRESHOLD_PCT, (
        f"Step A 剪切波粘度误差 {m.rel_err_pct:.4f}% ≥ {THRESHOLD_PCT}% "
        f"(ν_num={m.measured:.8f}, ν_exact={m.expected:.8f})"
    )


def test_step_a_negative_control_tau_perturbation(device="auto"):
    """负控：粘度偏 4% 时必须被判 FAIL —— 证明 1% 阈值真的有牙齿。"""
    nc = chain.negative_control_tau_perturbation(device)
    print(f"\n  负控: ν 偏离 {nc['deviation_pct']:.2f}%")
    assert nc["judged_fail"] is True, (
        f"负控失败：粘度偏 {nc['deviation_pct']:.2f}% 却仍被判 PASS，判据无灵敏度"
    )


# ---------------------------------------------------------------------------
# Step B: Couette 流 → 反弹位置（半路 0.5）+ 无滑移
# ---------------------------------------------------------------------------
def test_step_b_couette(device="auto"):
    """剖面 max|u-u_exact|/U < 1%（半路反弹，壁面在 y=0.5）。"""
    m = chain.run_couette(device)
    print(f"\n  Step B: max_err={m.rel_err_pct:.4f}% (device={m.settings['device']})")
    assert m.rel_err_pct < THRESHOLD_PCT, (
        f"Step B Couette 剖面误差 {m.rel_err_pct:.4f}% ≥ {THRESHOLD_PCT}%"
    )


def test_step_b_bounce_back_order_only_affects_transient(device="auto"):
    """反弹放在迁移之前/之后，定常 Couette 解相同（实测差可忽略）。

    m3579 结论：文档 ``LBM_VERIFICATION_TEACHING.md`` 把"反弹时机错"记为
    4.76% 误差，但 4.76% = 0.5/(ny-1.5)，是**壁面全路/半路约定的解析偏移**；
    在本链的定常通道里两种次序收敛到同一个解（实测差 <1e-3 个百分点）。
    """
    before = chain.run_couette(device).rel_err_pct
    after = chain.bounce_back_after_stream_err(device)
    print(f"\n  bounce-back order: before-stream={before:.4f}% after-stream={after:.4f}%")
    assert after < THRESHOLD_PCT
    assert abs(after - before) < 0.05, (
        f"反弹次序在定常 Couette 里产生了 {abs(after - before):.3f} 个百分点差异，"
        "与 m3579 的定常态不变性结论不符"
    )


def test_step_b_wall_convention_offset_is_the_documented_four_point_seven_six():
    """钉住 4.76% 的真实出处：0.5/(ny-1.5) @ ny=12 = 4.762%。"""
    val = chain.wall_convention_offset_pct(12)
    print(f"\n  wall-convention offset = {val:.4f}%")
    assert abs(val - 4.7619) < 1e-3, f"解析偏移应为 4.7619%，实际 {val:.4f}%"


# ---------------------------------------------------------------------------
# Step C: Poiseuille 流 → 体力项 + 速度剖面
# ---------------------------------------------------------------------------
def test_step_c_poiseuille(device="auto"):
    """抛物线剖面 max|u-u_exact|/u_max < 1%。"""
    m = chain.run_poiseuille(device)
    print(f"\n  Step C: max_err={m.rel_err_pct:.4f}% "
          f"(u_num={m.settings['u_max_numeric']:.6f}, u_exact={m.settings['u_max_exact']:.6f})")
    assert m.rel_err_pct < THRESHOLD_PCT, (
        f"Step C Poiseuille 剖面误差 {m.rel_err_pct:.4f}% ≥ {THRESHOLD_PCT}%"
    )


# ---------------------------------------------------------------------------
# Step D: 壁面摩擦 → 壁面剪切应力
# ---------------------------------------------------------------------------
def test_step_d_friction(device="auto"):
    """Cf = 2τ_w/U² vs 解析 2ν/(H·U)，相对误差 < 1%。"""
    m = chain.run_wall_friction(device)
    print(f"\n  Step D: Cf_num={m.measured:.8f} Cf_exact={m.expected:.8f} "
          f"err={m.rel_err_pct:.4f}%")
    assert m.rel_err_pct < THRESHOLD_PCT, (
        f"Step D 摩擦系数误差 {m.rel_err_pct:.4f}% ≥ {THRESHOLD_PCT}% "
        f"(Cf_num={m.measured:.8f}, Cf_exact={m.expected:.8f})"
    )


def main() -> int:
    print("\n" + "=" * 64)
    print(" LBM 解析验证链 Step A–D（绑定判据，阈值 <1%）".center(62))
    print("=" * 64)
    rows = chain.run_chain("auto")
    n_pass = 0
    for m in rows:
        ok = m.passed
        n_pass += int(ok)
        print(f"  {m.key:<16} err={m.rel_err_pct:>8.4f}%  {'PASS' if ok else 'FAIL'}")
    neg = chain.negative_control_tau_perturbation("auto")
    print(f"  负控(τ+{neg['deviation_pct']:.2f}%) → {'FAIL(正确)' if neg['judged_fail'] else 'PASS(判据失效)'}")
    print(f"  通过 {n_pass}/{len(rows)}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())