#!/usr/bin/env python3
"""Generate report JSON for laplace_cac_phasefield SDAA m293."""
import json
import base64

# Read the figure
with open("artifacts/laplace_cac_sdaa_diag.png", "rb") as f:
    png_b64 = base64.b64encode(f.read()).decode("utf-8")

report = {
    "title": "SDAA 诊断报告：laplace_cac_phasefield 臂覆盖 FAIL 根因分析（m292/m293）",
    "summary": "本案例 5 臂数值判据全部 PASS（per-arm err ≤0.445%, u_max ≤6.74e-05, drift ≤8.62e-13, NaN-free），但 SDAA 判定 HONEST_FAIL，根因为臂覆盖不足：B1 三臂均固定 128³、B3 两臂均固定 96³，缺少网格分辨率阶梯以证明收敛性。修复路径明确：为 B1/B3 各增补至少一档网格（如 96³/192³），不改业务代码。",
    "scenario_key": "laplace_cac_phasefield",
    "mission_id": 293,
    "tags": ["SDAA", "arm_coverage", "HONEST_FAIL", "phasefield", "CAC", "Laplace", "surface_tension"],
    "markdown": (
        "# SDAA 诊断报告：laplace_cac_phasefield 臂覆盖 FAIL 根因分析\n\n"
        "## ① 案例介绍\n\n"
        "**物理背景**：周期立方域中心放置静态液滴，CAC（conservative Allen-Cahn）相场两相 LBM 演化至稳态。"
        "液滴内外压差满足 Young-Laplace 方程（3D 球）：Δp = 2σ/R。通过测量 Δp 反推 σ_rec，"
        "验证 σ_rec 恢复目标值且 R 无关、密度比无关。\n\n"
        "**几何与参数**：\n"
        "- 域：周期立方，B1 臂 128³，B3 臂 96³\n"
        "- 液滴半径 R：B1 = {20, 32, 48}，B3 = 20\n"
        "- 密度比：B1 = 100 (ρ_l=1, ρ_g=0.01)，B3 = {10, 100}\n"
        "- 界面宽度 W = 4，迁移率 M = 0.05，ν = 0.1\n"
        "- Route-B σ 标定：σ_model = 0.01 / 0.923896 = 0.0108237\n\n"
        "**模型**：CAC/HCH 双分布 D3Q19，fp64，harmonic μ(φ) 混合律，"
        "tensorlbm.cac_lbm（Liang et al. PRE 97, 033309）。\n\n"
        "## ② 诊断方法\n\n"
        "1. 读取已落盘 result.json（m24 历史运行产物），提取 5 臂数值结果\n"
        "2. 对照 SDAA 判据口径，逐项核对 per-arm err、u_max、drift、NaN-free、B1 spread、臂覆盖\n"
        "3. 分析\"臂覆盖 FAIL\"根因：SDAA 要求网格分辨率阶梯（≥2 档网格/臂组）以证明数值收敛，"
        "但本案例 B1 三臂均固定 128³、B3 两臂均固定 96³\n"
        "4. 生成逐臂判据表与诊断图\n\n"
        "## ③ 计算结果（逐臂判据表）\n\n"
        "| 臂 | 类型 | 参数 | 网格 | σ_rec | 误差(%) | u_max | drift@10k | NaN-free | 数值PASS |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        "| b1_R20 | B1 | R=20 | 128³ | 0.0099938 | 0.062 | 6.73e-05 | 8.62e-13 | ✓ | ✓ |\n"
        "| b1_R32 | B1 | R=32 | 128³ | 0.0100004 | 0.004 | 6.14e-05 | 8.60e-13 | ✓ | ✓ |\n"
        "| b1_R48 | B1 | R=48 | 128³ | 0.0100445 | 0.445 | 6.08e-05 | 8.58e-13 | ✓ | ✓ |\n"
        "| b3_rho10 | B3 | ρ=10 | 96³ | 0.0099955 | 0.045 | 5.02e-05 | 8.62e-13 | ✓ | ✓ |\n"
        "| b3_rho100 | B3 | ρ=100 | 96³ | 0.0099938 | 0.062 | 6.74e-05 | 8.62e-13 | ✓ | ✓ |\n\n"
        "**汇总**：\n"
        "- per-arm err max = 0.445% ≤ 1.5% → PASS\n"
        "- B1 spread = 0.507% ≤ 1.0% → PASS（result.json 口径）\n"
        "- u_max max = 6.74e-05 ≤ 2e-4 → PASS\n"
        "- drift@10k max = 8.62e-13 ≤ 1e-6 → PASS\n"
        "- NaN-free = 5/5 → PASS\n"
        "- **臂覆盖 = FAIL**（缺少网格阶梯）\n"
        "- **B1 spread (SDAA 口径) = NOT_EVALUATED**\n\n"
        "## ④ 结果图\n\n"
        "![σ_rec per arm vs target and error breakdown](data:image/png;base64," + png_b64 + ")\n\n"
        "左图：5 臂 σ_rec 柱状图（蓝色=B1，橙色=B3），红色虚线为 σ_target=0.01。所有臂 σ_rec 与目标值偏差 ≤0.45%。\n"
        "右图：逐臂相对误差，红色虚线为 1.5% 阈值。最大误差 0.445%（b1_R48），远低于阈值。\n\n"
        "## ⑤ 注意点\n\n"
        "1. **HONEST_FAIL 含义**：数值判据全部合格，但 SDAA 流程要求臂覆盖（含网格阶梯）作为前置条件。"
        "缺少网格阶梯 → B1 spread 无法在 SDAA 口径下评估 → 臂覆盖 FAIL → overall FAIL。"
        "这是流程要求而非数值不合格。\n"
        "2. **Route-B 标定**：CAC 方案的 σ_eff(W) ≠ σ_input 是 W 依赖的标定常数，"
        "经一次性补偿后 σ_rec 恢复 0.01，跨 R、跨密度比迁移成立。\n"
        "3. **fp64 必要性**：相场模型对数值精度敏感，fp32 可能导致界面不稳定；本案例全程 fp64。\n"
        "4. **修复路径**：为 B1 至少一臂（建议 R=20）和 B3 至少一臂（建议 ρ=100）增加第二档网格分辨率"
        "（B1: 96³ 或 192³; B3: 64³ 或 128³），无需修改业务代码。\n\n"
        "## ⑥ 误差判定\n\n"
        "| 关键量 | 基准值 | 计算值 | 误差 | 是否<3% |\n"
        "|---|---|---|---|---|\n"
        "| σ_rec (b1_R20) | 0.01 | 0.0099938 | 0.062% | ✓ |\n"
        "| σ_rec (b1_R32) | 0.01 | 0.0100004 | 0.004% | ✓ |\n"
        "| σ_rec (b1_R48) | 0.01 | 0.0100445 | 0.445% | ✓ |\n"
        "| σ_rec (b3_rho10) | 0.01 | 0.0099955 | 0.045% | ✓ |\n"
        "| σ_rec (b3_rho100) | 0.01 | 0.0099938 | 0.062% | ✓ |\n"
        "| B1 spread | ≤1.0% | 0.507% | — | ✓ |\n"
        "| u_max | ≤2e-4 | 6.74e-05 | — | ✓ |\n"
        "| drift@10k | ≤1e-6 | 8.62e-13 | — | ✓ |\n\n"
        "**结论**：所有数值判据误差均远小于 3% 阈值。HONEST_FAIL 纯因臂覆盖（网格阶梯）不足，非数值不合格。"
    ),
    "figures": [
        {
            "name": "sigma_rec_per_arm_and_error",
            "caption": "左：5臂σ_rec柱状图（蓝=B1, 橙=B3），红线=σ_target=0.01；右：逐臂相对误差，红线=1.5%阈值。所有臂误差≤0.445%。",
            "png_base64": png_b64
        }
    ]
}

with open("/tmp/report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"Report saved to /tmp/report.json ({len(json.dumps(report))} bytes)")
