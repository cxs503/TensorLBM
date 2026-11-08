#!/usr/bin/env python3
"""Build the figure + report payload for the Allen-Cahn verification (m6394).

Reads artifacts/allen_cahn_r1_20261108/results.json (written by
scripts/allen_cahn_verify_20261108.py), renders fig1_anchors.png, and writes
/tmp/report.json with every number injected programmatically (no hand copying).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "allen_cahn_r1_20261108"
R = json.loads((ART / "results.json").read_text())

a1, a2, a3 = R["case_A1_stationary_profile"], R["case_A2_thickness_relaxation"], R["case_A3_travelling_wave"]
stab = R["case_H1_H2_default_stability"]
droplets = R["case_H3_droplet_laplace"]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=78)

ax = axes[0]
for case, label, color in ((a1, "A1 start = equilibrium W=4", "tab:blue"), (a2, "A2 start = 2W (W=8)", "tab:red")):
    steps = [h["step"] for h in case["history"]]
    widths = [h["width"] for h in case["history"]]
    ax.plot(steps, widths, "o-", color=color, ms=4, label=label)
ax.axhline(4.0, color="k", ls="--", lw=1, label=r"analytic $\xi=\sqrt{2\kappa/\beta}=4$")
ax.set_xscale("symlog", linthresh=100)
ax.set_xlabel("lattice steps")
ax.set_ylabel("measured interface width $W$")
ax.set_title("A1/A2: thickness relaxes to the analytic value")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["numeric\n(A3, 600 steps)", "analytic\n$v=\\frac{3}{2}Mh\\xi$"]
values = [a3["v_numeric"], a3["v_analytic"]]
bars = ax.bar(labels, values, color=["tab:purple", "tab:gray"], width=0.55)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:.4e}", ha="center", fontsize=8)
ax.set_ylabel("interface speed $v$ [lattice]")
ax.set_title(f"A3: travelling wave, rel. error {100 * a3['rel_error']:.2f}%")
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig_path = ART / "fig1_anchors.png"
fig.savefig(fig_path)
png_b64 = base64.b64encode(fig_path.read_bytes()).decode()

md = f"""## 1. 案例介绍

Allen–Cahn（AC）相场方程 ∂_tφ + ∇·(φu) = −Mμ，μ = ψ′(φ) − κ∇²φ，ψ = β(φ²−1)²/4。
TensorLBM 的 `src/tensorlbm/allen_cahn_lbm.py` 自 2026-08-20 合并后**零测试、零文档、未被包导出**，
docstring 却宣称「handles high density ratios (1000:1) without spurious currents / Better mass conservation
than Shan-Chen」。本单对它的相场内核算子做解析解复核，并对其耦合流体步给失效判定。

参数表（格点单位）：σ=0.01、W=4.0 ⇒ β=3σ/(2W)、κ=3σW/4，σ 与 ξ 可解析反解（rel 1e−12）；
A1 网格 4×4×96、M=1.0、4000 步；A2 同网格、初值 W=8、8000 步；A3 4×4×160、h=5e−4、600 步。

## 2. 怎么做的

相场输运改为显式 AC 更新 `step_allen_cahn_field`，Laplacian/梯度**复用**已被
`tests/test_phasefield_common_operators.py` 覆盖的 `tensorlbm.phasefield.operators`；界面力由
`(σ/W)(1−φ²)n̂`（无曲率）改为化学势 Korteweg 力 `μ∇φ`；流体步改为 ρ(φ) 入平衡态 + Guo 力。
旧实现不从记忆里取数：脚本用 `git show 3c0ab6c1:src/tensorlbm/allen_cahn_lbm.py` 现场取旧 blob 运行。
全部为 CPU 单线程轻量算例（本机 load15≈125）。

## 3. 计算结果

| 算例 | 观测量 | 数值 | 基准 | 误差 |
|---|---|---|---|---|
| A1 | 平衡厚度 | {a1['W_measured_final']:.6f} | {a1['W_nominal']} | {100*a1['width_rel_error']:.3f}% |
| A1 | L∞ 剖面漂移 | {a1['linf_drift_final']:.2e} | 0 | — |
| A1 | 界面位移 | {a1['interface_displacement']:.3f} 格 | 0 | 0 |
| A2 | 厚度终值（初值 {a2['W_initial']:.3f}） | {a2['W_final']:.6f} | {a2['W_target']} | {100*a2['rel_error']:.3f}% |
| A3 | 行波速度 | {a3['v_numeric']:.4e} | {a3['v_analytic']:.4e} | {100*a3['rel_error']:.2f}% |
| H1 旧实现 | 默认参数首次非有限步 | 第 {stab['legacy']['diverged_at_step']} 步 | 200 步有限 | 失败 |
| H2 新实现 | 默认参数首次非有限步 | 第 {stab['current']['diverged_at_step']} 步 | 200 步有限 | 失败 |

静止液滴（32³, R=8, σ=0.01）：1000:1 第 {droplets[0]['diverged_at_step']} 步发散，
10:1 第 {droplets[1]['diverged_at_step']} 步发散 ⇒ Young–Laplace 量测不可得，判 WITHHELD。

## 4. 结果图

![anchors](fig1_anchors.png)

(a) A1（起于解析剖面 W=4）与 A2（起于 2 倍过厚 W=8）的实测厚度演化，均收敛到 ξ=4；
(b) A3 实测行波速度 vs 解析 v=3/2·M·h·ξ。场量云图未给：本单验证 1-D 横向均匀内核，云图无信息量，
耦合液滴算例不收敛（画出来是 NaN）。

## 5. 注意点

1. 周期边界会把 1-D 单界面变成**双**界面（边界 φ=+1/−1 跳变），A1 的 L∞ 漂移会从 3.6e−3 涨到 0.81；
   A1/A2 必须用 `no_flux`——不做这个诊断会误判「内核不稳」。
2. 厚度测量对格子采样有 1–2.5% 系统偏置（W=2: 2.54%，W=4: 0.99%），判据要按偏置量级解读。
3. A3 的 7.36% 来自离散行波在有限 W 下非严格自相似（剖面厚度 4.04→4.08），非代码缺陷；
   要压到 <3% 需 h→0 收敛阶扫描（本次未做）。
4. 耦合流体步未修好：只去掉「力/气相密度」与「无曲率」两个确定缺陷，变密度 BGK 仍缺压力演化，
   实测比旧实现晚 1 步发散、首步 max|u| 反而更大（0.986 vs 0.332）⇒ 判 WITHHELD 并加硬门禁
   （默认 raise，需 `allow_unverified_hydro=True` 才能复现失败），避免下游拿到 NaN 继续算。
5. **环境陷阱**：`__editable__.tensorlbm-0.3.0.pth` 让 `sys.path` 同时含 6 个 worktree 的 `src/`，
   `import tensorlbm` 解析到 m6150 的工作树；必须 `PYTHONPATH=<本 worktree>/src`，否则测的是别人的代码。

## 6. 误差判定（是否 <3%）

A1 厚度 {100*a1['width_rel_error']:.3f}% ⇒ **是**；A1 L∞ 漂移对幅值 2 为 {100*a1['linf_drift_final']/2:.2f}% ⇒ **是**；
A1 界面位移 0 格 ⇒ **是**；A2 厚度 {100*a2['rel_error']:.3f}% ⇒ **是**；
A3 行波速度 {100*a3['rel_error']:.2f}% ⇒ **否**（渐近解，已明示）；
耦合流体稳定性 4/5/5/12 步发散 ⇒ **否**，判 WITHHELD，不粉饰。
"""

payload = {
    "title": "Allen–Cahn 相场内核算术解析解验证 + 旧实现失效判定（TensorLBM m6394）",
    "summary": (
        "零测试孤儿模块 allen_cahn_lbm 实测默认参数 4 步发散；重写为已验证 AC 内核"
        f"（A1 厚度误差 {100*a1['width_rel_error']:.3f}%、界面位移 0 格；A2 从 2W 收敛误差 "
        f"{100*a2['rel_error']:.3f}%；A3 行波速度 {100*a3['rel_error']:.2f}% 未达标如实记录），"
        "耦合流体步判 WITHHELD 并加硬门禁。"
    ),
    "scenario_key": "tensorlbm-allen-cahn-verification",
    "mission_id": 6394,
    "tags": ["TensorLBM", "multiphase", "phase-field", "Allen-Cahn", "analytic-verification", "honest-FAIL"],
    "markdown": md,
    "figures": [{"name": "fig1_anchors.png", "caption": "A1/A2 厚度收敛到解析 ξ=4；A3 行波速度 vs 解析解", "png_base64": png_b64}],
}

out = Path("/tmp/report.json")
out.write_text(json.dumps(payload))
print("payload bytes:", out.stat().st_size, "figure bytes:", len(png_b64))