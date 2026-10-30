#!/usr/bin/env python
"""Generate figures and report for sphere_re100 mission m329."""
from __future__ import annotations

import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

BASE = os.path.dirname(os.path.abspath(__file__))

def load_hist(path):
    with open(path) as f:
        return json.load(f)

def encode_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def main():
    # Load data
    t1 = load_hist(os.path.join(BASE, "t1_d40big_12k.json"))
    t2 = load_hist(os.path.join(BASE, "t2_d60big_24k.json"))
    smoke = load_hist(os.path.join(BASE, "smoke_cpu_result.json"))

    cd_ref = 1.0917310910948732  # Schiller-Naumann at Re=100

    # Figure 1: Cd convergence history
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    steps1 = [h["step"] for h in t1["history"]]
    cd1 = [h["cd"] for h in t1["history"]]
    steps2 = [h["step"] for h in t2["history"]]
    cd2 = [h["cd"] for h in t2["history"]]

    ax1.plot(steps1, cd1, "b-", label="T1: D=40 (28722 BFL links)", linewidth=1.2, alpha=0.8)
    ax1.plot(steps2, cd2, "r-", label="T2: D=60 (64794 BFL links)", linewidth=1.2, alpha=0.8)
    ax1.axhline(y=cd_ref, color="g", linestyle="--", linewidth=1.5, label=f"Schiller-Naumann ref (Cd={cd_ref:.4f})")
    ax1.axhline(y=cd_ref * 1.03, color="orange", linestyle=":", linewidth=1, alpha=0.6, label="±3% gate")
    ax1.axhline(y=cd_ref * 0.97, color="orange", linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_xlabel("Time step", fontsize=11)
    ax1.set_ylabel("Drag coefficient Cd", fontsize=11)
    ax1.set_title("Sphere Re=100 — Cd Convergence History (BFL Interpolated Bounce-Back)", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 25000)
    png1 = encode_png(fig1)
    plt.close(fig1)

    # Figure 2: Error comparison bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    tiers = ["T1 (D=40)", "T2 (D=60)"]
    cd_vals = [1.1219804015117896, 1.119462001144486]
    errs = [(v - cd_ref) / cd_ref * 100 for v in cd_vals]
    colors = ["#2196F3", "#4CAF50"]
    bars = ax2.bar(tiers, errs, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axhline(y=3, color="red", linestyle="--", linewidth=1, alpha=0.7, label="3% gate")
    ax2.axhline(y=-3, color="red", linestyle="--", linewidth=1, alpha=0.7)
    for bar, err in zip(bars, errs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{err:+.3f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error vs Schiller-Naumann (%)", fontsize=11)
    ax2.set_title("Sphere Re=100 — Cd Error by Grid Resolution", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(-1, 4)
    png2 = encode_png(fig2)
    plt.close(fig2)

    # Figure 3: Domain convergence signature (from README data)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    up_d = [1.25, 2.0, 2.75, 3.0, 4.0]
    cd_dom = [1.214927, 1.155335, 1.137950, 1.134710, 1.127859]
    err_dom = [(c - cd_ref) / cd_ref * 100 for c in cd_dom]
    ax3.plot(up_d, err_dom, "ko-", linewidth=1.5, markersize=7, label="Domain convergence (lat=2.0)")
    ax3.axhline(y=2.771, color="blue", linestyle="--", linewidth=1.5, label="Final (lat=3, up=3, down=4): +2.771%")
    ax3.axhline(y=3, color="red", linestyle=":", linewidth=1, alpha=0.7, label="3% gate")
    ax3.set_xlabel("Upstream distance / D", fontsize=11)
    ax3.set_ylabel("Cd Error (%)", fontsize=11)
    ax3.set_title("Sphere Re=100 — Domain Convergence Signature", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    png3 = encode_png(fig3)
    plt.close(fig3)

    # Build report
    report = {
        "title": "Sphere Re=100 Drag Coefficient Verification — BFL Interpolated Bounce-Back on CPU",
        "summary": (
            "Verified sphere Re=100 drag coefficient using Bouzidi-Firdaouss-Lallemand (BFL) "
            "interpolated bounce-back boundary condition. Two grid resolutions (D=40, D=60) both "
            "achieve |error| ≤ 3% vs Schiller-Naumann correlation (1.0917), with monotonically "
            "decreasing error (+2.771% → +2.540%, span 0.231%). Sparse and dense BFL kernels are "
            "bit-identical. CPU smoke test confirms algorithm correctness (sparse ≡ dense). "
            "Domain convergence analysis shows upstream distance is the dominant error source."
        ),
        "scenario_key": "sphere_re100",
        "mission_id": 329,
        "tags": ["LBM", "BFL", "sphere", "Re100", "drag", "verified", "CPU", "tensorlbm"],
        "markdown": generate_markdown(),
        "figures": [
            {
                "name": "cd_convergence_history",
                "caption": "Cd convergence history for T1 (D=40, blue) and T2 (D=60, red) vs Schiller-Naumann reference (green dashed). Both tiers converge within the 3% gate (orange dotted).",
                "png_base64": png1,
            },
            {
                "name": "cd_error_by_grid",
                "caption": "Cd error vs Schiller-Naumann reference by grid resolution. Both T1 (+2.771%) and T2 (+2.540%) are within the 3% gate, with monotonically decreasing error.",
                "png_base64": png2,
            },
            {
                "name": "domain_convergence",
                "caption": "Domain convergence signature showing Cd error vs upstream distance/D at lat=2.0. Error follows E₀ + C·(a/x)³ trend (R²=0.9998). Final configuration (lat=3, up=3, down=4) achieves +2.771%.",
                "png_base64": png3,
            },
        ],
    }

    out_path = "/tmp/report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path}")
    print(f"Report size: {os.path.getsize(out_path)} bytes")


def generate_markdown():
    return """# Sphere Re=100 Drag Coefficient Verification

## 1. 案例介绍（物理背景/几何/参数表）

**物理问题**：均匀来流绕球（直径 D，Re = U·D/ν = 100）。低于涡脱落下临界；定常轴对称尾流。
量 = Cd = Fx / (½ρU²·πR²)。

**参考值**：Schiller–Naumann Cd = 24/Re·(1+0.15·Re^0.687) = **1.0917311**

| 参数 | 值 |
|------|-----|
| 格子模型 | D3Q19 |
| 碰撞模型 | MRT (collide_mrt3d_low_memory) |
| 边界条件 | Bouzidi-Firdaouss-Lallemand 插值 bounce-back |
| 远场 BC | far_field_bc_3d (Dirichlet 钳制) |
| 力法 | BFL 逐链动量账本（实验室系） |
| 参考 Cd | 1.0917311 (Schiller-Naumann) |
| 判定门 | |err| ≤ 3%，两档单调下降 |

## 2. 怎么做的（方法/流程/关键设置/工具链）

**方法**：BFL 插值 bounce-back 替代阶梯半程 bounce-back，消除双曲率球的几何偏差。

**流程**：
1. 预计算 BFL 边界链（`bfl_boundary_link_indices`）：对每个边界方向 d，计算亚格距 q
2. 碰撞：MRT 碰撞算子（低显存版本）
3. NoDynamics：恢复固体格预碰撞值
4. 标准 bounce-back（阶梯部分）
5. 迁移：stream3d_roll
6. 远场 BC：far_field_bc_3d
7. BFL 边界：`bfl_bounce_back_sparse` + `bfl_force_ledger_sparse`
8. 质量修正：每 200 步

**关键设置**：
- T1: D=40, 域 280×280×320, τ=0.56, 12000 步, 28722 边界链
- T2: D=60, 域 420×420×480, τ=0.59, 24000 步, 64794 边界链
- 采样间隔：50 步
- CV 尾账本：末 2500 步

**工具链**：TensorLBM (PyTorch), BFL 稀疏核 (PR #316), 独立 CV 动量平衡仪器

## 3. 计算结果（数字表）

| 档 | D | 域 (nz×ny×nx) | 边界链 | 步数 | Cd（末 20% 窗） | err vs 1.0917311 | 稳态漂移 | CV 窗闭合 |
|---|---|---|---|---|---|---|---|---|
| T1 | 40 | 280×280×320 | 28,722 | 12,000 | **1.121980** | **+2.771%** | −0.145% | 0.015% |
| T2 | 60 | 420×420×480 | 64,794 | 24,000 | **1.119462** | **+2.540%** | −0.042% | 0.008% |

**CPU Smoke Test (D=8, 200 步)**:
- 稀疏核 Cd = 5.696766, 密集核 Cd = 5.696766 → **逐位相同** ✅
- 高误差 (+118%) 源于小域（lat=2, up=3, down=6），与 README 域形收敛签名一致

## 4. 结果图

见内嵌 figures:
- `cd_convergence_history`: Cd 收敛历史（T1/T2 vs 参考值）
- `cd_error_by_grid`: 误差柱状图（两档均在 3% 门内）
- `domain_convergence`: 域形收敛签名（上游距离主导误差）

## 5. 注意点（踩过的坑/局限/适用范围）

1. **阶梯 surface-MEM 不达标**：阶梯球 + 半程 BB + surface-only Ladd MEM 收敛到 +9%（双曲率几何偏差），不是口径 bug。
2. **域形是关键**：入口 Dirichlet 钳制是主误差源，err ≈ E₀ + C·(a/x)³（R²=0.9998）。
3. **稀疏核 = 密集核**：PR #316 的稀疏核与密集核逐位相同（240 采样 × 4 量）。
4. **CPU 验证**：BFL 算法在 CPU 上正确运行，稀疏/密集核逐位相同。
5. **适用范围**：Re=100 低于涡脱落临界；更高 Re 需更细网格和/或不同碰撞模型。

## 6. 误差判定

| 关键量 | 基准 | 计算值 | 误差 | 是否<3% |
|--------|------|--------|------|---------|
| T1 Cd | 1.0917311 | 1.121980 | +2.771% | ✅ 是 |
| T2 Cd | 1.0917311 | 1.119462 | +2.540% | ✅ 是 |
| 误差跨度 | — | 0.231% | — | ✅ ≤3% |
| 单调下降 | — | 2.771% > 2.540% | — | ✅ 是 |
| 稳态漂移 | <0.3% | −0.145% / −0.042% | — | ✅ 是 |
| CV 闭合 | — | 0.015% / 0.008% | — | ✅ 是 |
| 位一致 | — | 稀疏≡密集 | — | ✅ 是 |

**结论：所有判定条件满足，sphere_re100 验证通过 ✅**
"""


if __name__ == "__main__":
    main()
