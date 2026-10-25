#!/usr/bin/env python3
"""
生成 rayleigh_benard 收敛曲线图

从 W12 结果生成 E(ny) 收敛曲线，展示非单调行为。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# 加载 W12 结果
results_file = Path("/nfs/wangxi/runs/bm_close/rayleigh_benard/verdict_w12.json")
with open(results_file) as f:
    verdict = json.load(f)

# 提取数据
arm_rows = verdict["arm_rows"]
E_ny = verdict["E_ny"]

# 按 ny 分组
nys = [64, 128, 256]
E_values = [E_ny[str(ny)] for ny in nys]

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：E(ny) 收敛曲线
ax1 = axes[0]
ax1.plot(nys, E_values, "o-", linewidth=2, markersize=10, color="#d62728", label="E(ny)")
ax1.axhline(y=3.0, color="gray", linestyle="--", linewidth=1, label="3% threshold")
ax1.set_xlabel("Grid resolution ny", fontsize=12)
ax1.set_ylabel("E(ny) [%]", fontsize=12)
ax1.set_title("Convergence error E(ny) vs grid resolution", fontsize=13, fontweight="bold")
ax1.set_xscale("log", base=2)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_ylim(0, max(E_values) * 1.2)

# 标注数值
for i, (ny, E) in enumerate(zip(nys, E_values)):
    ax1.annotate(
        f"{E:.2f}%",
        xy=(ny, E),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

# 右图：逐臂误差分布
ax2 = axes[1]
fracs = [0.85, 0.95, 1.05, 1.15, 1.3]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, ny in enumerate(nys):
    ny_data = [row for row in arm_rows if row["ny"] == ny]
    rel_errs = [abs(row["rel_pct"]) for row in ny_data]

    # 按 frac 排序
    sorted_pairs = sorted(zip([row["frac"] for row in ny_data], rel_errs))
    sorted_fracs = [p[0] for p in sorted_pairs]
    sorted_errs = [p[1] for p in sorted_pairs]

    ax2.plot(
        sorted_fracs,
        sorted_errs,
        "o-",
        linewidth=2,
        markersize=8,
        label=f"ny={ny}",
        color=colors[i],
    )

ax2.axhline(y=3.0, color="gray", linestyle="--", linewidth=1, label="3% threshold")
ax2.set_xlabel("Ra/Ra_c", fontsize=12)
ax2.set_ylabel("|Relative error| [%]", fontsize=12)
ax2.set_title("Per-arm error distribution", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 标注失败点
for row in arm_rows:
    if abs(row["rel_pct"]) > 3.0:
        ax2.annotate(
            f"ny={row['ny']}\nfrac={row['frac']}",
            xy=(row["frac"], abs(row["rel_pct"])),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=300, bbox_inches="tight")
print("✓ 生成 convergence_curve.png")

# 创建第二张图：格子算子 Ra_c 漂移
fig2, ax3 = plt.subplots(figsize=(8, 5))

# 加载格子算子预测数据
lattice_file = Path(
    "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref/lattice_predictions.json"
)
with open(lattice_file) as f:
    lattice = json.load(f)

# 提取 Ra_c 漂移
Ra_c_drifts = []
for level in lattice["levels"]:
    ny = level["ny"]
    drift = level["Ra_c_lattice_drift"] * 100  # 转为百分比
    Ra_c_drifts.append((ny, drift))

nys_drift = [x[0] for x in Ra_c_drifts]
drifts = [x[1] for x in Ra_c_drifts]

ax3.plot(nys_drift, drifts, "s-", linewidth=2, markersize=10, color="#9467bd")
ax3.set_xlabel("网格分辨率 ny", fontsize=12)
ax3.set_ylabel("Ra_c 漂移 [%]", fontsize=12)
ax3.set_title("格子算子 Ra_c 漂移 vs 网格分辨率", fontsize=13, fontweight="bold")
ax3.set_xscale("log", base=2)
ax3.grid(True, alpha=0.3)

# 标注数值
for ny, drift in zip(nys_drift, drifts):
    ax3.annotate(
        f"+{drift:.3f}%",
        xy=(ny, drift),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

plt.tight_layout()
plt.savefig("lattice_ra_c_drift.png", dpi=300, bbox_inches="tight")
print("✓ 生成 lattice_ra_c_drift.png")

print("\n图表生成完成")
