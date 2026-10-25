#!/usr/bin/env python3
"""
DIT Turbulence 收敛性诊断图
- 图1: 谱斜率误差随网格分辨率变化
- 图2: 衰减指数随网格分辨率变化
- 图3: 能谱对比（N=128 vs N=256 vs -5/3 参考）
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 读取 result.json
result_file = Path("result.json")
with open(result_file) as f:
    data = json.load(f)

verdict = data["verdict"]
grids = verdict["grids"]
slope_errs = [verdict["slope_err_pct_per_grid"][str(n)] for n in grids]
n_decays = [verdict["n_decay_per_grid"][str(n)] for n in grids]

# 创建图形
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1: 谱斜率误差
ax1 = axes[0]
ax1.plot(grids, slope_errs, "o-", linewidth=2, markersize=8, label="slope error")
ax1.axhline(y=3.0, color="r", linestyle="--", label="3% threshold")
ax1.axhline(y=-3.0, color="r", linestyle="--")
ax1.set_xlabel("Grid resolution N", fontsize=12)
ax1.set_ylabel("Slope error vs -5/3 (%)", fontsize=12)
ax1.set_title("(a) Spectral slope error", fontsize=13)
ax1.set_xscale("log", base=2)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(grids)
ax1.set_xticklabels([str(n) for n in grids])

# 图2: 衰减指数
ax2 = axes[1]
ax2.plot(grids, n_decays, "s-", linewidth=2, markersize=8, color="green", label="decay exponent n")
ax2.axhline(y=1.2, color="r", linestyle="--", label="physical band [1.2, 1.5]")
ax2.axhline(y=1.5, color="r", linestyle="--")
ax2.axhline(y=1.35, color="gray", linestyle=":", alpha=0.5, label="target 1.35")
ax2.set_xlabel("Grid resolution N", fontsize=12)
ax2.set_ylabel("Decay exponent n", fontsize=12)
ax2.set_title("(b) Decay exponent", fontsize=13)
ax2.set_xscale("log", base=2)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(grids)
ax2.set_xticklabels([str(n) for n in grids])

# 图3: 能谱对比（从 data_N*.npz 读取）
ax3 = axes[2]
colors = ["blue", "orange"]
for i, n in enumerate(grids):
    data_file = Path(f"data_N{n}.npz")
    if data_file.exists():
        npz = np.load(data_file)
        times = npz["times"]
        energies = npz["energies"]
        spectra = npz["spectra"]

        # 选择中间时间快照
        mid_idx = len(times) // 2
        k = np.arange(spectra.shape[1])
        E_k = spectra[mid_idx]

        # 过滤掉 k=0
        mask = k > 0
        ax3.loglog(k[mask], E_k[mask], "-", linewidth=2, label=f"N={n}", color=colors[i])

# 添加 -5/3 参考线
k_ref = np.array([10, 40])
E_ref = 1e-2 * (k_ref / k_ref[0]) ** (-5 / 3)
ax3.loglog(k_ref, E_ref, "k--", linewidth=1.5, label="-5/3 reference")

ax3.set_xlabel("Wavenumber k", fontsize=12)
ax3.set_ylabel("E(k)", fontsize=12)
ax3.set_title("(c) Energy spectrum (t=t_mid)", fontsize=13)
ax3.grid(True, alpha=0.3, which="both")
ax3.legend()

plt.tight_layout()
plt.savefig("figures/dit_convergence_diagnosis.png", dpi=150, bbox_inches="tight")
print("✓ 生成 figures/dit_convergence_diagnosis.png")

# 创建 figures.json
figures_manifest = {
    "figures": [
        {
            "file": "figures/dit_convergence_diagnosis.png",
            "kind": "curve",
            "caption": "DIT 收敛性诊断：(a) 谱斜率误差随网格变化，(b) 衰减指数随网格变化，(c) 能谱对比",
            "source": "result.json + data_N128.npz + data_N256.npz",
        }
    ],
    "field_waiver": "DIT 为统计均匀湍流，无空间场量（仅能谱 E(k) 与时间序列 E(t)），豁免场图",
}

with open("figures.json", "w") as f:
    json.dump(figures_manifest, f, indent=2)
print("✓ 生成 figures.json")

print("\n图表说明：")
print("- 图(a): N=128 误差 +1.69% (达标)，N=256 误差 +17.05% (超差)")
print("- 图(b): N=128 n=1.449 (在 [1.2,1.5] 带内)，N=256 n=1.693 (超带)")
print("- 图(c): 能谱对比显示 N=256 在惯性区明显偏平（欠耗散）")
