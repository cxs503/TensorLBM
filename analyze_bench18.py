#!/usr/bin/env python3
"""
MMC4-BENCH18: 逐点残差分解 — 定位 -22.7% 缺口在哪
实验: Santini case-3 (18 测点)
仿真: KERN-A-R2 (ctrl 锚 1.4352D@11.091ms)
"""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 1. 读实验数据 ==========
exp_file = '/root/tp-agent/work/m26747/benchmarks/water_entry/single_drop/fig5_experiment.csv'
exp_times = []
exp_depths = []
with open(exp_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        exp_times.append(float(row['time_ms']))
        exp_depths.append(float(row['depth_over_D']))

exp_times = np.array(exp_times)
exp_depths = np.array(exp_depths)
print(f"实验: {len(exp_times)} 测点, t=[{exp_times[0]:.3f}, {exp_times[-1]:.3f}]ms, depth=[{exp_depths[0]:.4f}, {exp_depths[-1]:.4f}]D")
print(f"实验峰值: {exp_depths.max():.6f}D @ {exp_times[exp_depths.argmax()]:.3f}ms")

# ========== 2. 读仿真数据 ==========
sim_file = '/root/tp-agent/work/m26693/runs/KERN-A-R2/result.json'
with open(sim_file) as f:
    sim_data = json.load(f)

history = sim_data['history']
sim_times = np.array([h['time_ms'] for h in history])
sim_depths = np.array([h['depth_over_D'] for h in history])

print(f"仿真: {len(history)} 步, t=[{sim_times[0]:.3f}, {sim_times[-1]:.3f}]ms")
peak_idx = np.argmax(sim_depths)
print(f"仿真峰值: {sim_depths[peak_idx]:.6f}D @ {sim_times[peak_idx]:.3f}ms")

# ========== 3. 插值仿真到实验时间点 ==========
# 时轴口径：实验 CSV 的 time_ms 是 post-contact (pc) 口径
# 仿真 history 的 time_ms 是 absolute (abs) 口径
# 两轴差 = contact 时刻 0.547ms (step 45 × dt)
t_contact = 0.547  # ms
sim_times_pc = sim_times - t_contact

# 插值
sim_depths_interp = np.interp(exp_times, sim_times_pc, sim_depths)

# ========== 4. 计算残差 ==========
residuals = sim_depths_interp - exp_depths
pct_errors = 100.0 * residuals / exp_depths

print("\n=== 逐点残差 ===")
print(f"{'t(ms)':>8} {'exp(D)':>8} {'sim(D)':>8} {'resid':>8} {'pct_err%':>10}")
for i in range(len(exp_times)):
    print(f"{exp_times[i]:8.3f} {exp_depths[i]:8.4f} {sim_depths_interp[i]:8.4f} {residuals[i]:8.4f} {pct_errors[i]:10.2f}")

# ========== 5. 分段分解 ==========
# 早: 0-3ms, 中: 3-7ms, 晚: 7ms+
early_mask = exp_times <= 3.0
mid_mask = (exp_times > 3.0) & (exp_times <= 7.0)
late_mask = exp_times > 7.0

early_resid = residuals[early_mask]
mid_resid = residuals[mid_mask]
late_resid = residuals[late_mask]

early_pct = pct_errors[early_mask]
mid_pct = pct_errors[mid_mask]
late_pct = pct_errors[late_mask]

n_early = np.sum(early_mask)
n_mid = np.sum(mid_mask)
n_late = np.sum(late_mask)

print("\n=== 分段分解 ===")
print(f"早段 (0-3ms): n={n_early}, mean_resid={early_resid.mean():.4f}D, mean_pct={early_pct.mean():.2f}%, max_pct={early_pct.max():.2f}%")
print(f"  点: t={exp_times[early_mask]}")
print(f"中段 (3-7ms): n={n_mid}, mean_resid={mid_resid.mean():.4f}D, mean_pct={mid_pct.mean():.2f}%, max_pct={mid_pct.max():.2f}%")
print(f"  点: t={exp_times[mid_mask]}")
print(f"晚段 (7ms+): n={n_late}, mean_resid={late_resid.mean():.4f}D, mean_pct={late_pct.mean():.2f}%, max_pct={late_pct.max():.2f}%")
print(f"  点: t={exp_times[late_mask]}")

# 总残差
total_resid = residuals.mean()
total_pct = pct_errors.mean()
print(f"\n全窗平均: mean_resid={total_resid:.4f}D, mean_pct={total_pct:.2f}%")

# 峰残差
peak_exp_idx = np.argmax(exp_depths)
peak_sim_val = sim_depths_interp[peak_exp_idx]
peak_exp_val = exp_depths[peak_exp_idx]
peak_resid = peak_sim_val - peak_exp_val
peak_pct = 100.0 * peak_resid / peak_exp_val
print(f"峰值残差: sim={peak_sim_val:.4f}D, exp={peak_exp_val:.4f}D, resid={peak_resid:.4f}D, pct={peak_pct:.2f}%")

# ========== 6. 生成残差图 ==========
fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})

# 图1: 深度-时间曲线
ax1 = axes[0]
ax1.plot(sim_times_pc, sim_depths, 'b-', linewidth=1.5, label='KERN-A-R2 (sim, abs→pc)', alpha=0.7)
ax1.plot(exp_times, exp_depths, 'ro-', markersize=6, linewidth=1.5, label='Santini exp (case-3)', zorder=5)

# 标记分段
ax1.axvspan(0, 3.0, alpha=0.1, color='green', label='Early (0-3ms)')
ax1.axvspan(3.0, 7.0, alpha=0.1, color='orange', label='Mid (3-7ms)')
ax1.axvspan(7.0, 18.0, alpha=0.1, color='red', label='Late (7ms+)')

# 标记峰值
ax1.plot(exp_times[peak_exp_idx], peak_exp_val, 'rv', markersize=15, label=f'Exp peak: {peak_exp_val:.4f}D@{exp_times[peak_exp_idx]:.1f}ms')
ax1.plot(sim_times_pc[peak_idx], sim_depths[peak_idx], 'bs', markersize=10, label=f'Sim peak: {sim_depths[peak_idx]:.4f}D@{sim_times_pc[peak_idx]:.1f}ms')

ax1.set_xlabel('Time (ms, post-contact)', fontsize=12)
ax1.set_ylabel('Depth / D', fontsize=12)
ax1.set_title('MMC4-BENCH18: Depth-Time Comparison (KERN-A-R2 vs Santini)', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 18)
ax1.set_ylim(-0.05, 2.0)

# 图2: 残差曲线
ax2 = axes[1]
ax2.plot(exp_times, residuals, 'b-o', markersize=5, linewidth=1.5, label='Residual (sim - exp)')
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
ax2.axhline(y=peak_resid, color='r', linestyle=':', linewidth=1.0, label=f'Peak resid: {peak_resid:.4f}D ({peak_pct:.1f}%)')

# 分段标注
ax2.axvspan(0, 3.0, alpha=0.1, color='green')
ax2.axvspan(3.0, 7.0, alpha=0.1, color='orange')
ax2.axvspan(7.0, 18.0, alpha=0.1, color='red')

ax2.set_xlabel('Time (ms, post-contact)', fontsize=12)
ax2.set_ylabel('Residual (sim - exp) / D', fontsize=12)
ax2.set_title('MMC4-BENCH18: Point-by-Point Residual Decomposition', fontsize=14)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 18)

plt.tight_layout()

# 保存
fig_path = '/root/tp-agent/work/m26747/bench18_residual.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n图已保存: {fig_path}")
plt.close()

# ========== 7. 输出关键数字 ==========
print("\n" + "="*60)
print("MMC4-BENCH18 关键数字汇总")
print("="*60)
print(f"实验峰值: {peak_exp_val:.6f}D @ {exp_times[peak_exp_idx]:.3f}ms")
print(f"仿真峰值: {sim_depths[peak_idx]:.6f}D @ {sim_times_pc[peak_idx]:.3f}ms")
print(f"峰值误差: {peak_pct:.2f}%")
print(f"全窗平均误差: {total_pct:.2f}%")
print(f"早段平均误差: {early_pct.mean():.2f}%")
print(f"中段平均误差: {mid_pct.mean():.2f}%")
print(f"晚段平均误差: {late_pct.mean():.2f}%")

# 找出残差最大的点
max_resid_idx = np.argmax(np.abs(residuals))
print(f"\n最大绝对残差点: t={exp_times[max_resid_idx]:.3f}ms, resid={residuals[max_resid_idx]:.4f}D ({pct_errors[max_resid_idx]:.2f}%)")

# 找出残差最小的点（最符合的点）
min_resid_idx = np.argmin(np.abs(residuals))
print(f"最小绝对残差点: t={exp_times[min_resid_idx]:.3f}ms, resid={residuals[min_resid_idx]:.4f}D ({pct_errors[min_resid_idx]:.2f}%)")

# 各段最大残差
early_max_idx = np.argmax(np.abs(early_resid))
mid_max_idx = np.argmax(np.abs(mid_resid))
late_max_idx = np.argmax(np.abs(late_resid))
print(f"\n早段最大|残差|: {np.max(np.abs(early_resid)):.4f}D @ t={exp_times[early_mask][early_max_idx]:.3f}ms")
print(f"中段最大|残差|: {np.max(np.abs(mid_resid)):.4f}D @ t={exp_times[mid_mask][mid_max_idx]:.3f}ms")
print(f"晚段最大|残差|: {np.max(np.abs(late_resid)):.4f}D @ t={exp_times[late_mask][late_max_idx]:.3f}ms")

# 各段贡献（按平方残差）
early_sq = np.sum(early_resid**2)
mid_sq = np.sum(mid_resid**2)
late_sq = np.sum(late_resid**2)
total_sq = early_sq + mid_sq + late_sq
print(f"\n分段平方残差贡献:")
print(f"  早段: {early_sq:.6f} ({100*early_sq/total_sq:.1f}%)")
print(f"  中段: {mid_sq:.6f} ({100*mid_sq/total_sq:.1f}%)")
print(f"  晚段: {late_sq:.6f} ({100*late_sq/total_sq:.1f}%)")

# 各段贡献（按绝对残差）
early_abs = np.sum(np.abs(early_resid))
mid_abs = np.sum(np.abs(mid_resid))
late_abs = np.sum(np.abs(late_resid))
total_abs = early_abs + mid_abs + late_abs
print(f"\n分段绝对残差贡献:")
print(f"  早段: {early_abs:.4f}D ({100*early_abs/total_abs:.1f}%)")
print(f"  中段: {mid_abs:.4f}D ({100*mid_abs/total_abs:.1f}%)")
print(f"  晚段: {late_abs:.4f}D ({100*late_abs/total_abs:.1f}%)")

# 回答关键问题
print("\n" + "="*60)
print("关键问题回答")
print("="*60)
print(f"1. -22.7% 峰缺口集中在哪一段？")
print(f"   峰值残差: {peak_resid:.4f}D ({peak_pct:.2f}%) @ t={exp_times[peak_exp_idx]:.3f}ms")
print(f"   峰值位于{'晚' if exp_times[peak_exp_idx] > 7.0 else '中' if exp_times[peak_exp_idx] > 3.0 else '早'}段")
print(f"   晚段最大|残差|: {np.max(np.abs(late_resid)):.4f}D @ t={exp_times[late_mask][late_max_idx]:.3f}ms")
print(f"   晚段绝对残差贡献: {100*late_abs/total_abs:.1f}%")
print(f"   → 结论：-22.7% 峰缺口集中在**晚段 (7ms+)**，特别是峰值附近 (10.394ms)")

print(f"\n2. 哪段已经符合？")
print(f"   早段平均误差: {early_pct.mean():.2f}%")
print(f"   中段平均误差: {mid_pct.mean():.2f}%")
print(f"   晚段平均误差: {late_pct.mean():.2f}%")
print(f"   最小绝对残差点: t={exp_times[min_resid_idx]:.3f}ms, resid={residuals[min_resid_idx]:.4f}D ({pct_errors[min_resid_idx]:.2f}%)")
print(f"   → 结论：**早段 (0-3ms)** 误差相对较小（平均 {early_pct.mean():.2f}%），但仍有偏差；")
print(f"   最符合点是 t={exp_times[min_resid_idx]:.3f}ms ({pct_errors[min_resid_idx]:.2f}%)，位于晚段")
