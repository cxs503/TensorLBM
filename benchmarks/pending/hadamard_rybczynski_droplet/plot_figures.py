#!/usr/bin/env python3
"""生成 hadamard_rybczynski_droplet 终裁可视化图表"""
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取模拟数据
with open('/nfs/wangxi/runs/bm_close/hadamard_rybczynski_droplet/terminal_buoyancy.json') as f:
    data = json.load(f)

# 提取时间序列
steps = [p['step'] for p in data['trace']]
u_y = [p['u_y_drop'] for p in data['trace']]
y_com = [p['y_com'] for p in data['trace']]

# 参考值
u_ref = data['U_t_ref']

# 图表1: 终端速度收敛曲线
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(steps, u_y, 'b-', linewidth=2, label='Simulation $U_y$')
ax1.axhline(y=u_ref, color='r', linestyle='--', linewidth=2, label=f'Reference $U_t$ = {u_ref:.3e}')
ax1.axhline(y=np.mean(u_y[-100:]), color='g', linestyle=':', linewidth=2, 
            label=f'Measured $U_{{terminal}}$ = {np.mean(u_y[-100:]):.3e}')
ax1.set_xlabel('Time step', fontsize=12)
ax1.set_ylabel('Droplet velocity $U_y$', fontsize=12)
ax1.set_title('Hadamard-Rybczynski Droplet: Terminal Velocity Convergence', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, max(steps)])
plt.tight_layout()
plt.savefig('figures/terminal_velocity_convergence.png', 
            dpi=150, bbox_inches='tight')
plt.close()

# 图表2: 液滴质心轨迹
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(steps, y_com, 'b-', linewidth=2, label='Droplet COM')
ax2.set_xlabel('Time step', fontsize=12)
ax2.set_ylabel('Center of mass $y_{com}$', fontsize=12)
ax2.set_title('Hadamard-Rybczynski Droplet: Vertical Trajectory', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, max(steps)])
plt.tight_layout()
plt.savefig('figures/droplet_trajectory.png', 
            dpi=150, bbox_inches='tight')
plt.close()

# 图表3: 误差分析柱状图
fig3, ax3 = plt.subplots(figsize=(8, 6))
categories = ['G1\nLibrary', 'G2\nObservable', 'G3\nConvergence', 'G4\nRef Locked']
# G1: pass (1), G2: fail (0), G3: insufficient (0.5), G4: pass (1)
values = [1, 0, 0.5, 1]
colors = ['green', 'red', 'orange', 'green']
bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Gate Status', fontsize=12)
ax3.set_title('Hadamard-Rybczynski: Four-Gate Assessment', fontsize=14)
ax3.set_ylim([0, 1.2])
ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Pass')
ax3.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Insufficient')
ax3.axhline(y=0.0, color='red', linestyle='-', linewidth=1, alpha=0.5, label='Fail')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/gate_assessment.png', 
            dpi=150, bbox_inches='tight')
plt.close()

print("✓ 生成 3 张图表:")
print("  - terminal_velocity_convergence.png")
print("  - droplet_trajectory.png")
print("  - gate_assessment.png")
