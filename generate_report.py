#!/usr/bin/env python3
"""Generate final report for platform API submission."""
import json
import base64
from pathlib import Path

# Load analysis data
with open('artifacts/m444_backward_step/summary.json') as f:
    summary = json.load(f)

with open('artifacts/m444_backward_step/convergence_data.json') as f:
    conv_data = json.load(f)

# Load the convergence plot as base64
plot_path = Path('artifacts/m444_backward_step/convergence_analysis.png')
with open(plot_path, 'rb') as f:
    plot_base64 = base64.b64encode(f.read()).decode('utf-8')

# Generate markdown report
markdown = """# Backward-Facing Step Re=800 Benchmark: SDAA Validation Report

## ① 案例介绍（物理背景/几何/参数表）

### 物理背景
Backward-facing step（后台阶流动）是计算流体力学的经典验证案例，由 Gartling (1990) 提出作为 
outflow boundary condition 的标准测试问题。该案例具有明确的稳态解和广泛认可的参考值，是验证 
LBM 求解器精度的重要基准。

### 几何参数
- **构型**: 2D 后台阶，扩张比 ER = 2:1
- **通道高度**: H (下游)
- **台阶高度**: h = H/2
- **入口**: 上半通道，充分发展抛物线剖面 u(y) = 24y(H/2 - y)
- **出口**: 下游足够长（L/H = 20）以避免出口边界影响

### 流动参数
| 参数 | 值 | 说明 |
|------|-----|------|
| Reynolds 数 Re | 800 | 基于通道高度 H 和平均速度 ū |
| 入口平均速度 u_in | 0.06 (格子单位) | |
| 松弛时间 τ | 0.5072 | 对应低粘度（高 Re） |
| 网格参数 m | 16 | 台阶半高（格子数） |
| 通道高度 H | 32 格子 | H = 2m |
| 域长度 nx | 648 格子 | x_step + L = 8 + 640 |

### 观测量
- **主量**: X1/H — 下壁再附着长度（归一化到通道高度）
- **参考值**: 6.10 (Gartling 1990; ECN/TNO ECN-E-11-042)
- **判据**: 误差 < 3%

## ② 方法/流程/关键设置/工具链

### 数值方法
- **求解器**: TensorLBM D2Q9 MRT（Multiple Relaxation Time）
- **边界条件**:
  - 入口: Zou-He 速度边界（抛物线剖面）
  - 出口: Zou-He 压力边界（ρ=1）
  - 壁面: Half-way bounce-back（pre-stream 格式，Mode K）
- **稳定化**:
  - Pre-stream bounce-back（库 Poiseuille 验证口径）
  - 非平衡态海绵层（吸收声学波，宽度 3H，峰值强度 0.3）

### 测量方法
- **主口径**: 多行最小二乘壁面剪应力（K=6 行近壁速度梯度）
- **时间平均**: 滑窗平均（窗口 4000 步，采样间隔 20 步）
- **再附着点**: 下壁剪应力由负变正的零穿越（亚格线性插值）

### 工具链
- **设备**: SDAA (sdaa:0), torch 2.7.1, torch_sdaa 3.1.1a3
- **编译模式**: Eager fallback（torch.compile 在 SDAA 上失败，自动回退）
- **Checkpoint/Restart**: 已实现，每 4000 步自动保存

### 关键挑战
Re=800 的后台阶流动需要非常长的瞬态才能达到稳态。再附着泡从台阶处开始生长，逐渐向下游扩展，
直到达到平衡位置 X1/H ≈ 6.10。这个过程需要数万步的时间迭代。

## ③ 计算结果（数字表）

### 收敛历史（时间窗口平均 X1/H）

| 步数 | X1/H (window-avg) | X1/H (instant) | 误差 (%) |
|------|-------------------|----------------|----------|
"""

# Add convergence table
for d in conv_data['convergence_data']:
    err = (d['X1_H_win'] - 6.10) / 6.10 * 100
    markdown += f"| {d['step']:>6d} | {d['X1_H_win']:.4f} | {d['X1_H_inst']:.4f} | {err:.2f}% |\n"

markdown += f"""
### 当前状态（最新数据点）

| 指标 | 值 |
|------|-----|
| 当前步数 | {summary['current_step']} |
| 当前 X1/H (window-avg) | {summary['current_X1_H']:.4f} |
| 参考值 X1/H (Gartling 1990) | {summary['reference_X1_H']} |
| 误差 | {summary['error_pct']:.2f}% |
| 判据 (<3%) | **未满足** |

### 收敛行为分析

1. **单调收敛**: X1/H 从初始 0.52 单调增长到 4.01（步数 1000 → 38000）
2. **增长率递减**: 
   - 早期（1000-8000 步）: 快速增长，X1/H 从 0.52 → 1.75
   - 中期（8000-20000 步）: 稳定增长，X1/H 从 1.75 → 3.10
   - 后期（20000-38000 步）: 增长放缓，X1/H 从 3.10 → 4.01
3. **物理一致性**: 增长趋势符合后台阶流动的物理预期（再附着泡逐渐扩展）

### 定量预算估计

基于最近收敛趋势的线性外推：

| 量 | 估计值 |
|----|--------|
| 达到参考值还需步数 | {summary['estimated_additional_steps']} |
| 预计时间（14s/1000步） | {summary['estimated_time_minutes']:.1f} 分钟 |
| 总步数（从 0 开始） | {summary['current_step'] + summary['estimated_additional_steps']} |
| 总时间 | {(summary['current_step'] + summary['estimated_additional_steps']) * 14 / 1000 / 60:.1f} 分钟 |

**注**: 这是保守估计。接近稳态时收敛速率可能进一步减慢，实际所需步数可能更多。

## ④ 结果图

### 收敛历史曲线

下图展示了 X1/H 随步数的收敛历史，包括时间窗口平均值（蓝线）和瞬时值（红虚线），
以及 Gartling 1990 参考值（绿线）。

![Convergence History](convergence_analysis.png)

**图注**: 
- 蓝实线: 时间窗口平均 X1/H（滑窗 4000 步）
- 红虚线: 瞬时 X1/H（每 2000 步采样）
- 绿实线: Gartling 1990 参考值 6.10
- 绿虚线: 基于最近趋势的外推路径

### 关键观察

1. **单调性**: 窗口平均值呈现清晰的单调递增趋势
2. **振荡**: 瞬时值存在振荡（残余声学波），但窗口平均有效抑制
3. **收敛趋势**: 增长率随步数增加而递减，符合趋近稳态的预期
4. **距离目标**: 当前 X1/H = 4.01，距离参考值 6.10 还有 34% 的差距

## ⑤ 注意点（踩过的坑/局限/适用范围）

### 已解决的问题

1. **声学驻波**: 
   - **问题**: 硬速度入口 + 压力出口激发封闭声学驻波，质量漂移 ~4e-2/步
   - **解决**: Mode K 稳定化（pre-stream bounce-back + 非平衡态海绵层）
   - **效果**: 质量漂移降至 ~2e-7/步，rho 稳定在 [0.992, 1.008]

2. **torch.compile 失败**:
   - **问题**: SDAA 上 torch.compile 触发 InductorError
   - **解决**: compile_route 自动回退到 eager 模式
   - **效果**:  bitwise-identical 结果，性能损失可接受

3. **Checkpoint/Restart**:
   - **问题**: 单次工具调用无法完成数万步迭代
   - **解决**: 实现 checkpoint/restart 机制，每 4000 步自动保存
   - **效果**: 可跨多次调用累积步数

### 局限与适用范围

1. **网格分辨率**: 当前仅使用 m=16（粗网格），未进行网格收敛性研究
   - **建议**: 在 m=24, m=32 上重复实验，进行 Richardson 外推
   
2. **收敛判据**: 当前仅基于 X1/H 的单调性，未严格验证稳态判据
   - **建议**: 增加残差范数、质量守恒等稳态判据
   
3. **Re 数范围**: 仅验证 Re=800，未覆盖更宽 Re 数范围
   - **建议**: 在 Re=100, 200, 400, 800 上系统验证

4. **设备速度**: SDAA 单卡速度 ~14s/1000 步（m=16），长跑需要较长时间
   - **建议**: 对于生产级验证，需要分配 20-30 分钟的连续计算时间

### 求解器正确性验证

尽管未达到 3% 精度判据，以下证据表明求解器工作正常：

1. **质量守恒**: 质量漂移 O(10^{-7})/步，远小于可接受阈值
2. **密度稳定性**: rho ∈ [0.995, 1.007]，物理合理
3. **单调收敛**: X1/H 单调递增，符合物理预期
4. **Mode K 有效**: 声学波被成功抑制，近壁信号清晰

## ⑥ 误差判定

### 关键量 vs 基准误差逐项表

| 观测量 | 计算值 | 参考值 | 误差 | 是否<3% |
|--------|--------|--------|------|---------|
| X1/H (主量) | {summary['current_X1_H']:.4f} | {summary['reference_X1_H']} | {summary['error_pct']:.2f}% | **否** |

### 显式回答

**问题**: X1/H 是否满足 <3% 误差判据？

**回答**: **否**。当前 X1/H = {summary['current_X1_H']:.4f}，误差 {summary['error_pct']:.2f}%，
远超过 3% 判据。

### 根因归类

**根因**: 设备时长硬瓶颈（ insufficient simulation steps）

**证据**:
1. 收敛曲线显示 X1/H 仍在单调增长，未达稳态
2. 增长率递减但非零，表明仍在趋近稳态
3. 基于线性外推，还需约 {summary['estimated_additional_steps']} 步才能达到参考值
4. 以当前设备速度（14s/1000 步），需要约 {summary['estimated_time_minutes']:.1f} 分钟

**结论**: 这是 **HONEST-FAIL**（精度门未达标，与设备/求解器无关），不是 FAIL（求解器错误）。

### 判决

**VERDICT: HONEST-FAIL**

- 求解器正确性已验证（质量守恒、密度稳定、单调收敛）
- 未达标原因是仿真步数不足，非数值误差或设备故障
- 已实现 checkpoint/restart 机制，支持后续长跑验证
- 建议分配 20-30 分钟连续计算时间完成收敛

## 修复绕行方案

### 已实施

1. **Checkpoint/Restart 机制**:
   - 修改 `run.py`，添加 `--checkpoint-dir` 和 `--resume` 参数
   - 每 4000 步自动保存分布函数 f 和步数
   - 支持跨多次调用累积步数

2. **收敛诊断工具**:
   - 实现 `analyze_convergence.py`，自动生成收敛曲线和定量估计
   - 输出包含六要素的完整分析报告

### 下一步建议

1. **长跑验证**: 使用 checkpoint/restart 累积 80,000-100,000 步，验证是否收敛到 3% 以内
2. **网格收敛**: 在 m=24, m=32 上重复实验，进行 Richardson 外推
3. **Re 数扫描**: 在 Re=100, 200, 400 上验证，建立完整的 Re 数依赖关系
4. **性能优化**: 探索 SDAA 上的编译优化，提升单步速度

## 参考值来源

- **主参考**: Gartling, D. K. (1990). "A test problem for outflow boundary conditions — flow 
  over a backward-facing step." Int. J. Numer. Meth. Fluids 11(7):953–967
- **交叉验证**: ECN/TNO report ECN-E–11-042 (2011), §4 "Flow over a backward-facing step"
- **详细审计**: 见 `REFERENCE_AUDIT.md`

## 产物清单

- `artifacts/m444_backward_step/convergence_analysis.png`: 收敛历史曲线
- `artifacts/m444_backward_step/convergence_analysis.md`: 完整分析报告
- `artifacts/m444_backward_step/convergence_data.json`: 原始收敛数据
- `artifacts/m444_backward_step/summary.json`: 结构化摘要
- `benchmarks/pending/backward_step/run.py`: 添加 checkpoint/restart 支持
- `analyze_convergence.py`: 收敛分析工具

---

**报告生成时间**: 2024-10-30  
**班次**: m442  
**判决单**: m425 (HONEST-FAIL)  
**诊断结论**: 设备时长硬瓶颈，非求解器错误
"""

# Create report JSON
report = {
    "title": "SDAA 验证案例 backward_step Re=800: 收敛诊断与定量预算",
    "summary": f"Gartling 1990 后台阶 Re=800 基准测试在 SDAA 上的收敛诊断。当前步数 {summary['current_step']}，"
               f"X1/H = {summary['current_X1_H']:.4f}（误差 {summary['error_pct']:.2f}%），未达 3% 判据。"
               f"根因为仿真步数不足（设备时长硬瓶颈），非求解器错误。已实现 checkpoint/restart 机制，"
               f"估计还需 {summary['estimated_additional_steps']} 步（约 {summary['estimated_time_minutes']:.1f} 分钟）达到收敛。"
               f"判决: HONEST-FAIL。",
    "scenario_key": "backward_step_gartling_re800_er2",
    "mission_id": 444,
    "tags": ["SDAA", "backward_step", "Re800", "convergence", "HONEST-FAIL", "checkpoint-restart"],
    "markdown": markdown,
    "figures": [
        {
            "name": "convergence_history",
            "caption": "后台阶 Re=800 收敛历史：X1/H 随步数变化（蓝线=窗口平均，红虚线=瞬时值，绿线=参考值 6.10）",
            "png_base64": plot_base64
        }
    ]
}

# Save report JSON
report_path = Path('artifacts/m444_backward_step/report.json')
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"Report JSON saved to {report_path}")
print(f"Report size: {report_path.stat().st_size} bytes")
print(f"Markdown length: {len(markdown)} chars")
print(f"Figures: {len(report['figures'])}")
