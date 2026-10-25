# SIR 线交接笔记（m292 班次更新 2026-10-25）

## 当前状态

- m292 (SIR-1) 已交付：commit dde74d09
- 判决：CONDITIONAL-PASS（基于 tau_axis_20261015 已有证据，均匀网格验证未完成）
- 治愈旋钮已确认：全局 τ 是唯一治愈旋钮（D_QW 符号翻转，span_ratio=6.87）
- 生产点建议：τ ∈ [1.5, 4.0]（需验证 Re 耦合）

## 在飞事项

### G1 流体载荷精度封口（SIR-1 延续）

**已完成**：
- tau_axis_20261015 实验（AMR 生产系统）：τ 阶梯 {0.5,1,1.5,4} 验证 D_QW 从 -3.20e-3 拉过零到 +4.66e-4
- 终裁 #278：全局 τ 是唯一治愈旋钮，局部手术空间三代穷尽

**未完成（交后续班次）**：
1. 均匀网格 τ 阶梯验证：在 suboff_sail_resistance 案例上系统扫 τ
   - 驱动脚本已写：/nfs/wangxi/runs/tau_ladder_uniform_20261025/run_tau_ladder.py
   - 坑：驱动脚本使用 sys.executable（系统 Python）而非 venv Python
   - 修复：将 sys.executable 改为 /nfs/wangxi/venvs/tensor_eco/bin/python
   - τ 阶梯设计：{0.51, 0.55, 0.60, 0.70, 1.00}
   - run.py 已修改支持 SUBOFF_VISCOSITY_OVERRIDE 环境变量覆盖粘度
2. 跟踪 FA 锚族：冲量、E0、艏艉偶极、ct/e0、D_QW 自估
3. 验证 Re 耦合：τ 改变会改变有效 Re，需分析 τ-Re 耦合效应

## 坑与关键路径

- m292 实验失败根因：驱动脚本用 sys.executable（/usr/bin/python3）而非 venv
- 修复：替换为 /nfs/wangxi/venvs/tensor_eco/bin/python
- 判决报告：m292_delivery_report.md
- 判据工件：m292_validation_report.json
- 实验暂存根：/nfs/wangxi/runs/tau_ladder_uniform_20261025/
- 已有证据：/nfs/wangxi/runs/tau_axis_20261015/
- 终裁链：/nfs/wangxi/runs/suboff_sail_20261008/

## 身份与纪律

- 常驻记忆：suboff-icebreak-full
- 纪律：prereg 先冻结后跑、零手抄、honest FAIL 不假绿、GPU5 唯一
- 红线：不 git push；跨跑逐位门禁用

## 下一步

1. 修复驱动脚本（venv 路径）
2. 重跑均匀网格 τ 阶梯
3. 产出完整判决（含 FA 锚族跟踪）
4. 更新交接笔记
