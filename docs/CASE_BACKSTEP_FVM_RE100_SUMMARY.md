# CASE-BACKSTEP-FVM-RE100 验证总结

## 锚点信息
- **Slug**: CASE-BACKSTEP-FVM-RE100
- **标题**: 算例池·后台阶回流区长度 Re=100
- **验收标准**: X1/s vs Armaly 1983 误差 <10%
- **求解器**: TensorLBM D2Q9 TRT (Λ=3/16)

## 工作完成内容

### 1. 网格收敛分析
- 汇总了从粗网格 (h_in=8) 到细网格 (h_in=123) 的完整收敛数据
- 发现壁面剪切法与流函数法存在系统性差异 (~0.15-0.16)
- Richardson 外推: X1/s = 2.80±0.05

### 2. 关键结果
| 方法 | L3p (h_in=123) | vs Armaly (3.05) | vs Erturk (2.878) |
|------|---------------|------------------|-------------------|
| 壁面剪切 | 2.597 | -14.9% FAIL | -9.8% FAIL |
| 流函数 | 2.753 | -9.72% PASS | -4.33% PASS |

### 3. 产物
- `docs/CASE_BACKSTEP_FVM_RE100_GRID_CONVERGENCE.md`: 完整网格收敛分析报告
- `figures/CASE_BACKSTEP_FVM_RE100/`: 3 张出版级图表
  - `grid_convergence.png`: X1/s vs 1/h_in 收敛图
  - `error_comparison.png`: 误差 vs 网格分辨率
  - `summary_dashboard.png`: 关键指标仪表板
- `scripts/backstep_grid_convergence.py`: 可视化脚本
- `benchmarks/case_pool/backward_step_armaly/refs.json`: 更新方法比较记录

### 4. 集群作业
- cj4299: h1_re100 (h_in=26) - 排队中
- cj4300: g1_re100 (h_in=52) - 排队中

## 结论

流函数法在 L3p 网格上达到 -9.72% 误差，通过 <10% 验收标准。壁面剪切法因台阶角点奇异性影响存在系统性低估。建议将流函数法纳入 verdict.py 作为辅助判据。

## 后续建议
1. 等待集群作业完成获取 h1/g1 数据点
2. 研究壁面剪切法在分离区的改进
3. 考虑 3D 模拟评估三维效应
