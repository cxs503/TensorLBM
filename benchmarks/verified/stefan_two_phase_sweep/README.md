# B-verify: 二相 Stefan 问题（过热液相）—— 焓法相变 LBM 的 Stefan 数扫描

**状态：✅ 已验证（真实模拟，无人工修正，全部数字程序化生成 + sha256 溯源）**

节点：P272W2-N05。完整论文级报告见
[`docs/P272W2_N05_STEFAN_TWO_PHASE_SWEEP.md`](../../../docs/P272W2_N05_STEFAN_TWO_PHASE_SWEEP.md)。

本案例与 P155W1-N01（一相/等温远场 Stefan 验证面）是**不同问题类**：这里液相初始过热
（`T_i > T_m`），界面系数 λ 由**两个 Stefan 数的耦合超越方程**确定，潜热必须同时经固相与液
相导出；一相 Neumann 平面只是 `Ste_l = 0` 的退化极限（本案例把它当作回归检查，而不是主要靶子）。

## 验证链

| 判据 | 检验量 | 实测 | 门限 |
|---|---|---|---|
| C1 | 界面位置 `s(t)` vs 二相相似解 `2λ√(αt)`（3 个 Ste 工况，所有采样时刻最差） | 1.203% | < 5% |
| C2 | 温度剖面 `T(x,t)` vs 相似解（全域 L∞，所有采样时刻最差） | 2.734% | < 3% |
| C3 | 能量守恒：壁面输入 vs 显热 + 潜热（同一控制体 + 独立壁面通量算子） | 0.351% | < 2% |
| C4 | 液相分数 `0 ≤ f_l ≤ 1`（整段运行逐节点） | [0, 1] | 硬边界 |
| C5 | 网格收敛（17/33/66 前沿格）界面观测阶 | 1.40 / 1.37 | ≥ 1 |
| C1b | 单相极限 `Ste_l = 0` 回归 Neumann（同一求解器） | 1.433% | < 5% |
| — | 参考解自证：λ 单相极限 / 能量恒等式 / 界面平衡 / 相似解 ODE 残差 | 7.8e-16 / 6.1e-16 / 2.6e-18 / 1.3e-06 | 机器零 |
| C6 | 零手抄 JSON + 每数组 sha256 | `results.json` + `.sha256` | 存在 |

## 复现

```bash
PYTHONPATH=src python benchmarks/verified/stefan_two_phase_sweep/run.py          # ~95 s
PYTHONPATH=src python benchmarks/verified/stefan_two_phase_sweep/render_report.py
PYTHONPATH=src python -m pytest tests/test_stefan_two_phase_sweep.py
```

产物：`artifacts/stefan_two_phase_sweep/{results.json, results.json.sha256, lineage.json, figures/}`。