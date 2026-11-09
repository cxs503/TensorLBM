# B-verify: Stefan 相变传热（凝固/熔化移动边界）—— TensorLBM

**状态：✅ 已验证（真实模拟，无人工修正，全部数字程序化生成）**

节点：P155W1-N01。完整论文级报告见 [`docs/P155W1_N01_STEFAN_PHASE_CHANGE.md`](../../../docs/P155W1_N01_STEFAN_PHASE_CHANGE.md)。

## 验证链

| 链 | 检验量 | 结果 | 门限 |
|---|---|---|---|
| C1 一维凝固 | 前沿 `s(t)` vs Neumann 精确解（守恒相积分） | 0.504% | 2% |
| C1b 一维凝固 | 前沿 `s(t)` vs Neumann（`f_l=0.5` 穿越，独立口径） | 1.007% | 2% |
| C1c 一维 | 温度剖面 L2（排除界面带） | 1.04e-04 | 2% |
| C2 二维圆柱凝固 | 前沿半径 vs 独立锐界面参考解 | 1.446% | 3% |
| C2b 二维圆柱凝固 | 前沿形状（最大角向偏差 vs 圆） | 0.850% | 3% |
| C3 熔化 | 能量闭合（二阶壁面通量 vs 焓变） | 0.606% | 1% |
| C3b 熔化 | 壁面输入 vs Neumann 解析值 | 1.07e-05 | 1% |
| C4 收敛性 | 前沿观测阶 p | 1.00（理论 1） | 0.7–1.6 |
| C4 收敛性 | 剖面观测阶 p | 1.22 | ≥ 1 |

## 文件

* `run.py` — 全部数值 + 判据 + 图（写 `artifacts/stefan_phase_change/`）
* `render_report.py` — 由 `results.json` 渲染报告（零手抄）
* `result.json` — 精简结果（本目录）
* 测试：`tests/test_stefan_phase_change_verification.py`

## 复现

```bash
PYTHONPATH=src python benchmarks/verified/stefan_phase_change/run.py
python -m pytest tests/test_stefan_phase_change_verification.py -q
```

## 已知陷阱（详见报告 §5）

1. 圆柱相似解只对过冷熔体成立，而平衡焓法不能表示亚稳态液体 → 圆柱链改用冷壁构型 + 独立参考解；
2. PDE 残差测试不校验边界条件（曾漏掉熔化分支 `T(0)` 归一化错误）；
3. Planck 准稳态闭式解在 `Ste=0.5` 偏差达 20.3%，不能做 ≤3% 参考；
4. 一阶壁面通量与焓变是离散恒等式，不能当能量守恒证据。
