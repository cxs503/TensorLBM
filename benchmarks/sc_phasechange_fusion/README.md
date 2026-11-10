# 融合验证面：高密度比 Shan-Chen + 焓法相变（节点 p348w3-n04）

四条独立验证锚 + 7 条机器可算判据，全部由本目录脚本程序化产出（零手抄）。

| 锚 | 物理 | 基准 |
|---|---|---|
| C1 Neumann | 一维冰–水融化前沿 `s(t)=2λ√(α_l t)` | 两相 Neumann 相似解（含过冷项） |
| C2 Stefan 条件 | `ρL v_n = [k ∂T/∂n]` | 离散焓平衡 vs 独立通量外推 |
| C3 Laplace 律 | 静止液滴 `Δp = σ/R`，10:1/50:1/100:1 | 自洽 LSQ（无外部 σ 输入） |
| C4 Rayleigh–Taylor | 单模界面线性增长率 `σ=√(Agk)` | 无粘/等粘色散关系 |
| C5 质量 / C6 能量 | 守恒性 | 每步漂移 / 逐窗能量收支 |
| C7 覆盖 | 48 例 pytest（含判据自身反例） | — |

## 复现

```bash
PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part enthalpy
PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part laplace --ratio 10   # 50 / 100 同理
PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part rt
PYTHONPATH=src python benchmarks/sc_phasechange_fusion/run.py --part gates --test-count 48
PYTHONPATH=src python benchmarks/sc_phasechange_fusion/figures.py
PYTHONPATH=src python benchmarks/sc_phasechange_fusion/report.py
python -m pytest tests/test_sc_phasechange_fusion.py -q
```

## 结果（`artifacts/sc_phasechange_fusion/verdict.json`）

7 条判据 **6 过 1 FAIL**：C1 前沿误差 0.38%（<3%）、C2 Stefan 残差 1.39%（<2%）、
C6 能量 0.37%（<2%）、C4 RT 误差 ≤9.1%（<10%）、C5 每步质量漂移 9.5e-9、C7 48 例全过。
**C3 honest-FAIL**：10:1 档 `Δp=σ/R` 成立到 2.04%，但 50:1/100:1 档伪电流
Ma 升到 0.34/0.42 且有限汽相域被耗竭（实测密度比 247 vs 共存 100），
Laplace 拟合崩坏 —— 这是伪势 + EDM 格式的能力边界，已在报告 §3.3/§5 记录成因与后续方向。

论文级报告：`docs/P348W3_N04_SC_PHASECHANGE_FUSION.md`。