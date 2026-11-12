# T1 · 三维多相流物理模型重建（GAP-LBM-T1-MULTIPHASE）

第 1 轮 = **方法研究单（不含实现代码）**。基线 `a9870455`，分支 `agent/m15618`。

| 文件 | 内容 |
|---|---|
| [`01_MODEL_MATH_COMPARISON.md`](01_MODEL_MATH_COMPARISON.md) | 五模型（SCMP / SC-MCMP / CG / FE-CH / AC-cAC）数学模型 + 六轴离散误差系统比较 + OpenLB/waLBerla/Palabos/Lettuce 对应实现与引文（含证据级标注） |
| [`02_ACCEPTANCE_CRITERIA.md`](02_ACCEPTANCE_CRITERIA.md) | owner 五条物理判据的可执行口径（T1-C1…C5）、通过门、失败签名、反自宣布机制 |
| [`03_ANCHOR_CONSOLIDATION.md`](03_ANCHOR_CONSOLIDATION.md) | 11 条既有锚点/基准/在跑单点 → T1 伞下映射；**新增 slug = 0** |
| [`04_VERDICT_AND_ROADMAP.md`](04_VERDICT_AND_ROADMAP.md) | 模型选型裁决（cAC 主路线 / 化学势伪势备选 / SC-CG-AC 否决）+ −43% 根因假设表 H1–H6 + S0–S6 阶段门与工作量 |
| [`T1_NUMERIC_APPENDIX.md`](T1_NUMERIC_APPENDIX.md) | **程序生成**的全部数值表（Maxwell 共存、CG 声学上限、AC 崩塌时间、HR 恒等式与伪流预算、仓库实测归约） |
| `analysis/t1_phase_budget.py` | 上表生成器（`python3 docs/multiphase/T1/analysis/t1_phase_budget.py`）；机器可读输出 `artifacts/T1_multiphase/t1_phase_budget.json` |
| `figs/fig1..fig5` | ①CG 声学天花板 ②伪流预算 $S<0.1Ca$ ③SC 共存密度比 ④HR 有效窗 $Ca$–$Bo$ ⑤Model A 崩塌时间 |

## 三条可复算的核心结论

1. **Color-Gradient 是结构性不足**：$p=\rho c_s^2$ ⇒ $\Delta\rho_{max}=2\sigma/(Rc_s^2)$。
   $\sigma=10^{-2},R=12$ ⇒ 上限 0.005；要密度比 2 需 $\sigma=2.0$（越稳定界）。仓库两例实测
   平台 1.0027/1.0024 反解 σ = 4.1e-3/6.0e-3，与该式自洽。
2. **非守恒 Allen–Cahn 必然溶解**：$t^*=R_0^2/(4\tilde M\sigma)$，用仓库探针反演 $\tilde M\sigma\approx0.16$
   ⇒ 203 步，观测 200 步；与网格无关。
3. **HR 的伪流预算是 $S=u_{sp}\mu_o/\sigma<0.1Ca$，而 $Ca=Bo(1+\lambda)/(6(2+3\lambda))$ 不是自由参数**
   ⇒ $Bo=0.005,\lambda=1$ 时 $S<3.3\times10^{-5}$；SC 族实测 $S=7.0$，差 $2\times10^5$ 倍。

## 本单实跑

```
python3 -m pytest tests/test_cac_smoke.py tests/test_phasefield_static_droplet.py -q -x   # 14 passed, 93.85 s
python3 docs/multiphase/T1/analysis/t1_phase_budget.py                                    # JSON + 附录 + 5 图
```

**未做**：任何新数值算例、任何求解器代码改动、−43% 的根因实测定位（属 S4）。