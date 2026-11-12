# CASE-LBM-RB-ONSET-RA1708 — Rayleigh-Bénard 起始临界 Rayleigh 数：生产运行与判定（m16539）

> 锚点 `CASE-LBM-RB-ONSET-RA1708`（TensorLBM 算例池）｜ref: Chandrasekhar (1961) *Hydrodynamic and Hydromagnetic
> Stability* §II，刚-刚等温壁 2-D 辊 `Ra_c = 27π⁴/4 = 1707.7597`、`a_c = 3.11632`｜容差 <2%
> 本文档全部数字由脚本从 `benchmarks/case_pool/rayleigh_benard_onset/artifacts/summary.json`、
> `route*.json`、`independent_control.json` **程序化注入**（零手抄）。

## 判定（诚实、按运行前冻结的口径）

**VERDICT = FAIL**（`summary.json: pass = False`）。冻结门 6 项逐项：

| 门 | 结果 | 含义 |
|---|---|---|
| primary_grid_within_tol | True | 主网格 ny=64×128 的 Ra_c = 1731.86（+1.411%）< 2% |
| extrapolated_within_tol | False | 1/ny 两参数外推 Ra_inf = 1755.32（+2.785%）**超线** |
| control_ok | True | 控制臂（连续 N-S-Boussinesq 线性稳定性）Ra_c = 1707.7685（+0.00051%）< 0.05% |
| bisection_consistent | True | 每网格二分链收敛、拟合合格 |
| sigma_sign_change_present | True | 每网格 σ*(Ra) 括号两端异号（真起稳穿越） |
| ma_sensitivity_within_tol | False | α 减半臂 Ra_c = 1758.09（+2.947%）**超线** |

两门 FAIL ⇒ **honest FAIL，口径不移动**。同时如实登记三项强子结果：(a) 三档生产网格 Ra_c 全部 <2%
（+0.233/+1.323/+1.411%）；(b) 控制臂独立复算 +0.00051%，
另有 RK4 时域独立路径复算 rigid Ra_c = 1700.755（-0.410%）、free-slip 标定 656.869
（理论 657.511，-0.098%）；(c) 亚临界/超临界符号在 3 网格 × 3 臂 = 9 条
独立时间推进上全部正确。

## ① 案例介绍

两层无限大刚性平行板间的 Boussinesq 流体，下热上冷、x 向周期（Γ=nx/ny=2、m=1 ⇒ kH=π）；
纯导热静止基态在 `Ra > Ra_c` 后线性失稳，先出现二维定常辊。观测量 = 该本征模温度扰动投影幅 A(t)
的指数增长率 σ* = σH²/α 的**零点位置**（Ra_c）。参数：Pr=0.71（α=1/6 ⇒ τ_T=1.0）、τ_f=0.855、
格点阶梯 32×64/48×96/64×128、传播壁 gap=ny−2。

## ② 怎么做的

- **Route A（本案指标）**：库内双分布热 LBM（D2Q9 流场 + 热标量、BGK + Boussinesq 力、pre-streaming 半程反弹壁）
  时间推进测 σ*，对 Ra 做先验括号校验 + 保安全插值二分（停在全 σ* 边带 |σ*|≤0.05 或括号宽 <0.2%）。
- **Route B（交叉）**：同一格点步进算子的 FD Jacobian 谱 → 二分 γ(Ra)=0（不推进时间）。
- **Route C（控制）**：连续 N-S-Boussinesq 边际稳定性（二阶 FD + QZ 全谱 + 三网格 Richardson），必须先复现文献值才允许读 Route A 的偏差。
- **独立复核**：`independent_check_control.py` 用 RK4 时域积分 + 二分重推边际 Ra，并以有闭式解的 free-slip 臂标定实现。
- 敏感性臂（运行前声明）：波数盒 56/64/72（ny=32、Ra=1.05Ra_c）、α 减半（τ_T=0.75，降 Ma）。
- 总墙钟 545 s（本机 3 进程并行）+ 独立复核 94 s。

## ③ 计算结果

Route A（主表）：

| ny | nx | τ_f | τ_T | gap | Ra_c | 误差 % | 不确定度 % | 臂数 | 判决 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 64 | 0.855 | 1.00 | 30 | 1711.74 | +0.233 | 0.403 | 3 | ok |
| 48 | 96 | 0.855 | 1.00 | 46 | 1730.35 | +1.323 | 0.407 | 3 | ok |
| 64 | 128 | 0.855 | 1.00 | 62 | 1731.86 | +1.411 | 0.412 | 3 | ok |

- 1/ny 两参数外推 Ra_inf = **1755.32**（+2.785%），残差 ['-1.115', '3.344', '-2.229']
  （max|r|=3.344）——三点不共线于 1/ny，说明该拟合**形式**欠妥；仅用最细两点的两点外推 = 1736.37（+1.676%，门内）。
- Route B（ny=16，H=ny 口径）Ra_c = 2673.4；乘文档化换算 ((ny−2)/ny)³ 后 1790.99（+4.874%）。
- Route C（控制）Ra_c(Richardson) = 1707.76848（+0.00051%）；本案 kH=π 处 = 1706.61593。
- 波数敏感性（ny=32、Ra/Ra_c=1.05）：kH=3.590: σ*=+0.5177 ; kH=3.142: σ*=+0.6106 ; kH=2.793: σ*=+0.2153；抛物线峰 a_peak = 3.304 vs a_c = 3.11632（+6.033%）。
- α 减半臂：Ra_c = 1758.09（+2.947%），τ_f=0.6775/τ_T=0.75。

## ④ 结果图

`benchmarks/case_pool/rayleigh_benard_onset/artifacts/figures/`（`make_figures.py` 从 summary 现算）：
fig1 σ*(Ra) 各网格穿越与零点插值；fig2 Ra_c(ny) 与 1/ny 外推（含两点外推）；fig3 波数扫描抛物线；
fig4 控制臂 Richardson 收敛 + a_c 标位。**无场量云图**：本案观测量是增长率标量（临界态解即导热基态、
临界模幅 ~1e-4），以 σ*–Ra 相图替代，理由如实登记。

## ⑤ 注意点（坑与局限）

1. **驱动器致命缺陷（本单修复）**：`run_case.py::extrapolate_grids` 原写 `(n1, r1), (n2, r2) = ok[-2], ok[-1]`，
   而 `ok` 装的是整条臂记录 dict（十余键）——生产路径在 Route A 三网格跑完（约 6 min 计算）后必抛
   `ValueError: too many values to unpack (expected 2)`，**summary.json 从未产出过**，这正是本锚点长期 open 的根因；
   原 `tests/…::test_grid_extrapolation_form`（2 键 dict）也因字符串减法 `TypeError` 为红。已改显式取
   `ok[-2]["ny"]/["Ra_c"]`，并新增 `test_production_verdict_matches_recorded_adjudication` 把判定钉死防漂移。
2. **梯子非单调**：ny=32 档最好（+0.233%）、48/64 档几乎持平且更差
   （+1.323/+1.411%）：与"格点有效 Ra_c 上移"的既有诊断一致，误差不随加密消失，
   故 1/ny 形式不成立；两点外推（+1.676%）可作修正口径候选，交评审席裁定。
3. **Route B/A 口径不可直接比**：Route B 用 H=ny、Route A 用 gap=ny−2，须乘 ((ny−2)/ny)³（ny=16 时 0.67）；且 Route B 只在 ny=16。
4. **α 减半把穿越推高 +2.95%**：残余偏移含格点（Ma/τ_T）成分而非纯几何分辨率，该臂即主观不确定度上界。
5. kH=π 与 a_c=3.1163 有 +0.81% 失谐（Γ=2 整数模约束），故 Route A 的参照点是"该盒内模态的临界 Ra"。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 数值 | 相对基准 | <2%（本锚点口径） | <3%（通用线） |
|---|---|---|---|---|
| Ra_c（主网格 ny=64） | 1731.86 | +1.411% | ✅ | ✅ |
| Ra_c（ny=32/48） | 1711.74 / 1730.35 | +0.233% / +1.323% | ✅ | ✅ |
| Ra_inf（1/ny 三点外推） | 1755.32 | +2.785% | ❌ | ✅ |
| Ra_c（α 减半臂） | 1758.09 | +2.947% | ❌ | ✅ |
| Route C 控制臂 | 1707.76848 | +0.00051% | ✅（门 0.05%） | ✅ |
| 独立 RK4 复核（rigid） | 1700.755 | -0.410% | ✅ | ✅ |
| free-slip 标定 | 656.869 | -0.098% | ✅ | ✅ |

**结论**：全部量落在 <3% 内，但本锚点自定的 <2% 门有两项（Ra_inf、α 减半臂）不过 ⇒ **honest FAIL，不粉饰**。

## 复现

```bash
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/run_case.py --out-dir runs/case_rayleigh_benard_onset --parallel 3
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/independent_check_control.py
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/make_figures.py --run-dir runs/case_rayleigh_benard_onset
PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/make_report.py  --run-dir runs/case_rayleigh_benard_onset
python3 -m pytest tests/test_case_pool_rayleigh_benard_onset.py -q
```
