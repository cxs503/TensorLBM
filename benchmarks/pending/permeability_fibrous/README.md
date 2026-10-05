# W9-C: Gebart (1992) 横向渗透率 — 周期方阵/六角圆柱阵列 — FAIL（诚实）+ 方阵 verified 候选注记

## 一句话结论

**两阵列在冻结门下 FAIL**（sq: FAIL / hex: FAIL），三种失效模式全部定源：hex 近堆积极限 = **参考侧**（Gebart 渐近式偏差 +2.4%/+3.9%（0.75/0.70），已由方阵精确表双通道定量锁定）、sq 0.40 单调破缺 = **几何量化地板振荡**（gap-96 扩展案深入门内 -0.65%）、S1 = **真实 O(Re) 惯性**（继承力标下 Re_cell=0.5）。方阵 4 主档中 3 档（0.30/0.50/0.60）逐档全过门且严格单调 = **verified 候选**（晋级 owner 决策）。

## 口径

- D2Q9 BGK τ=1（ν=1/6 精确），全周期单元胞；sq：W×W 单圆柱居中（W4-A 继承）；hex：W×H 超胞（H=round(√3·W)），折叠角+中心双位= 三角格子。体力只加流体节点，步序 collide→stream→masked force→bounce-back；物理核全部来自 tensorlbm 库入口（铁律 grep 零命中）。
- 主参考（按档）：**方阵全部档 = Sangani & Acrivos 1982 精确 f 表**（k_ref=π/(vf·f)·R²，W4-A 双转录逐位一致）；**hex 近堆积（0.70/0.75 主档）= Gebart 1992 闭式** K/R²=C1·(√(vf_max/vf)−1)^2.5（sq C1=16/(9π√2)、hex 16/(9π√6)）。Gebart 同时对每档报告（一致性通道，方阵阶梯须原位复现锁定的 Gebart-vs-精确偏差：Vf=0.70 +2.978%、0.75 +0.268%）。
- 判据：err（对档主参考）随加密严格单调下降 AND finest ≤3% AND 全档稳态；Darcy/上皮口径 K_sim=(1−vf_actual)·ν·⟨u_x⟩_fluid/a_body；无外推（nominal 几何，不做有效半径再标定）。
- **hex 精确表不可得**：NOT OBTAINABLE——no exact hex transverse-permeability table is obtainable: Koch & Ladd 1997 and Wang & Sangani 1997 sit behind paywalls (8-channel hunt in prereg section 3); the hex main gate therefore references the Gebart near-packing asymptote, which the measurements show is biased HIGH by 2-4% at s = sqrt(vf_max/vf)-1 = 0.10-0.14

## 判决表（冻结 report，逐档）

| 档 | 阶梯 err (%) | 单调 | finest (%) | 全稳态 | 过门 | 档类 |
|----|--------------|------|-----------|--------|------|------|
| sq_vf0.3 | 15.23, 4.78, 2.18, 0.22 | True | 0.22 | True | True | main |
| sq_vf0.4 | 17.25, 6.92, 2.42, 2.63 | False | 2.63 | True | False | main |
| sq_vf0.5 | 16.03, 5.08, 2.96, 0.24 | True | 0.24 | True | True | main |
| sq_vf0.6 | 17.19, 5.99, 1.64, 0.88 | True | 0.88 | True | True | main |
| sq_vf0.7 | 17.12, 6.47, 3.85 | True | 3.85 | True | False | disclosure |
| hex_vf0.3 | 5.32, 2.74, 1.22 | True | 1.22 | True | True | disclosure |
| hex_vf0.45 | 5.55, 2.58, 2.16 | True | 2.16 | True | True | disclosure |
| hex_vf0.6 | 3.89, 2.09, 0.62 | True | 0.62 | True | True | disclosure |
| hex_vf0.7 | 1.66, 0.47, 0.13, 2.10 | False | 2.10 | True | False | main |
| hex_vf0.75 | 2.38, 0.34, 0.85, 2.03 | False | 2.03 | True | False | main |

sq 主档：0.3 过、0.4 未过、0.5 过、0.6 过。hex 主档 0.70/0.75 全 FAIL（vs Gebart，参考侧偏差见定源 1）。

## 失效定源（三种，全部机读）

1. **hex = 参考侧**。
   - resolved LBM K at hex 0.70/0.75 is genuinely BELOW Gebart by O(2-4%): error stays negative and keeps descending as gap grows 12->24->48->72->96; a staircase artifact would decay toward 0, a reference bias is revealed as the staircase part shrinks
   - 阶梯分解（1+signed = (1+staircase)*(1+gebart_bias)，DIAGNOSTICS.md D1 (controller-verified; this file ships verbatim in the archive)）：hex 0.75 阶梯项 [4.9, 2.9, 1.6, 0.5]%（gap 12/24/48/72）、Gebart 偏差 +2.4%；hex 0.70 阶梯项 [6.3, 4.4, 4.0, 1.8]%、偏差 +3.9%。
   - 排除项：premature steady (S2 20/20 pass, tol bounds hidden drift << 0.1%)；inertia (S1: 0.073% per dRe 0.5 at hex0.75)。
   - these are the first resolved LBM measurements of hex 0.70/0.75 transverse permeability; solver validated by four independent channels (square exact tiers, hex dilute tiers, gebart-consistency bit-closure, S1/S2)
2. **sq 0.40 = 量化地板振荡**。
   - formal ladder +17.25/+6.92/+2.42/+2.63% breaks strict monotonicity by +0.21pp between rungs 3 and 4; the gap-96 extension returns -0.6475% (deep under the gate) -> rung-3->4 uptick is geometry-quantization oscillation (dR/R ~ 0.5/R amplified by |dlnK/dlnR| ~ 6.7 at phi=0.4), not a divergent trend
3. **S1 力标敏感性与 sq 0.70 披露补全**。
   - fs 10->20 linearity fails at sq0.3 W64 (0.256% > 0.1% tol): real O(Re) inertia at the inherited Re_cell=0.5; hex0.75 pair passes (0.073%). Physical, correctly measured, recorded per frozen protocol (FAIL, no re-run). Diagnostics at fs=20 run at Re_cell ~ 1.0 because the W4-A fp32-margin guard rejected fs=10 at gap 96 (margins 8.67-11.35 < 14)
   - 3-rung formal ladder 17.12/6.47/3.85% (frozen non-gating disclosure); gap-72 extension -1.4686% crosses under the gate; kept non-gating per frozen prereg (tier frozen at 3 rungs)（扩展案 gap-72 err -1.4686%）。

S1 力线性敏感性（fs 10→20，K_sim 不变性，容差 0.1%）：

| 案 | K_base | K_scaled | Δ (%) | 过 |
|----|--------|----------|-------|----|
| case_hex_vf0.75_w265_fs20.json | 10.5463 | 10.5386 | 0.0732 | True |
| case_sq_vf0.3_w64_fs20.json | 41.7095 | 41.6028 | 0.2557 | False |

### 扩展案（gap-72/96 定源跑，冻结判决外，非门）

| 案 | 阵列 | vf | W | gap (lu) | K_sim | K_ref(主) | 符号 err (%) |
|----|------|----|----|---------|-------|-----------|--------------|
| case_hex_vf0.75_w1058_fs20.json | hex | 0.75 | 1058 | 95.7 | 161.386 | 167.58 | -3.6963 |
| case_hex_vf0.7_w790_fs20.json | hex | 0.7 | 790 | 96.0 | 194.389 | 197.606 | -1.6284 |
| case_sq_vf0.4_w335_fs20.json | sq | 0.4 | 335 | 95.9 | 511.719 | 515.053 | -0.6475 |
| case_sq_vf0.7_w1293_fs20.json | sq | 0.7 | 1293 | 72.3 | 121.841 | 123.657 | -1.4686 |

## 方阵 verified 候选注记

the square array at its exact Sangani-Acrivos reference is a VERIFIED CANDIDATE on three of four main tiers (0.30/0.50/0.60): every gated tier passes finest <= 3% with strict monotonicity; promotion is an owner decision (the array-level FAIL comes only from the 0.40 quantization oscillation)

| vf | 阶梯 err (%) | 单调 | finest 符号 err (%) | 过门 |
|----|--------------|------|--------------------|------|
| 0.3 | 15.23, 4.78, 2.18, 0.22 | True | -0.22% | True |
| 0.5 | 16.03, 5.08, 2.96, 0.24 | True | +0.24% | True |
| 0.6 | 17.19, 5.99, 1.64, 0.88 | True | -0.88% | True |

## R3 参考自产门（hex_ref_gate/）

- 做什么：self-production of the square reference at 0.5% three-estimator gate as the prerequisite for producing an exact hex reference (Schur-complement Stokes solver on the square array, ladder pool 51 cases)
- 协议：fixed-alpha LSQ 1.0/1.5 (finest windows 4/5/6) + worst-window point value + Aitken3 on the last three rungs; pass = all three |err| <= 0.5%
- 判决：**FAIL** — vf0.3/0.4/0.5 PASS, vf0.6 FAIL by 0.052pp (fixed15 -0.5521%), vf0.7 FAIL (aitken3 -0.7967%, tail effective order collapses 1.34->0.58), vf0.75 attempt tier FAIL (not blocking). Per prereg 11.5: stop — no hex values produced

| vf | ref_f | fixed-α1.0 (%) | fixed-α1.5 (%) | Aitken₃ (%) | 过门 |
|----|-------|----------------|----------------|-------------|------|
| 0.3 | 102.9 | -0.169 | -0.228 | -0.073 | True |
| 0.4 | 217.89 | -0.085 | -0.250 | -0.339 | True |
| 0.5 | 532.55 | +0.199 | -0.245 | -0.390 | True |
| 0.6 | 1763 | -0.188 | -0.552 | -0.283 | False |
| 0.7 | 13520 | +0.156 | -0.432 | -0.797 | False |
| 0.75 | 126300 | +1.762 | -0.942 | -7.671 | False |

- 控制器交叉核对：controller's own fitter recomputed all 18 estimator values and 6 tier verdicts bit-identical。
- owner 决策菜单（hex 精确参考的取得路径）：
  - (a) M1 cut-cell rerun of the full square array (prereg section 3 registered variant; recommended by the track)
  - (b) widen the gate to 1% (vf0.6/0.7 then pass; hex error band propagates equally wide)
  - (c) purchase Koch-Ladd 1997 / Wang-Sangani 1997 tables
  - (d) keep the Gebart gate and accept the W9-C FAIL as recorded (zero cost; reference-side bias +2.4%/+3.9% already source-attributed)
- 工件：hex_ref_gate/ (NOTES.md, prereg.md, src/, out/cases 51 case JSONs, sq_gate.json, sq_gate2.json, gate2_owner_report.md)

## 运行

```bash
python run.py case --array sq --vf 0.5 --w 238 --device cuda:0   # 单案
python run.py report --out-dir out                              # 聚合判决
```

## 交付物（本目录）

- `run.py`（case/report 双子命令；库核入口 + compile_route 适配）/ `result.json`（冻结 report 原文 + wave9_record 附录：失效定源、扩展案、方阵候选、R3 门）/ `out/`（40 份 case 机器档案 + 机器 README）/ `out_diag/`（gap-72/96 定源扩展 4 案）/ `prereg.md` / `DIAGNOSTICS.md` / `hex_ref_gate/`（R3 全套：NOTES.md + prereg.md + src/ + out/（51 case + 双门 JSON + owner report））。

## 判定

- benchmark_overall = **FAIL**（sq: FAIL、hex: FAIL；冻结判据原样，无事后放宽）。
- 附加价值：方阵三档 verified 候选 + hex 参考侧偏差定量钉死 + 首次 hex 0.70/0.75 横向渗透率解析测量。
