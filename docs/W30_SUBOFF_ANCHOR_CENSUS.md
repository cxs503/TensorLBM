# W30 · DTRC SUBOFF 外部锚逐点核销 + 阻力系数误差表

- 任务号 `m21512`，仓库 `TensorLBM`，生成器 `benchmarks/anchors/suboff_anchor_census.py`，git commit `e3bf2f4c097a619c1e85d5d8bc4456371ac15b70`（构建时 HEAD，重建会刷新）
- 本期清偿的任务键：`paper-exp-lbm-suboff-tau-3`, `paper-exp-lbm-suboff-tau-4`, `paper-exp-lbm-suboff-tau-6`（这三条此前 mission succeeded 但零报告、零工件）
- 主工件：`benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json`
- 锚件：`benchmarks/anchors/dtrc-suboff-1989.json`（构建器 `benchmarks/anchors/dtrc_suboff_1989_anchor.py`）
- 全部数值由 `benchmarks/anchors/suboff_anchor_census_report.py` 从工件程序化注入；工件内登记了 11 个源文件的 sha256，测试逐条核验。

## ① 案例介绍

**物理背景**：DARPA SUBOFF 裸艇（AFF-1，轴对称回转体）拖曳水池阻力。这是潜艇总体性能「阻力」学科最常用的公开外部锚：DTRC/SHD-1298-01 给几何，CRDKNSWC/HD-1298-11 给拖曳阻力表。本课题 `lbm-suboff-tau` 的全部阻力判定都以它为上限基准。

**几何（先验证再谈阻力，criterion C4）**

| 量 | 值 | 检查 | 允许带 | 结论 |
|---|---|---|---|---|
| L_max / D_max | 8.5748（记录 8.57） | 偏差 +0.0560% | ±1.0% | PASS |
| Cb = ∇/(A_mid·L) | 0.801915（标称 0.8） | 偏差 +0.2393% | ±3.0% | PASS |
| WSA | 5.998 m²（解析 5.9915 m²） | 偏差 +0.1091% | ±1.0% | PASS |

**换算条件（全部是声明假设，不是测量）**：ρ = 1000.0 kg/m³，ν = 1.139e-06 m²/s，1 kn = 0.514444 m/s，Re 以 L_max 计，C_T = R_T/(0.5ρV²S)，参考面积 S = WSA = 5.998 m^2 (Table 1 transcription)。锚点 Re 区间 11 点覆盖 1.165e+07 … 3.539e+07。

## ② 怎么做的

1. **零手抄解析**：`dtrc_suboff_1989_anchor.py` 在建件时用正则在仓内两个转录档案里解析每个 `(V, R_T)`：`docs/suboff_experimental_resistance_validation_2026-08-01.md`（Table 14 单行转录，族 T14）与 `benchmarks/pending/suboff_highre_resistance/refs_lock.md`（2024 年第三方论文Table 3 EFD 列转录 + 机器锁 `.json` 交叉核对，族 S1）。任何数值都不来自记忆或二次摘要。
2. **逐点登记**：每点带 verbatim 引句、文件定位（行号/字符偏移/表号/报告页码）与引用档案 sha256；拿不到原文的点登记 `acquisition_status=WITHHELD` 并写明缺什么。
3. **独立复算（双算术路径）**：C_F 用 ITTC-1957 `C_F = 0.075/(log10Re−2)²` 分别在 50 位 `Decimal` 与 `math.log10` 浮点两条独立路径上求值，再与仓内**别的**登记件已记的 C_F 逐点比残差；C_T 由 `R_T/(0.5ρV²S)` 独立重算。
4. **误差表**：把 LBM 现有 verified 工件（`benchmarks/verified/suboff_sail_resistance/`，Re_lattice = 1000 层流，参考 Blasius）与锚**逐点对齐**；同时把仓内唯一落在锚 Re 域内的分支（`benchmarks/pending/suboff_highre_resistance/`，壁模型 LES）与它实际跑过的 5 个锚点对齐。两个分支分别标 `regime_match` 与 `admissible_for_the_anchor`，不混算。
5. **敏感性**：网格阶梯 3 档（L=96/112/128）× 2 种仪器口径，加 τ 阶梯 5 臂；明确哪一档进入误差表。
6. **工具链**：纯 python（`decimal`/`math`/`json`）+ 读仓内 JSON/npz 工件；图由 matplotlib 生成；守卫测试 `pytest benchmarks/anchors/test_dtrc_suboff_anchor_census.py`。

## ③ 计算结果

### 3.1 锚件逐点登记表（族 T14 / 族 S1）

| id | 族 | V [kn] | V [m/s] | R_T [N] | Re | C_T | C_F(ITTC-57) | 1+k = C_T/C_F | 等级 | 获取 |
|---|---|---|---|---|---|---|---|---|---|---|
| T14-P1 | T14 | 5.920 | 3.0455 | 87.40 | 1.164727e+07 | 0.003142 | 0.002922 | 1.0753 | C | WITHHELD |
| T14-P2 | T14 | 10.000 | 5.1444 | 242.20 | 1.967443e+07 | 0.003052 | 0.002676 | 1.1403 | C | WITHHELD |
| T14-P3 | T14 | 11.840 | 6.0910 | 332.90 | 2.329453e+07 | 0.002992 | 0.002603 | 1.1492 | C | WITHHELD |
| T14-P4 | T14 | 13.920 | 7.1611 | 451.50 | 2.738681e+07 | 0.002936 | 0.002537 | 1.1574 | C | WITHHELD |
| T14-P5 | T14 | 16.000 | 8.2311 | 576.90 | 3.147909e+07 | 0.002839 | 0.002481 | 1.1444 | C | WITHHELD |
| T14-P6 | T14 | 17.990 | 9.2548 | 697.00 | 3.539431e+07 | 0.002713 | 0.002436 | 1.1140 | C | WITHHELD |
| S1-P1 | S1 | 5.929 | 3.0500 | 102.30 | 1.166444e+07 | 0.003667 | 0.002921 | 1.2552 | C | WITHHELD |
| S1-P2 | S1 | 9.999 | 5.1440 | 283.80 | 1.967275e+07 | 0.003576 | 0.002676 | 1.3363 | C | WITHHELD |
| S1-P3 | S1 | 11.857 | 6.1000 | 389.20 | 2.332888e+07 | 0.003488 | 0.002603 | 1.3399 | C | WITHHELD |
| S1-P4 | S1 | 13.918 | 7.1600 | 526.60 | 2.738276e+07 | 0.003425 | 0.002537 | 1.3502 | C | WITHHELD |
| S1-P5 | S1 | 15.998 | 8.2300 | 675.60 | 3.147487e+07 | 0.003326 | 0.002481 | 1.3405 | C | WITHHELD |

等级一律 **C**（拿不到原文，且 S1 族的源论文 PDF 也不在本检出中 → 连「B = 转录 + 源文档sha256」都够不到）。这是**拒发** A/B，不是「暂时没有」。

### 3.2 C_F 独立复算残差（判据 2）

- 点数 11，最大绝对残差 **4.337e-19** （门限 1e-09）→ **达标**
- 两条独立算术路径之间的最大差：4.337e-19
- 诚实边界：登记件里的 C_F 本来就是同一个闭式算出来的，所以这条残差核验的是**转录与换算链条**（V→Re→C_F），不是公式的物理正确性。物理正确性另有旁证：仓内已知答案 Re=1.2e7 → C_F=0.00291±5% 与本表 0.00292（Re=1.16e7）一致。

### 3.3 两族转录冲突（本轮新增的关键发现）

| S1 点 | T14 点 | 速度 [kn] | 速度差 | R_T(S1)/R_T(T14) |
|---|---|---|---|---|
| S1-P1 | T14-P1 | 5.929 | 0.1473% | 1.1705 |
| S1-P2 | T14-P2 | 9.999 | 0.0086% | 1.1718 |
| S1-P3 | T14-P3 | 11.857 | 0.1473% | 1.1691 |
| S1-P4 | T14-P4 | 13.918 | 0.0148% | 1.1663 |
| S1-P5 | T14-P5 | 15.998 | 0.0134% | 1.1711 |

比值 **1.1698 ± 0.0021**（即两族系统性差 17.0%），且与速度无关（速度错配本身按 R_T∝V^1.9 只能解释 ≤0.281%）。两族**不可能**同时是同一张 Table 14 的 AFF-1 裸艇值；判据 C3 判 `CONFLICT_EXCEEDS_GATE`：**17% > 15% 门**，因此在本环境里任何「LBM 与 DTRC 数据 15% 以内」的结论都无法被裁。

排名假设（详见工件 `cross_family_conflict.hypotheses_ranked`）：
- **H1** one of the two transcriptions is not the AFF-1 bare-hull row (wrong configuration column, e.g. a with-appendages or a different model-scale row)
  - 支持：the offset is a pure multiplicative constant 1.170 independent of speed (sd 0.0021); appendages alone explain only ~2.6% (in-repo appendage share), so a column mix-up rather than a physics difference is the better fit
  - 证伪试验：read Table 14 page 23 and match both the speed list and the R_T list; also compare against Table 15/16 configuration rows
- **H2** one family was normalised at a different density or reference scale before being tabulated
  - 支持：rho band 997-1000 gives only 0.3% and a model-scale change scales as L^2 (factor 4 or 0.25), so neither closes 1.17 cleanly
  - 证伪试验：check the primary for a reference-temperature / density footnote; both families would then need a re-derivation
- **H3** the 6-significant-digit transcription pair is right and the third-party paper rescaled EFD to a different reference area for its C_T comparison
  - 支持：S1's Table 3 is titled 'resistance in comparison with EFD' and may tabulate a rescaled force; but the in-repo S1 transcription is dimensional N, not a coefficient
  - 证伪试验：compare S1 Table 3 numbers with the paper's own CFD column and with its stated reference area

### 3.4 LBM vs 锚 逐点误差表（判据 3）

符号约定：`dev = (C_T,LBM / C_T,anchor − 1) × 100`；门限 |dev| < 15.0%。

**分支 A — verified 层流算例（`benchmarks/verified/suboff_sail_resistance/`，Re_lattice = 1000）**：raw 偏差 +952% … +1462%，33 行中通过 0 行 → **不可接受**。该分支 `admissible_for_the_anchor = False`（Re 差 4.07 个量级）。

| 锚点 | 族 | Re_anchor | C_T,anchor | LBM 档 | C_D,LBM | dev raw [%] | 门 | dev Re-折算 [%]（仅诊断） |
|---|---|---|---|---|---|---|---|---|
| T14-P1 | T14 | 1.1647e+07 | 0.003142 | L96 | 0.038570 | +1127.5 | FAIL | -14.6 |
| T14-P1 | T14 | 1.1647e+07 | 0.003142 | L112 | 0.040769 | +1197.5 | FAIL | -9.7 |
| T14-P1 | T14 | 1.1647e+07 | 0.003142 | L128 | 0.042396 | +1249.3 | FAIL | -6.1 |
| T14-P2 | T14 | 1.9674e+07 | 0.003052 | L96 | 0.038570 | +1164.0 | FAIL | -19.5 |
| T14-P2 | T14 | 1.9674e+07 | 0.003052 | L112 | 0.040769 | +1236.0 | FAIL | -14.9 |
| T14-P2 | T14 | 1.9674e+07 | 0.003052 | L128 | 0.042396 | +1289.3 | FAIL | -11.5 |
| T14-P3 | T14 | 2.3295e+07 | 0.002992 | L96 | 0.038570 | +1189.1 | FAIL | -20.1 |
| T14-P3 | T14 | 2.3295e+07 | 0.002992 | L112 | 0.040769 | +1262.6 | FAIL | -15.5 |
| T14-P3 | T14 | 2.3295e+07 | 0.002992 | L128 | 0.042396 | +1317.0 | FAIL | -12.2 |
| T14-P4 | T14 | 2.7387e+07 | 0.002936 | L96 | 0.038570 | +1213.8 | FAIL | -20.6 |
| T14-P4 | T14 | 2.7387e+07 | 0.002936 | L112 | 0.040769 | +1288.7 | FAIL | -16.1 |
| T14-P4 | T14 | 2.7387e+07 | 0.002936 | L128 | 0.042396 | +1344.1 | FAIL | -12.8 |
| T14-P5 | T14 | 3.1479e+07 | 0.002839 | L96 | 0.038570 | +1258.5 | FAIL | -19.7 |
| T14-P5 | T14 | 3.1479e+07 | 0.002839 | L112 | 0.040769 | +1335.9 | FAIL | -15.2 |
| T14-P5 | T14 | 3.1479e+07 | 0.002839 | L128 | 0.042396 | +1393.2 | FAIL | -11.8 |
| T14-P6 | T14 | 3.5394e+07 | 0.002713 | L96 | 0.038570 | +1321.5 | FAIL | -17.6 |
| T14-P6 | T14 | 3.5394e+07 | 0.002713 | L112 | 0.040769 | +1402.5 | FAIL | -12.9 |
| T14-P6 | T14 | 3.5394e+07 | 0.002713 | L128 | 0.042396 | +1462.4 | FAIL | -9.4 |
| S1-P1 | S1 | 1.1664e+07 | 0.003667 | L96 | 0.038570 | +951.9 | FAIL | -26.8 |
| S1-P1 | S1 | 1.1664e+07 | 0.003667 | L112 | 0.040769 | +1011.8 | FAIL | -22.7 |
| S1-P1 | S1 | 1.1664e+07 | 0.003667 | L128 | 0.042396 | +1056.2 | FAIL | -19.6 |
| S1-P2 | S1 | 1.9673e+07 | 0.003576 | L96 | 0.038570 | +978.5 | FAIL | -31.3 |
| S1-P2 | S1 | 1.9673e+07 | 0.003576 | L112 | 0.040769 | +1040.0 | FAIL | -27.4 |
| S1-P2 | S1 | 1.9673e+07 | 0.003576 | L128 | 0.042396 | +1085.5 | FAIL | -24.5 |
| S1-P3 | S1 | 2.3329e+07 | 0.003488 | L96 | 0.038570 | +1005.9 | FAIL | -31.5 |
| S1-P3 | S1 | 2.3329e+07 | 0.003488 | L112 | 0.040769 | +1068.9 | FAIL | -27.5 |
| S1-P3 | S1 | 2.3329e+07 | 0.003488 | L128 | 0.042396 | +1115.6 | FAIL | -24.7 |
| S1-P4 | S1 | 2.7383e+07 | 0.003425 | L96 | 0.038570 | +1026.1 | FAIL | -32.0 |
| S1-P4 | S1 | 2.7383e+07 | 0.003425 | L112 | 0.040769 | +1090.3 | FAIL | -28.1 |
| S1-P4 | S1 | 2.7383e+07 | 0.003425 | L128 | 0.042396 | +1137.8 | FAIL | -25.2 |
| S1-P5 | S1 | 3.1475e+07 | 0.003326 | L96 | 0.038570 | +1059.7 | FAIL | -31.5 |
| S1-P5 | S1 | 3.1475e+07 | 0.003326 | L112 | 0.040769 | +1125.8 | FAIL | -27.6 |
| S1-P5 | S1 | 3.1475e+07 | 0.003326 | L128 | 0.042396 | +1174.7 | FAIL | -24.7 |

**分支 B — 同 Re 域壁模型分支（`benchmarks/pending/suboff_highre_resistance/`，状态：pending (precision fails at single-GPU reachable grid)）**：这是仓内唯一落在锚 Re 域内的 LBM 计算，`admissible_for_the_anchor = True`，5 行中通过 0 行。

| 锚点 | Re | C_T,anchor | C_T,LBM | dev [%] | C_F,LBM/C_F,ITTC | y+_mean |
|---|---|---|---|---|---|---|
| S1-P1 | 1.1664e+07 | 0.003667 | 0.051173 | +1296 | 2.522 | 1878 |
| S1-P2 | 1.9673e+07 | 0.003576 | 0.075394 | +2008 | 2.515 | 3031 |
| S1-P3 | 2.3329e+07 | 0.003488 | 0.061713 | +1669 | 2.491 | 3519 |
| S1-P4 | 2.7383e+07 | 0.003425 | 0.075225 | +2096 | 2.502 | 4080 |
| S1-P5 | 3.1475e+07 | 0.003326 | 0.101863 | +2963 | 2.525 | 4660 |

### 3.5 分辨率 / τ 敏感性（判据 4）

**网格阶梯（真正的离散化旋钮）**

| 档 | τ（导出） | C_D,主口径(E2 standard) | 相对最细档 | C_D,共存档(mix50) | 对本案参考误差(主) |
|---|---|---|---|---|---|
| L96 | 0.5144 | 0.038570 | -9.02% | 0.042649 | -8.15% |
| L112 | 0.5168 | 0.040769 | -3.84% | 0.044485 | -2.92% |
| L128 | 0.5192 | 0.042396 | +0.00% | 0.045862 | +0.95% |

阶梯展开 9.92%，随 L 单调增。

**τ 阶梯（5 臂）—— 注意：在 LBM 里 τ 就是雷诺数，不是分辨率旋钮**

| 臂 | τ | ν_lb | Re_lattice | C_D total | 相对 A 臂 |
|---|---|---|---|---|---|
| A | 0.51 | 0.00333333 | 1200 | 0.035299 | +0.0% |
| B | 0.55 | 0.0166667 | 240 | 0.106751 | +202.4% |
| C | 0.60 | 0.0333333 | 120 | 0.183337 | +419.4% |
| D | 0.70 | 0.0666667 | 60 | 0.325955 | +823.4% |
| E | 1.00 | 0.166667 | 24 | 0.739868 | +1996.0% |

展开 1996% —— 那不是数值误差，是 Re 从 1200 扫到 24（50 倍）的物理效应；该阶梯自身判决为 `HONEST-FAIL`（G1 与 G3 通过，G2 在冻结口径下严格判 FAIL（仅 A 档带内），overall 由 G2 决定。同时披露：均匀网格上『负阻力病理』根本不存在，故 G1 的通过不等于『治愈成功』。）

**进入误差表的档**：`L128` + `standard (erratum E2, frozen primary caliber)`，C_D = 0.042396。理由：finest grid of the only verified LBM artifact; its own reference gate (G1 |err| <= 3%) passes at 0.95%. The co-shipped mix50 archive at the same tier (0.045862) is reported alongside, and the whole ladder is disclosed — no tier shopping。τ 臂不入表（not applicable (tau is not a resolution knob; the case's own derived tau at L128 is 0.5192)）。

## ④ 结果图

![fig1](figures/w30_suboff_anchor_census/fig1_anchor_ct_vs_re.png)

图 1 锚点 C_T(Re) 与 ITTC-1957 闭式线：两族转录给出两条平行线，间距即 17% 的系统性冲突。

![fig2](figures/w30_suboff_anchor_census/fig2_deviation_gate.png)

图 2 逐点偏差与 15% 门（symlog）：红=同 Re 域壁模型分支（+1296…+2963%），灰=verified 层流分支 raw（+952…+1462%），绿=Re 折算诊断（不作证据）。三条全不在门内。

![fig3](figures/w30_suboff_anchor_census/fig3_resolution_tau_sensitivity.png)

图 3 左：网格阶梯（3 档 × 2 口径，含本案层流参考线）；右：τ 阶梯（τ 增大 = Re 减小，C_D 随 Re 下降而上升）。

![fig4](figures/w30_suboff_anchor_census/fig4_form_factor_discriminator.png)

图 4 两族隐含形状因子 1+k：T14 给 1.08–1.16，S1 给 1.26–1.35，两者都在物理可辩范围内，算子层无法裁决。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **τ 不是分辨率旋钮**。LBGK 里 ν_lb=(τ−0.5)/3，案例冻结 u_lb 与 L 后，扫 τ 就是扫 Re（本仓 5 臂跨 Re 1200→24）。把 τ 阶梯当网格收敛证据是范畴错误；本报告把它单列并标注。
2. **verified 工件不是锚验证**。`benchmarks/verified/suboff_sail_resistance/` 是 Re_lattice=1000 的层流验证算例，参考是 Blasius 1.328/√Re，且几何是带帆（AFF-3 族）而锚是 AFF-1 裸艇。它对自己的参考误差 0.95%（最细档、主口径），但这**不能**当作对 DTRC 拖曳数据的验证——两者相差约 4 个 Re 量级。
3. **同 Re 域分支是 honest-FAIL**：壁模型分支的 y+_mean 落在 1878…4660，远超壁律可信带；其 C_F,fric/C_F,ITTC 在 5 个锚点上近乎常数（结构性水平误差，不是逐锚漂移），根因是单卡可达网格的分辨率饥饿，晋级路径=多节点 ~10⁸ 胞。
4. **锚本身有 17% 内部冲突**，大于 15% 门。这不是求解器的问题，是证据链的问题：在拿到 CRDKNSWC/HD-1298-11 Table 14 原文（p.23）之前，任何 <15% 的对标结论都不可裁。
5. **Re 折算列（绿色）是诊断不是证据**：把层流的 (1+k) 搬到湍流 ITTC 线上，按构造偏向低 1+k 的族（此处 T14），因此它对两族冲突**没有裁决力**。工件里 `renormalisation_discriminating_power` 字段写死了这一点，测试也锁住。
6. **等级天花板是 C**：本环境离线，A（读原文）/B（源文档 sha256）都够不到。工件 `claim_boundary.refused_labels` 显式拒发 A/B/PRIMARY-VERIFIED。
7. **适用范围**：本工件登记的是「本检出里能恢复到的 DTRC 数据」以及「本仓现存的 LBM 阻力证据」，不构成对 DTRC 数据的复现，也不构成对高 Re LBM 能力的宣称。

## ⑥ 误差判定（逐条，显式回答是否达标）

| 判据 | 内容 | 结果 | 是否达标 |
|---|---|---|---|
| Q1 | 锚件逐点 ≥3 点且 sha256 可核 | 11 点 / 2 族，全部引句+定位+sha256 可核 | **达标** |
| Q2 | C_F 独立复算残差 ≤1e-9 | 最大 4.337e-19 | **达标** |
| Q3 | LBM vs 锚逐点 |dev|<15% | 同 Re 分支 0/5（+1296…+2963%）；verified 分支 0/33 raw | **不达标** |
| Q4 | 网格/τ 敏感性 ≥3 档并标明入选档 | 网格 3 档 + τ 5 臂，入选 L128/standard | **达标** |

**关键量误差是否 <3%？**

- 对 **ITTC-1957 闭式**：是，最大残差 4.337e-19（≪3%），但它只证明换算链条自洽，不证明物理。
- 对 **LBM 自己的层流参考（Blasius）**：是，最细档主口径 0.95% < 3%（该案例自己的门 G1 通过）。
- 对 **DTRC 拖曳锚**：**否**。同 Re 域分支偏差 +1296…+2963%，远超 15% 门；且锚自身有两族 17% 冲突。此处如实判 **honest-FAIL**，不粉饰。
- 锚族之间：17.0% —— 也 >15%，同样 **不达标**。

**下一步实验（工件 `criteria[Q3].next_experiments`）**
- E1 (anchor side): obtain the primary HD-1298-11 PDF; re-register both families; the 17% conflict must be adjudicated before any 15% claim
- E2 (solver side): wall-model path on a grid with y+_mean in the valid band (multi-node submission) — the only admissible route to the anchor's Re
- E3 (scheme side): laminar Re ladder (1e3 -> 1e5) of the verified case to separate the form factor from the friction law

**复现**

```bash
python benchmarks/anchors/dtrc_suboff_1989_anchor.py           # 锚件（--check 只核）
python benchmarks/anchors/suboff_anchor_census.py              # 误差表 + 敏感性（--check 只核）
python benchmarks/anchors/suboff_anchor_census_figs.py         # 图
python benchmarks/anchors/suboff_anchor_census_report.py       # 本文件（程序化注入）
python -m pytest benchmarks/anchors/test_dtrc_suboff_anchor_census.py -q
```

