# W28 · DTRC SUBOFF 锚件落地 main + 分级如实（TRANSCRIBED-SECONDARY）+ 三档 GCI 复算守卫

任务 `paper-exp-lbm-suboff-tau-13`（m19903，仓 **TensorLBM**，runner=native）。
所有数字由 `build_anchor_landing.py` 写入 `anchor_landing.json`，本报告再由 `publish_report.py` 从该工件
注入（零手抄）。

| 工件 | 作用 | sha256 |
|---|---|---|
| `benchmarks/anchors/lbm-suboff-tau.json` | 外部锚件（W27 blob 逐字节落地 + 单点插入分级字段） | 落地前/剥离后 = `b452bccd7d35ce1c294b159342b2d0e0fbb75c6fd5c574975df732abe7b54f88`；落地后 = `146ed124a961983fd6638c59fe39820d190011fec393b51389f968fd50ee11df` |
| `papers/artifacts/w28/lbm-suboff-tau/anchor_landing.json` | 判据逐条裁决 + 复算表 + 溯源 | 见仓库 |
| `tests/test_w28_lbm_suboff_tau_anchor_landing.py` | 16 项守卫测试 | — |

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

**对象**：DARPA SUBOFF AFF-1 裸艇（configuration 1），回转体，`L_max = 4.356 m`、
`D_max = 0.508 m`（L/D = 8.5748），排水体积
`0.708 m³`，湿表面积 `WSA = 5.998 m²`。DTRC 拖曳水池实测总阻力
`R_T(V)` 是潜体阻力领域的标准外部锚。

**约定**（锚件自身声明）：`ρ = 1000.0 kg/m³`（声明假设）、`ν = 1.139e-06 m²/s`
（引句自 `docs/IN12_S6_WALL_LADDER.md:35`）、
`kn→m/s = 0.514444`、`Re = V·L_max/ν`、`C_T = R_T/(½ρV²S)`、`C_F = 0.075/(log₁₀Re−2)²`。

**先验几何门（不做几何验证不进阻力比对）**：L/D 8.574803（登记 8.57，+0.056%）、
Cb 0.8019（登记 0.80，+0.24%）、解析湿面积 5.99146 m²（登记 5.998 m²，−0.109%）—— 全部在
1%/3% 带内（W27 已跑，本单复核）。

**锚件 6 点（逐字转录自仓内 Table 14 档，报告 p.23；非原始报告）**：

| point | V [kn] | R_T [N] | Re | C_T | C_F 登记(ITTC-57) | C_F 复算(50 位) | |Δ| |
|---|---|---|---|---|---|---|---|
| AFF1-P1 | 5.92 | 87.4 | 1.1647e+07 | 0.003142066 | 0.002922083 | 0.002922083 | 0.00e+00 |
| AFF1-P2 | 10 | 242.2 | 1.9674e+07 | 0.003051557 | 0.002676144 | 0.002676144 | 0.00e+00 |
| AFF1-P3 | 11.84 | 332.9 | 2.3295e+07 | 0.002991973 | 0.002603496 | 0.002603496 | 0.00e+00 |
| AFF1-P4 | 13.92 | 451.5 | 2.7387e+07 | 0.002935801 | 0.002536624 | 0.002536624 | 0.00e+00 |
| AFF1-P5 | 16 | 576.9 | 3.1479e+07 | 0.002839278 | 0.002481123 | 0.002481123 | 0.00e+00 |
| AFF1-P6 | 17.99 | 697 | 3.5394e+07 | 0.002713425 | 0.002435803 | 0.002435803 | 0.00e+00 |

---

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **锚件落地（本单第一件事，修 W27 的「工件滞留未合并分支」缺陷）**
   W27 锚件只存在于未合并分支 `agent/m18960`（head `71c90fd13e591a2bec1a021b438c2556e54c36c4`），main 工作树没有
   `benchmarks/anchors/`，因此 W28 锚门判 NO-ANCHOR-ARTIFACT。本单用 `git show agent/m18960:…`
   取回 blob（**不联网**），先原样写入 main 工作树并核对 sha256：
   `b452bccd7d35ce1c294b159342b2d0e0fbb75c6fd5c574975df732abe7b54f88` = 分支件 sha256（逐字节一致，31108 B）。
2. **分级字段以「单点插入」方式补齐，可机械还原**
   分级不能不改文件（gate-v3 要读 `/anchor_grade`），所以采用**唯一一段连续字节插入**：在字节偏移
   `41` 处插入 3885 B（两个顶层键
   `anchor_grade` / `grade_rationale`），**既有内容 0 字节改动**。还原律：
   `landed[:off] + landed[off+len(inserted):] == 分支 blob`（sha256 逐字节复现，见判据 C1 与守卫测试）。
   差异逐条登记于 `landing.amendment.diff_itemization`（3 条：2 条 insert + 1 条「无删除/无重排/无空白改动」）。
3. **分级规则（写死在脚本里，不由人临时决定）**
   `anchor_grade ← f(status, acquisition_status)`；`(TRANSCRIBED, WITHHELD) → TRANSCRIBED-SECONDARY (C)`；
   主文档为 WITHHELD 时 A/B 属禁用语（脚本内 `FORBIDDEN_GRADES_WHEN_PRIMARY_WITHHELD`，触发即抛错）。
   本件两篇外部文献在本环境**均不可得**（离线：`curl apps.dtic.mil` DNS exit 6、`curl google.com` 超时 exit 28、
   全盘 `find *1298*` 无原件），故：TRANSCRIBED-SECONDARY，`claim_boundary.may_be_cited_as_published_experimental_benchmark = False`。
4. **ITTC-1957 独立复算（两条算术路径）**
   路径 A：`math.log10` 双精度；路径 B：`Decimal` 50 位有效数字、`log10 = ln(Re)/ln(10)` 全 Decimal 路径
   （不经 libm）；另留路径 C（`ln(Re)/ln(10)` 双精度）作交叉敏感性。逐点与登记 `C_F` 闭合，门槛 1e-9。
5. **同 Re 口径换算（预登记三算子，沿用 W27 冻结版本，不改阈值）**
   OP0 恒等（只用于量化 4 个数量级 Re 鸿沟，预先声明不是达标路径）；
   OP1 常形状因子 `Cd^anchor = (Cd_total(Re=1e3)/Cf_Blasius(1e3))·Cf_ITTC57(Re_anchor)`（**主算子**）；
   OP2 定压差 + ITTC 摩擦 `Cd^anchor = Cd_p(1e3)·(S_bench/S_anchor) + Cf_ITTC57(Re_anchor)`
   （把 πDL 圆柱代理面积显式换算到锚的解析湿面积，作敏感性）。反挑数条款：两算子差 >15 pp 则该点判
   `OPERATOR_DEPENDENT`，不得当作验证结论。
6. **三档 GCI（真实不等加密比）**：`h ∝ 1/L`，`r21 = 128/112 = 1.142857`、
   `r32 = 112/96 = 1.166667`，
   由 `ε32/ε21 = ((r21r32)^p − r21^p)/(r21^p − 1)` 二分（400 步）解观测阶，`GCI = Fs·|ε21/φ1|/(r21^p − 1)`，
   `Fs = 1.25`。**等加密比捷径式只作敏感性**（它会把前渐近三元组伪装成 p≈2.2）。
   三臂全报：A=E2 `standard`（任务点名的三档）、B=归档 `mix50`、C=裸艇 `diag_bare`（几何与锚一致）。
7. **工具链/成本**：纯 CPU、只读归档件；**未递交集群作业**（预算 100 调用 / 14400 s 未动用）。
   理由：`gci_v4.json` 实测 15–20 s/step @L96 → 锚 Re 档单档 61–144 h，本单不做新算例，只复算归档梯子。

---

## ③ 计算结果（数字表，程序化注入）

### ③.1 判据逐条裁决

| 判据 | 任务项 | 阈值 | 实测 | 裁决 |
|---|---|---|---|---|
| C1_anchor_landed_byte_identical_modulo_declared_amendment | 1 | sha256 逐字节一致(+逐条差异) | {"identical_modulo_amendment": true, "diff_items": 3, "bytes_added": 3885} | PASS |
| C2_anchor_grade_consistent_with_acquisition_status | 2 | WITHHELD 禁 A/B | {"grade": "TRANSCRIBED-SECONDARY", "expected": "TRANSCRIBED-SECONDARY", "all_checks_pass": true} | PASS |
| C3_ittc57_independent_recompute | 3 | |ΔC_F| ≤ 1e-9 | {"max_abs_residual": 4.336808689942018e-19, "tolerance": 1e-09} | PASS |
| C4_three_tier_gci_and_observed_order | 4 | GCI<5% 且 p≥1.8 | {"A_e2_standard_with_sail": {"p_observed": 1.0914768608320355, "gci_fine_pct": 30.575370945328785, "pass_gci_lt_5pct": false, "pass_p_ge_1p8": false,  | FAIL |
| C5_pointwise_deviation_vs_anchor_CT | 5 | |dev| ≤ 15% 且非算子依赖 | {"A_e2_standard_with_sail": {"operator_primary": "OP1_constant_form_factor", "max_abs_dev_pct_OP1_by_tier": {"L96": 20.643020447602424, "L112": 16.120 | FAIL |

### ③.2 三档 GCI 与观测收敛阶（三臂全报）

| 梯子臂 | r21 / r32 | φ_c / φ_m / φ_f (=Cd_total) | R=ε21/ε32 | p 观测 | GCI_fine [%] | 捷径 p(等 r) | 裁决 |
|---|---|---|---|---|---|---|---|
| A·E2 standard（带帆） | 1.142857 / 1.166667 | 0.038570 / 0.040769 / 0.042396 | 0.74 | 1.09 | 30.58 | 2.25 | HONEST_FAIL |
| B·mix50 归档（带帆） | 1.142857 / 1.166667 | 0.042649 / 0.044485 / 0.045862 | 0.75 | 1.00 | 26.30 | 2.15 | HONEST_FAIL |
| C·裸艇 diag_bare | 1.142857 / 1.166667 | 0.042110 / 0.044002 / 0.045306 | 0.69 | 1.59 | 15.21 | 2.79 | HONEST_FAIL |

### ③.3 同 Re 口径逐点偏差（锚 C_T 六点 × 三档，OP0/OP1/OP2）

**A·E2 standard（带帆）**（Re_case=1000，仪器：standard）

| L | 锚点 | Re_anchor | C_T 锚 | Cd(OP1) | dev OP1 [%] | dev OP2 [%] | 算子散布 [pp] | |<15%| | 算子依赖 |
|---|---|---|---|---|---|---|---|---|---|
| 96 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00268379 | -14.59 | 1705.59 | 1720.17 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.0024579 | -19.45 | 1751.08 | 1770.53 | ✗ | OPERATOR_DEPENDENT |
| 96 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00239118 | -20.08 | 1785.52 | 1805.60 | ✗ | OPERATOR_DEPENDENT |
| 96 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00232976 | -20.64 | 1819.31 | 1839.96 | ✗ | OPERATOR_DEPENDENT |
| 96 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00227879 | -19.74 | 1882.61 | 1902.35 | ✗ | OPERATOR_DEPENDENT |
| 96 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00223716 | -17.55 | 1972.89 | 1990.45 | ✗ | OPERATOR_DEPENDENT |
| 112 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00283675 | -9.72 | 2149.45 | 2159.17 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00259799 | -14.86 | 2208.11 | 2222.97 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00252747 | -15.53 | 2251.65 | 2267.17 | ✗ | OPERATOR_DEPENDENT |
| 112 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00246255 | -16.12 | 2294.36 | 2310.48 | ✗ | OPERATOR_DEPENDENT |
| 112 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00240867 | -15.17 | 2373.81 | 2388.97 | ✗ | OPERATOR_DEPENDENT |
| 112 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00236467 | -12.85 | 2486.87 | 2499.73 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00294997 | -6.11 | 2753.00 | 2759.11 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00270168 | -11.47 | 2829.56 | 2841.03 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00262834 | -12.15 | 2885.47 | 2897.63 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00256083 | -12.77 | 2940.32 | 2953.09 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.0025048 | -11.78 | 3041.72 | 3053.50 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00245905 | -9.37 | 3185.77 | 3195.14 | ✓ | OPERATOR_DEPENDENT |

**B·mix50 归档（带帆）**（Re_case=1000，仪器：mix50）

| L | 锚点 | Re_anchor | C_T 锚 | Cd(OP1) | dev OP1 [%] | dev OP2 [%] | 算子散布 [pp] | |<15%| | 算子依赖 |
|---|---|---|---|---|---|---|---|---|---|
| 96 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00296756 | -5.55 | 1705.59 | 1711.14 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.0027178 | -10.94 | 1751.08 | 1762.02 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00264402 | -11.63 | 1785.52 | 1797.15 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.0025761 | -12.25 | 1819.31 | 1831.57 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00251974 | -11.25 | 1882.61 | 1893.86 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00247371 | -8.83 | 1972.89 | 1981.73 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00309533 | -1.49 | 2149.45 | 2150.94 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00283481 | -7.10 | 2208.11 | 2215.21 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00275785 | -7.82 | 2251.65 | 2259.47 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00268702 | -8.47 | 2294.36 | 2302.84 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00262823 | -7.43 | 2373.81 | 2381.24 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00258022 | -4.91 | 2486.87 | 2491.78 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00319116 | 1.56 | 2753.00 | 2751.44 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00292258 | -4.23 | 2829.56 | 2833.79 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00284324 | -4.97 | 2885.47 | 2890.45 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00277021 | -5.64 | 2940.32 | 2945.96 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.0027096 | -4.57 | 3041.72 | 3046.29 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00266011 | -1.97 | 3185.77 | 3187.74 | ✓ | OPERATOR_DEPENDENT |

**C·裸艇 diag_bare**（Re_case=1000，仪器：mix50）

| L | 锚点 | Re_anchor | C_T 锚 | Cd(OP1) | dev OP1 [%] | dev OP2 [%] | 算子散布 [pp] | |<15%| | 算子依赖 |
|---|---|---|---|---|---|---|---|---|---|
| 96 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.0029301 | -6.75 | 1257.22 | 1263.97 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00268348 | -12.06 | 1289.42 | 1301.48 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00261064 | -12.75 | 1314.66 | 1327.40 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00254358 | -13.36 | 1339.45 | 1352.81 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00248793 | -12.37 | 1386.43 | 1398.80 | ✓ | OPERATOR_DEPENDENT |
| 96 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00244248 | -9.99 | 1453.70 | 1463.68 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00306175 | -2.56 | 1542.21 | 1544.77 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00280406 | -8.11 | 1582.86 | 1590.97 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00272794 | -8.82 | 1613.95 | 1622.77 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.00265787 | -9.47 | 1644.46 | 1653.93 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00259972 | -8.44 | 1701.81 | 1710.25 | ✓ | OPERATOR_DEPENDENT |
| 112 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00255223 | -5.94 | 1783.71 | 1789.65 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P1 | 1.165e+07 | 0.00314207 | 0.00315244 | 0.33 | 1951.83 | 1951.50 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P2 | 1.967e+07 | 0.00305156 | 0.00288712 | -5.39 | 2004.63 | 2010.02 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P3 | 2.329e+07 | 0.00299197 | 0.00280874 | -6.12 | 2044.11 | 2050.24 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P4 | 2.739e+07 | 0.0029358 | 0.0027366 | -6.79 | 2082.86 | 2089.64 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P5 | 3.148e+07 | 0.00283928 | 0.00267672 | -5.73 | 2155.11 | 2160.84 | ✓ | OPERATOR_DEPENDENT |
| 128 | AFF1-P6 | 3.539e+07 | 0.00271342 | 0.00262783 | -3.15 | 2258.04 | 2261.19 | ✓ | OPERATOR_DEPENDENT |


---

## ④ 结果图

（图内标签一律 ASCII：本容器无 CJK 字体，中文会渲染成方框；中文说明放在图题中。）

![w28_fig1_anchor_and_ittc57.png](figures/w28_fig1_anchor_and_ittc57.png)

*图1 锚件 6 点 C_T(V) 与 ITTC-1957 C_F(Re)：独立复算与登记值逐点闭合*
![w28_fig2_transposition_vs_anchor.png](figures/w28_fig2_transposition_vs_anchor.png)

*图2 三档×三臂同 Re 口径换算（OP1）与锚 C_T ±15% 带：细化使偏差朝同一方向增大，且 OP2 相差数百个百分点（算子依赖）*
![w28_fig3_gci_and_order.png](figures/w28_fig3_gci_and_order.png)

*图3 三档 GCI 与观测收敛阶（三臂全报）：p≈1.0–1.6 ≪ 1.8，GCI 15–31% ≫ 5%，等加密比捷径会把 p 伪装成 >2*

---

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **W27 的教训已闭环**：锚件此前只在未合并 mission 分支上，main 工作树读不到 → 下一波锚门判
   NO-ANCHOR-ARTIFACT。本单把 blob 落到 main 的声明路径，并把 sha256、来源 ref/commit、剥离还原律全部写进
   工件，任何后续读者都能证明「落地的就是这个 blob」。
2. **分级不是修辞而是机检**：`anchor_grade` 由 (status, acquisition_status) 表推出，脚本在主文档 WITHHELD 时
   **拒绝**输出 A/B；守卫测试同时断言「A/B 不可能从这张表里生成」。
3. **等加密比捷径会骗人**：三臂的捷径式分别给出 p =
   2.25, 2.15, 2.79
   （>2，看起来「很好」），用真实 `r21≠r32` 解则是
   1.091, 0.999, 1.589。
   同一批数据两种处理，一个假过、一个真不过 —— 所以判决按真实加密比。
4. **GCI 标度不变**：OP0/OP1/OP2 三列的相对网格误差相同，不能靠线性换算「洗掉」离散误差。
5. **Re 换算本身不适定**：OP1 与 OP2 在同一点可差数百个百分点（见 ③.3 的算子散布列），
   说明「把 Re=1e3 的结果换算到 1.16e7 再和实验比」的结论**主要由换算假设决定，而不是由求解器决定**。
6. **几何不一致要披露**：任务点名的三档是**带帆**（AFF-3 类）构型，锚是**裸艇 AFF-1**；因此并列跑
   `diag_bare`（几何一致）三档。两条梯子仪器不同（mix50 vs standard），不能互相当重复实验。
7. **锚是二手**：即使某天达成一致，也只能说「与 Table 14 的转录一致」，不能说「复现了 DTRC 试验」。
8. **单例限制**：结论建立在「同一算例、同一协议、仅网格变密」的三元组 + 单 Re 上，未新增多工况系综。

---

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）

| 关键量 | 本文值 | 基准/阈值 | 偏差 | 是否 <3% | 备注 |
|---|---|---|---|---|---|
| 锚件几何 Cb = ∇/(L·πD²/4) | 0.708 m³ | 0.80（登记） | 0.24% | ✅ <3% |  |
| 锚件 L/D | 8.5748 | 8.57（登记） | 0.056% | ✅ <3% |  |
| 锚件湿面积（解析 0.31576L²） | 5.99146 m² | 5.998 m² | −0.109% | ✅ <3% |  |
| ITTC-57 已知答案 Cf(Re=1.2e+07) | 0.00290719 | 0.00291 | -0.096% | ✅ <3% |  |
| ITTC-57 已知答案 Cf(Re=2.0e+06) | 0.0040543 | 0.00405 | +0.106% | ✅ <3% |  |
| ITTC-57 C_F 六点独立复算 | max|Δ| = 0.00e+00 | 登记 C_F | 0.00e+00 (abs) | ✅ 远 <3%（≪1e-9 门槛） |  |
| Re / C_T 由 V、R_T、S、ν 反算 | max rel 0.00e+00 | 登记 Re / C_T | 0.00e+00 | ✅ <3% |  |
| LBM Cd → 锚 C_T（A·E2 standard（带帆），OP1 主算子） | max|dev| = 20.64% | 锚 C_T | 20.64% | ❌ 不达标 | OP2 同表 max|dev| = 3185.8% |
| LBM Cd → 锚 C_T（B·mix50 归档（带帆），OP1 主算子） | max|dev| = 12.25% | 锚 C_T | 12.25% | ❌ 不达标 | OP2 同表 max|dev| = 3185.8% |
| LBM Cd → 锚 C_T（C·裸艇 diag_bare，OP1 主算子） | max|dev| = 13.36% | 锚 C_T | 13.36% | ❌ 不达标 | OP2 同表 max|dev| = 2258.0% |
| 离散误差（A·E2 standard（带帆）） | GCI=30.58%, p=1.091 | GCI<5% 且 p≥1.8 | GCI 30.58% / p 1.091 | ❌ 不达标（honest FAIL） |  |
| 离散误差（B·mix50 归档（带帆）） | GCI=26.30%, p=0.999 | GCI<5% 且 p≥1.8 | GCI 26.30% / p 0.999 | ❌ 不达标（honest FAIL） |  |
| 离散误差（C·裸艇 diag_bare） | GCI=15.21%, p=1.589 | GCI<5% 且 p≥1.8 | GCI 15.21% / p 1.589 | ❌ 不达标（honest FAIL） |  |

**一句话**：算子层（几何门 / ITTC-57 复算 / Re·C_T 反算 / 已知答案）全部 **<3% 且可机检**（其中 ITTC-57 复算
残差 0.0e+00，远低于 1e-9 门槛）；物理层（LBM Cd vs 锚 C_T）与离散层
（三档 GCI/观测阶）**不达标、如实记 honest FAIL** —— 三档网格未进入渐近区
（p = 1.091, 0.999, 1.589 < 1.8，GCI =
30.6%, 26.3%, 15.2% > 5%），同 Re 换算被证明算子依赖。
`overall_verdict = HONEST_FAIL`，该 FAIL 写入工件并由 16 项守卫测试保护，**不会被后续轮悄悄覆盖**
（翻转 C4 判决会立刻挂掉 3 个测试，已实测）。

---

## ⑦ 溯源（命令 / 输入 / sha256 / 引句位置）

```
python papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py
python papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py --check
python -m pytest tests/test_w28_lbm_suboff_tau_anchor_landing.py -q
```

锚件 sha256 链：分支 `agent/m18960`@71c90fd13e591a2bec1a021b438c2556e54c36c4 → blob `b452bccd7d35ce1c294b159342b2d0e0fbb75c6fd5c574975df732abe7b54f88`
→（单点插入 3885 B，0 字节改动）→ 落地 `146ed124a961983fd6638c59fe39820d190011fec393b51389f968fd50ee11df`
→ 剥离 `b452bccd7d35ce1c294b159342b2d0e0fbb75c6fd5c574975df732abe7b54f88`（= 分支 blob ✔）。

锚件所引仓内档案的 sha256 在 main 工作树逐条复核（`landing.anchor_input_sha256_recheck`，全部 match =
True）。引句定位：Table 14 / 报告 p.23 行
`docs/suboff_experimental_resistance_validation_2026-08-01.md:23`；
ν 约定 `docs/IN12_S6_WALL_LADDER.md:35`。
