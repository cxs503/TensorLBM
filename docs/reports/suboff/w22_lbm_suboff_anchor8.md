# W22 / m14504 — DTRC SUBOFF AFF-8 外部锚逐字段机读抄录 与 ITTC-1957 闭合检验

**任务**：`paper-exp-lbm-suboff-tau-w22-anchor8`（mission 14504，领域仓 TensorLBM，学科域：阻力）

**前身**：`paper-exp-lbm-suboff-tau-4`（零报告欠账，本单显式承接；本报告 §7 逐条对齐其判据）

**总判决**：**HONEST-FAIL** — 抄录与复算两条判据 PASS，闭合与 k 双口径两条判据 FAIL，按任务书要求如实落账，不改判据、不粉饰。

| 项 | 值 |
|---|---|
| 本单工件 | `benchmarks/verified/suboff_anchor_w22/anchor_transcription.json`、`benchmarks/verified/suboff_anchor_w22/ittc57_closure.json`、`benchmarks/verified/suboff_anchor_w22/figures/`（4 图 + manifest） |
| 抄录源 | 8 个工件，2342 个标量叶节点 |
| 抄录完整率 | 100.0000%（缺失/失配 0） |
| 生成时 git commit | `3619d287450a11dad98babbafbc1302e152e3918`（分支 `agent/m14504`） |
| 输入 sha256（抄录件） | `6f210931f13b5937c086571ac6fef9327f3aa2d59ad96d987f10deaaab97a488` |
| 盲评合规 | EVO-1 盲评期：本单未读取任何 task_key 以 evo1- 开头的报告或工件 |

## 1. 案例介绍（物理背景 / 几何 / 参数）

外部锚 `AN-SUBOFF`：DARPA SUBOFF（DTRC Model 4740）全附体 AFF-8 拖曳水池总阻力，属模型尺度深潜（无自由面兴波）工况；锚值只来自仓内已归档转录件，未凭记忆写数。

| 几何/口径 | 值 | 来源（指针） |
|---|---|---|
| 艇长 $L$ | 4.356 | `benchmarks/verified/suboff_anchor/anchor.json` → `/geometry/L_m` |
| 最大直径 $D$ | 0.508 | `benchmarks/verified/suboff_anchor/anchor.json` → `/geometry/D_m` |
| 湿表面积 $S$ | 5.998 | `benchmarks/verified/suboff_anchor/anchor.json` → `/geometry/wetted_surface_m2` |
| 排水体积 $\nabla$ | 0.708 | `benchmarks/verified/suboff_anchor/anchor.json` → `/geometry/displacement_volume_m3` |
| $C_T$ 定义 | CT = RT / (0.5 * rho * U^2 * S) —— 逐字口径：CT = RT / (0.5 · ρ ·… | `/conventions/CT_definition` |
| 锚最点 $Re$（$\nu$=1.139e-6 m²/s 声明假设） | 1.16644e+07 | `/anchors/0/Re_Lmax_nu1139` |
| 锚最点 $C_T$ | 0.00366690 | `/anchors/0/CT` |

内部被检对象：仓内案例 `suboff_sail_resistance`（AFF-3 with_sail，三档网格 L96/L112/L128）：

| 网格档 | 域（nx×ny×nz） | 步数 | 附体固体胞 | $C_T$（E2 主口径窗均值） | 归档 mix50 $C_T$ | 记录 $C_F$（本案层流参考） |
|---|---|---|---|---|---|---|
| L96 | 576×203×203 | 14400 | 48 | 0.038570 | 0.042649 | 0.031405 |
| L112 | 672×237×237 | 16800 | 70 | 0.040769 | 0.044485 | 0.034140 |
| L128 | 768×271×271 | 19200 | 126 | 0.042396 | 0.045862 | 0.035900 |

案例口径（逐条由工件注入，禁手抄）：

| 口径项 | 值 / 文本 | 指针 |
|---|---|---|
| 案例 $Re$ | 1000 | `benchmarks/verified/suboff_sail_resistance/tau_ladder_verdict_m298.json` → `/supplement_tau_re_coupling/production_point/re` |
| 案例参考式 | Blasius laminar flat plate Cf = 1.328/sqrt(Re), Re=1000 | `/reference/formula` |
| 观测定义 | Cd_total window mean over the last 1000 force samples (last 10000 steps), total-wetted-area normalization (frozen in prereg.md) | `/observable` |
| 案例自身记录的『不适用交叉校验』 | `ittc1957/darpa-aff8 turbulent` | `/reference/cross_checks_recorded_not_applicable` |
| 几何构型文本证据 | ``benchmarks/pending/suboff_sail_resistance`（AFF-3 with_sail，Re=1000）按严…` (`prereg.md:5`, sha256 `cfcb3f17e0d33119…`) | — |

→ 本案 $Re$ 比 AFF-8 锚 $Re$ 低 4 个数量级以上，且上游工件 `benchmarks/verified/suboff_anchor/anchor_points.json` 的 C4 已判『NOT REACHABLE at credible resolution』——**同 Re 直接对比在两个方向上都不可行**，这是本单闭合判据失败的物理前提，不是数值实现缺陷。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

### 2.1 锚件逐字段机读抄录

`benchmarks/verified/suboff_anchor_w22/build_anchor_transcription.py`：

- JSON 源：深度优先遍历全部标量叶节点（str/int/float/bool/null），指针为 **RFC-6901**（`~0`/`~1` 转义），空串=文档根；
- Markdown 源：正文/表格/代码块三类的**全部数值 token**（正则带左右标识符边界，避免把 `L96`/`AFF-8`/`m298`/`GPU5` 误判为数），指针为文档序`/md_tables/<t>/rows/<row_key>/<col>/tok<j>`、`/md_prose/tok<j>`、`/md_code_blocks/<k>/tok<j>`；
- 每个叶节点带 `source_path` + `source_sha256` + 逐字符定位（行/列/raw）；
- **回读**：JSON 侧用独立实现的指针解析回原文件比对 `type` 与 `repr`；Markdown 侧按记录的行列**重读源文件字符**再 `float()` 比对——不是把内存里同一对象写两遍；
- 自检：指针唯一性、叶数守恒、失配清单。

### 2.2 AN-ITTC57 独立复算

$C_F = 0.075/(\log_{10}Re-2)^2$，三路实现（`math.log10` / `math.log\math.log(10)` / 60 位 `Decimal.ln`），三路互差见工件；`Re=10^2` 处为极点，代码显式返回 `inf` 而非崩溃。

### 2.3 闭合结构式

$C_T = (1+k)\,C_F + C_W$，按四个口径算：

- **A（主口径）**：$C_F$ = ITTC-57 @ 案例 $Re$，$k$ = 记录的 AFF-8 形式因子，$C_W=0$；
- **B**：$C_F$ = 案例自身记录参考（Blasius @ 案例 $Re$），$k$ = 记录 AFF-8；
- **C**：$C_F$ = 案例参考，$k$ = 记录的 AFF-1 裸艇体值（with-sail 几何的下界）；
- **D（定义性，非证据）**：锚自分比 $C_T/C_F$ 在锚 $Re$ 上自洽——$k$ 本身由该比定义，残差恒零，只作诚实披露，不作证据。

$C_W$ 取零的依据（程序化扫描）：全部 2342 个抄录叶中匹配 `/c_w|/cw|/wave|/rw|/ct_wave` 的命中共 **0** 条，即没有任何工件承载兴波分量；案例自身的参考结构也只有单一摩擦项。此外把残差反解成『闭合所需 $C_W$』，其符号即判决。

### 2.4 k 的两口径

- 记录值：AFF-8 锚最点 $(1+k)=C_T/C^{ITTC}_{F}$，两次独立转录双录（S5 与 S6）；
- Prohaska：$C_T/C_F=(1+k)+c\,Fr^{n}$，$Fr=U/\sqrt{gL}$，$n=4$ 固定，对 AFF-8 阶梯 5 点做无权 OLS，截距即 $(1+k)$；
- 敏感性：最低 3 / 4 点窗口 + $n\in[3.0,5.0]$ 步长 0.05 扫描，全量披露。

### 2.5 预声明判据（动手前固定于脚本常量 `CRITERIA_PREDECLARED`）

| 判据 | 阈值 | 主口径 |
|---|---|---|
| C1 抄录完整率 | = 100.0% | 全源全叶 |
| C2 ITTC-57 复算 vs 记录 | < 1.0% | 锚件中带 ITTC-1957 记录的每一档 |
| C3 三档闭合残差 | < 3.0% | A: C_F = ITTC-1957 at the case Reynolds number, k = recorded AFF-8 form factor, C_W = 0 |
| C4 k 双口径差异 | < 5.0% | k (form factor = (1+k)-1) — the literal reading of the criterion |

Prohaska 主规则预声明：all transcribed AFF-8 ladder points (no outlier removal, no window selection), wave exponent fixed at n = 4 (Prohaska standard), unweighted OLS of C_T/C_F on Fr^n, intercept = 1+k；敏感性规则：lowest-Fr windows of 3 and 4 points and a free-exponent scan n in [3.0, 5.0] step 0.05 — reported in full, including the windows where the criterion would pass；thresholds are the task-statement values; they were not adjusted after any measured value was computed。

工具链：Python 3.11（`json`/`math`/`decimal`）、matplotlib 出图、pytest 验证；复现命令 `python3 benchmarks/verified/suboff_anchor_w22/build_report.py` → `python3 benchmarks/verified/suboff_anchor_w22/build_anchor_transcription.py` → `python3 benchmarks/verified/suboff_anchor_w22/build_ittc57_closure.py` → `python3 benchmarks/verified/suboff_anchor_w22/make_figures.py` → `pytest tests/test_suboff_anchor_w22.py`。

## 3. 计算结果（数字表，程序化注入）

### 3.1 抄录普查

| 源 | 类型 | 叶节点数 | 回读失配 | sha256（前 16） |
|---|---|---|---|---|
| `benchmarks/verified/suboff_sail_resistance/verdict.json` | json | 207 | 0 | `8407056397d402e8…` |
| `benchmarks/verified/suboff_sail_resistance/e2_verdict.json` | json | 60 | 0 | `12dd4838baebe0ef…` |
| `benchmarks/verified/suboff_sail_resistance/tau_ladder_report_m298.md` | md | 333 | 0 | `8a8a141fc4ba5852…` |
| `benchmarks/verified/suboff_sail_resistance/tau_ladder_verdict_m298.json` | json | 327 | 0 | `7bfcf64ec9fff0d9…` |
| `benchmarks/pending/suboff_highre_resistance/refs_lock.json` | json | 110 | 0 | `fe31e9dc414425fa…` |
| `benchmarks/verified/suboff_anchor/anchor.json` | json | 1021 | 0 | `61d7841a971926a5…` |
| `benchmarks/verified/suboff_sail_resistance/result.json` | json | 126 | 0 | `8a8cf4b96934d57d…` |
| `benchmarks/verified/suboff_sail_resistance/prereg.md` | md | 158 | 0 | `cfcb3f17e0d33119…` |
| **合计** | — | **2342** | **0** | — |

### 3.2 ITTC-1957 复算 vs 锚件记录（C2，逐档）

| 档（τ 臂） | $Re$ | 记录 $C_F^{{ITTC57}}$ | 独立复算 | 相对偏差 | <1%? | 记录指针 |
|---|---|---|---|---|---|---|
| A | 1200 | 0.06439801549 | 0.06439801549 | +0.000e+00% | 是 | `/supplement_tau_re_coupling/per_arm/A/ittc1957_cf_at_re` |
| B | 240 | 0.518809 | 0.5188136052 | +8.877e-04% | 是 | `/supplement_tau_re_coupling/per_arm/B/ittc1957_cf_at_re` |
| C | 120 | 11.963 | 11.96235261 | -5.412e-03% | 是 | `/supplement_tau_re_coupling/per_arm/C/ittc1957_cf_at_re` |
| D | 60 | 1.5239 | 1.523867802 | -2.113e-03% | 是 | `/supplement_tau_re_coupling/per_arm/D/ittc1957_cf_at_re` |
| E | 24 | 0.19524 | 0.1952422815 | +1.169e-03% | 是 | `/supplement_tau_re_coupling/per_arm/E/ittc1957_cf_at_re` |

最大绝对相对偏差 **5.412e-03%** < 1% → **C2 PASS**。（偏差全部来自锚件记录本身的 6 位有效数字舍入：A 档全精度为 0，B–E 档为摘要值。）

雷诺阶梯自洽：各臂 `Re·ν_lb` = A:4, B:4, C:4, D:4, E:4，案例点 `Re·ν` = 4 —— 同一 $u_{lb}L$ 冻结值，证明该阶梯**是雷诺阶梯而非等 Re 扫描**（与 m298 报告一致）。

Markdown 转录件与 JSON 工件同量对照（转录保真度正交检查）：最大相对偏差 2.406e-05%（6 位有效数字舍入量级）。

### 3.3 案例 Re 口径冲突（披露行 C2'，不参与 C2 判据）

| 量 | 值 |
|---|---|
| 案例 $Re$ | 1000 |
| ITTC-57 @ 案例 $Re$ | 0.075000 |
| 案例自身参考 $C_F$（层流 Blasius） | 0.041995047 |
| 相对偏差 | **+78.592%** |
| 案例参考式复算残差（1.328/√Re） | -1.65e-14% |

### 3.4 闭合检验（C3，逐档 × 逐口径）

| 网格档 | $C_T$ 实测 | 口径 A 预测 | A 残差 | 口径 B 残差 | 口径 C 残差 | A 所需 $C_W$ 才能闭合 |
|---|---|---|---|---|---|---|
| L96 | 0.038570 | 0.094141 | **-144.08%** | -36.67% | -17.08% | -0.055570 |
| L112 | 0.040769 | 0.094141 | **-130.91%** | -29.30% | -10.76% | -0.053372 |
| L128 | 0.042396 | 0.094141 | **-122.05%** | -24.33% | -6.51% | -0.051745 |

| 口径 | 定义 | 最大绝对残差 | 全部 <3%? |
|---|---|---|---|
| A（主） | $C_F$=ITTC-57@案例$Re$, $k$=记录 AFF-8, $C_W$=0 | 144.08% | 否 |
| B | $C_F$=案例参考, $k$=记录 AFF-8 | 36.67% | 否 |
| C | $C_F$=案例参考, $k$=记录 AFF-1 | 17.08% | 否 |
| D（定义性，非证据） | 锚自分比自洽 | 3.32e-10% | 定义使然 |

$C_T$ 口径敏感性（E2 窗均值 vs 归档 mix50）逐档残差：L96/e2_standard_window -144.08%；L96/archived_mix50 -120.74%；L112/e2_standard_window -130.91%；L112/archived_mix50 -111.62%；L128/e2_standard_window -122.05%；L128/archived_mix50 -105.27%。

口径 C 揭示的机理：把 AFF-8 的 $k$=0.2552 换成 AFF-1 裸艇体的记录值 $k$=0.0753 后，最细网格残差由 -24.33% 收到 -6.51% —— **形式因子的取值是差距主项**（附体/口径差异）；剩余项与 §3.6 的 $k_{eff}pprox+0.01$ 一致，即低 $Re$ 层流流动本身提供的形状阻力增量只有 1% 量级，远小于任何湍流口径的形式因子。

### 3.5 k 的双口径（C4）

| 量 | 值 | 说明 |
|---|---|---|
| 记录 $k$（AFF-8，$Re$=1.16644e+07） | 0.255211 | 主记录（15 位） |
| 双录值（独立第二转录） | 0.255211 | 相对差 +7.95e-10% |
| Prohaska 拟合 $(1+k)$ | 1.300114 ± 0.025073 | $n$=4，5 点 OLS，$R^2$=0.351 |
| Prohaska $k$ | 0.300114 | 截距 − 1 |
| 差异（$k$ 口径，主判据） | **+17.595%** | 阈值 5% → **FAIL** |
| 差异（$(1+k)$ 口径，披露） | +3.577% | 同数据、不同基，<5% |
| 点估计距记录值 | +1.79σ | 记录值落在拟合 ±2σ 带内 |

| 窗口敏感性 | 点数 | $(1+k)$ | $k$ | 差异（$k$ 基） | <5%? |
|---|---|---|---|---|---|
| lowest 3 Fr points | 3 | 1.264089 | 0.264089 | +3.479% | 是 |
| lowest 4 Fr points | 4 | 1.283110 | 0.283110 | +10.932% | 否 |
| 全 5 点（预声明主规则） | 5 | 1.300114 | 0.300114 | +17.595% | 否 |

自由指数扫描 $n\in[3.0,5.0]$：截距区间 [1.291026, 1.305677]，对应 $k$ 区间 [0.291026, 0.305677] —— 记录 $k$=0.255211 落在区间下方，即拟合主规则下两口径不可同化。
跨族交叉检（AFF-1 裸艇体 6 点，$R^2$=0.013）：$k_{Prohaska}$=0.126529 vs 记录 0.075283，差异 +68.07% —— 说明该 Prohaska 口径在仓内阶梯数据上本身可辨识性极弱（$R^2$ 接近 0），不是选点问题。

### 3.6 三档网格走向（诊断量）

| 网格档 | $C_T$（E2） | 隐含 $k_{eff}=C_T/C_F^{case}-1$ |
|---|---|---|
| L96 | 0.038570 | -0.0815 |
| L112 | 0.040769 | -0.0292 |
| L128 | 0.042396 | +0.0095 |

$k_{eff}$ 随网格加密从负数收敛到约 +0.01：本案的『形式因子』是本征的层流低 Re 量级，与 AFF-8 的湍流 0.255 完全不同量级——闭合失败的机理在数字上直接可见。

## 4. 结果图

![fig1](benchmarks/verified/suboff_anchor_w22/figures/fig1_closure.png)

**图 1**（`benchmarks/verified/suboff_anchor_w22/figures/fig1_closure.png`，sha256 `52e58a695a4f0e02…`）：Left: measured $C_T$ per grid vs the three $(1+k)C_F$ predictions. Right: the closure residual per grid for calibers A/B/C; every bar is far outside the ±3% gate.

![fig2](benchmarks/verified/suboff_anchor_w22/figures/fig2_prohaska.png)

**图 2**（`benchmarks/verified/suboff_anchor_w22/figures/fig2_prohaska.png`，sha256 `76fd9648433a2df5…`）：Left: Prohaska regression $C_T/C_F$ on $Fr^4$ over the five transcribed AFF-8 ladder points, with the recorded $(1+k)$ for comparison. Right: the per-point $k=C_T/C_F-1$ along the ladder, showing why the fit is weakly constrained ($R^2$ well below 1).

![fig3](benchmarks/verified/suboff_anchor_w22/figures/fig3_ittc57_domain.png)

**图 3**（`benchmarks/verified/suboff_anchor_w22/figures/fig3_ittc57_domain.png`，sha256 `fa06c2ae2819e61b…`）：AN-ITTC57 as a function of Re with the case point (Re=1e3), the AFF-8 anchor point (Re≈1.17e7) and the laminar case reference. The ITTC-57 line is a turbulent correlation: it has a pole at Re=1e2 and lies ~79% above the laminar reference at the case Reynolds number, which is the structural reason the closure cannot hold.

![fig4](benchmarks/verified/suboff_anchor_w22/figures/fig4_transcription_census.png)

**图 4**（`benchmarks/verified/suboff_anchor_w22/figures/fig4_transcription_census.png`，sha256 `12929d53636e372e…`）：Left: number of scalar leaves transcribed per source artifact. Right: readback mismatches per source — zero everywhere, i.e. every transcribed value resolves back to its source with identical type and value.

图中所有点数均由上述两件工件注入（见 `figures/manifest.json` 的 `data_evidence` 指针清单），无手工录入曲线。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **ITTC-1957 不是本案例 Re 段的物理参考**：它是湍流相关线，$Re=10^2$ 处有极点，$Re=10^3$ 处值恰为 0.075（=0.075/1），比本案层流参考高 78.6%。本单把它作为**抄录/复算保真度**的检验对象（C2），而非精度基准（C3 因此必然失败），并在 §3.3 显式披露该口径冲突。
2. **τ 阶梯就是雷诺阶梯**：$u_{lb}$、$L$ 被案例冻结，$\nu=(\tau-0.5)/3$ ⇒ $\tau\uparrow$ 则 $Re\downarrow$（本单用 `Re·ν_lb` 恒等式程序化验证，见 §3.2）。因此五档不是等 Re 重复，跨档比较必须带 Re 口径。
3. **k 在案例侧不可辨识**：三档网格同一 $Re$、同一 $Fr$，Prohaska 只能在锚阶梯上做；两口径不是同一物理 $k$ 的独立估计，这本身是判据 C4 的方法论上限。
4. **Prohaska 在仓内阶梯上统计力弱**：AFF-8 $R^2$=0.35、AFF-1 $R^2$=0.01，截距标准误 0.025；记录值落在 ±2σ 内（1.79σ），即点估计过不了 5% 门但统计上不可区分。本单按预声明主规则（全点、$n$=4）报 FAIL，并把窗口/指数敏感性全量列出，不择其利。
5. **锚 Re 不可达（上游结论，直接影响本单结论的适用范围）**：`benchmarks/verified/suboff_anchor/anchor_points.json` C4 判『NOT REACHABLE at credible resolution』（可信档 ~4276× 预算）。故『同 Re 闭合』在当前算力下无法证真；本单只证伪了跨 Re 的相似性换算式闭合。
6. **$C_W=0$ 是案例自身口径，不是本单加的假设**：抄录全量扫描无兴波字段命中；且残差反解出的所需 $C_W$ 三档全为负（L96: -0.055570、L112: -0.053372、L128: -0.051745），物理上非负的兴波阻力不可能闭合该恒等式 —— 闭合失败**不可能靠补一个兴波项救回**。
7. **锚是二手转录**：离线环境无外网，DTRC/ITTC 原始报告不在工作树内，全部外部锚来自仓内已归档文本（该限制由上游 `benchmarks/verified/suboff_anchor/anchor.json :: external_evidence.offline_disclosure` 承载）。
8. **Markdown 记录只有 6 位有效数字**：md 与 JSON 同量对照偏差 2.4e-5 量级，属舍入而非漂移；需要比特级复核时以 JSON 工件为准（本单已如此做）。
9. **案例几何是 AFF-3（with sail）**，锚记录只有 AFF-1 与 AFF-8 两个形式因子，故 0.0753 / 0.2552 构成 $k$ 的上下界，而非该几何的真值。

## 6. 误差判定（关键量 vs 基准，逐项给数与显式回答）

| 关键量 | 基准 | 误差 | 是否 <3%？ | 证据指针 |
|---|---|---|---|---|
| ITTC-57 $C_F$（复算） | 锚件记录 ITTC-57 $C_F$（5 档） | max 5.412e-03% | **是**（判据 1% 亦通过） | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/recompute_checks` |
| 案例参考 $C_F$ | Blasius $1.328/\sqrt{Re}$ | 1.65e-14% | **是** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/case/blasius_check` |
| $C_T$ 闭合（口径 A，L96） | $(1+k)C_F+C_W$ | 144.08% | **否** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/closure/caliber_A_primary/per_tier/0` |
| $C_T$ 闭合（口径 A，L112） | 同上 | 130.91% | **否** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/closure/caliber_A_primary/per_tier/1` |
| $C_T$ 闭合（口径 A，L128） | 同上 | 122.05% | **否** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/closure/caliber_A_primary/per_tier/2` |
| $C_T$ 闭合（口径 C，L128） | 用 AFF-1 $k$ 的最优情形 | 6.51% | **否** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/closure/caliber_C/per_tier/2` |
| $k$ 双口径（$k$ 基） | 记录 AFF-8 $k$ | 17.59% | **否** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/k_identifiability/prohaska` |
| $k$ 双口径（$(1+k)$ 基） | 记录 AFF-8 $(1+k)$ | 3.58% | **是**（披露口径） | 同上 |

**显式回答**：三档网格 $C_T$ 与 $(1+k)C_F+C_W$ 结构的闭合误差**不满足 <3%**（主口径 144.1% / 130.9% / 122.1%），且所需 $C_W$ 三档为负，**不是可修的数值误差，而是口径不可比**：ITTC-57（湍流线）+ AFF-8（$Re\approx1.17\times10^7$）的组合不能用来预测 $Re=10^3$ 的层流总阻力。

## 7. 判据逐条裁决表

| 判据 | 量值 | 阈值 | 裁决 | 证据指针 |
|---|---|---|---|---|
| **C1** 锚件抄录完整率 | 100.0000%（2342/2342 叶，回读失配 0） | 100% | **PASS** | `benchmarks/verified/suboff_anchor_w22/anchor_transcription.json` `/completeness` |
| **C2** ITTC-57 复算 vs 记录（逐档） | 最大偏差 5.412e-03% | <1% | **PASS** | `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` `/recompute_checks` |
| C2'(披露) 案例 Re 口径冲突 | +78.592% | <1%（不参与 C2 判定） | FAIL | `/case_caliber_gap` |
| **C3** 三档闭合残差 | -144.08% / -130.91% / -122.05% | <3% | **FAIL** | `/closure/caliber_A_primary` |
| **C4** k 双口径差异 | $k$ 基 17.59%（$(1+k)$ 基 3.58%） | <5% | **FAIL** | `/k_identifiability/prohaska` |

**总判决 = HONEST-FAIL**（C1/C2 PASS，C3/C4 FAIL；未改判据、未改阈值）。

### 7.1 与前身任务 `paper-exp-lbm-suboff-tau-4` 的判据对齐

前身零报告、零工件，其判据只能按本单任务书转述的版本逐条对齐（该继承事实如实披露）：

| 前身判据（任务书转述） | 本单回应 | 裁决 |
|---|---|---|
| ① 锚件抄录完整率 100%、缺失逐条列因 | §3.1 全源逐叶抄录 + 独立回读，缺失 0 条 | **PASS** |
| ② ITTC-57 $C_F$ 与记录逐档偏差 <1% | §3.2 五档全量对照 | **PASS** |
| ③ 三档 $C_T$ 闭合 <3% | §3.4 三档 × 四口径（含定义性口径披露） | **FAIL** |
| ④ k 双口径差异 <5% | §3.5 记录值 vs Prohaska，两基双披露 + 敏感性 | **FAIL** |

## 8. honest FAIL 说明（做了什么 / 没做到什么 / 不声称什么）

**已交付**：
- 锚件逐字段机读抄录（2342 叶，8 源，100% 完整率，独立回读 0 失配）
- ITTC-1957 三路实现 + 与仓内记录逐档比对（最大偏差 5.412e-03%，A 档全精度为 0，B–E 档受记录值 6 位有效数字舍入所限）
- 三档网格闭合检验（三个口径 + 一个定义性口径）
- k 两口径（记录值 / Prohaska 拟合）与窗口/指数敏感性
- C_W 的零值依据与『所需 C_W 为负』的结构性论证

**未达标（如实）**：
- C3 三档闭合残差全部远超 3%（主口径 122.05%–144.08%）
- C4 按 k 本身口径差异 17.59% > 5.0%

**不声称**：
- 未声称外部锚在 <3% 带内得到 LBM 验证
- 未声称 Prohaska 拟合给出与记录一致的 k（按主口径为否）
- 未把口径 D 的定义性零残差当作证据

**归因（逐条）**：
- **C3**：闭合在三档全部失败（主口径 A）：ITTC-57 在 Re=1e3 给出 C_F=0.075（湍流线），与本案层流参考 C_F=0.041995 相差 78.6%；预测 C_T 比实测高 144.1%（L96）到 122.1%（L128）。把 C_W 放开也无解：使恒等式闭合所需的 C_W 在三档均为负值（见 /closure/caliber_A/per_tier[*]/required_C_W_to_close），物理上非负的兴波阻力不可能闭合该恒等式。
- **C4**：Prohaska 截距被抬高，因为 AFF-8 阶梯覆盖 Fr = 0.467–1.259，Fr^4 兴波项在该区间并非单一幂律（高档已接近/越过峰阻区）；取最低 3 档时 k 口径差异降到 3.3%（PASS），全部 5 档时为 17.6%（FAIL）。窗口选择在 k 口径上翻转判据，故本单按预声明主规则（全部点）报 FAIL 并全量披露敏感性。
- **C2_vs_physical**：C2 通过（最大偏差 5.412e-03%，即记录值舍入量级）只证明公式实现与抄录正确，不证明 ITTC-57 在本案 Re 段适用；C2' 披露行给出 +78.6% 的口径冲突。

**要闭合需要什么**（下一班可选路线，均不改本单判据）：
1. **同 Re 侧**：在 $Re\sim10^6$ 量级、壁面分辨可接受的网格上直接模拟 AFF-8 外形——上游预算评估为可信档 ~4276× 预算，当前不可行；
2. **低 Re 侧**：把 $C_F$ 换成与案例同 Re 的层流参考（本案自身参考），并用**案例自身 Re 阶梯**做 Prohaska（需要至少 3 个不同 $u_{lb}$ 的同网格跑次，且 $\tau$ 保持固定），这正是 m298 报告 §6 的 R1 路线（解耦 $\tau$ 与 $Re$）；
3. **式子侧**：若坚持 ITTC-57 结构式，必须把对比点搬到 $Re\gtrsim10^5$，并同时给出相似性换算误差预算（上游已量化：最有利 $Re_s=10^6$ 仍有 10.87%）。

## 9. 溯源（工件路径 + commit + sha256 + 复现）

| 工件 | sha256 |
|---|---|
| `benchmarks/verified/suboff_anchor_w22/anchor_transcription.json` | `6f210931f13b5937c086571ac6fef9327f3aa2d59ad96d987f10deaaab97a488` |
| `benchmarks/verified/suboff_anchor_w22/ittc57_closure.json` | `8e617715ca0e623b0420bc0a4ef4079df2bfbd6d48cade4792d6d61ae771925c` |
| `benchmarks/verified/suboff_anchor_w22/figures/manifest.json` | `846fd7b3aab0573fb28cd35920f9ee0f73b62220ecb2326b004745f3cbb3b52e` |
| `papers/reports/w22_lbm_suboff_anchor8.md` | 本文件不自我记录 sha256（自指）；提交记录中的 commit 即其身份 |

| 输入源 | sha256 |
|---|---|
| `benchmarks/verified/suboff_sail_resistance/verdict.json` | `8407056397d402e8a8f58618720ed8c58095901068bbae995111106ae0cbc5d9` |
| `benchmarks/verified/suboff_sail_resistance/e2_verdict.json` | `12dd4838baebe0efe363fe5e9028d8dc234fdd04782c541926aa8490206dcf8a` |
| `benchmarks/verified/suboff_sail_resistance/tau_ladder_report_m298.md` | `8a8a141fc4ba585266bc33c9c4823e5cc48665812c6517f6718ce4223aaea60b` |
| `benchmarks/verified/suboff_sail_resistance/tau_ladder_verdict_m298.json` | `7bfcf64ec9fff0d90447e6227a7bac7f74cb422cc13c8c1003412a567e7f88e5` |
| `benchmarks/pending/suboff_highre_resistance/refs_lock.json` | `fe31e9dc414425fa929593444a2d82f57fbf25a9cec764938137c644b08afc3c` |
| `benchmarks/verified/suboff_anchor/anchor.json` | `61d7841a971926a57fa0ecaddb44b4d2dafe13e33d77f6736768b8b17f222e00` |
| `benchmarks/verified/suboff_sail_resistance/result.json` | `8a8cf4b96934d57d72d37a442fc4771d62e3bb550ffb1cea07e44939a6f155ef` |
| `benchmarks/verified/suboff_sail_resistance/prereg.md` | `cfcb3f17e0d33119992aa53da73e7b13424de62b0187eb1ef029e8df589a953d` |

git commit（工件写入时）：`3619d287450a11dad98babbafbc1302e152e3918`；分支 `agent/m14504`。

复现：

```bash
python3 benchmarks/verified/suboff_anchor_w22/build_anchor_transcription.py   # 抄录 + 回读
python3 benchmarks/verified/suboff_anchor_w22/build_ittc57_closure.py        # 复算 + 闭合 + Prohaska
python3 benchmarks/verified/suboff_anchor_w22/make_figures.py                 # 4 图 + manifest
python3 benchmarks/verified/suboff_anchor_w22/build_report.py                 # 本报告
python3 -m pytest tests/test_suboff_anchor_w22.py -q            # 13 项不变量
```

