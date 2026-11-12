# 交差审计：m15464 · CASE-RAYLEIGH-BENARD-RA1E4（自然对流方腔 Ra=1e4 · TensorLBM）

**审计单**: m15814（交差核验，非重做） · **被审单**: m15464（succeeded, accepted, PR #1821 merged） · **仓库**: TensorLBM
**审计者**: board-audit · **日期**: 2026-11-12 · **裁定级别**: **L2**
**一句话**: 工件真、复现位级一致、误差分析诚实（Nu FAIL 如实报），但**主判据 Nu 未过 5% 门（−9.10% @nx=48）**、且"u_max PASS"是极值挑选伪影（同一解 |min u_x| = +8.97% → FAIL）；无平台接入/无基准台账行 → 不达 L3/L4。

图：`figures/audit_m15814_ra1e4.png`（三面板：Nu vs 网格 + 两条 Richardson 外推；u 反对称性；左右壁热流与镜像不变量）

---

## ① 案例介绍（物理背景 / 几何 / 参数）

- **物理**：二维差分加热方腔自然对流（左热右冷、上下绝热、L/H=1），空气 Pr=0.71，Ra=1e4。
- **参照**：de Vahl Davis (1983) *Natural convection of air in a square cavity: a benchmark numerical solution*, Int. J. Numer. Methods Fluids **3**, 249–264：Nu=2.243、u_max/(α/L)=16.178、v_max/(α/L)=19.617。
- **锚点命名注意**：锚点 slug 写 `RAYLEIGH-BENARD`，而 ref 与实现都是 **de Vahl Davis 差分加热方腔**（不是底热顶冷的水平层 RB 构型）。被审单已在其文档 §① 显式澄清此混用（仓内 `docs/P67N01_HEAT_CONVECTION_BENCHMARKS.md` 有 11% 误差前科）——**澄清是加分项，但 slug 与物理的错配仍未在台账层修正**（F6）。
- **参数表**（从工件 JSON 读，非手抄）：τ=0.90 → ν=0.133333、α=ν/Pr=0.187793、τ_T=3α+0.5=**1.06338**、gβ=Ra·ν·α/H³（H=nx）、Ma(nx=64→32)≈0.16、float64 CPU。

## ② 怎么做的（审计方法与证据链）

1. 读被审工作树 `/root/tp-agent/work/m15464`，取其真实工件 `runs/case_rayleigh_benard_ra1e4/local_nx{24,32,48}_tau0.9.json`。
2. **独立重算**全部指标（不采信报告表格）：Nu 误差、u/v 极大值误差、左右壁通量不平衡、镜像（反对称）不变量。
3. **复现验证**：在本审计工作树内重跑 `run.py --nx 24 --tau 0.9 --steps 40000`（3 档中最省时的一档），与交付工件逐字段比对。
4. **回归验证**：`pytest tests/test_thermal.py tests/test_heat_convection_benchmarks.py tests/test_conjugate_thermal_ddf.py -q`。
5. **台账裁判核对**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-RAYLEIGH-BENARD-RA1E4'` → **0 行**（该表共 22 个 anchor，无本锚点）。
6. **合入状态核对**：`git ls-files benchmarks/case_rayleigh_benard_ra1e4` 在 main 上只有 6 个源码文件 + docs，**无 runs/*.json、无 figures/*.png**（`runs/` 被 .gitignore:108 忽略、png 属合并期工件滤除类）——即仓库账面上无原始工件，只有工作树里的。
7. **独立补充探针**：递交集群 `cj1185`（nx=64, τ=0.9, 160000 步）以定 Richardson 阶；审计窗口内仍在排队 → **未跑，不计入证据**（honest not_run）。

## ③ 计算结果（审计独立复算，全部来自工件 JSON）

| nx | Nu(grad2) | vs ref 2.243 | Nu_left | Nu_right | 左右不平衡 | max u_x | vs ref 16.178 | \|min u_x\| | vs ref | u 镜像差 | v_max | vs ref 19.617 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1.7193 | **−23.35%** | 1.7897 | 1.6490 | 8.53% | 16.935 | +4.68% | 19.013 | **+17.53%** | 12.27% | 20.691 | +5.48% |
| 32 | 1.8897 | **−15.75%** | 1.9448 | 1.8346 | 6.01% | 16.438 | +1.60% | 17.629 | **+8.97%** | 7.25% | 19.791 | +0.89% |
| 48 | 2.0389 | **−9.10%** | 2.0788 | 1.9991 | 3.99% | 16.102 | −0.47% | 16.662 | **+2.99%** | 3.48% | 19.320 | −1.51% |

- **主判据（预注册 <5%）**：Nu **三档全 FAIL**；最好档 nx=48 仍 −9.10%。
- **Richardson（32→48，两点）**：p=1 → 2.337（+4.21%），p=2 → 2.158（−3.77%），区间包含 ref。被审单的"区间包含 ref"陈述**数值正确**，但 p 未定 → 只是 bracket，不是收敛证明。
- **复现（审计实跑）**：nx=24 重跑结果与交付 JSON **逐字段全等**（Nu=1.719326、20 个采样点逐点 max|ΔNu|=0.0、elapsed 128.6s 亦同）→ 确定性复现成立。
- **回归**：53 passed in 93.63s（被审单自报 53 passed/92s，一致）。
- **状态**：三档均在 resid_u≈1e-15、resid_T≈5e-14 到达定常平台（Nu 漂移 <2e-12），"未稳态"不是误差来源——这点被审单判断正确。

## ④ 图

`figures/audit_m15814_ra1e4.png`：
(a) Nu vs nx（ref 线 + ±5% 带 + 两条 Richardson 外推），(b) max(u_x) 与 |min(u_x)| 双序列对 ref（暴露极值挑选），(c) 稳态守恒不变量违例幅度（Nu_left/Nu_right−1、镜像差）。
**缺场量云图**：库 `simulate_natural_convection` 不返回场量（被审单已如实登记为摩擦），审计侧同样无法补图；此处按实写明。

## ⑤ 注意点（审计发现，按严重度）

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **高（阻断 L4）** | 主判据 Nu 未过 5% 门：nx=48 −9.10%、nx=32 −15.75%、nx=24 −23.35%。被审单如实报 FAIL——**诚实无问题，但"基准验证"这一功能实质上未成立** | 工件 JSON `result.nu_grad2`；docs §④ |
| F2 | **中高** | 报告"u_max +1.61% PASS"是**极值挑选伪影**：同一档 |min u_x|=17.629 → **+8.97% FAIL**；理想解满足 u(x,y)=−u(L−x,L−y)，故镜像极值应等幅。工件里 `u_max_nd_neg` 字段早已存在却未进对照表 | nx32/nx48 JSON `u_max_nd_neg`；本审计表 ③ |
| F3 | **中** | 稳态**左/右壁热流不平衡 6.01%(nx32)/3.99%(nx48)**：定常解必须通量相等。这意味着 Nu 观测量本身带 ~±3% 口径不确定度，与被审单自称的墙位置不一致缺陷同源；"Nu 的 −9.10% 主要来自粗网格"因此**未被证明**（报告 §⑥ 有该推断，但两点 Richardson + 未定阶不足以支撑） | 工件 `nu_left_grad2/nu_right_grad2`；图 (c) |
| F4 | 中 | **平台基准台账缺行**：`tensor_benchmark_cases` 对本 anchor **0 行** → "唯一裁判"不存在，5% 门只写在单内文档里 | psql 直查（22 个 slug 全列） |
| F5 | 中 | **合入面缺工件**：main 上无 `runs/*.json`、无 `figures/*.png`（runs/ 被 .gitignore:108 忽略，png 属合并期工件类滤除）。后果：从 main 检出无法直接核对数字，只能靠重跑；而 `repro.sh` **只覆盖 nx=24/32，不覆盖头牌 nx=48**（nx=48 需手敲命令，798.5s；README 写"2×4.5min"与实际 4.3+2.1min 不符） | `git ls-files`；`repro.sh` |
| F6 | 中低 | **锚点 slug 与物理构型错配**（RAYLEIGH-BENARD vs de Vahl Davis 方腔）：文档澄清了，台账/锚点层未改，后续若有人按 RB 标度律读 Nu 会系统性误判 | docs §①；F4 表 |
| F7 | 中低 | **文档参数表错误**：docs §③ 写 τ_T=0.78213，工件实际 **1.06338**（=3α+0.5, α=0.187793）。若按 0.78213 复算 α 会得 0.094（Pr≈1.42），是会被下游直接抄错的口径错误 | docs §③ vs JSON `config.tau_T` |
| F8 | 低 | **平台报告 §⑥ 网格趋势标签错**："nx=24 1.7193 → nx=32 2.0389（+18.59%）"——2.0389 是 **nx=48** 的值（nx=32 为 1.8897）；紧随的 Richardson 实际用的是 32→48 对，标签与算式不自洽 | report id=14542 §⑥ |
| F9 | 低 | **无本算例级测试**：`tests/` 无 `test_case_rayleigh_benard*`；`tests/test_thermal.py:169` 只断言 `nu_grad2>0`、`u_max>1e-4`（`nu_grad2` 恒正，等于无断言）→ 判据没有测试守卫 | grep tests/ |
| F10 | 低（口径） | 被审单状态 **succeeded / accepted / fail_stage 空**，而其自身报告标题即写 "Nu FAIL"；平台质量审计自身亦标 `gate_valid=false, key_present=false`。任务书要求"失败=honest-FAIL 留 fail_stage"→ 记账面与实质不一致 | mission 15464 result json |

**过度声明清单（被审单）**：
- "速度幅值判据 PASS（<2%）"——仅对挑选出的正极值成立，镜像极值 +8.97% 不成立（F2）。
- "说明 −15.75% 主要来自粗网格离散误差而非物理错误"——未证明；左右壁通量不平衡提供了强替代解释（F3）。
- README "**状态：honest 部分达成**" 与 "**致：u_max/v_max 过 5% 门**" 表述偏乐观（同上）。
- 其余声明（三档稳态、复现、历史 SDAA 记录 0.14% 一致、库改动默认行为不变）经审计**属实**。

## ⑥ 误差判定（逐项，显式回答是否 <5% / <3%）

| 量 | ref | got(nx=48) | 相对误差 | <5%? | <3%? |
|---|---|---|---|---|---|
| Nu（grad2 平均） | 2.243 | 2.0389 | **−9.10%** | ✗ 否 | ✗ 否 |
| u_max（+ 极值，报告口径） | 16.178 | 16.102 | −0.47% | ✓ 是 | ✓ 是 |
| \|u\|_max（镜像极值，不变量口径） | 16.178 | 16.662 | **+2.99%** | ✓ 是 | ✓ 是（nx=32 档为 +8.97% ✗） |
| v_max | 19.617 | 19.320 | −1.51% | ✓ 是 | ✓ 是 |
| 左右壁热流不平衡（应为 0） | 0 | 3.99% | 3.99% | ✓ | ✗ 否 |

**结论**：关键量 Nu 误差 **−9.10%，不满足 <5% 门，也不满足 <3%**；仅速度幅值与 v_max 达标（且 u 需按镜像口径复核后 nx=32 档不达标）。

---

## 判级：L2（不是 L3/L4）

- **L2 成立**：真代码输入→计算→输出（`run.py` / `ladder.py` / `make_figures.py` / `publish_report.py` 非占位）；真实运行工件（3 JSON，含逐 2000 步 Nu 历史）；**审计位级复现**（nx=24 全字段一致）；入口存在。
- **L3 不成立**：交付物是**对既有公共 API 的算例外壳**，未接入任何生产/平台路径；库侧增量只有返回字典诊断键 + 可选提前终止参数（`stop_tol_nu` 默认 None，不改任何算子）；无平台基准台账行（F4）；无算例级测试守卫（F9）。
- **L4 不成立**：完整算例 + 误差分析 + 可复现的形式都在，但**基准判定实质失败**（Nu 三档全 FAIL，最好 −9.10%；F1），且"速度 PASS"经不变量复核后不稳健（F2）。按 board 惯例（m15457 同型：预声明判据 FAIL + 台账 0 行 → 不达 L4），本单封顶 L2。

## 建议（给第 2 轮接棒）

1. **先立不变量门再谈精度**：稳态 `|Nu_left/Nu_right−1| < 0.5%` 与 `|(u_min+u_max)/u_max| < 0.5%` 作为 pass/fail 前置门（当前 3.99% / 3.48% 即证温度壁节点 vs 动量 half-way 壁几何不一致）。
2. 修 `apply_temperature_boundaries` 的壁几何一致性（ghost 节点 / half-way 温度反反弹），`thermal_params` 的 H 与实际壁距同源。
3. nx=64/96 三点定阶（本审计已递交集群 `cj1185`=nx=64，排队未跑）；仅当 nx≥64 直接值进入 5% 门才算达标，**不接受只靠外推区间覆盖 ref**。
4. 台账：`tensor_benchmark_cases` 增本 anchor（metric=Nu, tolerance=0.05；同时登记 u/v 与两条不变量），并修正 slug 与物理构型的一致性。
5. 文档：修 docs §③ τ_T、report §⑥ 网格标签；`repro.sh` 覆盖 nx=48 并给出真实耗时。
6. 记账：本单实质为"部分达成"，应用 honest-FAIL/stage 语义记账，避免 succeeded 与结论矛盾（F10）。

## 未做（honest not_run）

- 未重跑 nx=32/48 全档（审计定位为核验；nx=24 位级复现已足以确立可复现性）。
- 集群 `cj1185`（nx=64 定阶探针）审计窗口内仍在排队（前方 369 单）→ 无结果，不据此下任何结论。
- 未改任何被审代码（交差核验单，非返修）。
- 未验证 SDAA/GPU 路径（本单一律 CPU float64）。