# 交差审计 · m15519 CASE-BACKWARD-STEP-LBM（TensorLBM）

**审计单**: m15924 · **被审单**: m15519（succeeded / auto-approve，PR #1831 merge 为 `0ac198e4`）
**被审 commit**: `b63efb0e` · **工作树**: `/root/tp-agent/work/m15519` · **判级**: **L2**（独立功能实现）

---

## 0. 处置（disposition）

**request changes → verdict L2 / 基准面 L4 WITHHELD**。

被审单的主线是**真代码 + 真跑 + 真失败登记**（这部分成立）：`run.py` 是真输入→计算→输出实现，
复跑确定性通过（row1 复现到 0 位差、shear 复现到 0.04%）；作者也主动把 Re=400 判为无效、
把原文不可达写成 `PRIMARY_PDF_UNREACHABLE`。**不构成造假**。

但两处**材料缺陷**使"Re=100/200 达标（<15%）"这一核心结论**不能按交付口径采信**：

1. **F1 雷诺口径矛盾（HIGH）**：交付表所依据的参考数值链
   `benchmarks/pending/bstep_3d/result_H40.json` 明确写着
   `"re_definition": "Re = U_max * step_h / nu (最大入口速度, 台阶高)"`，
   而 `run.py` 用 `Re = ū·H/ν`（H=2S，抛物线 U_max=1.5ū）⇒ 两者相差 **4/3 倍**：
   `Re_ref = 0.75·Re_sim`。要么继承来的 3.0/5.5/8.0 不是在它自己声明的口径下，要么
   逐行比较时的 Re 错了 25%——**两者必有一错，而交付从未察觉/披露**。
   X1/S 随 Re 单调增，因此 −10%/−24% 的系统性偏低**正是**在更低有效 Re 下该出现的方向。
2. **F2 基准裁判缺席（HIGH，结构性）**：`tensor_benchmark_cases` 中
   `anchor_slug='CASE-BACKWARD-STEP-LBM'` **0 行**（全库 25 个锚点无此锚）。
   指标定义/容差只存在于散文（`refs.json::tolerance_pct=15.0`），唯一机器裁判沉默 ⇒ **L4 按规则不予授予**。

## ① 案例介绍

后台阶（backward-facing step）层流一级再附长度 `X1/S`（台阶高 S 归一），Armaly et al.,
*J. Fluid Mech.* **127**(4):473–496 (1983)，ER=1.9423，Re 以入口通道水力直径定义。

| 量 | 被审单取值 |
|---|---|
| 几何 | S=m 格，入口高 m，出口 H=2m ⇒ ER=**2.0**（原文 1.9423，+2.9%，已披露） |
| 入口 | 充分发展抛物线，按离散行均值归一到 ū=u（0.1） |
| Re | `ū·H/ν`，H=D_h；跑 100 / 200 / 400 |
| ν, τ | ν=ūH/Re；τ=0.5+3ν ∈ [0.512, 0.596] |
| 域 | 下游 L/H=12（≈24S），上游 8 格，出口海绵层 σ=0.3 |
| 网格/时间 | S=8/16；12000–16000 步；末 30% 时间窗、步长 100 |

## ② 审计方法（不重做，只核验 + 独立复算）

1. **取三件套**：`run.py`(193 行)/`refs.json`/`runs/*.json`(13 件)/`results_table.json`/`docs/CASE_BACKWARD_STEP_LBM.md`/`figures/CASE_BACKWARD_STEP_LBM.png`。
2. **复跑 repro_cmd**（本审计执行）：`python3 benchmarks/case_pool/backward_step_armaly/run.py --re 100 --m 16 --steps 12000 --u 0.1` → rc=0, 71.9 s，
   `x1_over_s_row1` 复现 **0.000% 差**，`x1_over_s` 复现 0.038% 差 ⇒ **确定性成立**（`artifacts/m15924_audit/repro_check.json`）。
3. **程序化重算交付表**（`recompute.py` → `audit_matrix.json`）：从 6 份 raw JSON 重算 got/err%，与交付 `results_table.json` 逐行一致（`recompute_matches_delivered_got=true` ×6）⇒ **表内数字不是手抄**。
4. **独立量测算子探测**（`metric_sensitivity.py`，复用被审 `run.py` 的辅助函数与 TensorLBM 内核，未改被审树）：同一 LBM 场用 5 种算子求再附点。
   自校验：本探测的 K 行剪切算子 = 1.8835117488 vs `run.py` 落盘 1.8835117495（相对差 4e-10）⇒ 场是同一个场。
5. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BACKWARD-STEP-LBM'` → **0 行**。
6. **参考链溯源自查**：`refs.json::provenance` → `result_H40.json` / `result_re200_H49.json` / `result_re400_H49.json`，逐个读其 `reference` 段。

## ③ 计算结果（全部由 `audit_matrix.json` 程序化生成）

### 交付表逐行复核（ref 3.0/5.5/8.0，容差 ±15%）

| Re | S | steps | τ | ρ_max−ρ_min | got X1/S (row1) | err% row1 | 判定 | run.py 打印口径 shear | err% shear |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 16 | 12000 | 0.596 | 0.048 | 2.6932 | **−10.23** | ✅ | 1.8835 | −37.22 ✗ |
| 100 | 8 | 12000 | 0.548 | 0.065 | 2.4706 | −17.65 | ✗ | 1.2420 | −58.60 ✗ |
| 200 | 16 | 12000 | 0.548 | 0.028 | 4.6892 | **−14.74** | ✅（贴线） | 3.3240 | −39.56 ✗ |
| 200 | 8 | 14000 | 0.524 | 0.074 | 4.1976 | −23.68 | ✗ | 1.9372 | −64.78 ✗ |
| 400 | 16 | 12000 | 0.524 | **0.755** | 0.3430 | −95.71 | ✗ 无效 | 5.2890 | −33.89 |
| 400 | 8 | 16000 | 0.512 | NaN | NaN | NaN | ✗ | NaN | NaN |

`row1` PASS 2/6，`shear` PASS 0/6。

### 同一场的算子敏感性（本审计独立探测，S=16）

| 算子 | Re=100 值 | err% | Re=200 值 | err% |
|---|---|---|---|---|
| K 行最小二乘剪切（`run.py` docstring 声明的观测量、RESULT 行打印值） | 1.8835 | −37.2 | 3.3240 | −39.6 |
| row1（交付主口径；单帧瞬态场） | 2.6913 | −10.3 | 4.6819 | −14.9 |
| row2（时间平均场） | 2.6297 | −12.3 | 4.5827 | −16.7 |
| row3（时间平均场） | 2.5274 | −15.8 | 4.4139 | −19.7 |
| 4 行抛物线拟合壁面斜率 | 2.8280 | **−5.7** | 4.9004 | −10.9 |
| **算子带宽 / ref** | **31.5%** | | **28.7%** | |

**带宽 29–32% ≫ 容差 15%** ⇒ "PASS" 由算子选择决定，不由物理决定。

## ④ 图

- `fig1_operator_spread.png`：同一场 5 算子 vs ±15% 带 —— 绿色带内/带外一目了然（row1 与抛物线拟合在带内，剪切口径在带外）。
- `fig2_error_and_rho.png`：交付表 err%(row1 vs shear) 随 Re + 压缩性诊断柱（Re=400 ρ 摆幅 0.755 = 失效）。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

- **F1 雷诺口径矛盾**（HIGH）：见 §0；`refs.json` 自称 Armaly `ū·D_h/ν` 口径，却从声明 `U_max·S/ν` 口径的旧档取数。
- **F2 裁判缺席**（HIGH）：`CASE-BACKWARD-STEP-LBM` 无 `tensor_benchmark_cases` 行 ⇒ 无机器容差、无指标定义，L4 结构性不可授予（平台账本缺口，建议由算例池战役统一补登记）。
- **F3 观测量与"文档观测量"不一致**（HIGH→MEDIUM）：`run.py:14-16` docstring 与 `run.py:188` RESULT 行都是 K 行剪切（1.884），交付表却判 row1（2.693）——**复跑者按打印行对照交付表会差 42.9%**。且 row1 取的是**末帧瞬态场**（`run.py:162`），绕过了本单自己的时间平均机制；Re=400 的 0.343 正是单帧瞬态被台阶角伪信号俘获的产物。
- **F4 invariants 未断言**：`run.py` 无收敛判据（滑窗变化阈值）、无 ρ 摆幅门、无非有限值门；Re=400 因此静默产出"物理无效值"而不是中止。建议加 `rho_max-rho_min<2%`、`isfinite`、`u_max<0.15` 三道硬门。
- **F5 工件可追溯性**（MEDIUM）：`runs/*.json`（13 件）**未入库**（`git ls-tree b63efb0e` 只有 run.py/refs.json/docs/fig）；`results_table.json` **在仓库内没有任何生成脚本**（`grep -rn results_table --include=*.py` = 0），与交付自述"由脚本程序化注入"不符（数字本身可重算，故不算造假）；判分字段 `x1_over_s_row1` 因 numpy.float32 被 `default=str` 落成**字符串** `"2.6931608"`，下游做算术即崩（本审计探测首次读取就因此 TypeError）。
- **F6 继承链卫生**（LOW）：`result_re400_H49.json` 的 `source` 串复制自 Re=200（写的是"Re=200 X_r/h≈5.5"，而 `xr_h_ref=8.0`）⇒ 8.0 这个数在链上没有自洽出处。
- **F7 外部锚 WITHHELD**：Armaly 原文表离线不可得（作者已诚实登记并做了可复核替代：arXiv:2507.16509 p.2 机器核实 ER=1.942 / Re 口径 / 2D 有效至 Re≤400）。
- **局限**：本审计未取得 Armaly Table 1 原文，故"3.0/5.5/8.0 是否等于原表"仍为 WITHHELD；未重跑 Re≥200 的收敛研究（非本单范围）；`Pi_xy` 输运量算子首次穿越落在台阶角（0.0315S，两 Re 相同）= 角点伪信号，本审计据此**判该算子不可用**，不计入带宽统计。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 基准 | 交付 | 本审计核验 | 是否 <3% |
|---|---|---|---|---|
| Re=100 S=16 X1/S | 3.0（SECONDARY_IN_REPO，原文未核） | 2.6932 | 复现 2.6932（0 位差） | **否**（10.2%） |
| Re=200 S=16 X1/S | 5.5（同上） | 4.6892 | 复现 4.69 | **否**（14.7%） |
| Re=100 S=16 X1/S（物理更稳的抛物线拟合算子） | 3.0 | —（未报） | 2.8280 | **是**（5.7%） |
| 复跑确定性（本审计 vs 交付） | 交付落盘 | — | row1 0.000%，shear 0.038% | **是** |
| 雷诺口径一致性 | `Re_ref=U_max·S/ν`（继承档自述） | `Re=ū·H/ν` | 差 4/3 | **否**（25% 口径错位） |

**结论**：交付的"<15% 达标"在其自选算子下成立、在文档声明的算子和更稳的拟合算子下不成立（前者 −37%/−40%，后者 −5.7%/−10.9%）；**任何一项都未达 3% 论文级门槛**；且 F1 使"在标称 Re 下达标"这一说法本身未成立。故：**L2 成立，L4 WITHHELD**，建议返修项 = ①口径归一（重跑在 `U_max·S/ν` 或 Armaly `ū·D_h/ν` 其中一种口径下并与同口径参考比）②壁面剪切真判据（动量交换法）+ 收敛判据 + 三道硬门 ③登记 `tensor_benchmark_cases` 行让裁判上线 ④runs 工件与生成脚本入库。

---

**产物**：`artifacts/m15924_audit/{recompute.py,metric_sensitivity.py,figures.py,audit_matrix.json,repro_check.json,metric_re*_m16.json,re100_m16_rerun.json,fig1_operator_spread.png,fig2_error_and_rho.png}`；verdict SQL：`verdict_m15924.sql`。