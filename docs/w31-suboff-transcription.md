# W31 · DTRC SUBOFF 外部锚两族转录冲突的第三方裁决

- 任务键 `paper-exp-lbm-suboff-tau-15`，mission `22289`，仓库 `TensorLBM`，分支 `agent/m22289`，git `d90b11fa43ac001e723d0ff3b18bb169c61c5544`
- 工件（本文件所有数字的唯一来源，禁手抄）：[裁决 JSON](../benchmarks/anchors/evidence/suboff_w31/w31_suboff_transcription_verdict.json)
- 构建器 `benchmarks/w31_suboff_transcription_verdict.py`；正文由 `--doc` 从工件注入；守卫测试 `tests/test_w31_suboff_verdict.py`
- 图：由 `python benchmarks/w31_suboff_transcription_verdict.py --figs` 从归档生成，输出在 `outputs/w31-suboff-transcription/figures/`
- 锚等级 `TRANSCRIBED-SECONDARY` / `C`（原始 DTRC 报告仍 WITHHELD；本工作树离线，无外网）

## ① 案例介绍

物理背景：DARPA SUBOFF Model 4740 拖曳水池阻力外部锚（Lmax = 4.356 m，参考湿面积 WSA = 5.998 m² 为转录值；ρ = 1000.0 kg/m³、ν = 1.139e-06 m²/s 为声明假设，非测量）。

上一单状态（原文保留，不改写）：W30 工件 `papers/artifacts/w30/lbm-suboff-tau/suboff_anchor_census.json` 的判据 `Q3_lbm_vs_anchor` 判为 `不达标`；其登记的竞争假设原文如下：

- rank 1：regime mismatch: Re_lattice = 1000 (laminar, Blasius reference) vs anchor Re = 1.2e7..3.5e7 (turbulent). The raw deviation is the Reynolds gap. —— 量化支持原文：raw dev +952..+1462% at Re gap 4.07 decades; after transferring the LBM form factor (1+k) onto the anchor-Re ITTC-57 line the residual shrinks to -32.0..-6.1%, i.e. the bulk of the deviation is the Reynolds number, not the scheme
- rank 1：resolution starvation of the wall model: y+_mean 1878..4660 sits far outside the wall-law validity band, so the delivered friction is the *resolved-Re* (1e5) friction, not the anchor-Re friction —— 量化支持原文：C_F_fric/ITTC(anchor Re) = 2.491..2.525 and nearly constant across the five anchors, i.e. a structural wall-chain level error, not a per-anchor drift; dev range +1296..+2963%
- rank 2：anchor itself is ambiguous: the two in-repo transcription families differ by 17.0%, larger than the 15% gate, so no simulation can be adjudicated to 15% against it —— 量化支持原文：ratio 1.1698 +- 0.0021

本件按同一分母口径（裸艇值）复算，逐点最大差 **+17.176%**，与 W30 的读数一致；本件补的是 W30 未做的那一步：引入第三族 F3（带构型标签）与第三文档 F4 作裁决方。

四个数据源（全部仓内文本，脚本正则直读，零手抄）：

| 源 | 文件 | 角色 | 条目数 |
|---|---|---|---|
| F1（族 A） | `docs/suboff_experimental_resistance_validation_2026-08-01.md` | Table 14 转录，含 AFF-1 与 AFF-8 两行 | 6 + 6 |
| F2（族 B） | `benchmarks/pending/suboff_highre_resistance/refs_lock.md` | 第三方论文 Table 3 的 EFD 列，未标构型 | 5 |
| F3（族 C·裁决方） | `src/tensorlbm/suboff_reference_data.py` | 仓内维度力注册表，**带构型标签** | 6 + 6 |
| F4（补充第三文档） | `benchmarks/pending/suboff_highre_resistance/refs_lock.md` | Flow360 DDES 页转录反推 EFD 力（derived） | 6 |

## ② 怎么做的

1. 预注册：决策 = 两个构型各绑定哪条转录；四个竞争假设 H1 构型口径错配 / H2 密度或尺度归一化 / H3 参考面积重标定 / H4 ΔC_F 是否计入；门槛先固定（15% 门、3% 带、速度配对容差 1.5%）。
2. 换算链：Re = U·Lmax/ν，C_T = R_T/(0.5ρU²WSA)，C_F 用 ITTC-1957 `0.075/(log10Re-2)²`；余下参数全部由 `refs_lock.json` 直读。
3. 逐点对表：五条候选序列（F1-AFF1、F1-AFF8、F2、F3-AFF1、F3-AFF8）在同一速度索引下求两两相对差。
4. 裁决投票：F2 的 EFD 列分别与 AFF-1 行、AFF-8 行、F3 两行、F4 派生序列比平均绝对差，取最小者所属构型。
5. 归因：每个冲突点给四类归因之一，并对另外三类给出**量化上界**后显式否证；换算因子由裁决方 F3 独立给出，不用观测比值反推。
6. 判别性检验：比值随速度（即随 Re）的散布用来区分 H1（构型/几何 → 比值恒定）与 H4（ΔC_F → 比值随 Re 变）。
7. 同 Re 门：把仓内唯一落在锚 Re 域的 LBM 分支（`benchmarks/pending/suboff_highre_resistance/formal/formal_verdict.json`）与裁决后锚逐点对齐。

## ③ 计算结果

### 3.1 四源逐点对表（力域 [N]）

| 对 | V(F1-AFF1) [kn] | V(F2) [kn] | R_T F1-AFF1 | R_T F2 | R_T F1-AFF8 | R_T F3-AFF8 | R_T F4(derived) |
|---|---|---|---|---|---|---|---|
| 1 | 5.920 | 5.929 | 87.40 | 102.30 | 102.30 | 102.30 | 102.55 |
| 2 | 10.000 | 9.999 | 242.20 | 283.80 | 283.80 | 283.80 | 286.11 |
| 3 | 11.840 | 11.857 | 332.90 | 389.20 | 389.20 | 389.20 | 397.06 |
| 4 | 13.920 | 13.918 | 451.50 | 526.60 | 526.60 | 526.60 | 535.85 |
| 5 | 16.000 | 15.998 | 576.90 | 675.60 | 675.60 | 675.60 | 680.00 |

关键三项的逐点相对差（完整 10 组两两矩阵在工件 `three_family_pointwise[*].pairwise_rel_diff_pct`）：

| 对 | 逐点 R_T 相对差 [%] | 最大绝对 [%] |
|---|---|---|
| F1_AFF1\|F2_paper_EFD | -14.565 / -14.658 / -14.466 / -14.261 / -14.609 | 14.658 |
| F1_AFF8\|F2_paper_EFD | +0.000 / +0.000 / +0.000 / +0.000 / +0.000 | 0.000 |
| F2_paper_EFD\|F3_AFF8 | +0.000 / +0.000 / +0.000 / +0.000 / +0.000 | 0.000 |
| F1_AFF1\|F3_AFF1 | +0.000 / +0.000 / +0.000 / +0.000 / +0.000 | 0.000 |
| F1_AFF8\|F3_AFF8 | +0.000 / +0.000 / +0.000 / +0.000 / +0.000 | 0.000 |

### 3.2 裁决：F2 属于 AFF-8 行

投票表（相对差以 F2 为分母：|a/F2 − 1|×100，与 3.3 表的‘分母 F1’列互为倒数关系）：

| 对照 | 平均绝对相对差 [%] | 最大绝对 [%] |
|---|---|---|
| F2 vs F1_AFF1 | 14.512 | 14.658 |
| F2 vs F1_AFF8 | 0.000 | 0.000 |
| F2 vs F3_AFF1 | 14.512 | 14.658 |
| F2 vs F3_AFF8 | 0.000 | 0.000 |
| F2 vs F4_flow360_derived_vs_AFF8 | 1.098 | 2.019 |
| F2 vs F4_flow360_derived_vs_AFF1 | 18.259 | 19.273 |

速度列指纹：F2 速度与 AFF-8 行最大差 0.063%，与 AFF-1 行最大差 0.147%（两者均远小于 1%，故速度列只作旁证，不作决定性证据）。

比值随速度的平坦性（判别 H1 与 H2/H4）：

- R_T(F2)/R_T(F1-AFF1) 观测均值 **1.1698**，sd 0.0021（相对散布 **0.183%**）
- 裁决方 F3 预测的 AFF-8/AFF-1 因子均值 1.1698（相对散布 0.183%），与观测均值差 **+0.000%**
- ΔC_F 路径（ITTC-78 模型尺度）要求的比值均值 2.0534、相对散布 3.182% → 与观测的平坦性不相容，H4 被否证
- **独立性声明**：F1 与 F3 的 AFF-8 行数值逐一相同，故 arbiter_mean 与 observed_mean 精确相等；这是同一条 Table 14 转录链的自洽性，**不构成独立测量**。真正的独立性来自 F4（另一份公开文档的反推力，对 AFF-8 平均差 1.10%、对 AFF-1 平均差 18.26%）

### 3.3 冲突点归因表（每点四类取一，未归因 = 0）

| 冲突点 | F1-AFF1 R_T | F2 R_T | 原始差(分母 F1) [%] | 原始差(分母 F2) [%] | 原始差(\|a-b\|/max) [%] | 归因类 | 换算因子(F3) | 换算后差 [%] |
|---|---|---|---|---|---|---|---|---|
| WC1 | 87.40 N @ 5.920 kn | 102.30 N @ 5.929 kn | +17.048 | -14.565 | 14.565 | `scale` / SCALE_REFERENCE_BODY | 1.17048 | +0.0000 |
| WC2 | 242.20 N @ 10.000 kn | 283.80 N @ 9.999 kn | +17.176 | -14.658 | 14.658 | `scale` / SCALE_REFERENCE_BODY | 1.17176 | +0.0000 |
| WC3 | 332.90 N @ 11.840 kn | 389.20 N @ 11.857 kn | +16.912 | -14.466 | 14.466 | `scale` / SCALE_REFERENCE_BODY | 1.16912 | +0.0000 |
| WC4 | 451.50 N @ 13.920 kn | 526.60 N @ 13.918 kn | +16.633 | -14.261 | 14.261 | `scale` / SCALE_REFERENCE_BODY | 1.16633 | +0.0000 |
| WC5 | 576.90 N @ 16.000 kn | 675.60 N @ 15.998 kn | +17.109 | -14.609 | 14.609 | `scale` / SCALE_REFERENCE_BODY | 1.17109 | +0.0000 |

> 分母约定披露：W30 报的 17% 是以裸艇值为分母（+17.05%）；若以 F2 的实验参考值为分母则为 −14.58%，已落到 15% 门内。**门的原始读法依赖分母约定**，因此本单不以原始读法销项，而以与分母约定无关的口径换算残差（0.0000%）销项。两个原始数字均保留在工件里，不美化。

四类归因的量化否证（以 WC1 为例，逐点数值在工件）：

- `unit`（单位）：观测比值 1.1705；已知单位因子 lbf→N 4.448222、kgf→N 9.806650 均不匹配；速度配对差 0.147% 经 R_T∝V^1.9 只解释 0.280% → 否证。
- `wetted_area`（湿面积定义）：力域比较中参考面积整体约去（解释力 0%）；即便按附体 own 面积占比 2.6% 折算也只能解释 2.6% → 否证。
- `delta_cf`（ΔC_F 是否计入）：模型尺度 ITTC-78 ΔC_F = 2.7760e-03（k_s = 150 µm，L = 4.356 m），相对 C_F 达 **95.0%**；要闭合观测所需的 ΔC_F 在 4.2193e-04–4.9816e-04 间变化（随 Re 变 18.1%），与观测比值的平坦性矛盾 → 否证。
- `scale`（尺度/口径，参考体）：换算因子由裁决方 F3 逐点独立给出，换算后最大残差 0.0000% → **接受**。

### 3.4 同 Re 的 LBM 分支 vs 裁决后锚

分支 `benchmarks/pending/suboff_highre_resistance/formal/formal_verdict.json`（AFF-8 全附体壁模型 LES(pending 层)，与 AFF-8 锚同一 Re 域；分支自评 `FAIL`，M1 规则 `|CT_sim/CT_EFD - 1| <= 5% at >= 4/5 anchors` 通过 0/5）。

| tag | Re | C_T LBM | C_T 裁决后锚(AFF-8) | 偏差 [%] | <15%? | <3%? | y+ | 奇偶 cv | 观察者差 [%] |
|---|---|---|---|---|---|---|---|---|---|
| r1_slip_a1 | 1.1664e+07 | 0.051173 | 0.003665 | +1296.1 | FAIL | FAIL | 1878 | 0.326 | 59.0 |
| r2_slip_a2 | 1.9673e+07 | 0.075394 | 0.003576 | +2008.5 | FAIL | FAIL | 3031 | 0.234 | 73.2 |
| r3_slip_a3 | 2.3329e+07 | 0.061713 | 0.003492 | +1667.2 | FAIL | FAIL | 3519 | 0.696 | 65.7 |
| r4_slip_a4 | 2.7383e+07 | 0.075225 | 0.003424 | +2096.9 | FAIL | FAIL | 4080 | 0.604 | 72.0 |
| r5_slip_a5 | 3.1475e+07 | 0.101863 | 0.003325 | +2963.5 | FAIL | FAIL | 4660 | 0.511 | 82.4 |

C3 读数：最大 |ΔC_T| = **2963.5%**；是否 <15%：否；**是否 <3%：否**。

诊断（同一 Re 下差 2 个量级的原因）：(a) 摩阻系数在锚 Re 上是 ITTC-57 的 2.51 倍，而按网格可分辨 Re 折算后为 0.797 倍；(b) 由 C_f 反解的等效 Re 均值 2.373e+05，比锚 Re 均值 2.270e+07 低约 2.0 个量级；(c) 奇偶时间步相对 C_T 变异 0.23-0.70，观察者差 59.0-82.4% -> 统计未收敛。因此该分支 admissible_for_anchor=False，不能用于 15% 或 3% 声明。

锚约定核对：该分支工件自记的 `CT_EFD` 与本次裁决后锚的最大相对差 1.26e-01%（逐点见 `lbm_same_re_vs_arbitrated_anchor.rows[*].CT_EFD_vs_arbitrated_rel_diff_pct`），说明该分支本来就用 AFF-8 锚 —— 裁决不改变 C3 的结论。

另：benchmarks/verified/suboff_sail_resistance (laminar, Re_lattice=1000) —— 与锚 Re(~1.2e7-3.1e7)相差约 4 个量级，不同 Re 下不能用于 15%/3% 判定

### 3.5 对上单假设的裁决（含被否证项）

| 假设 | 状态 | 机制 | 证据键 |
|---|---|---|---|
| W30 Q3 root_cause_hypotheses rank 2 (anchor itself is ambiguous ... 17.0% > 15% gate) | **REFUTED** | 两族差不是锚的固有歧义，而是构型配对错误：F2 的 EFD 列逐字等于 Table 14 的 AFF-8 行(F1/F3 两处独立转录)，按构型配对后残差 0.0%，锚对 AFF-8 与 AFF-1 各自都是确定量的 | arbitration.votes, conflict_attribution[*].post_conversion_rel_diff_pct, criteria[C1].measured.post_conversion_max_rel_diff_pct |
| W30 Q3 rank 2 的支持论据: appendages alone explain only ~2.6% (in-repo appendage share) | **REFUTED** | 该 2.6% 是附体 own 湿面积份额(面积口径)，不是阻力份额；Table 14 自身给出的 AFF-8/AFF-1 阻力比即为观测比值，故 17% 与构型差完全相容。附体增阻的摩阻/形状/干扰分解仍 WITHHELD | arbitration.ratio_stats, conflict_attribution[0].alternatives_rejected.wetted_area |
| H4 delta_cf (ITTC-78 roughness allowance) credited as the 17% | **REFUTED** | 模型尺度 ITTC-78 dCF 相对 C_F 达 95-112%，量级过大且随 Re 变 18%；观测比值相对散布仅 0.18% 且与速度无关 | conflict_attribution[*].alternatives_rejected.delta_cf, arbitration.ratio_stats.delta_cf_implied_sd_rel_pct |
| H1 configuration caliber mismatch (F2 = AFF-8 row) | **CONFIRMED** | F2 与 AFF-8 行逐点差 0.000%(两处独立转录)；第三文档 F4 反推力对 AFF-8 平均差 1.10%、对 AFF-1 平均差 18.26% | arbitration.votes, conflict_attribution[*].conversion |

## ④ 判据逐条读数

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| C1 | 三条转录两两逐点相对差 <=15%，或显式归类为口径差并给出量化换算(换算后 <=15%) | 原始差(分母=裸艇) +17.176%；原始差(分母=F2 参考值) 14.658%；原始差(\|a-b\|/max) 14.658%；换算后残差 2.22e-14%；门 15%；三对族对中冲突 1 对 | **PASS** |
| C2 | 每条冲突点给出单位/尺度/湿面积/dC_F 四类归因之一，未归因点计数为 0 | 冲突点 5；未归因 0；被量化并否证的替代归因 15 条 | **PASS** |
| C3 | LBM 分支与裁决后锚在同一 Re 下 \|dC_T\| <= 15%，并回答是否 <3% | max\|dC_T\| = 2963.5%；<15%: False；<3%: False；对齐锚数 5 | **honest FAIL** |
| C4 | 全部判据数字由脚本从锚件直读(零手抄)，pytest 全绿 | 哈希源 7 个；手抄数值字面量 0；正文由工件注入 `docs/w31-suboff-transcription.md`；测试 `tests/test_w31_suboff_verdict.py` | **PASS** |

- **C1** PASS：三口径下的原始两族差：以裸艇值为分母 +17.176%(>15% 门，保留不美化)、以 F2 参考值为分母 14.658%(<15%)、|a-b|/max 14.658%(<15%) —— 原始读法本身依赖分母约定，故不用原始读法销项；口径换算后最大残差 0.000%(与分母约定无关)<= 15% 门 -> 走判据的『口径差 + 量化换算』分支通过
- **C2** PASS：5 点全部归入 scale(SCALE_REFERENCE_BODY)；unit / wetted_area / delta_cf 三类均给量化上界并显式否证(见 conflict_attribution[*].alternatives_rejected)
- **C3** honest FAIL：同 Re 下最大 |dC_T| = 2963.5%(原始数字保留)远超 15% 门；是否 <3%：否(差 2 个量级以上)。原因见 lbm_same_re_vs_arbitrated_anchor.diagnosis
- **C4** PASS：族数据全部由正则从仓内文件解析；正文与判据表由 --doc 从工件注入；pytest 复核哈希/判据/可重建性

## ⑤ 注意点 / 局限

- 未证明 Table 14 的物理正确性(原始报告仍 WITHHELD，无法读原表)
- F3 与 F1 同属 Table 14 转录链，二者数值一致只证明转录链自洽，不构成独立测量
- 附体增阻 17% 的物理分解(摩阻 vs 形状/干扰)仍不可核，标 WITHHELD
- 图 `w31_ratio_vs_speed.png`（左：四源 R_T(V) 与裁决后配对；右：观测比值 vs ΔC_F 路径要求值）；`w31_lbm_same_re.png`（左：C_T(Re) 对照；右：|ΔC_T| 与 15%/3% 门）。

## ⑥ 误差判定（逐项，含是否 <3%）

| 量 | 对照 | 误差 | <15% | <3% |
|---|---|---|---|---|
| F2 vs 裁决后 AFF-8 锚 | F3 registry AFF-8 行 | 最大 0.000% | 是 | 是 |
| F2 vs F1 AFF-8 行 | 同文件另一构型行 | 最大 0.000% | 是 | 是 |
| F4(Flow360 派生) vs AFF-8 | 独立文档 | 最大 2.019% | 是 | 是 |
| LBM C_T vs 裁决后锚（同 Re） | AFF-8 锚 | 最大 2963.5% | 否 | 否 |

结论：**外部锚的转录层已裁清**（口径 = 构型 AFF-8/AFF-1；换算后残差 ≤ 0.0000%，远小于 3%）；**LBM 同 Re 层仍是 honest FAIL**（最大 2963.5%，非 <3%），原因是分支等效 Re 偏低与统计未收敛，不是锚的问题。

## 复现

```bash
python benchmarks/w31_suboff_transcription_verdict.py --all
pytest tests/test_w31_suboff_verdict.py -q
```
