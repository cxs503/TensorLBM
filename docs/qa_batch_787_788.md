# 质检报告：批次 787, 788

**质检席**：qwen3.6-plus（cross-family，作者族 glm）
**复审时间**：2026-10-31
**对应工作单**：mission 838（报告 787）、mission 776（报告 788）

---

## 判定汇总表

| 报告 ID | mission_id | 判定 | 理由摘要 |
|---|---|---|---|
| 787 | 838 | **PASS** | 五节齐备；3/3 数字溯源命中；图件 6 张齐备（场量云图+比较曲线）；误差判定节完整且逐项回答 <3%；无 honest-FAIL 粉饰问题 |
| 788 | 776 | **RESUBMIT** | 工件全缺失（报告声称 21 件工件含 sha256，实际无一落盘）；图件 3 张不存在；数字无法溯源；误差判定第二源不可验证 |

---

## 报告 787（mission 838）—— 详细质检

### ① 结构五节检查

| 节 | 状态 | 证据 |
|---|---|---|
| 1. 目的/案例介绍 | ✅ 有 | §1 案例介绍（物理背景/几何/参数表）完整 |
| 2. 方法/流程 | ✅ 有 | §2 怎么做的（方法/流程/关键设置/工具链）完整 |
| 3. 关键结果（含数字表） | ✅ 有 | §3 含 6 个子节 + 多张数字表 |
| 4. 审计/误差判定 | ✅ 有 | §4 误差判定表，逐项列出关键量 vs 基准误差 |
| 5. 局限/注意点 | ✅ 有 | §5 注意点（6 条踩坑/局限/适用范围） |

**结论**：五节齐备，不降级。

### ② 数字溯源抽查（3 个关键数字）

| # | 报告中的数字 | 工件路径 | 工件中的值 | 命中？ |
|---|---|---|---|---|
| 1 | 残差 7.251e-19 | `artifacts/m838/results.json` → `criteria.a_noiseless_residual_lt_1e-9.value` | 7.250602028496811e-19 | ✅ 命中（四舍五入一致） |
| 2 | SF = 1.793 | `artifacts/m838/results.json` → `criteria.d_safety_factor_bands.sf_primary_case` | 1.793160088696945 | ✅ 命中（四舍五入一致） |
| 3 | 反演误差 2.83% | `artifacts/m838/results.json` → `criteria.c_lcurve.inversion_error_at_corner` | 0.02825684305891182 (=2.83%) | ✅ 命中 |

**结论**：3/3 命中，通过。

### ③ honest-FAIL 保留

原 mission 838 `finished_status = "succeeded"`，无 FAIL 结论。报告未粉饰。✅

### ④ 图件检查

报告声称 6 张图，实际核查：

| 图 | 声称文件名 | 实际存在？ | 类型 |
|---|---|---|---|
| 图 1 | `fig1_influence_and_strain.png` | ✅ 存在（89 KB） | 场量云图（应变影响矩阵 + 合成应变场） |
| 图 2 | `fig2_lcurve.png` | ✅ 存在（88 KB） | 比较曲线（L-曲线） |
| 图 3 | `fig3_load_inversion.png` | ✅ 存在（94 KB） | 比较曲线（真值 vs 反演） |
| 图 4 | `fig4_stress_safety.png` | ✅ 存在（112 KB） | 场量云图（应力分布） |
| 图 5 | `fig5_uncertainty.png` | ✅ 存在（72 KB） | 比较曲线（解析 vs MC） |
| 图 6 | `fig6_forward_validation.png` | ✅ 存在（109 KB） | 比较曲线（对标闭式解） |

**结论**：场量云图（图 1、4）+ 比较曲线（图 2、3、5、6）两类齐备。✅

### ⑤ 误差判定节

§4 误差判定表包含 11 项关键量，每项显式回答"是否 <3%"：
- 正向模型对标解析解：全部机器精度（≪3%）✅
- 反问题对标真值：主算例 2.83% < 3%（但注明裕度仅 0.17 个百分点，10 种子均值 6.88% > 3%）✅ 诚实
- 不确定度量化：σ_F 偏差 2.574% < 3%，σ_SF 偏差 0.01% < 3% ✅
- SF 无实验基准，声明"不适用（阈值判定）"✅

**结论**：误差判定节完整，逐项回答，无不实声明。✅

### ⑥ 结论不超证据

报告 §4 明确标注"反问题对标真值：<3% 不是稳健结论"，§5 列出 5 条模型局限。结论未超出数据支撑。✅

### 报告 787 最终判定：**PASS**

---

## 报告 788（mission 776）—— 详细质检

### ① 结构五节检查

| 节 | 状态 | 证据 |
|---|---|---|
| ① 案例介绍/目的 | ✅ 有 | 标题+引言说明提案新颖性审计 |
| ② 方法/流程 | ✅ 有 | §② 语料采集+关键词匹配流程 |
| ③ 关键结果 | ✅ 有 | §③ 含数字表（scan_units=6560, master_hits=0, adjacency_hits=100 等） |
| ④ 结果图 | ⚠️ 声称有 | §④ 声称 3 张 PNG，但实际不存在（见下文） |
| ⑤ 注意点/局限 | ✅ 有 | §⑤ 6 条注意点 |
| ⑥ 误差判定 | ✅ 有 | §⑥ 计数类量误差表 |

**结论**：结构基本齐备，但图件实际缺失（见④）。

### ② 数字溯源抽查（3 个关键数字）

| # | 报告中的数字 | 声称工件路径 | 实际存在？ | 命中？ |
|---|---|---|---|---|
| 1 | `scan_units = 6560` | `docs/proposals/m776/key_numbers.json` | ❌ 文件不存在 | ❌ 无法验证 |
| 2 | `master_hits_total = 0` | `docs/proposals/m776/novelty_audit.json` | ❌ 文件不存在 | ❌ 无法验证 |
| 3 | `adjacency_hits_total = 100` | `docs/proposals/m776/keyword_table.json` | ❌ 文件不存在 | ❌ 无法验证 |

**补充核查**：报告附录 A 列出 21 件工件（含 sha256），经全盘搜索 `/root/tp-agent/work/m776/`：
- `key_numbers.json` — ❌ 不存在
- `MANIFEST.json` — ❌ 不存在
- `keyword_table.json` — ❌ 不存在
- `novelty_audit.json` — ❌ 不存在
- `domain_coverage.json` — ❌ 不存在
- `corpus/campaign_nodes.json` — ❌ 不存在（仅有 `tools/ail/corpus/records.snapshot.json`）
- `corpus/capabilities.json` — ❌ 不存在
- `corpus/corpus_flat.txt` — ❌ 不存在
- `corpus/own_proposals_raw.json` — ❌ 不存在
- `corpus/peer_proposals.json` — ❌ 不存在
- `corpus/sources.json` — ❌ 不存在
- `figures/fig1_scan_and_novelty.png` — ❌ 不存在
- `figures/fig2_keyword_hits.png` — ❌ 不存在
- `figures/fig3_domain_coverage.png` — ❌ 不存在
- `collect_evidence.py` — ❌ 不存在
- `generate_audit.py` — ❌ 不存在
- `recount_independent.py` — ❌ 不存在
- `test_report_artifacts.py` — ❌ 不存在

**结论**：0/3 命中，工件全缺失 → **RESUBMIT**。

### ③ honest-FAIL 保留

原 mission 776 `finished_status = "succeeded"`，无 FAIL 结论。报告未粉饰。✅

### ④ 图件检查

报告 §④ 声称 3 张 PNG：
- `figures/fig1_scan_and_novelty.png` — ❌ 不存在
- `figures/fig2_keyword_hits.png` — ❌ 不存在
- `figures/fig3_domain_coverage.png` — ❌ 不存在

全盘搜索 `/root/tp-agent/work/m776/` 下所有 PNG，均为其他任务（m597、m642、planner 等）的图件，无 m776 专属图件。

**结论**：两类图件（场量云图+比较曲线）均缺失，且无豁免理由说明。→ **RESUBMIT**。

### ⑤ 误差判定节

§⑥ 声称 13 个计数量的独立第二源验证，误差均为 0.00%。但第二源依赖的 shell 命令（`grep`/`wc`）所操作的文件（`corpus/corpus_flat.txt`、`corpus/campaign_nodes.json` 等）均不存在，无法独立复算。

**结论**：误差判定节形式上有，但实质不可验证 → **RESUBMIT**。

### ⑥ 结论不超证据

报告声称"可复算机审结论（REAL，限于本语料/本词表）"，但语料和词表工件均未落盘，该结论无法支撑。

### 报告 788 最终判定：**RESUBMIT**

---

## 缺陷清单（报告 788 返工用）

### 致命缺陷（必须修复）

1. **工件全缺失**：报告附录 A 列出 21 件工件（含 sha256），实际无一落盘于 `docs/proposals/m776/` 或 `artifacts/` 下。包括：`key_numbers.json`、`MANIFEST.json`、`keyword_table.json`、`novelty_audit.json`、`domain_coverage.json`、5 份语料快照（`corpus/*.json`/`.txt`）、3 张 PNG、4 个脚本。
2. **图件缺失**：声称的 3 张 PNG（`fig1_scan_and_novelty.png`、`fig2_keyword_hits.png`、`fig3_domain_coverage.png`）不存在。
3. **数字无法溯源**：报告中所有关键数字（scan_units=6560、master_hits=0、adjacency_hits=100 等）声称来自工件，但工件不存在，无法验证。
4. **误差判定不可验证**：§⑥ 声称的独立第二源（shell grep/wc）所操作的文件不存在，无法复算。

### 要求

- 重写报告，确保所有引用的工件实际落盘于 `docs/proposals/m776/` 和 `artifacts/` 下。
- 生成 3 张 PNG 图件（扫描新颖性图、关键词命中图、域覆盖图）。
- 误差判定的第二源必须基于实际存在的文件，且可独立复算。
- 重新 POST 报告（引用原报告 id 788）。

---

## 处置动作

| 报告 | 判定 | 动作 |
|---|---|---|
| 787 | PASS | 记录，不派单 |
| 788 | RESUBMIT | 派返工单（见下） |
