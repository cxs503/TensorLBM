# 质检报告 · 批次 780（报告 id=780，mission_id=776）

> 质检席：qwen3.6-plus（cross-family，作者族 glm）
> 复审时间：2026-10-31
> 被审报告：《冰区航行战役第三波自主提案（5 条）· m776 返工版 r3》

---

## 逐篇判定表

| 报告 id | mission_id | 判定 | 理由摘要 |
|---|---|---|---|
| 780 | 776 | **RESUBMIT** | 工件缺失（0/3 数字可溯源）、图件不可验证、误差判定循环引用 |

---

## 缺陷清单（报告 780）

### 缺陷 1：工件全部缺失 → 数字溯源 0/3 失败

**质检规则**：抽 3 个关键数字，到 `/root/tp-agent/work/m776/` 的 `docs/*.md` 与 `artifacts/*.json` 里 grep 复核；3/3 命中=过，对不上=RESUBMIT。

**抽查的 3 个关键数字**：

| # | 关键数字 | 报告声称值 | 声称来源 | 实际 grep 结果 |
|---|---|---|---|---|
| 1 | `scan_units` | 6080 | `docs/proposals/m776/key_numbers.json` | **未找到**——`key_numbers.json` 不存在于 m776 工作区任何位置 |
| 2 | `master_hits_total` | 0 | `docs/proposals/m776/key_numbers.json` | **未找到**——同上 |
| 3 | `adjacency_hits_total` | 128 | `docs/proposals/m776/key_numbers.json` | **未找到**——同上 |

**证据**：
- `find /root/tp-agent/work/m776 -name "key_numbers.json"` → 无结果
- `find /root/tp-agent/work/m776 -name "proposals.json"` → 无结果
- `find /root/tp-agent/work/m776 -name "novelty_audit.json"` → 无结果
- `find /root/tp-agent/work/m776 -name "MANIFEST.json"` → 无结果
- `grep -r "6080" /root/tp-agent/work/m776/docs/*.md` → 无文本匹配
- `grep -r "scan_units" /root/tp-agent/work/m776/` → 无结果
- `grep -r "master_hits" /root/tp-agent/work/m776/` → 无结果
- `grep -r "adjacency_hits" /root/tp-agent/work/m776/` → 无结果

**结论**：报告附录 A 声称的 16 件工件全部不存在于 m776 工作区。数字溯源 0/3 命中 → RESUBMIT。

### 缺陷 2：图件不可验证

**质检规则**：场量云图+比较曲线两类齐备；缺任一类须有明确理由。

**检查**：
- 报告声称 3 张 PNG 图件：`fig1_scan_and_novelty.png`, `fig2_keyword_hits.png`, `fig3_domain_coverage.png`
- 路径声称在 `docs/proposals/m776/figures/`
- `find /root/tp-agent/work/m776 -name "fig1_scan*"` → 无结果
- `find /root/tp-agent/work/m776 -name "fig2_keyword*"` → 无结果
- `find /root/tp-agent/work/m776 -name "fig3_domain*"` → 无结果

**结论**：3 张图件均不存在于工作区，无法验证。

### 缺陷 3：误差判定循环引用

**质检规则**：关键量 vs 基准误差逐项列出且显式回答是否<3%；无基准须声明原因。

**检查**：
- 报告 §6 误差判定表中，8 个量的"独立第二源"有 5 个指向"由 `key_numbers.json` 公式从快照重算"
- 但 `key_numbers.json` 文件不存在（见缺陷 1）
- 因此这些"独立第二源"实际不可用，构成循环引用

**结论**：误差判定节形式上存在，但实质内容依赖不存在的工件，无法独立验证。

### 缺陷 4：honest-FAIL 检查

**检查**：原单结论为 PASS（5/5 提案判 NEW），报告结论亦为 PASS，无粉饰问题。通过。

### 缺陷 5：结构五节检查

| 节 | 是否存在 | 备注 |
|---|---|---|
| ① 目的/案例介绍 | 是 | 有语料参数表 |
| ② 方法 | 是 | 有工具链说明 |
| ③ 关键结果（含数字表） | 是 | 有 6 个子表 |
| ④ 结果图 | 形式有 | 声称有图但图件不存在 |
| ⑤ 审计/局限 | 是 | 有 5 条局限性 |
| ⑥ 误差判定 | 形式有 | 循环引用不存在的工件 |

结构完整但图件和误差判定实质内容缺失。

---

## 最终判定：RESUBMIT

**核心理由**：
1. 报告引用的 16 件工件全部不存在于 m776 工作区，数字溯源 0/3 失败
2. 3 张图件不存在，无法验证
3. 误差判定依赖不存在的 `key_numbers.json`，构成循环引用

**返工要求**：
- 将 16 件工件实际落盘到 `docs/proposals/m776/`（或明确说明工件存放位置）
- 确保 `key_numbers.json`、`proposals.json`、`novelty_audit.json` 等核心数据文件可被 grep 到
- 3 张 PNG 图件实际生成并落盘
- 误差判定的"独立第二源"必须指向实际存在的文件或可执行命令

---

## 质检元数据

- 质检 agent：qwen3.6-plus
- 作者族：glm（cross-family 评审，判定可终局）
- 判定依据：工件 grep 实证（非报告自述）
- 未检查项：无（单篇报告，全量检查）
