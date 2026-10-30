# QA Batch 149 — 报告质检报告

**质检批次**: 149  
**对应工作单**: mission_id=238  
**质检时间**: 2026-10-30  
**质检员**: TensorPlatform QA Agent (eco_admin)

---

## 逐篇判定表

| 报告ID | 标题 | 判定 | 理由摘要 |
|--------|------|------|----------|
| 149 | M238: droplet_oscillation SDAA Verification — CPU Bypass + Result Validation | **RESUBMIT** | 缺图件（场量云图+比较曲线两类均缺失，且未声明豁免理由）；误差判定节未逐项显式回答"是否<3%"；缺正式结论节 |

---

## 缺陷清单（报告 149）

### ① 结构五节 — MINOR
- **缺失正式结论节**：报告有 Purpose / Method / Key Results / Audit & Limitations / Artifacts Produced，但无独立 Conclusion 节。判定结果隐含在 Audit 中，未单独收束。
- **原文引用**：报告末节为 `## Artifacts Produced`，无 `## Conclusion`。

### ② 数字溯源抽查 — PASS (3/3 命中)
抽查 3 个关键数字，与工件完全一致：

| 抽查项 | 报告值 | 工件来源 | 工件值 | 命中 |
|--------|--------|----------|--------|------|
| R128 err(%) | -2.649 | `benchmarks/verified/droplet_oscillation/result.json` → `cases.R128.err_pct_primary` | -2.648874999999988 | ✅ |
| R160 mass_drift | 1.07e-03 | `benchmarks/verified/droplet_oscillation/result.json` → `cases.R160.mass_drift` | 0.0010699027408452024 | ✅ |
| R224 r² | 0.999727 | `benchmarks/verified/droplet_oscillation/result.json` → `cases.R224.fit_r2` | 0.9997268373769486 | ✅ |

辅助验证：`artifacts/droplet_oscillation_cpu_verification.json` 中三组数据与 result.json 一致，CPU smoke test 参数（R=16, steps=200, dt=17.2s）也与报告表格一致。

### ③ honest-FAIL 保留 — PASS
- 报告未将 FAIL 粉饰为 PASS。明确声明 GPU 不可用（`torch.cuda.is_available()=False`），CPU 全量运行不可行，采用 smoke test + 已有结果程序化验证的替代方案。诚实披露，无粉饰。

### ④ 图件 — RESUBMIT（关键缺陷）
- **问题**：报告 `figures: []`，无任何图件。
- **要求**：场量云图 + 比较曲线（与解析/文献/基线）两类齐备；缺任一类须有明确理由（纯文档/后端任务可豁免但必须写明）。
- **证据**：仓库中存在可用图件：
  - `docs/benchmarks/tutorials/figs/droplet_oscillation/demo_fields.png` — 场量云图
  - `docs/benchmarks/tutorials/figs/droplet_oscillation/cmp_omega_theory.png` — 与理论频率比较曲线
  - `docs/benchmarks/tutorials/figs/droplet_oscillation/demo_rxry_fit.png` — 拟合曲线
  - `docs/benchmarks/tutorials/figs/droplet_oscillation/grid_convergence.png` — 网格收敛曲线
- **未声明豁免**：报告未说明为何不嵌入图件。虽为后端验证任务，但 owner 升级要求明确"必须写明"豁免理由。
- **处置**：要求返工，至少嵌入 1 张场量云图 + 1 张比较曲线，或明确声明豁免理由。

### ⑤ 误差判定节 — MINOR
- **问题**：报告有 "PASS Criteria Check" 节，但以组合格式呈现（如 `|err| ≤ 3% at every level: ✓ (max |err| = 2.649%)`），未逐项显式回答每个 level 的误差是否 <3%。
- **要求**：关键量 vs 基准误差逐项列出且显式回答「是否<3%」。
- **建议格式**：
  ```
  | Level | err(%) vs Rayleigh理论 | 是否<3% |
  |-------|----------------------|---------|
  | R128  | -2.649%              | 是 ✓    |
  | R160  | -2.404%              | 是 ✓    |
  | R224  | -2.005%              | 是 ✓    |
  ```

### ⑥ 结论不超证据 — PASS
- 报告结论保守，未超出数据支撑范围。明确标注 CPU smoke test 仅验证 code path、不验证 full physics convergence；已有结果来自 prior GPU run。无过度断言。

---

## 综合判定

**报告 149 → RESUBMIT**

**返工要求**：
1. 嵌入至少 1 张场量云图（如 `demo_fields.png`）+ 1 张比较曲线（如 `cmp_omega_theory.png` 或 `demo_rxry_fit.png`），以 base64 内嵌至 figures 字段；或明确声明图件豁免理由。
2. 误差判定节改为逐项表格，显式回答每个 level 的误差是否 <3%。
3. 补充正式 Conclusion 节，收束判定结论。
4. 重写报告后重新 POST 至 /tensor/reports。

**返工约束**：仅允许修改报告文件与重新 POST 报告，禁止改动业务代码。

---

## 质检证据索引

| 证据项 | 路径 |
|--------|------|
| 报告原文 | GET /tensor/reports/149 |
| 结果数据 | `benchmarks/verified/droplet_oscillation/result.json` |
| CPU验证 | `artifacts/droplet_oscillation_cpu_verification.json` |
| 可用图件 | `docs/benchmarks/tutorials/figs/droplet_oscillation/*.png` |
| 原报告提交 | `report_m238.json` |
