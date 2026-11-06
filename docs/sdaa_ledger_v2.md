# SDAA 工件台账 v2 —— 方法、结果与审计发现

> 交付物：`tools/sdaa_ledger_v2.py`（抽取器/视图生成器）、`artifacts/sdaa_ledger_v2.json`（台账）、
> `artifacts/sdaa_ledger_v2.html`（只读静态视图）、`tests/test_sdaa_ledger_v2.py`（契约测试）。
> 生成时间：2026-11-06T15:14:09+00:00（UTC）。全部数字由脚本从台账 JSON 注入，零手抄。

## 1. 口径与最重要的审计发现

台账口径：一条 entry = 磁盘上一份 `artifacts/**/sdaa*.json` 工件实例。语料根共
**193** 棵（本工作树 + 全部 mission worktree 的 `artifacts/`，只读；口径外副本
18 份不计）。

| 指标 | 值 |
|---|---|
| 工件实例（entries） | 3190 |
| 唯一内容工件（sha256 归一） | 57 |
| 副本实例 / 放大倍数 | 3133 / 55.96× |
| 案例数（唯一内容视图） | 50 |
| parse_ok | 100.00%（解析失败 0） |
| 缺失字段默认填充计数 | 0（null+missing_reason 口径） |
| 冲突条目（实例 / 唯一内容） | 550 / 10 |
| 登记 needs_rerun（实例 / 唯一内容） | 275 / 5 |

**发现 1（首要）**：任务书里的“约 3132 份工件”是**副本放大**的结果——同一份内容在多棵 mission
worktree 里各自检出（平均 55.96×，最大单份复制 55 次）。按 sha256 归一后
**唯一工件仅 57 份**。仅用本工作树 `artifacts/` 时 entries=58
（`--strict-local`），可见台账若不显式去重，会把“份数”当“成果量”高估一个数量级。
台账因此同时给两个视图：`entries`（实例，磁盘真实存在）与 `entries_canonical`（按内容去重的
canonical 实例），机器判据与实质分析都以 canonical 为口径复核。

**发现 2**：约 9% 的实例（385 份）不足以支撑“SDAA 已验证”——
含 `cuda` 设备结果被归档成 `sdaa_*`（`110` 份）、
no-evidence 诚实失败、报告类工件、案例定义类工件。
**发现 3**：550 条实例（10 份唯一内容）与自述/历史终态冲突，
其中 275 条登记重跑；冲突类型计数见下（`self_inconsistent` 最多）。

## 2. 方法

1. **发现**：glob `artifacts/**/sdaa*.json`，跨本工作树与 `/root/tp-agent/work/*/artifacts`（只读）。
   台账自身产物（`sdaa_ledger*`）排除，避免自指。
2. **归一抽取**：case（case/benchmark/case_dir 字段 → 退化为文件名）、device 块、判据块（gates/criteria/
   judgment/arms）、墙钟、mission_id、timestamp、基准块，以及结果类量化指标（排除 steps/tau/grid 等配置键）。
3. **证据定级**（确定性规则，顺序即优先级）：
   `NOT_SUPPORTED`（显式 not supported/未实现，且无量化结果）→ `MOCK`（mock|synthetic|hardcoded|fake|
   placeholder|dummy|fabricated 标记）→ `WITHHELD`（报告类工件 / 案例定义类工件 / no-evidence /
   有声明但无结果类指标 / 缺 SDAA 设备证据，含 CUDA·CPU 归属不成立）→ `STUB`（无结果指标且只剩结构/状态）
   → `REAL`（SDAA 设备证据 + ≥3 项程序化量化结果）。
4. **缺失字段纪律**：任何抽取不到的字段一律 `null` 并在 `missing.<field>` 写原因；脚本内断言
   `default_fill_count == 0`（不填默认值）。
5. **冲突重裁**（只重裁不重跑）：强终态字段（verdict/overall_pass/all_pass/verified）与弱字段
   （status/mission_status/gate）分级处理；比对“派生等级 vs 自述终态 vs 历史文档终态
   （docs/sdaa_sweep/*.md）”。类型：`claim_unsupported` / `self_inconsistent` / `claim_understated` /
   `history_drift` / `weak_claim_unsupported` / `weak_status_vs_history` / `history_conflict`→
   案例级复核后降级为 `non_evidence_artifact`（同案例另有 REAL 工件时，不重复登记重跑，避免假阳性）。
6. **只读视图**：单文件静态 HTML（内联 CSS，零外链、零后端调用），含分布/案例/冲突/重跑/缺失字段明细。

## 3. 结果

### 3.1 evidence_level 分布

| 等级 | 实例 | 实例占比 | 唯一内容 | 唯一占比 |
|---|---|---|---|---|
| REAL | 2750 | 86.2% | 49 | 86.0% |
| MOCK | 0 | 0.0% | 0 | 0.0% |
| STUB | 55 | 1.7% | 1 | 1.8% |
| WITHHELD | 385 | 12.1% | 7 | 12.3% |
| NOT_SUPPORTED | 0 | 0.0% | 0 | 0.0% |

### 3.2 案例台账（唯一内容视图，按需重跑排序，前 16）

| 案例 | 唯一工件 | 主导等级 | 等级构成 | 需重跑 |
|---|---|---|---|---|
| `droplet_oscillation` | 1 | STUB | STUB:1 | 1 |
| `floating_body_free_decay` | 1 | WITHHELD | WITHHELD:1 | 1 |
| `poiseuille_3d_annulus` | 2 | REAL | REAL:1/WITHHELD:1 | 1 |
| `sphere_re100_mem` | 1 | WITHHELD | WITHHELD:1 | 1 |
| `womersley_sdaa_sweep_v2` | 1 | WITHHELD | WITHHELD:1 | 1 |
| `<unknown>` | 1 | WITHHELD | WITHHELD:1 | 0 |
| `B18_kovasznay_2d_sdaa_sweep` | 1 | REAL | REAL:1 | 0 |
| `acoustics` | 1 | REAL | REAL:1 | 0 |
| `aircraft_icing_naca0012_rime` | 1 | REAL | REAL:1 | 0 |
| `cavity_3d_full` | 2 | REAL | REAL:2 | 0 |
| `cavity_natural_convection` | 1 | REAL | REAL:1 | 0 |
| `cavity_re100_ghia_sdaa` | 1 | REAL | REAL:1 | 0 |
| `cavity_re400_sdaa_sweep` | 1 | REAL | REAL:1 | 0 |
| `couette_2d` | 1 | REAL | REAL:1 | 0 |
| `couette_2d_sdaa` | 1 | REAL | REAL:1 | 0 |
| `couette_3d` | 1 | REAL | REAL:1 | 0 |

### 3.3 冲突重裁明细（唯一内容视图，共 10 条）

| 案例 | 工件 | 冲突类型 | conflict_reason |
|---|---|---|---|
| `cavity_natural_convection` | `sdaa_cavity_natural_convection.json` | history_drift | 工件终态 WEAK_PASS（status=ok，弱字段）与历史文档终态 FAIL（docs/sdaa_sweep/cavity_natural_convection.md）不一致，历史终态需以最新工件重裁 |
| `cylinder_3d_re40` | `sdaa_cylinder_3d_diag.json` | self_inconsistent | 工件自述 PASS（verdict=PASS），但内部判据含 5 条 FAIL / 3 条 PASS，自述终态与判据明细矛盾（重裁取明细：gate_pass=False） |
| `laplace_cac_phasefield` | `sdaa_laplace_cac_phasefield.json` | self_inconsistent | 工件自述 PASS（verdict=PASS），但内部判据含 3 条 FAIL / 4 条 PASS，自述终态与判据明细矛盾（重裁取明细：gate_pass=False） |
| `laplace_droplet` | `sdaa_laplace_droplet.json` | history_drift | 工件终态 FAIL（overall_pass=False，强字段）与历史文档终态 PASS（docs/sdaa_sweep/laplace_droplet.md）不一致，历史终态需以最新工件重裁 |
| `poiseuille_3d_annulus` | `sdaa_poiseuille_3d_annulus_v3.json` | claim_unsupported | 工件自述 PASS（verdict=PASS），但派生证据等级为 WITHHELD：设备归属为非 SDAA（CUDA/NVIDIA），工件却以 sdaa_ 前缀归档，SDAA 证据不成立；PASS 无 SDAA 数值支撑，重裁为 WITHHELD |
| `poiseuille_3d_pipe` | `sdaa_poiseuille_3d_pipe_case.json` | non_evidence_artifact | 历史文档终态 PASS（docs/sdaa_sweep/poiseuille_3d_pipe.md），但最新运行证据工件未携带可判 PASS 的证据（WITHHELD：缺 SDAA 设备证据（无 tcaicard / sdaa:<id> / sdaa_available=true），设备归属不明）； |
| `sphere_re100_d3q27` | `sdaa_sphere_re100_d3q27.json` | self_inconsistent | 工件自述 PASS（verdict=PASS），但内部判据含 7 条 FAIL / 12 条 PASS，自述终态与判据明细矛盾（重裁取明细：gate_pass=False） |
| `sphere_re100` | `sdaa_sphere_re100_envelope.json` | history_drift | 工件终态 WEAK_PASS（status=SUCCESS，弱字段）与历史文档终态 FAIL（docs/sdaa_sweep/sphere_re100.md）不一致，历史终态需以最新工件重裁 |
| `thermal_cavity_3d` | `sdaa_thermal_cavity_3d.json` | self_inconsistent | 工件自述 PASS（overall_pass=True），但内部判据含 1 条 FAIL / 4 条 PASS，自述终态与判据明细矛盾（重裁取明细：gate_pass=False） |
| `womersley` | `sdaa_womersley.json` | history_drift | 工件终态 FAIL（overall_pass=False，强字段）与历史文档终态 PASS（docs/sdaa_sweep/womersley.md）不一致，历史终态需以最新工件重裁 |

### 3.4 缺失字段（null + missing_reason）

| 字段 | 条目数 | 占比 |
|---|---|---|
| `mission_id` | 2750 | 86.2% |
| `timestamp` | 2530 | 79.3% |
| `reference` | 2365 | 74.1% |
| `wall_clock_s` | 2090 | 65.5% |
| `criteria_gates` | 1100 | 34.5% |
| `verdict_claimed` | 935 | 29.3% |
| `device` | 440 | 13.8% |
| `numeric_metrics` | 165 | 5.2% |
| `case` | 55 | 1.7% |

## 4. 注意点与局限

- **语义边界**：`REAL` 只表示“有 SDAA 设备证据 + 程序化量化结果”，**不表示判据通过**；
  判据是否通过另存于 `gate_pass`/`gate_fail`/`conflict`。反过来 `WITHHELD` 不等于造假，
  多数是诚实失败（OOM/超时）或非证据载体工件。
- **弱终态字段**：`status: ok/success` 这类字段不当 PASS 用（弱字段单独打标
  `verdict_claim_strength`），否则会把“任务状态正常”误读成“物理判据通过”。
- **案例级复核**：同案例已有 canonical `REAL` 工件时，非证据载体工件（案例定义、摘要）
  不产生“历史 PASS 无证据”冲突，也不重复进重跑队列——否则 55 份副本会把同一件事报 55 次。
- **未做**：不重跑任何案例（任务约定只重裁）；不解析 `docs/sdaa_sweep/*.md` 的表格数值
  （仅取终态词用于冲突）；不含非 JSON 工件（png/csv/npz）与口径外 `/tmp` 副本。

## 5. 结论

- 机器判据：entries=3190（≥3000）、parse_ok=100.00%（≥90%）、
  无 evidence_level 空值、冲突条目全部带 conflict_reason、default_fill_count=0、HTML 可打开 —— 全过
  （`python3 tools/sdaa_ledger_v2.py --check`）。
- honest-FAIL 纪律未触发（parse_ok 远高于 60% 红线），因此**不提**“工件格式无契约 / schema 强制”P7 提案；
  但建议把“设备归属字段 + 判据块 + 量化指标”三件套写入工件 schema，以消除 `WITHHELD` 里
  设备归属不明的那部分（110 份实例）。
