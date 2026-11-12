# `benchmarks/` 验证状态台账机器化（m16798）

## ① 案例介绍
`benchmarks/` 的验证债不是"哪个算例不准"，而是**"谁验证过、验证结论是什么" 无法机器核对**：
三份真相源各自演化——(a) `benchmarks/pending/README.md` 人工总表（状态图例 ❌/⛔/⚠️/🟡/🔵/⏸️），
(b) `benchmarks/STATUS.md` 人工 verified 表 + 顶部 advertised 计数，(c) 每个算例目录里的机器证据
（`result.json` / `validation_status.json`）。三份从不互相校验，于是出现：目录已晋级 `verified/` 但
pending 表还在、新算例目录从未登记、顶部计数停在旧值、同一算例出现多行互相矛盾的状态。

## ② 怎么做的
新增 `benchmarks/status_ledger.py`（只读扫描 + `--write` 生成）：

- **发现**：扫 `benchmarks/pending/`、`benchmarks/verified/` 全部算例目录；
- **解析**：pending 总表行（case、状态图例 emoji→机器类 `pass_candidate/running/unverified/fail/closed`），
  `STATUS.md` 的 verified 表与 advertised 计数；
- **机器判决抽取**：按优先级读 `result.json` → `validation_status.json` 的
  `verdict/status/passed/passed_3pct_and_converged/passed/verified/ok`，只认预定义词表
  （pass/verified/true ↔ fail/not_verified/false）；**认不出就报 `unknown`，绝不猜测**，
  并记录来源文件与键名（可回溯）；
- **比对**：hard（阻断 CI）= 未登记 pending 目录 / 未列入 verified 台账的目录 / advertised 计数漂移；
  soft（已披露存量债）= 重复索引行 / 索引行指向不存在目录（已迁移或已删除）/ 索引状态与机器判决冲突 /
  无 README 且无判决；
- **反腐烂**：`--write` 把台账写进两个索引文件的
  `<!-- STATUS_LEDGER:BEGIN/END -->` 生成块（全量目录登记表 + 计数行），并落
  `benchmarks/STATUS_LEDGER.json`；写入迭代到不动点，二次运行报 `unchanged`（幂等）；
- **登记语义**：算例"已登记" = 人工行 **或** 生成块行，因此生成块一旦建立，登记不会腐烂，
  也不会在"列出缺失/列出存在"之间振荡。
- 工具只读、不删任何人工行；`--check` 在 hard 违规时退出码 1，可直接进 CI。

## ③ 计算结果（程序化注入）
| 量 | 值 |
|---|---|
| pending 目录 / verified 目录 | 56 / 79 |
| pending 人工索引行 | 32 |
| STATUS.md 人工 verified 行 | 36 |
| advertised 计数（scan 时） | verified=79 · pending=56 |
| pending 带 result.json / README | 27 / 40 |
| pending 机器判决 pass/fail/unknown | 1 / 16 / 39 |

违规 before→after（before 由 `git show HEAD:` 的两份索引 + 同解析器程序化重算，非手抄）：

| id | severity | before (HEAD) | after (this branch) |
|---|---|---|---|
| `stale_counts` | hard | 1 | 0 |
| `unlisted_verified_case` | hard | 43 | 0 |
| `unregistered_pending_case` | hard | 29 | 0 |
| `duplicate_index_row` | soft | 2 | 2 |
| `missing_machine_verdict` | soft | 56 | 11 |
| `stale_index_row` | soft | 3 | 3 |
| `status_claim_conflict` | soft | 0 | 1 |

本次清掉的 hard 违规明细：未登记 pending 目录 **29** 个（acoustic_pulse_2d, aircraft_icing_rg15_ipw2, cavity_3d_re100, channel_turb_3d, channel_turb_re180, cylinder_re100_dfg, dam_break_2d_multimodel, dam_break_ritter_ci …），
verified 未入台账 **43** 个（cavity_natural_convection, cavity_re100, compressible_shock_family, couette_3d, couette_poiseuille, cylinder_re100 …），
advertised 计数 38/35 → 79/56。
剩余 soft 存量债 **17** 条（逐条见 `benchmarks/STATUS_LEDGER.json` 与
pending/README 生成块折叠区），如实保留未粉饰：

- `duplicate_index_row` `cavity_3d_full` — 2 index rows (lines [41, 42])
- `missing_machine_verdict` `cavity_3d_re100` — no README and no machine verdict on disk
- `status_claim_conflict` `cylinder_re40_st` — index class pass_candidate implies pass, machine verdict fail (verified=False in /root/tp-agent/work/m16798/benchmarks/pending/cylinder_re40_st/result.json)
- `duplicate_index_row` `dit_turbulence` — 2 index rows (lines [45, 47])
- `missing_machine_verdict` `dtmb5415_resistance` — no README and no machine verdict on disk
- `missing_machine_verdict` `kcs_resistance` — no README and no machine verdict on disk
- `missing_machine_verdict` `nwt_stokes2_2d` — no README and no machine verdict on disk
- `missing_machine_verdict` `porous_ladder` — no README and no machine verdict on disk
- `missing_machine_verdict` `pseudopotential_edm_upgrade` — no README and no machine verdict on disk
- `missing_machine_verdict` `series60_resistance` — no README and no machine verdict on disk
- `missing_machine_verdict` `ship_wave_michell` — no README and no machine verdict on disk
- `missing_machine_verdict` `staircase_wall_offset` — no README and no machine verdict on disk
- `missing_machine_verdict` `trt_wall_accuracy` — no README and no machine verdict on disk
- `missing_machine_verdict` `two_phase_poiseuille_viscous` — no README and no machine verdict on disk
- `stale_index_row` `cavity_natural_convection` — index row at line 43 points to a non-pending case (moved to verified/)
- `stale_index_row` `channel_turb` — index row at line 44 points to a non-pending case (no such directory)
- `stale_index_row` `channel_turb` — index row at line 46 points to a non-pending case (no such directory)

## ④ 结果图
本单是**证据基础设施/后端工具**任务，观测量是台账一致性与违规计数，无物理场量
（无速度/压力场、无与解析解/文献的曲线比较），故不产出云图/比较曲线；数值证据以
`benchmarks/STATUS_LEDGER.json`（机器可复算）与上表呈现。此为无图的理由声明。

## ⑤ 注意点
- **机器判决抽取是白名单式的**：`result.json` schema 在仓内不统一（`verdict`/`passed`/
  `passed_3pct_and_converged`/`verified` 都出现过），认不出的键一律 `unknown`；`unknown` 不等于 FAIL，
  只表示"磁盘上没有可判读的判决档"。
- **soft 债不清零是设计选择**：重复/过期索引行含人工撰写的结论文字（如 `channel_turb` 的
  ⏸️/❌ 两行是不同轮次的裁决），工具不删除、只枚举，交由归档 PR 人工收敛。
- **登记 ≠ 达标**：生成块只是"目录已入账"，pending 仍然是 pending；晋级仍需对的预注册判据。
- 生成块位置若被移动到 `## verified/` 小节内部，人工表解析会把它当成人工行——
  `strip_block()` 已规避，且测试用"块落在小节尾部"的夹具覆盖该退化情形。

## ⑥ 误差判定
本单无物理量、无基准误差概念，因此不适用"<3%"判定。可判定的正确性指标是
**台账一致性**（机器可复核）：

- hard 违规：0 条（要求 0，实测 0）；
- advertised 计数 vs 目录列举：verified 79 = 79 ✓，
  pending 56 = 56 ✓；
- `--write` 幂等性：二次运行两个索引文件均 `unchanged` ✓；
- 测试：`pytest tests/test_benchmarks_status_ledger.py` → 10 passed（含 8 条合成夹具缺陷检出 +
  2 条仓内不变量 + 1 条 JSON 产物新鲜度，其中夹具覆盖 unregistered/unlisted/stale_counts/
  duplicate_row/stale_row/claim_conflict/unknow-verdict/idempotency）。

## 复现
```bash
python benchmarks/status_ledger.py                # 人读摘要 + hard/soft 明细
python benchmarks/status_ledger.py --check        # CI 门（hard 违规 → exit 1）
python benchmarks/status_ledger.py --write        # 刷新 STATUS_LEDGER.json + 两个生成块
python -m pytest tests/test_benchmarks_status_ledger.py -q
```
