# 交差核验：m20071 · GAP-DIFFLBM-SOLID-LIMIT（TensorLBM）

**审计单** m20183（board-audit，非重做）· 核验对象：m20071（已 succeeded，第 1 轮·部分交付）
**被审提交** `23399cb023c1576f1c21240b24e1198c4ab1deb5`（分支 `agent/m20071`，**未合入 main**；main HEAD `0414e8681`）
**被审工作树** `/root/tp-agent/work/m20071`（只读）· 复制件 + sha256：`docs/audit-m20183/refs/m20071/SHA256SUMS`
**审计复算脚本** `docs/audit-m20183/audit_checks.py` → `docs/audit-m20183/audit_checks.json`（全部数字程序化读出，无手抄）
**等级裁定**：**L2**（独立功能实现）。L3 不上（无主求解器调用方）；L4 不上（14 个算例只落地 1 个，锚点两条判据 D1/D2 均未测）。
**交差结论**：三件套里 `impl_files`/`tests` 真存在且可跑；`run_artifacts` 只有 1/14、且与其自带代码不自洽；`benchmark` 在 `tensor_benchmark_cases` 里 **0 行**；`repro_cmd` 语法与执行都通。交付自身已诚实写 "1 of 14 shards / NOT MEASURED"，**未虚报**；但按锚点判据属 **honest-FAIL 未完成**。

---

## ① 取证：三件套逐项核

| 要件 | 声称 | 实测（审计复算） | 判定 |
|---|---|---|---|
| **run_artifacts** | "14 个 (arm,D,scale) 组合，`artifacts/.../runs/*.json`（one per combo）" | `runs/` 只有 **1** 个文件 `amplitude_D20_s1.json`（sha256 `f74b2d1f…`）；arm=amplitude, D=20, s=1.0, steps=50000, stop_reason=max_steps | **部分**：1/14 |
| **benchmark ref vs got** | Cd ref 5.579535 / got 6.33239 / +13.49% | 复算 `(6.332395−5.579535)/5.579535 = 13.493228%`（与工件 `err_cd_cv_pct` 逐位相同）；`ref_cd` 与仓内 `benchmarks/verified/cylinder/re20_st/result.json`（`ref_cd=5.57953523384`，ref_source="Nabh 1998 谱方法 / FeatFlow ST 2D-1 Re=20"）一致 | **数值真**，但 **+13.49% > 3%（发表线），也 > 5%（D1 带）** |
| **benchmark 登记** | 锚点判据以 `tensor_benchmark_cases` 为唯一裁判 | `SELECT count(*) … WHERE anchor_slug='GAP-DIFFLBM-SOLID-LIMIT'` → **0**（最近行为 `129:GAP-DIFFLBM-3D-COST:verified`） | **缺**：基准判据无行可裁 |
| **repro_cmd** | `bash cases/GAP-DIFFLBM-SOLID-LIMIT/repro.sh --full` + 分片 campaign 模板 | `bash -n` 通过；审计实跑单臂入口 `run_case.py --arm amplitude --D 20 --scale 1.0 --max-steps 300`（`wall_s=59.5`）成功产出 JSON；`campaign.py --index i` 与 `summarize.py` 均存在 | **可执行**（但复现出的不是交付记录，见 F1） |
| **报告六段** | docs/…md + tensor_reports#18370 | DB 检查：①–⑥ 六段标记全 True，figures=1（仅 `solid_representation.png`；`cd_vs_mask_scale.png`/`criteria.png` 不存在，因为 `summarize()` 没跑过） | **六段齐**（图缺 2 张） |
| **tests** | "9 invariants all green" | 审计重跑 `pytest tests/test_volume_solid.py` → **9 passed in 4.73 s**（二进制极限 ↔ `bounce_back_cells` 用 `torch.equal` 位级断言、力恒等 `<1e-12`、质量/动量账 `<1e-10`、VOF 体积 0.0339%） | **真** |

实算支撑（审计复算，非转抄）：VOF 体积 314.265625 vs πr² 314.159265 → 0.0339%；配对式动量交换 6.332395085221632 vs 控制体 6.3323946780185505 → 相对差 6.43e-8；尾窗 80 样本首半/次半 6.332343 vs 6.332446 → 1.6e-5。

## ② 怎么核的（方法/流程/工具链）

1. 读被审工作树（只读）＋其 commit diff（8 文件，+1372 行）；把关键文件复制到 `docs/audit-m20183/refs/m20071/` 并双侧 sha256 对齐（`audit_checks.py:sha256`）。
2. 重跑单元测试（`pytest -q`，9 passed）。
3. 从工件 JSON **程序化**重算 Cd 窗口均值、误差、估计器一致性、块漂移，并与工件字段逐项比对（`audit_checks.json:delivered_record`）。
4. 按交付代码 `run_case._block_drift` 与 `n = avg_window // cv_every` 的定义，**反推交付代码在同参数下会走到哪一步/什么停机理**（F1）。
5. 实跑单臂入口（300 步）验证入口/JSON 输出可执行；`bash -n` 验 repro 脚本。
6. SQL 侧：`tensor_benchmark_cases`（锚点行数）、`tensor_reports`（六段与图数）、`tensor_maturity_verdicts`（此前无本 scope 行）。
7. 生成两张审计图（`figures/cd_ref_vs_got.png`、`figures/cd_history_and_drift.png`）。

**独立性限制**：同一台 99 控制面机器、同一份只读工作树；未在集群上重跑 50000 步全算例（单臂本地实测约 0.19 s/步 → 3.5 h/臂，属重计算，按计算调度令不本机跑）；`agent/m20071` 未合入 main，故仓内 mainline 无 `src/tensorlbm/volume_solid.py`，`integration_callers` 只能在被审工作树内 grep。

## ③ 严重度排序的发现（含证据与复现）

**F1（高·可复现性）交付工件与其自带代码不自洽 —— artifact 由修 bug 前的旧代码产生。**
工件记 `steps=50000, stop_reason=max_steps, cd_plateau_drift=1.3333450329703842`。按交付版 `run_case.py:281-284`（`avg_window=2000, cv_every=25` → 窗口 `n=80` 样本），对其自带 `series.cd` 复算：
`_block_drift(series[-80:]) = 2.4708855e-05 ≤ plateau_tol(1e-3)`，且**首次**落入 ≤1e-3 发生在第 782 个样本（**step 19550**）。即交付代码在同参数下应在 ~19.5k 步以 `stop_reason="plateau"` 停，而不是 50k 步 `max_steps`；存储漂移 1.3333 无法由交付 `_block_drift` 从该序列的任何窗口/块数（2/4/8、头/尾 80…2000）产生（最近候选差 ≥0.9）。
复现：`python3 docs/audit-m20183/audit_checks.py` → `delivered_record.delivered_window_drift_recomputed / predicted_plateau_stop_step / artefact_provenance_note`。
**影响**：Cd 本身不受影响（尾窗都稳态：预测 plateau Cd=6.332638 vs 交付 6.332395，差 **0.0038%**），但"工件可复现"这条**只能算部分成立**：重跑会得到不同的 steps/stop_reason/drift 三元组。属第 2 轮应修的记录缺陷（重跑一次 19.5k 步即可对齐三元组，约省 60% 墙钟）。

**F2（高·判据）锚点两条验收判据一条也没测。**
D1（s=0.5→1.0 阻力变化 ≤5%）与 D2（与硬壁动量交换偏差 ≤10%）在工件里**均无数据**：缺 s=0.5/0.75 与 hardwall 臂。交付文档自己写 "NOT MEASURED (shards pending)"、`FAIL_STAGE = cluster_queue_waiting`——**诚实**，但按总账判据这是未完成，L4 不成立。

**F3（中·登记）锚点在基准表里 0 行。**
`tensor_benchmark_cases` 无 `GAP-DIFFLBM-SOLID-LIMIT` 行，故"唯一裁判"无容差可裁；目前唯一外部锚是仓内 `benchmarks/verified/cylinder/re20_st/result.json`（该文件本身带 D=40/80 的 −0.558%/−2.875% 三级验证）。第 2 轮必须把 D1/D2 与 Cd 误差登记成行，否则 gate 永远只能给过程分。

**F4（中·精度）唯一实测点 +13.49%，超发表线（<3%）4.5 倍。**
D=20、blockage 0.244、s=1.0、amplitude 臂：Cd=6.332395 vs 5.579535。硬壁对照臂缺失 → **无法**把误差拆成"(a) 闭包加厚（24 个半满单元，theta=2m>0）"还是"(b) D=20 分辨率本身"。注意仓内同算例硬壁在 D=40 已 −0.558%、D=80 −2.875%，说明 D=20 分辨率贡献可能很大；因此"体积罚闭包把阻力抬了 13.5%"**不能**从这份数据推出——交付报告也没这么说（这点没有夸大）。

**F5（中·集成）无主求解器调用方。**
`grep -rn "volume_solid" --include=*.py` 全树仅 4 处：`src/tensorlbm/volume_solid.py`、`cases/GAP-…/run_case.py`、`cases/…/summarize.py`、`tests/test_volume_solid.py`。闭包只用在一个 case 驱动里手工插入 stream/collide 之后，没有进入 `tensorlbm.solver`/边界 API，也没有默认参数化的可复用入口 → **L3 不上**。

**F6（低·诚实性核对）报告/文档没有虚报。**
"9 invariants all green"、"VOF 体积 0.03%"、"pair-form 一致 4e-8"、"1 of 14 shards"、"D1/D2 NOT MEASURED" 逐条核对**全部属实**（见 ① 表）。`cd_vs_mask_scale.png`/`criteria.png` 未生成而文档明说"as soon as the shards land / not fabricated here"——不是造假。**唯一不实之处是 F1 的元数据三元组**（旧代码残留），且不改变 Cd。

**未做的检查（记 unknown，不算 pass）**：D1/D2 未测；硬壁臂 Cd 未测；两分辨率（D=32）未测；s=0.5/0.75 未测；interface 臂一次也没跑过（该臂是"修法"本体，其 `interface_bounce_back` 只有单元测试覆盖，**从未在完整算例里验证过**）；集群复跑 50000 步未做；`agent/m20071` 未合 main 故 mainline gate 的 L3 grep 必然为空。

## ④ 结果图

`figures/cd_ref_vs_got.png` — 唯一实测点 vs Schäfer–Turek 2D-1 参考（3%/5%/15% 三条带）。
`figures/cd_history_and_drift.png` — 工件 Cd 历史（每 25 步）+ 交付平均窗内 4 块均值；标题标出交付代码应停步位置与工件元数据不符。

## ⑤ 注意点（坑与范围）

* `summarize.py::evaluate` 只会用**落地的**记录算 D1/D2，缺臂时表为空——所以 `results.json`/`table.csv` 根本不存在，不能把"跑过 summarize"当成"判据算过"。
* 工件 `wall_s=871.85 s / 50000 步`（17.4 ms/步）与本地实测（300 步 59.5 s ≈ 0.19 s/步，128 核单进程）差 11 倍。计时不可跨机比较，**不能**据此说工件是编的；但也说明"871 s/run → 14 分片"的可行性估计偏乐观（本地等价约 2.6 h/分片）。本条只作提示，不入级别裁定。
* `interface_bounce_back` 把 `phi>=0` 的体内部全部重置为静止平衡态——文档已声明"不是固体状态模型"，不可用于体内非定常压力史；此限制正确披露。
* 判级只看功能实质：文档 156 行、报告六段齐、两张审计图都**不加级**。

## ⑥ 误差判定（是否 <3%）

| 量 | ref | got | 误差 | <3%? |
|---|---|---|---|---|
| Cd（amplitude, D=20, s=1.0） | 5.579535 | 6.332395 | **+13.493%** | **否**（>5% D1 带，>3% 发表线；仅在 15% owner 带内） |
| D1 掩膜尺度不变量 | — | — | 未测 | **无数据，不回答** |
| D2 硬壁一致性 | — | — | 未测 | **无数据，不回答** |
| 估计器一致性（控制体 vs 配对动量交换） | 6.3323947 | 6.3323951 | 6.4e-8 相对 | 是（但这是恒等式级自洽，不是精度） |
| VOF 几何体积 | 314.159265 | 314.265625 | +0.0339% | 是（几何前置项） |
| 单元不变量（9 项） | — | — | 9 passed | 是（L1 级，不可当算例精度） |

## 处置（disposition）

**request changes / honest-FAIL（未达锚点判据）**，级别 **L2**，`gate` 预期 `reopened`（消不掉账）。返修清单（给第 2 轮，按性价比排序）：

1. **先补 hardwall 臂 + s=0.5/0.75 + interface 臂**（同 `campaign.py --index`，14 个分片）；跑完 `summarize.py` 出 D1/D2 表——这是锚点唯一判据，其他都是准备。
2. 用 D=20 的 hardwall Cd 做分母算 D2；若要判"闭包加厚"机制，再补 D=32 或 D=40 一档，否则 13.5% 里分辨率的份额无法分离（F4）。
3. 修 F1：重跑一次交付记录（预计 ~19.5k 步即 plateau），使 `steps/stop_reason/cd_plateau_drift` 与交付代码自洽；顺手把 `_block_drift` 的窗口语义写进 docstring。
4. 把 D1/D2/Cd 误差登记进 `tensor_benchmark_cases`（锚点 slug 一致），否则 gate 无行可裁。
5. `interface_bounce_back` 至少跑通 1 个完整算例（目前只有单元测试）后再声明它是"修法"。
6. 合入 main（当前分支未合，mainline 无该模块，任何 L3 级 grep 都为空）。

**级别说明**：`impl_files` 真实（302 行、公式带文献、二进制极限位级还原 `bounce_back_cells`）、`tests` 真跑 9 passed、`entry` 可执行、`run_artifacts` 有 1 个真工件 → **L2 成立**；L3 缺主求解器调用方、L4 缺完整算例与判据数字 → 不上调。