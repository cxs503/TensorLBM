# VERDICT m21879 — 交差核验：m21790 `REPRO-LBM-HOU1995`（TensorLBM）

**性质**：交差审计（非重做）。**对象**：m21790（已 succeeded，commit `f186b9c21`，merged as `990f7e83f` → merge `908d863c7` Merge #2503）。
**裁判**：`tensor_benchmark_cases`（anchor_slug）为唯一判据，报告好看不算数。

## 0. 结论（一句话）

**level = 2（独立功能实现）**。三件套真的存在且**可复现**：sha256 侧车与工件一致、审计独立复算 ref-vs-got 误差 **0.000e+00 差异**、审计重跑 n=128/6000 步得到**位级一致**中心线场（88 s）。但**不授 L3**（无生产调用方）、**不授 L4**（唯一打分网格是 6000 步瞬态、`converged=false`，误差 96.44 % 是**截断伪影**而非稳态精度；生产网格 cj3716/3717/3718 审计时仍 `queued`；且 **DB 判据行 0 行**）。交付本身**诚实**：报告 ⑥ 明确「是否<3%？**否**」，未冒充 L4。

## 1. 取证结果（逐项）

| 检查项 | 结论 | 证据 |
|---|---|---|
| impl 真代码 | ✅ | `TensorLBM/src/tensorlbm/verification/cavity_re3200/{runner.py:1-306,scoring.py:1-125,reference.py:1-130,__main__.py:1-194}`；驱动库内 D2Q9 RLBM `CavitySolver` |
| entry | ✅ | `benchmarks/verified/cavity/re3200/run.py:1-36`（:27 `sys.path.insert(REPO/TensorLBM/src)`）；`benchmark.py:1-201` 出 `benchmark.json/.csv/.md` |
| run_artifacts | ⚠️ 部分 | `out/re3200_n128.json`（sha256 `27b32743…` = 侧车，审计复算一致）、`re3200_n128_fields.npz`、`run_manifest.json`（argv/env/git_sha/ref 哈希）**真存在**；但只有 **1 个网格**，192/256 = WITHHELD；两张 PNG 在 mission 工作树存在、**在 main 被门禁滤除**（`git ls-tree HEAD \| grep re3200/figures` = 0 命中） |
| benchmark ref vs got | ❌ FAIL（诚实） | n=128 wall：`max|err| = 96.442088 %`、`rmse = 42.984798 %`（tol 5 %）；最差 u 行 coord 0.9766 ref +0.53236 / got −0.4320609（−96.44 %），最差 v 行 coord 0.9453 ref −0.54053 / got +0.0235467（+56.41 %） |
| 裁判行（DB） | ❌ 0 行 | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-LBM-HOU1995'` → **0 rows**；`tensorbench_results WHERE mission_id=21790` → 0 rows |
| tests | ✅ | 审计重跑 `pytest tests/test_cavity_re3200_runner.py tests/test_cavity_re3200_reference.py -q` → **28 passed, 2 skipped, 30.37 s**（含「runner 真驱动求解器」的独立直算对照、参照哈希门、隔离格排除、WITHHELD/FAIL 路径） |
| repro_cmd | ⚠️ | 记录里的 `python -m tensorlbm.verification.cavity_re3200 …` **照抄跑不通**（repo 根与 `TensorLBM/` 均 `ModuleNotFoundError`，exit 2：包体只在 `TensorLBM/src`，而环境里 `tensorlbm` 解析到**别的任务工作树** `/srv/tensor/agent/work/m21384/…`）。可用入口是 `run.py`（审计已跑通：n=32/300 步 → `err(wall)=52.75 %`，工件齐） |
| 报告六段 | ✅ | report id **19695**（mission 21790）：1 案例 / 2 方法 / 3 结果表 / 4 图（2 张内嵌）/ 5 注意点 / 6 误差判定（显式「是否<3%？否」），无 L4 冒充 |
| 外部锚 | ✅ | Ghia 1982 Table I/II Re=3200 列，OCR 文本层 `ghia1982_tables_I_II.ocr.txt`（sha256 `3637ef9e…`, PDF `1d76e463…`）；审计抽验 mantissa `24427/34323/41933/39017/54053/86636` 均在 OCR 文本中（如 :109 `-0.86636`、:110 `a.24427`、:267 `-0.54053`） |
| 集成调用方 | ❌ 无 | `grep -rl cavity_re3200`（py/md/toml，排除实现/benchmark/ shim）→ 仅 `docs/REPRO-LBM-HOU1995_re3200_reference-admission.md` 与 `tests/test_cavity_re3200_runner.py` |

## 2. 审计独立复核（不调 scoring.py）

* **复算**：用 numpy 独立插值（`np.interp` + wall 约定 `y=(j−0.5)/(n−2)`）从原始记录重算：`u` 96.442088 %、`v` 56.407675 %，`rmse` 42.984798 %，逐行 `|shipped_got − independent_got|` 最大 **0.000e+00** ⇒ `benchmark.json` 忠于原始记录，非编造。
* **重跑**：`run.py --n 128 --steps 6000 --monitor-every 1000 --ramp-steps 1000 --threads 4`（**87.96 s**）：`u_centerline` / `v_centerline` / `u_min` / `u_min_y` / `v_max` / `v_max_x` / `final_resid` / `tau` 与已发布记录**逐位相同**（记录文件 sha256 不同仅因 timestamp/env 字段）⇒ 物理可复现。
* **排队证据**：`cj3716 state=queued age=66.8 min` ⇒ 生产网格（150k 步、resid_tol=2e-6）**无工件**，`analysis.json` 的 `grid_convergence=WITHHELD` 属实。

## 3. 分级（L0–L5）

| 级别 | 判 | 依据 |
|---|---|---|
| L1 数学/单元 | 已有 | 28 测试含打分器精确性/单位不变量/参照哈希门 |
| **L2 独立功能实现** | **✅ 授予** | 真代码 输入→计算→输出（6000 步 RLBM 实算，88 s 审计重跑位级一致），机器可读哈希工件，KPI 独立复算一致 |
| L3 系统集成 | ❌ | 无生产调用方（grep 仅 docs+tests）；runner 消费库求解器，库不消费 runner |
| L4 基准验证 | ❌ | 完整算例✗（单网格瞬态，25× 缺步数）、误差分析虽有但为截断伪影（96.44 % ≫ 15 %）、裁判行 0 行；三件套不同时成立 |
| L5 科研工程 | ❌ | 非该类成果 |

## 4. 缺陷 / 夸大（严重度排序）

1. **F1（对 L4 阻塞，交付本身诚实）**：交付的「复现」是 6000 步瞬态，`final_resid` 非单调（步 4000/5000/6000 = 0.121/0.094/**0.362**）。报告 ⑤⑥ 已披露。
2. **F2（对 L4 阻塞，平台侧）**：anchor 无 DB 裁判行（0 行）；5 % 容差只存在于 `scoring.py:33` 与散文里 ⇒ gate 无带可比对，L4 结构性不可达。建议注册 `cavity_re3200_ghia / max|err| per U_lid / 5 % / Ghia 1982 Table I·II Re=3200 / pending`。
3. **F3（minor，可复现元数据）**：`run_manifest.json`/`benchmark.md` 的 `repro_cmd` 照抄不可执行（见 §1）。修法：发布 `run.py` 调用式，或在命令里带 `PYTHONPATH=TensorLBM/src`。
4. **F4（minor，main 证据完整性）**：两张结果图被 matured 提交滤除，main 内 `re3200/figures/` 不存在（DB 报告内嵌图仍可查）。
5. **夸大 1**：`benchmark.md` 的「how to reproduce」首条 `run.py --n 128 192 256 --steps 150000` 被摆在已发布数字后面，但该命令**从未跑出**已发布数字（数字来自 6000 步 n=128）。建议显式标注哪条命令对应哪张表。
6. **证据质量旁注**：witness 门按 5 位尾数匹配（`build_reference.py:174-204`），符号字形损坏仍算 exact（u(0.2813) 追到 token `a.24427`）；本例由括号门 `[−0.24427, −0.04272]` 佐证符号，数值不变 ⇒ 无需返修，但「exact 104/105」的表述对符号不自证。

## 5. 未做的检查（unknown ≠ pass）

* 生产网格 n=128/192/256 × 150k 步（resid_tol=2e-6）：**未跑**（超本机 <10 min 预算，且与队列中 cj3716–3718 重复）⇒ **Re=3200 稳态精度仍未被测量**。
* 网格收敛研究（需 ≥2 张收敛网格）：**未做**。
* Ma=0.173 压缩性 / τ→0.524 稳定极限敏感性：**未做**。
* 从 PDF 本体重新数字化 Ghia Re=3200 列：**未做**（只核了仓内 OCR 层）。

## 6. 处置

**request changes**（对*级别*不阻塞，对 *L4* 阻塞）：① 落地排队中的生产网格并用同一代码路径重打分；② 注册 `tensor_benchmark_cases` 行给 gate 一条裁判带；③ 修 `repro_cmd`（F3）。已交付工件**诚实、带哈希、位级可复现** ⇒ **level 2 确认**（非虚标，无降级扣分理由）。

**审计独立性限制**：审计与被审同仓同环境（共享 site-packages 与集群），未做独立第三方环境重建；生产网格的终态以队列回执为准，未做长窗轮询。
## 7. Gate 机器裁定预演（maturity_gate.py 的 judge()，非人工意见）

evidence 已按 gate 的 SCHEME 契约改为**机器可核指针**（`impl_files`/`entry`/`run_artifacts`/`tests` 为真实路径列表，散文移入 `*_detail` 键）。用生态 gate 本体复核：

```
import maturity_gate; judge(2, evidence, 'TensorLBM', 21879) -> (2, [])
judge(3, ...) -> (2, ['L3 缺: integration_callers'])
judge(4, ...) -> (2, ['L4 缺: integration_callers', 'L3 缺: integration_callers'])
judge(5, ...) -> (2, ['...L3/L4/L5 缺'])
```

即：声明 L2 无任何 fail 理由 ⇒ gate_level=2（≥2 ⇒ `confirmed`，不返修）；声明 L3/L4/L5 会被机器降级到 2，因为 `integration_callers` 为空（grep 在生态仓 0 命中）。
落表：`tensor_maturity_verdicts` id=892（repo=TensorLBM, scope=`m21790 REPRO-LBM-HOU1995`, mission_id=21879, level=2, verdict_by=board-audit），`gate_state` 在审计单 pass 前的周期里仍为 `pending`（gate 每 30 min 一轮、只信 succeeded 审计单），下一轮应判 `confirmed`。
