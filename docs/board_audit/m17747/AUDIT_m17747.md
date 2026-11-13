# 交差核验 · m17679 — XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS

| 项 | 值 |
|---|---|
| 被核对象 | mission 17679（状态 `succeeded`，decision_note = `auto-approve TP_AUTO_APPROVE_DEV_ACCEPTANCE`） |
| 仓 / 分支 | TensorLBM / `agent/m17679`，commit `3161a1fc` *xval(vivsim): port continuous error-print vs DOI refs with 15% gate* |
| 工作树 | `/root/tp-agent/work/m17679`（工作树干净，`git status --porcelain` 空） |
| 对标锚 | `XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS`，对标 `haimingz/vivsim @ fb84b12f02e18f3b9dfdf738d6a1a318d81539f9` `examples/2d/flow_pass_cylinder.py:78-85` |
| 核验人 | board-audit（mission 17747，本审计自建、独立重算，不采信被审方任何汇总数字） |
| 判级 | **L2**（独立功能实现）；L3 / L4 未达 |
| 处置 | **request_changes（reopened）**：L2 保留，锚点主交付（口径匹配的对标数字）仍缺 |

**一句话结论**：harness 是真的——代码能跑、11 条测试真断言、honest-FAIL 的 NaN 记录可复现（审计方独立重放 D=10/u=0.1，`nan_at=2000` 完全一致、崩前 `Cd` 五阶一致）；
但锚点要的那件东西（TensorLBM 在 **vivsim 口径**下 vs `1.389/0.718` 的误差）**一个有效数字都没拿到**（3/3 NaN），
且 `tensor_benchmark_cases` 里该锚 **0 行裁判**。所以「<15%」不能记在这个锚点上；`FROZEN` 档的 −3.63%/−5.13%/−4.47%/−7.43% 来自**另一套设置**的既有 verified 算例，只能算间接旁证。

---

## ① 核验范围与方法（独立性声明）

审计方**没有**重做物理，只做三件事：**(a) 逐项查三件套是否真存在；(b) 从原始工件独立重算每个被判数字；(c) 对最关键的 honest-FAIL 记录做独立重放**。

```bash
# 全部核验动作（可复跑）
python3 docs/board_audit/m17747/audit_m17747.py       # 生成 audit.json（A1–A8）
python3 docs/board_audit/m17747/gen_figure.py         # 生成 fig_m17747.png
#   其中 A7 会替审计方重跑被审方测试：
#   cd /root/tp-agent/work/m17679 && python3 -m pytest tests/test_xval_vivsim_cylinder_re200.py -q
#   A3b 会替审计方重放 NaN 记录：
#   cd /root/tp-agent/work/m17679 && python3 benchmarks/xval/xval_vivsim_cylinder_re200.py \
#       --D 10 --u 0.1 --steps 2000 --threads 16 --out /tmp/m17747_repro_D10.json
```

证据快照（`evidence/`，含 sha256 清单 `MANIFEST.sha256`）——本审计把所有被引工件**拷贝进本分支**，因为其中 3 个 NaN 记录在被审分支上是 gitignore 的（见 F5）：

| 快照文件 | 来源 |
|---|---|
| `evidence/oss_flow_pass_cylinder.py`, `evidence/oss_LICENSE.txt` | `/tmp/vivsim_src/examples/2d/flow_pass_cylinder.py @ fb84b12`（MIT） |
| `evidence/xval_vivsim_cylinder_re200.py`, `evidence/test_xval_vivsim_cylinder_re200.py` | `agent/m17679` 交付代码与测试 |
| `evidence/result.json` | `benchmarks/verified/cylinder/re200/result.json`（verified_date 2026-08-19） |
| `evidence/summary.json` | `artifacts/xval_vivsim_re200/summary.json` |
| `evidence/xval_D10_u010.json` / `xval_D12_u010.json` / `xval_D12_u030.json` + 三份 `log_*.txt` | `runs/`（honest-FAIL 原始记录） |
| `evidence/m17747_repro_D10.{json,log,npz}` | **审计方重放**产出（独立证据） |
| `evidence/XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS.md` | 被审方交付文档 |

**独立性限制（必须读）**：三档 NaN 记录只重放了 D=10/u=0.1 一档（单档 32.5 s @16 线程，原记录 244 s @20 线程；D=12 两档每档 ~250 s 未重放）；
审计方**未**定位失稳根因（被审方已自我否证"低 τ"假设，审计方也未验"低 D + 10% 播种"假设）；
vivsim **从未被执行**（本机无 JAX，上游算例 3.9×10¹¹ 格点更新），OSS 侧全部结论仅源码级。

---

## ② 三件套逐项点名

| 三件套 | 是否真存在 | 证据（真实指针） |
|---|---|---|
| **run_artifacts** | ✅ 真存在，且是**真跑**的 | `runs/xval_D10_u010.json`（`wall_s=244.4`, `mlups=0.49`, `nan_at=2000`, `steps_done=1999`）；`runs/xval_D12_u010.json`（`wall_s=316.7`, `nan_at=2500`）；`runs/xval_D12_u030.json`（`wall_s=177.1`, `nan_at=500`）；`runs/xval_D10_u010.npz` 内 `cd/cl` 199 点中 28 点为 NaN；`artifacts/xval_vivsim_re200/summary.json`；`docs/figures/xval_vivsim_re200_frozen_ladder.png`（74 KB） |
| **benchmark ref vs got** | ⚠️ 只存在**异口径**的一半 | ref = `1.389 / 0.718`（审计 A1 从上游源码**解析**得到，非采信转述）；got 只有 FROZEN 档：D48 `Cd=1.3386`(−3.63%) / `√2·Cl_std=0.6859`(−4.47%)，D64 `1.3177`(−5.13%) / `0.6647`(−7.43%)；**口径匹配档 got = 无**（3/3 NaN） |
| **repro_cmd** | ✅ 可执行（审计方亲跑） | `python3 benchmarks/xval/xval_vivsim_cylinder_re200.py --D 10 --u 0.1 --steps 2000 --threads 16 --out /tmp/m17747_repro_D10.json` → 32.5 s，`nan_at=2000`，gate `FAIL`；**但**被审文档结语指的第二轮入口 `benchmarks/xval/README` **不存在**（F4） |

---

## ③ 核验项与重算结果（A1–A8，数字全部来自 `audit.json`）

### A1 参考值出处（零手抄核查）— **通过**
从 `evidence/oss_flow_pass_cylinder.py` 直接解析：`CD_REF=1.389`、`CL_REF=0.718`、三篇 DOI `10.1016/j.ijmecsci.2024.108961`、`10.1016/j.camwa.2014.05.013`、`10.1017/S0022112086003014`；
与交付代码里的常量 `cd_match=True`、`cl_match=True`、`doi_match=True`。
上游几何 `D=60, NX=30*D, NY=20*D, CYL_X=CYL_Y=10*D, U0=0.05, RE=200, TM=int(30*5/U0*D)` 与交付文档 ③ 参数表一致；
上游口径 `cd_mean=mean(cd[TM//2:])`、`cl_max=max|cl[TM//2:]|` 与交付的 `vivsim_calibers()` 一致；上游 `assert`/`pytest` 出现次数 = 0（"只 print 不 assert" 的说法属实）。

### A2 FROZEN 档独立重算 — **数字对得上，但不能算本锚点的验证**
从 `result.json` 原始值重算（不读被审方 summary）：

| 档 | Cd | err vs 1.389 | √2·Cl_std | err vs 0.718 | St | <15%? | <3%? |
|---|---|---|---|---|---|---|---|
| D48 | 1.3386 | **−3.63%** | 0.6859 | **−4.47%** | 0.1962 | ✓ | ✗ |
| D64 | 1.3177 | **−5.13%** | 0.6647 | **−7.43%** | 0.1936 | ✓ | ✗ |

网格趋势：Cd −1.56%、St −1.33%、Cl 幅值 −3.09%（同向趋近参考值），但**只有 2 档 → 无收敛阶**，且每档都是 `result.json` 里 70000/80000 步的运行。

**致命口径问题（本审计最看重的判据）**：这两个档是 `40D×40D` 域、圆柱在 `12D`、`u=0.08`、`sponge=10D`、Braza-1986 锚；而 vivsim 是 `30D×20D`、`10D`、`u=0.05`、无 sponge。
另外 `√2·Cl_std` 是**正弦假设下的换算**，不是测力得到的 `max|Cl|`。三处偏差叠加 → 这 4 个百分比只能证明"我们这条链在**自己**的设置里跟 1.389/0.718 相距 3–8%"，**不能**证明"vivsim 的做法被复刻成功"。

### A3 口径匹配档（决定锚点的那一档）— **0/3 有效**

| 档 | 设置 | 终点 | 有效数字 |
|---|---|---|---|
| D=10 | u=0.1, τ=0.515, 300×200, 15000 步预算 | **NaN @2000**（`steps_done=1999`，244.4 s） | 无 |
| D=12 | u=0.1, τ=0.518, 360×240, 18000 步 | **NaN @2500**（316.7 s） | 无 |
| D=12 | u=0.3, τ=0.554, 360×240, 6000 步 | **NaN @500**（177.1 s） | 无 |

**审计方独立重放（A3b）**：`--D 10 --u 0.1 --steps 2000 --threads 16` → `nan_at=2000`、`verdict` 字典与归档记录**逐键相同**（含 `st_vs_lit_err_pct=-49.238578680203055`）；崩前 `cd_half1` 1.2805861 vs 1.2805931，相对差 **5.45e-6**（差异来自 20 vs 16 线程的 float32 归约顺序）。
→ **honest-FAIL 属实**：记录不是编的，是可重放的。

### A4 报告面 / claim 交叉核查
- `tensor_reports` id=16662：六段①–⑥**齐全**，`figures` 非空（1 张），且报告**显式回答**了"是否 <3%"（答：不能声称）。
- 但存在两处报告面瑕疵：**St 幻觉数**与**表格漂移**——见 F2 / F3。

### A5 裁判登记 — **0 行**
`SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS'` → **0**；`ILIKE '%continuous%'` 亦无近邻行。
→ 本锚点在此前**没有任何机器可裁的 ref/got 对**，唯一的 gate 是被审方自己脚本里的 15% 门，而这个门从未被喂进口径匹配的数字。

### A6 交付耐久性 — **代码/测试/文档/摘要已入库，FAIL 原始记录被 gitignore**
`agent/m17679` 已跟踪：`benchmarks/xval/*.py`、`tests/test_xval_vivsim_cylinder_re200.py`、`docs/XVAL-...md`、`artifacts/xval_vivsim_re200/summary.json`、`docs/figures/xval_vivsim_re200_frozen_ladder.png` ✅
未跟踪：`runs/xval_D*.json`、`runs/log_*.txt` — 被 `.gitignore:108 runs/` 规则排除（仓级策略，非本单违纪），后果是**三份 honest-FAIL 原始记录不随分支走**，只有 `summary.json.runs_fresh` 里的 `nan_at` 存活。本审计已把它们快照进 `docs/board_audit/m17747/evidence/`。

### A7 测试复跑 — **11 passed**
`python3 -m pytest tests/test_xval_vivsim_cylinder_re200.py -q` → `11 passed, 1 warning in 4.35s`。
断言质量抽查（不是"跑过即绿"）：`test_vivsim_calibers_selects_last_half_only`（窗口选择）、`test_vivsim_cl_peak_is_single_sample_sensitive`（钉住 `max|Cl|` 毛刺缺陷，同一毛刺抬高 >1.25× 而谐波幅值只偏 <0.15）、`test_gate_fails_outside_tolerance_and_at_the_boundary`（15% 本身判 FAIL，严格不等号）——**断言的是不变量**，合格。

### A8 集成面 — **0 个调用者**
`grep -rn xval_vivsim` 在 `TensorLBM/` 与 `src/` 下 0 命中；只有 `benchmarks/xval/` 内部、测试与 docs 引用。→ L3 不达。

---

## ④ 问题清单（按严重度，含复现）

| # | 严重度 | 问题 | 证据 / 复现 |
|---|---|---|---|
| **F1** | **blocking（对锚点）** | 锚点主交付缺失：口径匹配的 got 数字 0/3。判据（<15%）**无法被裁决**，任何"通过"结论都无据 | A3 表；重放 `--D 10 --u 0.1 --steps 2000` → `nan_at=2000` |
| **F2** | **medium** | **St 幻觉数**：三个 NaN 记录的 `verdict.st_vs_lit_err_pct` 都是**有限值**（−49.24% / −51.27% / −18.78%），来源是 `harmonic_amplitude()` 退化后回落到频率地板 `1/T=0.001` → `St=0.1`。读者会把 −49% 当成实测误差 | `audit.json → A4_claim_checks.st_floor`；复现：同一重放的 `ours.st=0.09999999999999999` |
| **F3** | **medium** | **口径匹配档的 `Cl` 判据天生不稳**：`cl_peak=max|Cl|[TM//2:]` 对单点毛刺可抬 2.6×（被审方自己测出来并写进测试），叠加参考值 `0.718` 自身"三篇 DOI 平均、平均方式上游未写"——即使 F1 修好，`Cl` 仍可能因口径而非物理越过 15% | `evidence/test_xval_vivsim_cylinder_re200.py:71-85`；上游 `flow_pass_cylinder.py:78-85` 注释仅 "Averaged from" |
| **F4** | **low-medium** | **repro 指针断链**：文档结语指向 `benchmarks/xval/README`（第二轮接棒命令"见 README 摘要行"），该文件**不存在**；`benchmarks/xval/` 只有 2 个 `.py` | `ls benchmarks/xval/`；文档 `:186` |
| **F5** | **low** | **FAIL 原始记录不入库**：`runs/` 被 `.gitignore:108` 排除 → 重放证据只活在本机；换机后只能看 `summary.json` 的转述 | `git check-ignore -v runs/xval_D10_u010.json` |
| **F6** | **low** | **"全部程序化注入"轻微夸大**：`summary.json` D64 的 Cl 误差是 −7.43%，文档 ④ 表里手写成 −7.42%；同一量两个值 → 该表是手抄不是注入 | `audit.json → A4_claim_checks.{summary_vs_recomputed_drift, doc_table_cl_d64_pct}` |
| **F7** | **info** | `ST_LIT=0.197` 标注 "Williamson & Brown 1998 / Williamson 1996" 但无 DOI/页号；虽被标为"非判据"，仍是一个不可溯源的二级锚 | `evidence/xval_vivsim_cylinder_re200.py:71` |

**未发现**：伪造 run（时间/吞吐自洽且可重放）、覆写 FAIL（原始记录保留）、把 FROZEN 数字冒充 FRESH（被审方在文档/报告/JSON 三处都标了证据级，`summary.json.fresh_caliber_matched_verdict="WITHHELD"`）。

---

## ⑤ 判级（L0–L5，判据=功能实质，报告厚度不计分）

| 级 | 判据 | 结论 | 依据 |
|---|---|---|---|
| L0 | 文档成果 | — | — |
| L1 | 数学/单元验证 | **达** | 11 条测试真断言不变量（窗口/幅值恢复/毛刺钉死/15% 边界），审计方复跑绿 |
| L2 | 独立功能实现（输入→计算→输出） | **达** | `xval_vivsim_cylinder_re200.py` 真跑（审计方重放 32.5 s / 1999 步 / 3.69 MLUPS），产出 JSON/CSV/NPZ + gate 判决 + 连续误差打印；参考值出处 A1 逐行核对 |
| L3 | 系统集成（进主求解器/平台） | **未达** | A8：0 调用者；未进 Registry/能力契约/平台 API |
| L4 | 基准验证（完整算例+误差分析+可复现） | **未达** | A5：0 行裁判；A3：0/3 有效数字；收敛阶缺（2 档且异设置）；FROZEN 档口径不符（40D×40D / u=0.08 / √2·Cl_std 换算） |
| L5 | 科研工程成果 | **未达** | 无实验/独立锚点，全链无有效对标数字 |

**L = 2**。锚点名的能力（"连续误差打印 vs DOI 参考值"）在**做法层**已被真正实现并加了门；但锚点的**验证语义**（口径匹配下的 <15% 裁决）没有落地——这正是 L2 与 L4 的分界，不能靠一份 186 行文档补齐。

---

## ⑥ 处置与返修工单（round 2 接棒）

**处置 = request_changes（reopened）**，L2 保留，不撤销、不追打。

1. **（阻断项）定位 NaN 根因**：审计方建议按被审方否证后的假设做 **(D, 播种幅值) 二维扫描**——最小可判组合：`D∈{10,12}` × `seed_amp∈{0.00,0.02,0.05,0.10}`，每格 3000 步；`seed_amp=0` 仍崩 ⇒ 主因在边界/算子（`far_field_bc_2d` + `collide_mrt` 近 τ=0.5），否则在播种。*(本审计未做，未做即 not_run。)*
2. **（阻断项）口径匹配数字**：根因修好后跑 `D=16/24`（域 30D×20D，`u=0.05` 保 Ma 低）到 150 D/U₀；同时报 `max|Cl|` 与谐波幅值两套口径（F3）。
3. **（机制）登记裁判**：本审计已在 `tensor_benchmark_cases` 插入本锚点的 referee 行（`status=pending`，容差**事前**声明 15%，见 `verdict_m17747.sql`），round 2 只需填 got 即自动落判。
4. **（修表）** 补 `benchmarks/xval/README`（F4）；把 `runs/xval_D*.json` 三份 FAIL 记录复制进 `artifacts/xval_vivsim_re200/`（tracked）或引本审计的 `evidence/`（F5）；文档 ④ 表改为脚本生成（F6）；`ST_LIT` 补 DOI 或删（F7）；`verdict()` 在序列含 NaN 时把 `st`/`err_st` 置 `None` 而不是频率地板值（F2）。
5. **（口径上限）** `CL_REF=0.718` 的"三篇 DOI 平均"不可溯源 —— round 2 应至少把三篇 DOI 各自的 Re=200 取值列出来，量化 0.718 的不确定带，否则 15% 判据里有一半是参考值噪声。

---

## ⑦ 本审计未做的检查（不得读作"通过"）

- D=12/u=0.1 与 D=12/u=0.3 两档**未重放**（读记录）。
- 失稳**根因未定位**（三档 NaN 的物理解释保持被审方原文，未独立验证）。
- vivsim **未执行**（无 JAX）；上游 `git` 检出无可用 HEAD 对象，`fb84b12` 身份仅靠工作树内容与引用行号一致。
- 平台层 friction（`run_cluster` 4 作业 40 min 未派发、`knowledge/search` engine_offline）**未复核**——属平台域，非本单判据。
- 未评估 `xval_vivsim_re200_report.py` 的三张图是否与 `summary.json` 逐点一致（仅核对了 ④ 表的两个数）。

---

## ⑧ 产物

```
docs/board_audit/m17747/
├── AUDIT_m17747.md            # 本文件
├── audit.json                 # A1–A8 机读结果（含 level_decision / audit_limits）
├── audit_m17747.py            # 生成器（重算全部被判数字；A7 会替审计方重跑测试；A3b 重放 NaN）
├── gen_figure.py              # 图生成器（数据全部来自 audit.json / evidence）
├── fig_m17747.png             # 左：FROZEN 档误差 vs 15%/3% 带；右：重放 Cd/Cl 与三档 NaN 位置
├── verdict_m17747.sql         # 裁判行 + verdict 落表 SQL（可复跑）
├── publish_report.py          # 报告发布器（六段数字全部读 audit.json）
└── evidence/                  # 17 个被引工件快照 + MANIFEST.sha256

落表结果：`tensor_maturity_verdicts` id=628（TensorLBM / m17679 XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS /
mission 17747 / **level 2** / board-audit / gate_state=pending）；裁判行 `tensor_benchmark_cases` id=63（status=pending）。
本单报告：`tensor_reports` id=16689（六段齐全 + 双面板图内嵌）。
```