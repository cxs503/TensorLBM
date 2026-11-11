# Board audit — CASE-POISEUILLE-2D / m15240 (TensorLBM)

Audit mission: **m15277** (repo TensorLBM, branch `agent/m15277`)
Audited mission: **m15240** (`succeeded`), audited commit: `agent/m15240 @ ecff590da9c4292bf2b42160740f9b534d0a91e8`
Verdict row: `tensor_maturity_verdicts (repo='TensorLBM', scope='m15240 CASE-POISEUILLE-2D', mission_id=15277, verdict_by='board-audit')`
Mode: **交差核验（非重做）** — 只读审计 + 独立重跑，不改被审仓任何文件。

---

## ① 判据：谁说了算（先查裁判，再看报告）

| 裁判来源 | 查询 | 结果 |
|---|---|---|
| `tensor_benchmark_cases` | `WHERE anchor_slug='CASE-POISEUILLE-2D'` | **0 行** |
| `tensor_case_pool`（算例池，本锚点的实际登记处） | `case_code='POISEUILLE-2D'` | id=1, TensorLBM, ref_source=「流体力学解析解（抛物线剖面）」, **tol_policy=「解析 <2%」**, status=dispatched, mission_id=15240 |
| `feature_ledger` | `slug='CASE-POISEUILLE-2D'` | acceptance=「算例池｜ref: 流体力学解析解（抛物线剖面）｜容差: 解析 <2%」, status=open |

结论：本锚点的机器判据是 **解析抛物线 <2%**（算例池 + 台账登记），但 `tensor_benchmark_cases`
对算例池锚点 0 行 —— 任务榜要求的「SQL 唯一裁判」在本锚点上**不存在**（见 F2，系统级缺口）。
因此本审计不采信交付自报的误差数字，而是**在审计侧重推解析解并重算误差**（`verify_repro.py` C2）。

## ② 三件套取证（逐项核存在性 + 内容）

| 件 | 声称 | 实核 | 判定 |
|---|---|---|---|
| run_artifacts | `benchmarks/verified/poiseuille_2d_bodyforce/{result.json,res_raw.json,sens_raw_*.json,figs/*.png,report.md}` | 全部存在；`result.json` 32 896 B（sha256 `ba221c7f…0422`）；4 张 png（profile/convergence/τ/defect_control）可打开 | ✅ REAL |
| benchmark ref vs got | `u_max` 0.012788 vs 0.012776（-0.09 %）、`eps_inf` 8.99e-04、`eps_2` 1.23e-03 | 审计侧从解析式**重算** ref：`u_max=gH²/(8ν)=0.0127875`、`u_bar=0.00853333`；从 `result.json` 的 `y/u_numerical` 重算 `eps_inf=8.993144e-04`、`eps_2=1.230212e-03`、`amp=−8.9929e-04`，与交付值**逐位一致**（rel diff 0.0） | ✅ REAL |
| repro_cmd | `PYTHONPATH=TensorLBM/src python benchmarks/verified/poiseuille_2d_bodyforce/run.py res` 等 8 条 | 审计侧在 `/tmp/audit_p2d`（被审 commit 的干净副本）**真跑** `run.py res`：H=8/16/32/64 四网格 36 s 跑完，产出 `eps_inf=1.460e-02/3.608e-03/8.993e-04/2.247e-04`，与存档 `result.json` 的 `reference_case` 全部键 **rel diff = 0.00e+00**（见 `logs/repro_res.log`） | ✅ REAL |

报告六段（`audited_report.md`）：①案例介绍 ②怎么做的 ③计算结果 ④结果图 ⑤注意点 ⑥误差判定 —— 六段齐全，
且明写「是否<3%？是（+0.090 %）」。第④段声明「本算例场量沿 x 无关、无二维云图可言」，属物理真话（非缺图）。

## ③ 级别判定 L0–L5

| 级 | 判据 | 证据（真实指针） | 结论 |
|---|---|---|---|
| L1 数学/单元验证 | 测试名 | `TensorLBM/tests/test_poiseuille_2d_bodyforce.py`（129 行，5 项：<2 % 判据、质量守恒 ≤1e-8、步数不变性、二阶收敛、**负对照：截断源项必须被检出**）。审计侧重跑：**5 passed in 26.06 s**；同时 `test_ice_hydro_breakup.py` **14 passed in 13.40 s**（修复未破既有冰盖用例） | ✅ |
| L2 独立功能实现 | 非占位源码 + 入口 + 运行工件 | `TensorLBM/src/tensorlbm/ice_hydro/lbm_channel.py`（369 行，`D2Q9Channel._stream/_collide` 为真 numpy D2Q9 BGK + Guo 源项，无 `NotImplementedError`/`pass` 占位；sha256 `e1dd9fb2…9236`）；入口 `run.py`（374 行，`res/sens/merge` 三模式）；运行工件见 ② | ✅ |
| L3 系统集成 | 生态仓真调用方 | `TensorLBM/src/tensorlbm/ice_hydro/run_study.py:35` `from .lbm_channel import poiseuille_channel, wavy_cover_pressure_gain`，`:40` `r = poiseuille_channel(ny=ny, force=1.0e-5, steps=12000)`（`stage_lbm()`，产出 ice_hydro 文档/报告的数字源）；包级导出 `ice_hydro/__init__.py:30`；`tests/test_ice_hydro_breakup.py` 亦调用。生态仓 `/root/tensor-ecosystem/TensorLBM` grep `poiseuille_channel(` 命中 3 文件。**注意**：无平台侧（ship-performance-platform）调用方，集成深度=库内 pipeline，不含船舶平台 | ✅（库内） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ②全项：参考工况 6 指标 vs 预登记 2 % 门（全过）；网格 H=8→64 收敛，逐段阶数 **2.017 / 2.004 / 2.001**（拟合 2.00，理论 2）；τ/forcing/steps 三组敏感性；负对照（修复前核函数 H=64/30k：质量漂移 **-1.136e-02**、`curv_res` 6.56e-03、`pass=false` vs 修复后 -1.9e-12）；repro_cmd 审计侧真跑复现（rel diff 0） | ✅ |
| L5 科研工程成果 | L4 + 外部锚（文献/实验引用或前后对照问题记录） | 外部锚=解析解（`analytic_anchor.md` 在审计侧重新推导）成立；但**无文献/实验页级引用**（环境无外网，交付与审计双方均已如实声明），且成果性质=「基准验证 + 一处源项缺陷修复」，无新增物理/方法结论 | ❌ 不判 L5 |

**判定级别 = L4（基准验证）**。附百页报告不改变级别；L5 缺外部文献锚与研究级新结论，不虚标。

## ④ 独立复核（审计侧真跑，不只读报告）

| 复核项 | 做法 | 结果 |
|---|---|---|
| C1 复现一致性 | 被审 commit 干净副本重跑 `run.py res`，与存档 `result.json` 逐键对照 | 10/10 键 rel diff = 0（bit-identical） |
| C2 解析重判 | 审计侧从 `g/(2ν)y(H−y)` 重推 ref，重算误差（不取报告任何数字） | `eps_inf=8.9931e-04`、`eps_2=1.2302e-03`、`amp=-8.99e-04`，均 ≪2 %；`Re=2.731`、`Ma=0.0221` |
| C3 收敛阶 | 用 `resolution_study` 的 (H, eps_inf) 逐段求阶 | 2.017/2.004/2.001（全网格均 <2 %） |
| C4 负对照 | 修复前核函数（截断 Guo 源项）质量漂移 | -1.136e-02（≫1e-6 门）→ 缺陷是真缺陷，非叙事 |
| C5 勘误复核 | 修复版 `poiseuille_channel(ny=34/66, 12000 步)` | 最大相对偏差 **0.1023 % / 0.0239 %**，与 P338W3 文档勘误「0.102 % / 0.024 %」相符 |
| C6 主干基线 | `main`（审计 worktree）同函数对比 | 主干仍是 `cst*WT*(CX[i]-ux_eq)`（截断式）→ 修复尚未并入 main |

## ⑤ 发现（按严重度）

- **F1（info，需跟进）** 修复只在 `agent/m15240`，`main` 仍是截断源项。合并前，主干所有 `poiseuille_channel` 调用方
  （`run_study.py::stage_lbm`）继续带 ~0.1–1 % 的质量汇。**建议**：本单 gate `confirmed` 后尽快并入 main。
- **F2（medium，系统级）** `tensor_benchmark_cases` 对算例池锚点无行 → 交差审计的「SQL 唯一裁判」空转，
  只能退回 `tensor_case_pool.tol_policy`。**建议**：`case_feeder` 派单时同事务写入 benchmark_cases 行（ref/tolerance 同 pool 行），
  否则每个 CASE-* 锚点的审计都得靠审计员自推 ref（不可复用、易漂移）。非本单交付缺陷。
- **F3（low）** 报告 ⑥ 把 `curv_res`（无量纲 `max|νu''+g|/g`）列在「相对误差 —」列，未写有限差分 stencil；
  该项随分辨率非单调（H=64 时 3.21e-07 > H=32 的 3.39e-08），仍是 ≪2 % 门，不影响结论。
- **F4（low，已核无回归面）** 同一提交顺带改了 `forcing='simple'` 的平衡态速度位移（0.5g → 0），CHANGELOG 未单列；
  仓内无 `forcing='simple'` 调用方（grep 仅 docstring），故无回归面，但属语义变更应显式记录。

## ⑥ 未做 / 不可判

- 文献页级引用未核（无外网）：`analytic_anchor.md` 只用可离线推导的解析解判决，Guo 2002 仅作方法出处声明。
- 船舶平台侧（ship-performance-platform）集成**未做**，故 L3 只按「库内 pipeline 集成」记，不声称平台集成。
- `docs/benchmarks/tutorials/poiseuille_2d_bodyforce.md` 的 docx/pdf 未生成（表中为 `—`），属交付自述的合理省略，未核。
- 未审计 `figs/*.png` 的绘图数据链（只核 `result.json` → `make_figures.py` 读同一文件）。

## ⑦ Disposition

**approve，级别 L4（confirmed 预期）**。三件套真存在、判据真达标（<2 %，实测 0.09 %）、复现真 bit-identical；
缺陷修复（Guo 源项零和）有负对照背书且未破既有 14 项冰盖用例。F1 需在合并侧跟进，F2 为系统级登记缺口，交平台机制层处理。

## 复现本审计

```bash
cd /root/tp-agent/work/m15277
python artifacts/audits/m15240-poiseuille-2d/verify_repro.py     # C1–C6，写 independent_check.json
python artifacts/audits/m15240-poiseuille-2d/make_figure.py      # 审计图
# 被审交付自身的复现（在 agent/m15240 检出的干净副本内）
PYTHONPATH=TensorLBM/src python benchmarks/verified/poiseuille_2d_bodyforce/run.py res
```

工件：`artifacts/audits/m15240-poiseuille-2d/`（`manifest.json` 记录被审文件到审计副本的 sha256 与 byte-identical 标记）。