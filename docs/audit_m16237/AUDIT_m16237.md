# 交差审计 · m16237 `CASE-LBM-STOKES-FIRST-PROBLEM`（TensorLBM）

**审计单**：m16256（任务榜交差核验 · board-audit）· **被审对象**：m16237，状态 `succeeded`，commit `60dd85a2`
（worktree `/root/tp-agent/work/m16237`，分支 `agent/m16237`）
**审计日期**：2026-11-12 · **裁定**：**L4（基准验证）** · **disposition：confirmed（消账）**，附 2 条非阻塞缺陷与 1 条平台级入库缺口
**工件镜像**：本审计的运行产物规范副本在 `runs/m16256_audit/`（被 .gitignore 遮蔽，仅本机可核）；为让证据随分支入库，同一套文件已镜像到 `docs/audit_m16237/`（含被审 ship 工件 `table.csv`/`summary.json` 快照）。
**基准裁判**：`tensor_benchmark_cases` 对该锚点 **0 行**（无机器裁判行，见 F1）→ 判据回落到 case pool
`tensor_case_pool.id=118`：`tol_policy = "<2% | 指标: 无量纲速度剖面 u/U=erfc(eta) 逐点相对误差 vs 解析 (eta=0..2, t*=1.0)"`，
并由本审计**独立重跑全部 7 个配置**（非重做，只做复算裁判）。

---

## ① 被审交付物是什么

Stokes 第一问题（脉冲起动平板 / Rayleigh 问题）：半无限静止流体下 y=0 平板 t=0 突以 U 起动，
相似解 `u/U = erfc(η)`，`η = y/(2√(νt))`。判据要求 **η∈[0,2] 逐点相对误差 <2%**；`erfc(2)=4.678e-3`，
即误差是用 0.47% 板速量级判分的（尾部陷阱）。交付物 = 新 benchmark 脚本 `run_tstar1.py`（7 配置矩阵）、
资产生成器 `make_assets.py`（表 + 4 图）、pytest 回归、论文级文档、平台报告 15335。

## ② 三件套逐项核验（"存在"必须可指认，不能凭报告自述）

| 件 | 声明 | 核查结果 | 证据指针 |
|---|---|---|---|
| run_artifacts | `scan_tstar1/` = 7×配置 json + summary.json + table.csv + 4 png | **真存在**，且全部在 commit `60dd85a2` 内被跟踪（`git show --stat` 23 文件 +5198 行，工作树 `git status` 干净） | `benchmarks/verified/stokes_first_problem/scan_tstar1/{U1e-3_H200_bgk,U1e-3_H400_bgk_fine,U1e-3_H200_trt,U1e-3_H200_bgk_topnoslip,U1e-3_H200_bgk_tstar0.64,U0.01_H200_bgk,U0.05_H200_bgk}.json`、`summary.json`、`table.csv`、`fig_{profile,error,convergence,spacetime}.png` |
| benchmark ref vs got | 主配置 0.0781%（η∈[0.1,2]）、全窗口 1.1282%、η=2 处 0.0040% | **数值复算一致（bitwise）**：审计方独立重跑 7/7 配置，`u_over_U_num` 数组与 4 项误差指标逐位相同 | 本审计 `runs/m16256_audit/repro_compare.txt`；被重复算源 `runs/m16256_audit/{repro_main.json,repro_U0.05.json,repro_fine_dref50.json,scan_repro/*.json}` |
| repro_cmd | `python run_tstar1.py matrix --out-dir scan_tstar1` + pytest | **可执行**：审计方实跑 `single`（主配置/U=0.05/δ=50 加密）与 `matrix`（6 配置，第 7 个单独跑），全部一次成功；pytest `7 passed in 10.85s` | `runs/m16256_audit/matrix_repro.log`、`repro_compare.txt`、pytest 运行记录见 §④ |

## ③ 基准裁判：判据 vs got（唯一裁判，逐项打勾）

`tensor_benchmark_cases WHERE anchor_slug='CASE-LBM-STOKES-FIRST-PROBLEM'` → **0 行**（全表仅 29 行，其中与本族相关的只有
`GAP-LBM-MCMP-POISEUILLE`、`GAP-LBM-SPHERE`）。故按 case pool 118 的 `tol_policy` 判：

| 配置 | 判据量（<2%） | got（审计方重跑值） | 判定 |
|---|---|---|---|
| U1e-3_H200_bgk（主） | η∈[0,2] 逐点 max | **1.1282%**（含 η=0 钳位端点）/ 0.0781%（η∈[0.1,2]）/ η=2 尾部 0.0040% / 全离散节点 0.0895% | PASS |
| U0.01_H200_bgk | 同上 | 1.1282% / 0.0781% / 0.0040% | PASS |
| U0.05_H200_bgk | 同上 | 1.1282% / 0.0781% / 0.0040% | PASS |
| U1e-3_H200_bgk_topnoslip | 同上 | 1.1282% / 0.0781% / 0.0040% | PASS |
| U1e-3_H200_trt | 同上 | 1.1283% / 0.1904% / 0.1904% | PASS |
| U1e-3_H200_bgk_tstar0.64 | 同上 | 1.4101% / 0.0874% / 0.0064% | PASS |
| U1e-3_H400_bgk_fine（δ=50） | 同上 | **0.5642%** / 0.0154% / 0.0093% | PASS |

关键量 vs 基准误差（回答"是否 <3%"）：主配置 **0.0781%**（η∈[0.1,2]）/ 最严配置（含壁面端点全窗口）**1.4101%**
→ **是，全部 <3%，且全部 <2% 判据线**，无外推、无标定、无结果调参。二阶收敛复核：`p = log2(0.0781/0.0154) = 2.342`（审计方自算）。

**η=0 端点的性质（审计复核，不作扣分项）**：半步反弹壁面在 y=0.5，最近流体节点在 η₁=1/(4δ)=0.01，
故 η=0 无离散自由度，报告取值是节点值钳位，偏差有闭式恒等式 `1−erfc(1/(4δ))`（δ=25→1.1282%，δ=50→0.5642%，
审计方实测 1.128178% / 0.564165% 与恒等式一致）。即该 1.13% 是**坐标伪误差**，不是格式误差；
真实节点级误差 ≤0.0895%（δ=25）。报告 §⑦ 已如实披露，pytest `test_eta_zero_gap_is_the_half_way_wall_artifact` 固化。

## ④ 独立复算（审计方的判据，不是作者的）

1. `python -m pytest tests/test_stokes_first_problem_tstar1.py -q` → **7 passed, 1 warning in 10.85s**（与作者自述一致）。
2. `run_tstar1.py single --H 200 --tau 0.65 --U 1e-3 --tstar 1.0` → 与 `U1e-3_H200_bgk.json` **逐位相同**。
3. `single --U 0.05` → 与 `U0.05_H200_bgk.json` **逐位相同**；且与 U=1e-3 的差仅 1.4e-6 相对 ⇒ "速度剖面与 Ma 无关（压缩性缺陷不进入 u）"的结论复核成立。
4. `single --H 400 --tau 0.8 --tstar 1.0 --delta-ref 50` → 与 `U1e-3_H400_bgk_fine.json` **逐位相同**（0.0154%/0.5642%）。
5. `matrix --out-dir scan_repro`（6/7 配置；fine 单独跑）→ 7/7 全部 `bitwise_identical=True`，见 `repro_compare.txt`。

复算脚本与原始输出：`runs/m16256_audit/{gen_fig.py, repro_compare.txt, repro_*.json, scan_repro/, matrix_repro.log, report15335.md}`。

## ⑤ 平台报告 15335 六要素核查

| 要素 | 是否齐 | 说明 |
|---|---|---|
| ① 案例介绍（物理/几何/参数表） | 齐 | §① 判据与相似解、§③ 参数表（τ/ν/H/δ/U/Ma/Re/ρ 偏差/质量漂移/精度） |
| ② 方法（格式/设置/工具链） | 齐 | §② 含边界模型（Ladd 动壁 + 半步反弹 + 远场自由滑移）、float64、壁面 y−0.5 偏移、代码指针 |
| ③ 计算结果（数字表，程序化注入） | 齐 | §③ 逐点表 + `table.csv` 机器导出 |
| ④ 结果图 | 齐（4 图，base64 内嵌） | 一维单向流无二维场量云图，用 `u/U(η,t*)` 时空云图替代并写明理由——符合"无图须说明"的等价要求 |
| ⑤ 注意点/局限 | 齐 | §⑦ 坐标伪误差、float32 不可用、不外推三维、未做 ν 扫描、Bouzidi 未启用 |
| ⑥ 误差判定（显式回答 <3%） | 齐 | §⑥ 明写 **"是否 <3%？是"**，7 配置 5 类指标全 <2% |

（审计只核六要素与数字是否与工件一致：报告表中 11 个 η 点的数值与本审计 `repro_main.json` 一致。）

## ⑥ 判级（L0–L5）与理由

| 级 | 判据（功能实质，非报告厚度） | 是否达 | 依据 |
|---|---|---|---|
| L1 数学/单元验证 | 测试存在且跑过 | ✅ | `tests/test_stokes_first_problem_tstar1.py` 7 passed；含解析解 helper、钳位恒等式、壁面节点 erfc 匹配 |
| L2 独立功能实现 | 真代码：输入→计算→输出 | ✅ | `run_tstar1.py`（314 行，CLI：`single`/`matrix`→json），无占位/无外推；`make_assets.py`（235 行）生成表与图 |
| L3 系统集成 | 接入主求解器/平台 | ✅（有保留） | 直接 import 主库内核 `tensorlbm.d2q9.{equilibrium,macroscopic}`、`tensorlbm.solver.{collide_bgk,collide_trt,stream}`（`run_tstar1.py:51-52`），非自带私有 kernel；用例已登记 case pool 118；平台报告 15335 已发布；pytest 已入 `tests/`。**保留**：交付物尚未并入生态仓（见 F2），集成关系目前只在 worktree 内可核 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ✅ | 7 配置完整算例；逐点误差表 + 收敛阶 2.342 + Ma/精度/碰撞模型/远场 BC 敏感性；审计方 7/7 逐位复现 |
| L5 科研工程成果 | 新科研/工程能力（新方法、新现象、外部实验/文献锚） | ❌ 不主张 | 解析解为教科书（Stokes 1851 / Schlichting §5）；本质是**验证能力**（+ 两条工程结论：float32 不可用、Ma 缺陷结构性不入 u），非新物理/新算法 |

**裁定 level = 4**。附百页报告不改变级别；本单实际功能体量（314 行 solver-driving benchmark + 7 配置资产 + 回归测试）与 L4 相称。

## ⑦ findings（按严重度；均不阻塞确认，但需机制侧收口）

- **F1【中】无机器裁判行**：`tensor_benchmark_cases` 对该锚点 0 行 ⇒ "基准判据是唯一裁判"在本例无裁判可比，
  got 只能按 case pool 118 `tol_policy` + 审计方独立重跑裁。建议：把 case pool 的 core 锚点判据回填 `tensor_benchmark_cases`
  （case_name/metric/tolerance/repo/status=passed），否则同类锚点的"机器裁定"都是空转。
- **F2【高·平台侧】交付物未入库**：`git ls-remote origin agent/m16237` → **无该分支**；生态仓 `/root/tensor-ecosystem/TensorLBM`
  HEAD=`6911b748`（m16085），既无 `run_tstar1.py`、也无 `scan_tstar1/`、`tests/test_stokes_first_problem_tstar1.py`、`docs/stokes_first_problem_tstar1_20261112.md`。
  即：单子被自动批准 succeeded，但**成果只活在 worktree**。风险=worktree 回收即证据蒸发。建议：m16237 分支补推，
  或平台 auto-push 增加"分支是否真到远端"的核对（否则"入库"口径失真）。
- **F3【低】side artifact 不可由 repro_cmd 再生**：`scan_tstar1/U1e-3_H200_bgk_float32.json` 与 `dtype_sensitivity.json`
  不在 `MATRIX` 里，CLI 也无 `--dtype` 开关，`make_assets.py` 不产出它们 ⇒ "float32 尾部 98%"这条报告结论当前**不可复算**（我未复算，标 not_run）。
  另外 `dtype_sensitivity.json` 只含 U0.05(0.977%)/U0.01(24.39%) 两行，缺主配置 U1e-3(98.11%，见 float32 json) 那一行。
- **F4【低】CLI 默认值会静默生成第三种配置**：`single --H 400 --tau 0.8 --tstar 1`（不显式 `--delta-ref 50`）得到 δ=25、误差 0.0615%，
  标签却不是矩阵里的 "fine"。矩阵本身对（`delta_ref=50.0` 写死），但 README 的"复现命令"段仍只写老 `run.py`，
  未给 `run_tstar1.py` 的 `--delta-ref` 说明——易误用。建议 README 补一行并给 `--delta-ref` 默认值告警。
- **F5【信息】TRT 尾部反而差 47×**（0.1904% vs 0.0040%）：报告已如实写出，本审计复现了 TRT 数值（0.190415%）。属配置观察，不影响主结论。
- **F6【信息】判据取值的口径**：case pool 只写 "t*=1.0"，作者显式定义 `t*=νt/δ_ref², δ_ref=25` 并记录。审计认为**无实质歧义**：
  Stokes 第一问题是相似解，剖面只依赖 η，t* 的选择只改变离散分辨率（δ），不改变被比对的物理量；且 δ 加密一致性自洽（δ=20/25/50 同趋势）。

## ⑧ 未做 / 局限（诚实标注，非"通过"）

- 未做 RANS/三维外部实验对标：本题**只有解析解**可对标（无实验值需求），外部锚为教科书（Schlichting §5，DOI 10.1002/zamm.19800600419）——已核为正确引用，但非独立实验锚。
- 未复算 float32/dtype_sensitivity 资产（F3，缺生成入口）。
- 未复算 4 张 png 的**像素级**一致性（只核 png 存在且在 commit 内、且其数据源 json 已逐位复现）。
- 未跑全仓 pytest（只跑本锚点测试文件 + 依赖的 `run_tstar1` 导入链）；未在集群/loong64 架构复核（本任务为 CPU 秒级算例，x86 主机足够）。
- 复现的平台环境与作者一致（同一主机、同一 worktree 代码），**不构成跨机可复现性证明**（bitwise 一致在同机同 dtype 下是预期行为，非独立证据的替代）。

## ⑨ 产物（本审计）

- `runs/m16256_audit/AUDIT_m16237.md`（本文）
- `runs/m16256_audit/fig_audit_profile_error.png`（左：解析 vs 重跑剖面，含 1e-3 尾巴；右：逐点误差 vs 2% 判据线；`gen_fig.py` 程序化生成）
- `runs/m16256_audit/{repro_main.json, repro_U0.05.json, repro_fine_dref50.json, scan_repro/*.json, repro_compare.txt, matrix_repro.log, report15335.md}`
- `verdict_m16256.sql`（`tensor_maturity_verdicts` 落表语句）