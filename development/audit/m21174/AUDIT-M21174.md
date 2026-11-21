# 交差核验单 m21174 — m21138 / GAP-DIFFLBM-EXT-LIT-GHIA400（Ghia Re=400 文献锚补强，TensorLBM）

* 被审对象：mission m21138，分支 `agent/m21138`，commit `21bd5de8b`（交付）
  + `9f72af226`（tp-pusher 自动成熟的工件提交）。
* 审计方式：非重做。`git merge agent/m21138` 进审计工作树 `work/m21174`（merge commit `9c69493a9`），
  在审计树内**独立重跑**锚闸门 / 打分器 / 测试，并对求解器侧做反证探针。
* 级别裁定：**L3（系统集成）**，**L4（基准验证）WITHHELD**。
* 处置：**request changes** — 锚侧消账，精度臂返修。

## 1. 三件套（run_artifacts / benchmark ref-vs-got / repro_cmd）逐项核验

| 项 | 判定 | 证据 |
|---|---|---|
| run_artifacts 真存在 | **REAL** | `benchmarks/verified/cavity/re400/artifacts/{run_raw.json, ghia_re400_anchor.json, report.md, profiles.png}` 均在 commit 内（`git show 9f72af226 --stat`） |
| benchmark ref vs got | **REAL 且 FAIL** | Ghia BOUND 表 vs 实跑：M1 **19.89%**、M2 205.35%、M3 u_min 4.96% / v_min 44.21%（审计树重跑 `anchor_check.py`） |
| 独立复算 ref vs got | **一致** | 审计自写插值复算 max&#124;Δ&#124; = 0.19891·U_lid → **19.89%**，最差点 Table II v @ x=0.8594（ref −0.44993 / got −0.25102）；`audit/m21174/independent_rescore.txt` |
| repro_cmd 可执行 | **打分侧 PASS / 求解侧 FAIL** | `run_anchors.py`、`anchor_check.py` 秒级可跑且确定性；但 `report.md` §6 打印的求解命令 `run.py --nx 65 --steps 60000 --u-lid 0.1` 在本树 **NaN 发散**（3000 步即 NaN，`last_resid=nan`），无法再生 `run_raw.json` |
| benchmark_cases 机器裁判 | **缺失** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-GHIA400'` → **0 行** |
| 六段报告齐全 | **齐全但不自洽** | `docs/JOUR-DIFFLBM_GHIA400_ANCHOR_M21138.md` ①–⑥ 齐；但 §3/§6 数字取 n=97/12000（M1 26.46%），而工件实为 n=65/60000（M1 19.89%） |

## 2. 判级（L0–L5）

* **L2 通过**：真实代码链路 输入(OCR 见证 → provenance → CSV) → 计算(6 步闸门 + 三口径打分) → 输出(工件)。
* **L3 通过**：修复落在主库模块 `src/tensorlbm/lid_driven_cavity.py:GHIA_RE400`（被 `app/backend/services/generic_run.py`、
  `benchmarks/verified/cavity/*/run.py` 消费），并由 `benchmarks/verified/cavity/re400/run.py` 落 profile 供打分器读取。
* **L4 不通过（WITHHELD）**：基准验证要求算例 + 误差分析 + 可复现且**判定通过**。本单预声明门限 5%，
  最好成绩 19.89%（亦超 owner 15% 线），且运行工件无任何入库命令可再生产 → 不满足。
* **L5 不通过**：无。

## 3. 已核实为真的部分（不虚标）

1. 锚闸门 `benchmarks/anchors/run_anchors.py` 独立重跑 **6/6 PASS**；见证 OCR 文件在盘，
   sha256 = `3637ef9e…060f5b`（与 provenance 内声明一致）；4/4 锚 CSV sha256 通过。
2. **代码常量漂移缺陷真实存在且已修**：合并前 `GHIA_RE400['v']` 含 `-0.38598`（x=0.9063）且无 `-0.23827`；
   合并后 17+17 点与 BOUND CSV 逐点一致（max&#124;Δv&#124; < 5e-5）→ `audit/m21174/inline_constant_check.txt`。
3. 锚值与文献规范值一致（u_min −0.32726 @ y=0.2813，v_min −0.44993 @ x=0.8594）。
4. 打分器确定性：重跑仅 ISO 时间戳变化，其余工件字节稳定。
5. 回归：`tests/test_ghia_re400_anchor_lit.py` **9 passed**；`tests/test_lid_driven_cavity.py` **20 passed**（常量修正无回归）。

## 4. 发现（按严重度）

| ID | 严重度 | 结论 | 复现 |
|---|---|---|---|
| F1 | blocking | 预声明 5% 精度门限未达（M1 19.89%，M3 v_min 44.21%） | `python benchmarks/verified/cavity/re400/anchor_check.py` |
| F2 | blocking | 运行工件不可再生：打印的求解复现命令 NaN 发散，且无脚本/日志记录产出命令与 commit | `audit/m21174/runner_stability_probe.log` |
| F3 | blocking | `tensor_benchmark_cases` 未登记该锚 → 无机器裁判 | `audit/m21174/benchmark_cases_query.txt` |
| F4 | medium | 六段文档与入库工件所打分 run 不一致（n=97/12000 vs n=65/60000） | `git show 9f72af226` vs `docs/JOUR-…M21138.md` §3 |
| F5 | minor | M3 “v_min” 实为「参考极值坐标处的 got 值」（−0.25102 @ 0.8594），剖面极值为 −0.28601（36.43%）→ 标签应改，偏保守 | `audit/m21174/independent_rescore.txt` |
| F6 | minor | `run.py`（V3/MRT）当前配方 NaN，而 `result.json` 仍宣称 1.50%/0.83%；本单只在 §5.2 记录、未修未 xfail | `audit/m21174/runner_stability_probe.log` |

## 5. 未运行 / 独立性限制

* 未完成集群端再生：`cj3214`（n=65/n=97，120000 步）审计时仍 `running`（node n315，age 8 min），
  故 60000 步有限值剖面**未在第二台机器上再生成**；文中所有 NaN 复现均来自控制面（CPU，torch eager 与默认 compile 两路）。
* 未审 `profiles.png` 的像素级正确性（仅确认存在与重生成）。
* `tensor_benchmark_cases` 为 0 行，故「got 在容差内」只能对照本单自声明文件判定，不能做第三方裁判复核。

## 6. 第 2 轮建议输入（返修）

1. 先修 `benchmarks/verified/cavity/re400/run.py` 的 V3/MRT 配方发散（corners 双处理 / MRT 与 stream 约定），
   或把实际使用的 RLBM 驱动脚本入库并把 `report.md` §6 的 repro 指向它。
2. 用 n=129 或 n=193 + 步数 ≥2e5（RLBM + ramp）跑正式两档，收敛后再打分；65² 在 Re=400 不可能进 5%。
3. 登记 `tensor_benchmark_cases`（anchor_slug、metric、tolerance、ref_source=Ghia 1982 Table I/II）以便 gate 机器复核。
4. 让 `docs/JOUR-…M21138.md` §3/§6 与入库工件同步（同一 run id / 同一 steps）。