# 交差核验单 m21802 · m21672 / CASE-BACKSTEP-FVM-RE100（LBM 臂）判定 = **L3**

- **被审对象**：mission m21672（status=succeeded，headline「返修轮1 … 精度判定如实 WITHHELD」）
  - 交付 commit：`0fb4c2ebe` ("m21672: matured mission deliverables")，分支 `agent/m21672`（`parent cf178e1de`）
  - 工作树：`/srv/tensor/agent/work/m21672`（审计前后 `git status` 干净，未改动被审对象任何文件）
- **审计者**：m21802，`verdict_by=board-audit`；判定落表 `tensor_maturity_verdicts(scope='m21672 CASE-BACKSTEP-FVM-RE100', level=3)`
- **基准裁判**：`tensor_benchmark_cases`（唯一裁判）→ **该锚点无注册行**，详见 F2
- **图**：`fig_m21802_backstep_audit.png`（`make_figure.py` 程序化读取工件生成，零手抄）

---

## 1. 判定与理由

| 级别 | 是否达到 | 依据（真实指针） |
|---|---|---|
| L0 文档 | 是 | `docs/CASE_BACKWARD_STEP_LBM_ROUND4.md`、report id 19655 |
| L1 数学/单元验证 | 是 | `tests/test_case_pool_backward_step_armaly.py`：几何 ER 严格 101/52（:43）、抛物线入口离散行均值严格 =u_bar（:67）、二次剖面壁面剪切导数 1e-12（:80）、零点插值（:89）。审计者实跑：7 passed, 2 skipped |
| L2 独立功能实现 | **是（实测）** | `src/tensorlbm/bfs_armaly.py`（485 行 / 21 函数，非占位）+ `benchmarks/case_pool/backward_step_armaly/run_case.py`（108 行 CLI）。审计者执行 `run_case.py --smoke`（400 步）rc=0，落盘 `result.json`+`wall_shear.csv`(15 KB)+`field.npz`(19 KB)，X1/s=2.0761 → 输入→计算→输出闭合 |
| L3 系统集成 | **是（有条件）** | ① 库内真实调用者 `src/tensorlbm/backward_facing_step.py:78` `from .bfs_armaly import reattachment_from_wall_shear, wall_shear_from_near_wall`，`:482` 写入 `metadata["final_reattachment_xr_star_wallshear"]`；② 算例池驱动可发现：`run_pool.py --list` 输出含 `backward_step_armaly`（审计者实跑）。**条件**：驱动按文档调用时崩（F5，已在本单分支修复） |
| L4 基准验证 | **否** | 见 F1/F2：交付内无收敛点（唯一 run 工件是 20000/200000 步瞬态，`steady_at_step=null`），案例自带机器判分 `runs=0 · n_pass=0 · all_judged_pass=false`；裁判表无该锚点注册行；容差断言测试被 `pytest.skip` 跳过 |
| L5 科研工程 | 否 | — |

**判定 = L3**（不是 L2：集成调用者真实存在且可发现；不是 L4：裁判表无行 + 无收敛点，两者任一都足以卡死 L4）。

---

## 2. 三件套逐项核验（task ①）

### 2.1 run_artifacts —— 存在，但是 partial，不是基准点

| 工件 | bytes | 关键字段 | 结论 |
|---|---|---|---|
| `artifacts/case_backward_step_armaly/h1_re100/result.partial.json` | 1754 | `"partial": true`、`steps_run=20000`、`n_steps=200000`、`steady_at_step` 缺失（历史 5 点全为 None）、`elapsed_s=286.9` | 真实作业工件，但只跑完 14% 预算 |
| `artifacts/case_backward_step_armaly/h1_re100_partial/history.json` | 1172 | X1/s 瞬时 2.7649→2.9680→2.9420→2.9037→2.9298（20000 步），`max_speed` 0.15567→0.15311 单调下降 | 仍在瞬态，不能当基准值 |
| `artifacts/case_backward_step_armaly/h1_re100_partial/partial_manifest.json` | 1658 | `"kind": "partial"`、`"not_a_finished_point": true`、provenance `job_id=cj3613 / node n330 / cmd=…--steps 200000` | 标注诚实（这点做对了） |

### 2.2 benchmark ref vs got —— 裁判表无行（F2），只能用仓内 refs

- `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BACKSTEP-FVM-RE100'` → **0 行**（全表 140 行、TensorLBM 60 行；以 `%BACKSTEP%/%backward%/%armaly%/%reattach%` 模糊匹配同为 0 行）。
- 仓内 `benchmarks/case_pool/backward_step_armaly/refs.json`：主参考 `tier=SECONDARY_IN_REPO`，`acquisition_status=PRIMARY_PDF_CLOSED`（Armaly 1983 原文 PDF 四路抓取全失败，日志在文件内）；Re=100 → 3.05；交叉锚 Erturk 2008 (2D) 2.878（band 2.792–2.965）→ 实验 vs 2D 数值本身有 6.0% 不可约差。
- 交付工件内 `artifacts/.../verdict.json`：`runs=[] n_runs=0 n_pass=0 max_abs_error_pct=null all_judged_pass=false` —— 案例自带裁判自认无点可判。
- 唯一能对上的数字：瞬态 X1/s=2.9298 → 相对 3.05 = **−3.94%**、相对 Erturk 2.878 = **+1.80%**。**在容差内但不可计分**：瞬态值不是基准点（案例自己的 5 条件 gate 要求 steady drift ≤1%、mass flux ≤1e-3 等，未被评过）。

### 2.3 repro_cmd —— 入口可执行（审计者实测），但 repro_cmd 的主点从未跑完

- 文档 repro_cmd 的第一段（`--preset h1_re100 --steps 200000`）**没有产生收敛点**：cj3613 20000 步后终止（`partial_manifest.json::provenance`）。
- 审计者用同一入口 `--smoke` 实跑成功（rc=0，400 步，2 s，产物齐全）→ 入口本身可执行，失败在步数预算/排队，不在入口。

### 2.4 报告六段 —— 齐全且如实

report id **19655**（`GET /api/v1/tensor/reports/19655`）：`md_len=4626`、1 张图（`h1_partial_trace.png`），节标题 6 个：①what changed ②the run committed ③result vs reference honest judgement ④what is still missing ⑤figure ⑥machine evidence；显式写「**Is it <3 %? No verdict exists** … accuracy claim stays **WITHHELD**, not passed」。**无粉饰**。

---

## 3. findings（按严重度）

**F1（blocking，L4）交付内没有收敛基准点。** 交付树唯一 run 工件是 20000/200000 步瞬态（`steady_at_step` 不存在，`max_speed` 仍单调降 0.1557→0.1531），案例自带 gate `all_judged_pass=false`。→ 不可计 L4。

**F2（blocking，裁判）锚点未在 `tensor_benchmark_cases` 注册。** 无机器容差判据就没有裁判，任何「达标」都无法被机器复核。这一项需要单独落表（`anchor_slug='CASE-BACKSTEP-FVM-RE100'`，metric=`X1/s vs Armaly 3.05`，tolerance=`<15%`），本单为审计单不代劳（避免审计者给自己造裁判）。

**F3（medium，陈旧声明未标注）`docs/CASE_BACKWARD_STEP_LBM_ROUND3.md` 仍宣称「首个收敛解落地，Re=100 达标（<15%）PASS，X1/s=2.920 → −4.3%」并指向 `artifacts/case_backward_step_armaly/h1_re100/{result.json,wall_shear.csv,field.npz}` —— 这三个文件在交付 commit `0fb4c2ebe` 里**不存在**（同目录只剩 `result.partial.json`）。经查该收敛工件只存在于**未合并分支** `agent/m17034` 的 `f3a211d78`：
`git merge-base --is-ancestor f3a211d78 HEAD` → false；`f3a211d78:...h1_re100/result.json` = 88000 步、`steady_at_step=88000`、`X1/s=2.92021`、`mass_flux_rel_error=8.93e-5`（≤1e-3 ✓）、`rho_max=1.159`。
即：**数字是真的、但不在交付里、且之后代码变过**（`cf178e1de` 改了 `max_speed` 诊断、加了 `uy_mean`/fused-step），不能直接当交付证据。Round-4 报告已声明 WITHHELD 且注明「不重用」，但 round-3 文档没有任何 superseded 标注 → 读者会看到一条指向空路径的 PASS。建议：round-3 文档加 `SUPERSEDED by round 4 / WITHHELD` 抬头（无实质成本）。**这是本单唯一「overstated claim」**。

**F4（medium，可复现性）** 若要把 L4 补上，最小充分动作 = 用交付代码重跑 `h1_re100`（88k 步；按 cj3613 的 20000 步/287 s 外推约 21 min 单作业）+ 注册裁判行 + 让 `verdict.py` 产出 `runs>=1 且 all_judged_pass=true` 并把 `result.json`（非 partial）提交进来。

**F5（repo 级缺陷，已在本分支修复）`benchmarks/case_pool/run_pool.py` 三处：**
1. 文档调用 `PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --case <name>` 直接 `ModuleNotFoundError: No module named 'case_pool'`（`benchmarks/` 从未进 `sys.path`；**13/13 算例全中**，不只是本算例）；
2. `run_case()` 无条件 `mod.main(extra)`，而 3/13 算例（`couette_2d`、`karman_re100`、`mcmp_laplace`）的 `main()` 无参 → `TypeError: main() takes 0 positional arguments but 1 was given`，`--all` 必崩；
3. 修好 1+2 后，无参 `main()` 会自己去 `parse_args()` 看到驱动的 argv → `unrecognized arguments: --case couette_2d --smoke`。
修复：`run_pool.py` 加 `POOL_ROOT` 进 `sys.path`、`_main_accepts_argv()` 签名探测、无参算例在干净 `sys.argv` 下调用并打印「不转发 extra」；回归测试 `tests/test_case_pool_run_pool.py`（7 passed）。修复后文档调用实测：`--case backward_step_armaly --smoke` rc=0（此前 ModuleNotFoundError）。
> 影响判定：这正是 m21672 `maturity_evidence.json::L3.integration_callers` 声称「case-pool driver discovers/executes the case」的那条路径 —— 发现（`--list`）成立，执行按文档不成立。因此 L3 只给「有条件」。

**F6（minor）** `src/tensorlbm/bfs_armaly.py` 末尾对同一 `f` 调了两次 `macroscopic(f)`（`_rho_f` 与 `rho_final` 重复），纯冗余算力，无正确性影响；未改（不在审计单范围）。

**F7（minor，未修）** 驱动转发 `--smoke` 的契约与 3 个算例不匹配（`couette_2d` 用 `--quick`），修复后行为是「不转发 + 提示」，语义正确但契约不统一。

---

## 4. 审计做了/没做什么（独立性说明）

**做了（可复现的命令与结果）**
1. `psql` 查 `tensor_benchmark_cases`（锚点/模糊匹配）→ 0 行；`tensor_maturity_verdicts` 现存 887 行（m21548 判 L1）。
2. `git log/merge-base/ls-tree/show` 于被审树 `0fb4c2ebe` 与未合并 `f3a211d78`（见 F3）。
3. `pytest tests/test_case_pool_backward_step_armaly.py -q` → **7 passed, 2 skipped**（跳过者正是 `test_committed_benchmark_meets_reference_tolerance` 与 field 完整性两条 = 基准断言，理由 `no committed Re=100 artefact yet`）。
4. `run_case.py --smoke --out-dir /tmp/…` → rc=0，产物齐全（L2 实测）。
5. `run_pool.py --list` → 含 `backward_step_armaly`；驱动修复后 `--case backward_step_armaly --smoke` rc=0。
6. `verdict.py` 于被审树实跑 → `all_judged_pass=False`（跑完已 `git checkout --` 复原被它重写的 `verdict.json/md`，被审树保持干净）。
7. 读 report 19655 全文（六段 + WITHHELD 声明）。

**没做（限制，如实登记为 not_run）**
- **未重跑 88k 步收敛点**（审计单明令非重做；且算例收敛步数≈21 min/作业、当时队列 60–84 深度）。→ F3 的 2.92021 是**历史工件读数**，不是审计者的复现；本审计因此不对其计入 L4。
- 未跑 `--preset g1_re100` 精确 ER 档（1.9423077 精确网格）、未做网格收敛研究。
- 未覆盖 TensorFVM 臂（锚点 slug 里的 `FVM` 解读；本仓 = TensorLBM，LBM 臂是 m21548/m21672 的既定解读）。
- 未注册裁判行（F2 明确留给独立单，避免审计者自造裁判）。

## 5. 处置（disposition）

**判 L3 · 请求返修（request changes）以冲 L4。** 返修最小清单：
(i) 交付代码跑出收敛点（`steady_at_step!=null` 且 mass_flux_rel_error≤1e-3），把 `result.json`（非 partial）提交进 `artifacts/…/h1_re100/`；
(ii) 在 `tensor_benchmark_cases` 注册裁判行（tolerance `<15%`，metric `X1/s`）；
(iii) `verdict.py` 产出 `all_judged_pass=true` 并让被跳过的两条基准断言变绿；
(iv) 给 round-3 文档加 SUPERSEDED 抬头（F3）。