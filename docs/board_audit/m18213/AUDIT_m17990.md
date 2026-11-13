# 交差审计 m17990 — XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE

审计单：m18213（TensorLBM，board-audit）｜被审：m17990（status=succeeded，worktree `/root/tp-agent/work/m17990`，commit `5f81352b`）
审计脚本（可复跑）：`audit/m18213/verify_m17990.py`、`audit/m18213/nc_single_rung.py`
原始输出：`audit/m18213/verify.json`、`audit/m18213/nc_single_rung.json`

**独立性声明**：不 import 被审的判定代码来产生被核数字。KPI 差值、TRT τ₋ 由本审计从
`runs/*/result.json`、`runs/*/fail.json` 独立重算；被审代码只作"被核对象"跑一次
（analyze.py 复跑、pytest 复跑、负控），并逐项标注哪些是 not_run。

## ① 基准裁判（唯一裁判）先行 —— 判据缺失

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE';   --> 0 rows
SELECT count(*) FROM tensorbench_results WHERE mission_id=17990;   --> 0 rows
GET /api/v1/tensor/bench/l4-verdict/17990
 --> status=CLAIMED_NOT_PROVEN, l4_benchmark_pass=false,
     failed_checks=[execution_success, source_cluster_job, cluster_job_bound,
                    ref_got_error_present, within_tolerance],
     reason="no tensorbench_results row for this mission: the L4 benchmark claim is
             self-declared, not measured"
```

⇒ 本 anchor **没有任何平台登记判据**：不存在可机器裁定的 ref/got/tolerance 三元组。
按任务书「基准判据是唯一裁判」，got 数值**无从"入容差"**——不是失分，而是**判据面为空**，
因此 L4（基准验证）在本单上结构上不可达。报告里 46.57% 等数字是**相对同仓 A1 臂**的内部
对照（a cross-configuration contrast，不是对标实验/文献的 benchmark 验证）。

## ② 三件套核验（逐项，指针真实）

| 项 | 声明 | 本审计实测 | 判 |
|---|---|---|---|
| run_artifacts | runs/A3_D14/fail.json 等 | 存在：`runs/A3_D14/fail.json`(1.4 KB)、`runs/A{1,2,4}_D10/{result.json,trace.npz}`、`runs/A{3,5}_D10/fail.json`、`summary.json`(15 KB)、`table.csv`、5 张 png | ✅ |
| benchmark ref vs got | 内核 rel-L2 1.18e-07 / 2.71e-08；选型级 46.57% | 内核：`oss_crosscheck.json::k1_collision.rel_l2_collision=1.1757884876259976e-07`、`k2_streaming.rel_l2_streaming_interior=2.709855973692391e-08`（源文件 5 个 sha256 已钉 `Maarten-vd-Sande/lbm@573a01c`）✅；选型级：本审计从 `runs/A1_D10/result.json` 与 `runs/A2_D10/result.json` 独立重算 Δcl_amp = **-46.56648513639526%**，与 `summary.json::verdict.largest_abs_delta_pct` **逐位一致** ✅ | ✅（内部对照）／❌（平台判据 0 行） |
| repro_cmd 可执行 | analyze.py 全链复跑；pytest 6 passed | pytest 实跑 `6 passed in 54.67s` ✅；analyze.py 在 /tmp 副本重跑，重生成的 `summary.json` 与原文件 **结构全等（逐键数值比对无差异）** ✅ | ✅ |
| TRT τ₋ 自洽 | D=10 τ₋=25.50 @step600；D=14 τ₋=18.36 @step1000 | 独立复算 τ₋=0.5+Λ/(τ₊-0.5)，Λ=0.1875：D=10 → 25.500000000000163（match），D=14 → 18.357142857142936（match）；两分辨率均 `status=blowup` 于 step 600/21259、1000/29763 | ✅ |
| 报告六段 | id=16909 六段齐 + 3 图 | ①②③④⑤⑥ 标题全在；figures=fig3/fig4/fig5 内嵌；⑥ 逐项显式回答 <15%/ <3% | ✅ |
| 预注册时序 | predeclared_v3.json 写于任何 D=14/20 KPI 之前 | `predeclared_v3.json` mtime 10:56:32 < `runs/A3_D14/fail.json` mtime 10:58:11；且 D=10 臂工件属上一轮（main 已有 v2 summary），梯级前不存在 D=14/20 工件（`ladder_present`: A1_D20/A2_D20/…全为 null，A1_D14/A4_D14 为空目录） | ✅（结构证据；无独立第三方时间戳） |
| 落主线 | — | **未合并**：`git branch --contains 5f81352b` → 仅 `agent/m17990`；main 的 `xval/benchmark_method_selection/` 仍是 v2（无 predeclared_v3.json / run_ladder.sh / figures / 新 tests / 新 docs） | ❌ |

## ③ 分级裁定：**L2**（独立功能实现）

- **L2 成立**：真代码真计算。`run_arm.py` 走 `sys.path.insert(0, REPO/"src")` →
  `from tensorlbm.solver import collide_bgk, collide_mrt, collide_trt, stream`、
  `tensorlbm.boundary.boundaries`、`tensorlbm.d2q9.equilibrium`，即**驱动本仓真实内核**跑
  方柱绕流（Re=200，20D×20D，D=10 时 21259 步，KPI 取后 50%）；产出机器可读工件；
  6 个不变量型测试全过；聚合链可位级复现。
- **L3 不授**：无生产侧调用方。`grep -rl run_arm|benchmark_method_selection` 的命中只有
  `tests/`、`xval/benchmark_method_selection/` 自身；主求解器路径未被触及，xval harness 不是
  任何入口的被调方。（与 id=712/m17866 的 L2 先例同判据。）
- **L4 不授**：三条同时不成立 ——（a）平台判据 0 行、`tensorbench_results` 0 行、
  l4-verdict=CLAIMED_NOT_PROVEN；（b）收敛梯级 honest not_run（D=14 只完成 A3 失败臂，
  A1/A2/A4 的 D=14 为空目录/仅 state.npz，D=20 全无），预注册主假设 H_conv 未判；
  （c）外部锚（Norberg/Sohankar Re=200 Cd/Cl/St 原表）报告自认 WITHHELD，绝对精度未验证，
  A1 的 Cd=1.613 是否在文献带内**未被任何 referee 判定**。
- **L5 不授**：同上，且交付未落主线。

## ④ 发现（按严重度，含复现）

**F1（高，阻塞"第一证据"的语义边界）`verdict.H_gap_survives_refinement=true` 在只有 1 个分辨率时为真。**
`analyze.py:289-290`：`bool(gap_ladder) and all(v["out_of_band_at_any_rung"] ...)`，而
`out_of_band_at_any_rung` 是**单 rung 属性**（`analyze.py:150`）⇒ 一个 rung 即可 true。
发布件 `summary.json::verdict.H_gap_survives_refinement = true`，但 `gap_ladder.A2.rungs = [10]`。
**负控（端到端，未改被审代码）**：`audit/m18213/nc_single_rung.py` 把 D=14/20 工件移出 runs/ 后重跑
analyze.py → `published rungs=[10]`，`H_gap_survives_refinement=True`，而按预注册语义（需 ≥2 rung）
应为 `False`。对照证据：同文件 `H_fail_persists_at_higher_D`（`:291-293`）**正确**要求
`len(resolutions)>=2` —— 说明这是漏写，不是设计。影响：下游读者（与 maturity gate）会把
"差距在加密下存活"读成已证，而报告 ⑤ 明写梯级 not_run ⇒ **摘要/机器字段与正文自相矛盾**。
修复（2 行 + 1 测试）：`len(e["rungs"])>=2 and e["rungs"][-1] 仍越界`；测试加
"single-rung ⇒ flag 必须 False"。

**F2（中）交付未落主线。** `5f81352b` 只在 `agent/m17990`；main 侧 `predeclared_v3.json`、
`run_ladder.sh`、`run_ladder_cluster.sh`、`figures/`、新 tests/docs 均不存在，main 的
`summary.json` 仍是 `schema xval-method-selection/v2`（rows 无 A3_D14）。⇒ 可复现能力目前只存在
于一个未合并分支 + 一个已 published 的报告，主线不继承。

**F3（中）平台判据面为空。** 该 anchor 在 `tensor_benchmark_cases` 0 行 —— 这类 XVAL 单连续判据
缺失会让每一轮结论都只能自证。建议随修复登记一行（如 metric="A1-vs-A2 |Δcl_amp|% at D=10",
ref=同分辨率 A1, tolerance="<15%"），并登记内核项（<1e-6）。

**F4（低，诚实已披露）收敛梯级未跑完。** `run_ladder_cluster.sh` 递交的 `cj1744` 排队未派发；
报告 ⑤ 明写 not_run 且 ⑥ 写 WITHHELD —— 这是 honest not_run，**不作为缺陷扣分**，但它把
本单上限锁死在 L2（L4 需完整算例）。

**F5（低）测试不覆盖判定语义。** `tests/test_xval_benchmark_method_selection.py` 6 项均过，但
`test_published_ladder_is_consistent_when_present` 只校验 rung 取值/相邻 Δ/收敛标记，
**不校验 `verdict.H_gap_survives_refinement`** ⇒ F1 得以带绿发布。

## ⑤ 声明 vs 实测（overstated claims）

| 声明（摘要/报告） | 实测 | 处置 |
|---|---|---|
| "双分辨率 + 内核级 <1e-6" | 内核 <1e-6 ✅（1.18e-07 / 2.71e-08）；"双分辨率"仅对 A3 失败臂成立（D=10/D=14），KPI 梯级仍单分辨率 | 措辞需限定 |
| "选型级最大偏差 46.57%（>15%）" | 逐位复现 ✅（-46.56648513639526% vs A1） | 成立 |
| summary `H_gap_survives_refinement=true` | 单 rung 即真（F1 负控） | **不成立** |
| 报告六段/图 | 齐（3 图内嵌） | 成立 |
| "pytest 6 passed" | 实跑 6 passed in 54.67s | 成立 |

## ⑥ 未跑项（unknown，不算 pass）

- 交付的完整梯级（A1/A2/A4 × D=14/20）：not_run（本审计未跑 29763 步级算例；集群 cj1744 历史状态未复核）。
- 外部文献原表（Norberg 2003 / Sohankar 1998 Re=200）比对：not_run（无联网取数；报告自认 WITHHELD）。
- OSS 端到端全算例（121 h/臂）：not_run 且报告已判生态不可行（loongarch64 无 numba）。

## ⑦ 处置（disposition）

**request changes（L2 暂定，返修后重审 L3/L4 不自动升）**：
1. 修 F1（判定语义 ≥2 rung）+ 加单 rung 负控测试；
2. 合并 `agent/m17990` 到 main，让 predeclared_v3/ladder 脚本/figures/tests 进主线；
3. 登记 F3 的 `tensor_benchmark_cases` 判据行（否则 L4 永远不可达）；
4. 摘要措辞按 ⑤ 表限定"双分辨率"的适用范围。
honest not_run（F4）**不**列为返修要件——它已被如实披露，属于两轮法的接棒点。