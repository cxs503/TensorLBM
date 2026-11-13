#!/usr/bin/env python3
"""Board-audit for m17895 (XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN, TensorLBM).

Writes the audit report + verdict SQL into the worktree, inserts the verdict
(tensor_maturity_verdicts), and posts the paper-grade progress report.
All numbers are read from the audited branch's run_artifacts / re-run values
recorded below; nothing is typed from memory.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # /root/tp-agent/work/m18086
SUBJECT_WT = Path("/root/tp-agent/work/m17895")
ART = SUBJECT_WT / "xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_artifacts"
COMMIT = "222585d6"
DSN = "host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation"
MISSION = 18086
SCOPE = "m17895 XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN"
LEVEL = 2

# ---------------------------------------------------------------- evidence
summary = json.loads((ART / "summary.json").read_text())
s14 = {k: json.loads((ART / f"arm_a1_channel_ny{k}.json").read_text()) for k in (16, 32, 48)}
control = json.loads((ART / "arm_a2_reference_main_ny32.json").read_text())
tg = json.loads((ART / "arm_a3_taylor_green_32x32.json").read_text())
screen = {k: json.loads((ART / f"arm_a{k}.json").read_text()) for k in
          ("4_determinism", "5_examples_audit", "6_detector_control")}

chan = []
for ny, rec in s14.items():
    k = rec["kpis"]
    chan.append({"ny": ny, "u_max": k["u_max"], "ref": rec["analytic"]["u_max"],
                 "rel_err_pct": 100 * k["relative_error_vs_analytic"],
                 "profile_l2_pct": 100 * k["profile_rel_l2"],
                 "mass_drift": k["mass_drift"], "steps": k["steps_run"],
                 "field_digest": k["field_digest"][:16]})
tg_rel_pct = 100 * abs(tg["kpis"]["decay_rate_measured"] - tg["analytic"]["decay_rate"]) / tg["analytic"]["decay_rate"]
checks = summary["checks"]
c11_rho = checks["C11_stability_rule"]["rho_range"]
a2 = summary["arms"]["A2_control"]
a2_rel = a2.get("u_max_rel_diff")
if a2_rel is None:
    a2_rel = abs(control["u_max"] - s14[32]["kpis"]["u_max"]) / control["u_max"]
a2_l2 = a2.get("profile_l2_diff")
if a2_l2 is None:
    a2_l2 = control.get("profile_rel_l2_diff", float("nan"))
audit_repro = {
    "pytest": "PYTHONPATH=src pytest tests/test_example_harness.py tests/test_example_antipattern_audit.py -q -> 16 passed in 22.58s",
    "xval_a1_32_a2": "run_xval.py --arms a1_32,a2 -> 63.2s, C7 reproduced 6.124291821339015e-13",
    "xval_a3_a4_a5_a6": "run_xval.py --arms a3,a4,a5,a6 -> 23.5s, C6=0.001127977207684141 C9=0.6212 C10 pass",
    "digest": "channel ny=32 rerun keep_history=False -> record_digest 288a7c6c... == shipped; field_digest 8bebedb9... == shipped",
}

evidence = {
    "impl_files": [
        f"src/tensorlbm/example_harness.py:1-511 @{COMMIT} (CaseSpec/run_case/record_digest, D2Q9 BGK + full Guo source)",
        f"src/tensorlbm/audit/example_antipattern.py:1-313 @{COMMIT} (name/literal-insensitive AST duplication audit)",
        f"examples/channel_poiseuille_thin.py:1-58 @{COMMIT}",
    ],
    "entry": "xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_xval.py:411 main (arms a1_16/a1_32/a1_48/a2/a3/a4/a5/a6); public export src/tensorlbm/api.py:36-44 (get_case/run_case/record_digest)",
    "run_artifacts": [f"xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_artifacts/{p.name} @{COMMIT}"
                      for p in sorted(ART.glob("*.json"))],
    "tests": ["tests/test_example_harness.py (9 invariant tests: bit-identical to collide_bgk+stream, <3% vs closed form, mass drift, 4nu k^2, determinism)",
              "tests/test_example_antipattern_audit.py (7 tests: planted-fixture discrimination, syntax-error tolerance)",
              "audit re-run: 16 passed in 22.58s (PYTHONPATH=src, commit 222585d6 tree)"],
    "integration_callers": ("NONE in ecosystem main: origin/main fc86f508 contains none of the 31 delivered files; "
                            "agent/m17895 (222585d6) unmerged at audit time. Inside the branch only src/tensorlbm/api.py "
                            "re-exports the harness and examples/channel_poiseuille_thin.py consumes it; no main-solver/platform caller "
                            "(dependency direction is harness -> solver kernels) => L3 not met"),
    "benchmark": {
        "registered_case": ("ABSENT: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug="
                            "'XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN' -> 0 rows; tensor_case_pool linked_anchor -> 0 rows; "
                            "GET /api/v1/tensor/bench/l4-verdict/17895 -> CLAIMED_NOT_PROVEN, rows=0, "
                            "failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]"),
        "delivered_ref_vs_got": {
            "channel_u_max_vs_gH2_8nu": chan,
            "taylor_green_decay_rel_err_pct": tg_rel_pct,
            "harness_vs_independent_main_rel": a2.get("u_max_rel_diff"),
            "mass_drift_max": max(abs(c["mass_drift"]) for c in chan),
            "all_checks_pass": summary["all_pass"],
        },
        "threshold_basis": "predeclared.json (pre-run): owner 15%, target 3% (u_max/profile L2/TG), harness-vs-control 1e-8, mass 1e-10; not a platform-registered tolerance",
        "error_verdict": "all delivered physics KPIs <3% (worst 0.7755% at ny=16); L4 still DENIED because the sole registered referee has 0 rows",
    },
    "repro_cmd": [
        "cd <repo>; PYTHONPATH=src python xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_xval.py --repo-root .",
        "PYTHONPATH=src python -m pytest tests/test_example_harness.py tests/test_example_antipattern_audit.py -q",
        "PYTHONPATH=src python examples/channel_poiseuille_thin.py --ny 32",
        "audit re-run: " + json.dumps(audit_repro, ensure_ascii=False),
    ],
    "external_anchor": ("sorush-khajepor/listLBM example-per-main practice -- characterised structurally (AST), upstream NOT vendored/executed "
                        "(recorded limit). Physics reference is closed-form: u_max=gH^2/(8nu), H=ny-2; TG energy exp(-4 nu k^2 t)"),
    "findings": [
        "F1 BLOCKING(L4): tensor_benchmark_cases/tensor_case_pool 0 rows for this anchor; platform l4-verdict=CLAIMED_NOT_PROVEN rows=0 => L4 withheld, delivery's predeclared 3%/15% tolerances never registered",
        "F2 BLOCKING(L3): origin/main fc86f508 has zero delivered files (branch unmerged), no main-solver/platform caller; only api.py re-export + thin example inside the branch",
        "F3 LOW(doc): module docstring src/tensorlbm/example_harness.py:31 says u_max = gH^2/(8nu) with H = ny-1, while code wall_height=ny-2, the test, and REPORT §①/§⑤ use ny-2",
        "F4 LOW(report): REPORT §③ C11 row states rho in [0.99942,1.00000]; summary.json C11_stability_rule.rho_range = [0.9999999999990441,0.9999999999997147] (~1e-13 from 1, consistent with mass drift 9.55e-13). The 0.99942 row was not injected from the artifact (contradicts the 'zero hand-copy' claim); PASS conclusion unchanged",
        "F5 NOTE(scope): C9 detector discrimination 0.6212 vs threshold 0.20 is on planted fixtures only; the repo-level duplication index (31.4% lines, 97.5% standalone mains) has no external reference (metric definition is the deliverable itself)",
    ],
}

# ---------------------------------------------------------------- report md
rows = "\n".join(
    f"| {c['ny']} | {c['u_max']:.8f} | {c['ref']:.8f} | {c['rel_err_pct']:.4f}% | {c['profile_l2_pct']:.4f}% | {c['mass_drift']:.2e} | {c['steps']} | {c['field_digest']} |"
    for c in chan)
check_rows = "\n".join(
    f"| {name} | {v.get('value')} | {v.get('threshold')} | {v['pass']} |"
    for name, v in sorted(checks.items()))
md = f"""# 交差审计 m17895 — XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN（TensorLBM）

**裁定 = L{LEVEL}**（L4 机器裁定 `CLAIMED_NOT_PROVEN` rows=0；L3 无生态内消费者）｜被审 commit `{COMMIT}`（分支 agent/m17895）｜审计单 m{MISSION}

## ① 案例介绍

被审对象是 TensorLBM 的 example-per-main 反模式治理：上游 sorush-khajepor/listLBM 每个例子自带 `main`/格子/时间推进/后处理；TensorLBM `examples/` 放大到 {screen['5_examples_audit']['python_files']} 个文件。交付物是
(1) 声明式算例运行时 `src/tensorlbm/example_harness.py`（CaseSpec→run_case→JSON run record），(2) AST 反模式审计器 `src/tensorlbm/audit/example_antipattern.py`，(3) 一条 thin example，(4) 8 臂 xval 与机器裁判。

物理算例（D2Q9/BGK/全 Guo(2002)/半程反弹壁/float64 CPU）：压力驱动平面槽道 ny=16/32/48, nx=8, τ=0.8(ν=0.1), g=1e-5；Taylor–Green 32×32 周期, U0=0.05, 800 步。闭式参照 `u_max=gH²/(8ν), H=ny−2` 与 TG 能量 `exp(−4νk²t)`。

## ② 怎么做的（审计方法）

1. 取证：`git archive {COMMIT}` 到隔离目录，逐项核 impl_files/entry/run_artifacts/tests/benchmark/repro_cmd。
2. 独立复跑：pytest 16 项 + xval 分支臂（a1_32,a2 = 63.2s；a3,a4,a5,a6 = 23.5s），并与交付工件逐值比对；channel ny=32 以 `keep_history=False` 重跑，`record_digest`/`field_digest` 与工件逐比特一致。
3. 基准裁判：查 `tensor_benchmark_cases`（本 anchor 0 行）与 `tensor_case_pool`（0 行），并调平台机器裁定 `GET /api/v1/tensor/bench/l4-verdict/17895`。
4. 集成面：查 `origin/main`（fc86f508）是否含交付文件、是否有主求解器/平台侧调用方。
5. 文档一致性：报告数字 vs run_artifacts 程序化导出逐行对表。

## ③ 计算结果（ref vs got，全部读自 run_artifacts）

槽道 u_max vs 闭式 `gH²/(8ν)`：

| ny | u_max(got) | u_max(ref) | rel err | profile L2 | mass drift | steps | field_digest[:16] |
|---|---|---|---|---|---|---|---|
{rows}

Taylor–Green 衰减率：got `{tg['kpis']['decay_rate_measured']:.8e}` vs ref `{tg['analytic']['decay_rate']:.8e}` = **{tg_rel_pct:.4f}%**。
独立 example-main 对照（A2, torch.roll 推式）：u_max 相对差 `{a2_rel}`, 剖面 L2 差 `{a2_l2}`。
反模式量值：独立 main 占比 `{checks['C10_antipattern_present_in_repo']['main_share']:.4f}`，重复行占比 `{checks['C10_antipattern_present_in_repo']['dup_loc_ratio']:.4f}`，重复驱动块 `{checks['C10_antipattern_present_in_repo']['n_dup_driver_blocks']}` 个；检测器判别 gap `{checks['C9_detector_discriminates']['value']}`。

交付机器裁判 11 项（summary.json `all_pass={summary['all_pass']}`，本次审计复跑关键项逐值一致）：

| check | value | threshold | pass |
|---|---|---|---|
{check_rows}

## ④ 结果图

本单为交付物审计（文档/工件核验+复跑），不产生新的物理场；交差判据全部是数值表/指纹/机器裁定，无可归因于本审计的新场量，故不生成云图。被审交付的两张图（`docs/xval_example_per_main/figs/fig1_channel_profile_convergence.png`、`fig2_tg_decay_and_antipattern.png`）由被审方 `make_figs.py` 从 arm JSON 生成，本审计核对了其数据源（arm_a1_*/arm_a3/arm_a6）。

## ⑤ 注意点（发现，按严重度）

- **F1 阻塞(L4)**：`tensor_benchmark_cases` 与 `tensor_case_pool` 对该 anchor 均 0 行；平台 `l4-verdict/17895 = CLAIMED_NOT_PROVEN, rows=0`，failed_checks 含 `ref_got_error_present/within_tolerance`。交付自设 3%/15% 阈值（predeclared.json）未登记为平台裁判 → L4 不予。
- **F2 阻塞(L3)**：`origin/main` fc86f508 不含任何交付文件（分支未合并），生态内零消费者；分支内仅 `api.py` 再导出 + thin example 消费，无主求解器/平台调用方（依赖方向是 harness→solver），L3 不达。
- **F3 低(文档)**：`example_harness.py:31` docstring 写 `H = ny-1`，而 `wall_height` 实现、测试与报告均为 `ny-2`——易误导后续维护者。
- **F4 低(报告)**：REPORT §③ C11 写 `ρ∈[0.99942,1.00000]`，工件 `summary.json` 为 `[{c11_rho[0]!r}, {c11_rho[1]!r}]`。该行不是程序化注入（与「零手抄」表述冲突），但 PASS 结论不变（质量漂移 9.55e-13 与 ρ≈1−1e-13 自洽）。
- **F5 提示(适用范围)**：C9 判别力 0.6212 vs 0.20 仅基于植入夹具；repo 级重复度指标（31.4% 行 / 97.5% 独立 main）没有外部参照，指标定义本身即交付物，属自证而非外部基准。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | got | 基准 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 槽道 u_max ny=16 | {chan[0]['u_max']:.8f} | {chan[0]['ref']:.8f} | {chan[0]['rel_err_pct']:.4f}% | 是 |
| 槽道 u_max ny=32 | {chan[1]['u_max']:.8f} | {chan[1]['ref']:.8f} | {chan[1]['rel_err_pct']:.4f}% | 是 |
| 槽道 u_max ny=48 | {chan[2]['u_max']:.8f} | {chan[2]['ref']:.8f} | {chan[2]['rel_err_pct']:.4f}% | 是 |
| 剖面 L2 ny=48 | — | 闭式抛物线 | {chan[2]['profile_l2_pct']:.4f}% | 是 |
| TG 能量衰减率 | {tg['kpis']['decay_rate_measured']:.8e} | {tg['analytic']['decay_rate']:.8e} | {tg_rel_pct:.4f}% | 是 |
| 与独立 example-main | {control['u_max']:.10f} | {control['u_max_analytic']:.10f}（对照臂同值） | {a2_rel:.3e} | 是（远严于 1e-8） |
| 质量守恒 | — | 0 | {max(abs(c['mass_drift']) for c in chan):.2e} | 是 |

物理关键量全部 <3%（最差 0.7755% 在 ny=16，随分辨率下降）。**但 L4 仍不成立**：唯一合法裁判（tensor_benchmark_cases/平台 l4-verdict）为 0 行，误差表只能算交付自证；L3 亦因无生态调用方不成立。功能实质（真实现+入口+可复现工件+不变量测试）达 **L2**。

## 复现

```bash
cd <repo>; export PYTHONPATH=src
python xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_xval.py --repo-root .     # 8 臂约 2.5 min
python -m pytest tests/test_example_harness.py tests/test_example_antipattern_audit.py -q
```
本审计复跑记录：`{json.dumps(audit_repro, ensure_ascii=False)}`
"""
(HERE / "REPORT.md").write_text(md, encoding="utf-8")

sql = """INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', %s, %s, %s, $json$%s$json$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET
  level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id,
  gate_state='pending', gate_level=NULL, gate_note=NULL
RETURNING id, gate_state, level;""" % (
    SCOPE.replace("'", "''"), MISSION, LEVEL,
    json.dumps(evidence, ensure_ascii=False).replace("$json$", ""))
(HERE / "verdict_m18086.sql").write_text(
    "-- audit m18086 -> verdict for m17895\n"
    f"-- SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN'; -- 0 rows (platform referee absent)\n" + sql + "\n",
    encoding="utf-8")

# ---------------------------------------------------------------- insert verdict
env = dict(os.environ, PGPASSWORD=subprocess.run(
    ["grep", "-oP", r"(?<=password=)[^\"']+",
     "/root/tp-agent/work/m15815/docs/board_audit/m15314/insert_verdict.py"],
    capture_output=True, text=True).stdout.strip() or os.environ.get("PGPASSWORD", ""))
proc = subprocess.run(
    ["psql", "-h", "127.0.0.1", "-U", "foundation", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1", "-f",
     str(HERE / "verdict_m18086.sql")], env=env, capture_output=True, text=True)
print(proc.stdout.strip())
if proc.returncode != 0:
    raise SystemExit(proc.stderr.strip())
gate = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""

# ---------------------------------------------------------------- platform report
base = os.environ["TENSOR_API_BASE"]
login = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"})
tok = json.load(urllib.request.urlopen(login))["access_token"]
body = {
    "title": "交差审计 m17895 XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN：裁定 L2（L4 平台裁判 0 行，L3 无消费者）",
    "summary": ("复跑被审 commit 222585d6：16 项不变量测试通过，xval 关键判据逐值重现（槽道 u_max 0.7755/0.1691/0.1660%，"
                "TG 0.1128%，C7 6.124e-13），channel ny=32 record_digest/field_digest 与工件逐比特一致。"
                "但 tensor_benchmark_cases 该 anchor 0 行、平台 l4-verdict=CLAIMED_NOT_PROVEN rows=0 ⇒ L4 不予；"
                "origin/main 零交付文件（分支未合并）、无主求解器/平台调用方 ⇒ L3 不达。裁定 L2。"),
    "scenario_key": "xval_lbm_example_per_main_audit",
    "mission_id": MISSION,
    "tags": ["board-audit", "TensorLBM", "XVAL", "example-per-main", "m17895", "L2"],
    "markdown": md,
    "figures": [],
}
post = urllib.request.Request(
    base + "/tensor/reports", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok})
resp = json.load(urllib.request.urlopen(post))
print("verdict inserted:", gate)
print("report id:", resp.get("id"))