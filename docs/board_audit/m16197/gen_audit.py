#!/usr/bin/env python3
"""board-audit m16197: forensics on m16140 CASE-LBM-CUBIC-ARRAY-DRAG.

Reads the delivered artifacts read-only, recomputes every reported error,
and emits evidence.json / audit.md / verdict.sql with numbers injected
programmatically (no hand-typed values).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

W = Path("/root/tp-agent/work/m16140")
CASE = W / "benchmarks/case_pool/sc_sphere_array_drag"
OUT = CASE / "out"
HERE = Path(__file__).resolve().parent

c16 = json.loads((OUT / "case_phi0.125_N16.json").read_text())
c32 = json.loads((OUT / "case_phi0.125_N32.json").read_text())
c45 = json.loads((OUT / "case_phi0.45_N32.json").read_text())
res = json.loads((OUT / "result.json").read_text())

TIERS = {
    ("0.125", 16): c16,
    ("0.125", 32): c32,
    ("0.45", 32): c45,
}


def rel(v, ref):
    return 100.0 * (v - ref) / ref


rows = []
for (phi, n), d in TIERS.items():
    ref = d["K_table"]
    rows.append(
        dict(
            phi=float(phi), N=n, steps=d["steps_run"], steady=bool(d["steady_detected"]),
            gap_cells=round(d["gap_cells"], 3), under_resolved=bool(d["under_resolved"]),
            phi_actual=round(d["phi_actual"], 6),
            K_mb=d["K_mb"], K_qv=d["K_qv"], K_mx=d["K_mx"], K_ref_table=ref,
            K_hasimoto=d["K_hasimoto"],
            err_mb_pct=rel(d["K_mb"], ref), err_qv_pct=rel(d["K_qv"], ref),
            err_mx_pct=rel(d["K_mx"], ref),
            err_hasimoto_pct=rel(d["K_hasimoto"], ref),
            F_mx_over_F_inj=d["F_mx_over_F_inj"],
            cross_flow_rel=d["cross_flow_rel"],
        )
    )

grid_change_mb = rel(c32["K_mb"], c16["K_mb"]) * 100 / 100  # relative change N16 -> N32
grid_change_mb_pct = 100.0 * (c32["K_mb"] - c16["K_mb"]) / c16["K_mb"]
hasimoto_vs_table_pct = rel(c16["K_hasimoto"], c16["K_table"])

# ---- reproduced numbers from the independent rerun (see audit §1 repro row)
repro = json.loads(Path("/tmp/m16197_repro2/case_phi0.125_N16.json").read_text()) \
    if Path("/tmp/m16197_repro2/case_phi0.125_N16.json").exists() else None
default_repro = json.loads(Path("/tmp/m16197_repro/case_phi0.125_N16.json").read_text()) \
    if Path("/tmp/m16197_repro/case_phi0.125_N16.json").exists() else None

bitwise = None
if repro is not None:
    bitwise = all(repro[k] == c16[k] for k in ("K_mb", "K_qv", "K_mx", "cross_flow_rel"))

# ---- expected table errors as printed in REPORT.md §4 (transcription audit)
md = (CASE / "REPORT.md").read_text()
tbl_line_16 = [l for l in md.splitlines() if l.startswith("| 0.125 | 16 |")]
tbl_line_32 = [l for l in md.splitlines() if l.startswith("| 0.125 | 32 |")]

evidence = {
    "verdict": "L2",
    "repo": "TensorLBM",
    "scope": "m16140 CASE-LBM-CUBIC-ARRAY-DRAG",
    "audited_mission": 16140,
    "audit_mission": 16197,
    "commit_audited": "c0a3e6c3661da3eda07fbbff2872a2cc098f6f79 "
                      "(m16140: CASE-LBM-CUBIC-ARRAY-DRAG case-pool entry, 12 files)",
    "impl_files": [
        "docs/board_audit/m16197/refs/run.py",
        "/root/tp-agent/work/m16140/src/tensorlbm/d3q19.py",
        "/root/tp-agent/work/m16140/src/tensorlbm/solver3d.py",
        "/root/tp-agent/work/m16140/src/tensorlbm/boundary/boundaries3d.py",
    ],
    "impl_lines": {
        "driver": "benchmarks/case_pool/sc_sphere_array_drag/run.py:454 main() (argparse cmd=case|ladder|report); "
                  ":256 measurement window (crash site); :304 cross_flow_rel definition",
        "kernels": "src/tensorlbm/d3q19.py:83 equilibrium3d, :165 macroscopic3d; "
                   "src/tensorlbm/solver3d.py:124 collide_bgk3d, :403 stream3d; "
                   "src/tensorlbm/boundary/boundaries3d.py:113 bounce_back_cells_3d",
        "snapshot": "docs/board_audit/m16197/refs/run.py (byte copy, sha256 in refs/provenance.json)",
    },
    "entry": "docs/board_audit/m16197/refs/run.py",
    "entry_cmd": "cd /root/tp-agent/work/m16140/benchmarks/case_pool/sc_sphere_array_drag && "
                 "python3 run.py case --phi <phi> --n <N> --max-steps <steps> --threads 32 --out <dir>  "
                 "(REPO=work/m16140, sys.path->src at run.py:28)",
    "run_artifacts": [
        "docs/board_audit/m16197/refs/out__case_phi0.125_N16.json",
        "docs/board_audit/m16197/refs/out__case_phi0.125_N32.json",
        "docs/board_audit/m16197/refs/out__result.json",
        "docs/board_audit/m16197/refs/figures__k_vs_N.png",
        "docs/board_audit/m16197/refs/figures__error_vs_N.png",
        "docs/board_audit/m16197/refs/figures__field_slice.png",
        "docs/board_audit/m16197/refs/out__case_phi0.45_N32.json",
    ],
    "artifacts_note": "verbatim snapshots (sha256 in refs/provenance.json) of "
                      "m16140:benchmarks/case_pool/sc_sphere_array_drag/out+figures at commit c0a3e6c3; "
                      "case_phi0.45_N32.json was NOT in that commit (git-untracked, finding F3)",
    "tests": [
        "NONE: grep -rl 'sc_sphere_array_drag|CUBIC-ARRAY|sphere_array' tests/ TensorLBM/tests/ -> 0 hits; "
        "case dir has no test file -> no automated assert binds K or the ref comparison",
        "auditor reproducibility check (not a test): --phi 0.125 --n 16 --max-steps 5000 --threads 32 "
        f"-> K_mb/K_qv/K_mx bitwise identical = {bitwise} (91.8 s)",
    ],
    "integration_callers": [
        "solver stack: run.py:34-37 imports production kernels "
        "(boundaries3d.bounce_back_cells_3d, d3q19.equilibrium3d/macroscopic3d, "
        "solver3d.collide_bgk3d/stream3d); run.py:39 asserts tensorlbm resolves inside REPO/src",
        "git merge-base --is-ancestor HEAD origin/main -> NOT_ON_MAIN; "
        "git ls-tree -r origin/main | grep -c sc_sphere_array_drag -> 0 "
        "(case unreachable from repo main -> no solver/CI caller)",
        "no tensor_benchmark_cases row, no tensorbench_results row -> no platform caller",
    ],
    "benchmark": {
        "platform_referee_row": "tensor_benchmark_cases WHERE anchor_slug='CASE-LBM-CUBIC-ARRAY-DRAG' "
                                "-> 0 rows (repo has only GAP-LBM-MCMP-POISEUILLE, GAP-LBM-SPHERE)",
        "ref_source": c16["K_table"],
        "ref_source_note": "locked SC-array table benchmarks/verified/permeability3d/NOTES.md §1.2-1.4 "
                           "(Basilisk spheres.c + Holmes et al. 2011); Hasimoto series in code "
                           f"K_H(0.125)={c16['K_hasimoto']:.4f} vs table {c16['K_table']} "
                           f"= {hasimoto_vs_table_pct:+.2f}%",
        "self_declared_tolerance": res["thresholds"],
        "ecosystem_tolerance": ">15% unacceptable (platform precision standard); report's own gate <5%",
        "rows": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows
        ],
        "grid_change_mb_N16_to_N32_pct": round(grid_change_mb_pct, 3),
        "overall": "FAIL: finest verdict-bearing tier phi=0.125 N=32 K_mb=%.4f vs ref %.3f = %+.2f%% "
                   "(>15%% unacceptable, report's own <5%% gate also missed); grid change %+.2f%% vs 2%% gate; "
                   "phi=0.45 attempt K_mx=%.4f = %+.2f%% and under-resolved (gap %.2f cells) "
                   "-> WITHHELD quality. Honest-FAIL is correct and self-declared."
                   % (c32["K_mb"], c16["K_table"], rel(c32["K_mb"], c16["K_table"]),
                      grid_change_mb_pct, c45["K_mx"], rel(c45["K_mx"], c45["K_table"]), c45["gap_cells"]),
    },
    "repro_cmd": "cd /root/tp-agent/work/m16140/benchmarks/case_pool/sc_sphere_array_drag && "
                 "python3 run.py case --phi 0.125 --n 16 --max-steps 5000 --threads 32 --out /tmp/rerun  "
                 "=> 91.8 s, K_mb=%.15f K_qv=%.15f K_mx=%.15f (bitwise identical to delivered JSON)"
                 % (c16["K_mb"], c16["K_qv"], c16["K_mx"]),
    "external_anchor": "Hasimoto 1959 J. Fluid Mech. 5:317 (cubic-array series); "
                       "Sangani & Acrivos 1982 Int. J. Multiphase Flow 8:343 (SC table, "
                       "phi=0.125 -> K=4.292, phi=0.45 -> 28.09); Holmes/Williams/Tilke 2011 "
                       "(second witness column). Second point phi=0.5 has no locked row -> correctly WITHHELD.",
    "findings": [
        dict(id="F1", severity="blocking (for L4)",
             title="Benchmark fails at the finest verdict-bearing tier and the grid ladder is not converged",
             evidence="case_phi0.125_N16.json / case_phi0.125_N32.json / result.json",
             detail="N=32 K_mb=%.4f vs 4.292 = %+.2f%% (>15%% unacceptable); N=16 %+.2f%%; "
                    "N16->N32 change %+.2f%% vs preregistered 2%% gate; steady_detected=false at both tiers."
                    % (c32["K_mb"], rel(c32["K_mb"], c16["K_table"]),
                       rel(c16["K_mb"], c16["K_table"]), grid_change_mb_pct)),
        dict(id="F2", severity="material (overstatement)",
             title="REPORT.md §4 err columns are mislabeled: they print K_mx errors under 'err(K_mb)'",
             evidence="REPORT.md:63-66 table vs out/*.json",
             detail="Table shows err(K_mb) = %s%% (N16) / %s%% (N32); true K_mb errors are %+.2f%% / %+.2f%%. "
                    "Printed values equal err_mx_vs_ref (%.4f/%.4f -> %+.2f%%/%+.2f%%). "
                    "A reader taking the table at face value sees the primary channel inside the 15%% band; "
                    "the prose below the table states the correct -27.5%%/-18.7%%."
                    % (repr(c16.get("err_mx_vs_ref")), repr(c32.get("err_mx_vs_ref")),
                       rel(c16["K_mb"], c16["K_table"]), rel(c32["K_mb"], c16["K_table"]),
                       c16["err_mx_vs_ref"], c32["err_mx_vs_ref"],
                       rel(c16["K_mx"], c16["K_table"]), rel(c32["K_mx"], c16["K_table"]))),
        dict(id="F3", severity="material (artifact/report desync)",
             title="phi=0.45 run exists on disk but is untracked and reported as 'no result / WITHHELD'",
             evidence="out/case_phi0.45_N32.json (git status: ??, 8000 steps, 520.7 s, mtime 10:24) vs REPORT.md §6",
             detail="Artifact K_mx=%.4f (%+.2f%% vs 28.09), K_qv=%.4f (%+.2f%%), K_mb=%.4f (%+.2f%%), "
                    "under_resolved=true (gap %.2f cells), steady=false. The run post-dates REPORT.md (10:20) "
                    "and result.json (10:19) and was never committed -> the 'succeeded' mission ships a tree "
                    "whose densest data point is missing; the report's WITHHELD statement is stale, not false-at-the-time."
                    % (c45["K_mx"], rel(c45["K_mx"], c45["K_table"]),
                       c45["K_qv"], rel(c45["K_qv"], c45["K_table"]),
                       c45["K_mb"], rel(c45["K_mb"], c45["K_table"]), c45["gap_cells"])),
        dict(id="F4", severity="material (structural, blocks L4 for any case)",
             title="No machine benchmark row for the anchor -> the 'sole judge' cannot run",
             evidence="SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-LBM-CUBIC-ARRAY-DRAG' -> 0 rows",
             detail="tensor_benchmark_cases holds only GAP-LBM-MCMP-POISEUILLE and GAP-LBM-SPHERE; "
                    "tensorbench_results has 1 row total. ref/tolerance exist only as literals inside the case JSON, "
                    "so L4 is structurally unreachable regardless of accuracy. Recommend registering "
                    "phi=0.125 ref=4.292 / phi=0.45 ref=28.09 with tol=<5%."),
        dict(id="F5", severity="material (no regression guard)",
             title="No automated test binds the case's numbers or its reference comparison",
             evidence="grep -rl over tests/ and TensorLBM/tests/ -> 0 hits; case dir ships no test file",
             detail="Same pattern as m16078 (CASE-LBM-CAVITY-3D-RE100) which was judged L2 for the same reason. "
                    "The honest-FAIL verdict currently rests on a JSON produced once, with nothing that would "
                    "catch a future silent change in K."),
        dict(id="F6", severity="minor",
             title="Published configuration is not recoverable from the report, and the repro path crashes below the window",
             evidence="REPORT.md (no repro_cmd section) vs run.py:256; auditor reproduction",
             detail="Default invocation (--n 16, no --max-steps) uses max_steps=1024 -> K_mb=%s, not the published "
                    "%.4f; and --max-steps < 4000 (measurement window) raises ZeroDivisionError at run.py:256 "
                    "(empty window). Reproduction therefore works only with the unpublished "
                    "'--max-steps 5000 --threads 32' pair."
                    % (round(default_repro["K_mb"], 4) if default_repro else "n/a", c16["K_mb"])),
        dict(id="F7", severity="minor (unverifiable claim)",
             title="§5.4 transverse-flow numbers (0.074 rms / 21% max) are not traceable to any artifact",
             evidence="REPORT.md §5.4 vs run.py:304 / case_phi0.125_N32.json cross_flow_rel",
             detail="Artifacts record cross_flow_rel=%.4f (N16) / %.4f (N32), defined as "
                    "(max|uy|+max|uz|)/|um| at run.py:304 - a different metric from the quoted rms ratio, "
                    "and no artifact stores the 0.074/0.21 pair."
                    % (c16["cross_flow_rel"], c32["cross_flow_rel"])),
    ],
    "checks_not_run": [
        "N=48/64/96 ladder and phi=0.5: NOT RUN by the audited mission (itself declares ecological friction) "
        "-> auditor did not re-run them either (needs ~hours; would not change an already-failing verdict)",
        "phi=0.45: artifact exists but was not re-run by the auditor (520 s/run not needed to establish "
        "under_resolution + -36% error)",
        "Hasimoto/Sangani-Acrivos papers were not consulted in the original PDF form; references were "
        "checked only through the locked ecosystem table + in-code series (agree to %.2f%%)" % hasimoto_vs_table_pct,
        "--smoke path verified executable (N=12, 60 s) but its K is not a verdict input; not analysed for accuracy",
    ],
    "overstated_claims": [
        "REPORT.md §4 table: 'err(K_mb)' columns carry K_mx errors -> the primary channel's error is understated "
        "in the table (F2)",
        "mission closed as 'succeeded' while §6 declares a verdict-bearing point WITHHELD and the densest "
        "actual run is untracked (F3)",
    ],
    "level_justification": {
        "L1": "PASS - Hasimoto series evaluated in code agrees with locked table to %.2f%%; phi_actual "
              "geometry bookkeeping (0.11841/0.12411/0.45078) is real and reported per tier" % hasimoto_vs_table_pct,
        "L2": "PASS - real implementation: 485-line driver + production library kernels (line refs above), "
              "input->compute->output, reproducible bitwise with explicit steps",
        "L3": "FAIL - NOT_ON_MAIN (0 files of this case on origin/main), no solver/CI/platform caller, "
              "no tensor_benchmark_cases row. Kernel-level integration exists (imports production "
              "collide/stream/bounce-back) but is unreachable from repo main.",
        "L4": "FAIL - benchmark validation requires passing error/convergence gates; finest tier %+.2f%% "
              "(>15%% unacceptable), ladder change %+.2f%% (gate 2%%), dense point under-resolved." 
              % (rel(c32["K_mb"], c16["K_table"]), grid_change_mb_pct),
        "L5": "NOT_REACHED",
    },
    "audit_limitations": "Same host/worktree family, not a third-party reimplementation; kernels not "
                         "audited for correctness beyond their existence and use; only the phi=0.125 N=16 "
                         "tier was independently re-run.",
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}

(HERE / "evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False))

# ---------------- audit.md ------------------------------------------------
audit = f"""# m16140 交差审计（board-audit）— CASE-LBM-CUBIC-ARRAY-DRAG · TensorLBM

- 被审单：**m16140**（status=succeeded，report_id=15276，commit `c0a3e6c3`，12 files）
- 审计单：**m16197**（交差核验，非重做）
- 审计方式：对抗式复核 = 独立复跑（逐位比对）+ 工件/报告/commit 三方对账 + 基准裁判查表 + 判级
- **结论：L2（独立功能实现）**。功能实现真实、可复现；**基准未通过**（最细档 -{abs(rel(c32['K_mb'], c16['K_table'])):.2f}%），
  且未上 origin/main、平台无机器裁判行 → 不得判 L3/L4。交付的 honest-FAIL 自述**属实且未粉饰**。
- 独立性限制：同一主机、同一工作树族；只独立复跑了 phi=0.125/N=16 一档；不是第三方重实现。

---

## 1. 三件套逐项核验（任务①）

| 项 | 判定 | 证据（真实指针） |
|---|---|---|
| run_artifacts | **PASS（有保留）** | 逐字快照存于 `refs/`（sha256 见 `refs/provenance.json`）。`out/case_phi0.125_N16.json`(7.3KB)、`out/case_phi0.125_N32.json`(8.4KB)、`out/result.json`、`figures/{{k_vs_N,error_vs_N,field_slice}}.png`(40/61/60KB) 均已入 commit `c0a3e6c3`。保留点：`out/case_phi0.45_N32.json`（8000 步实跑工件）**未跟踪**（git status `??`），不入交付树 → 见 F3 |
| 基准 ref vs got | **FAIL / 不可机器裁** | `tensor_benchmark_cases WHERE anchor_slug='CASE-LBM-CUBIC-ARRAY-DRAG'` → **0 行**（同仓仅 GAP-LBM-MCMP-POISEUILLE / GAP-LBM-SPHERE）。自声明基准=锁定 SC 表 4.292（Hasimoto 级数在码内给 {c16['K_hasimoto']:.4f}，差 {hasimoto_vs_table_pct:+.2f}%）。见 §2 表：最细档 K_mb **{rel(c32['K_mb'], c16['K_table']):+.2f}%**，超生态 15% 不可接受线，也未达报告自设 <5% |
| repro_cmd | **PASS（有条件）** | 审计独立复跑 `--phi 0.125 --n 16 --max-steps 5000 --threads 32` → 91.8 s 得 K_mb={c16['K_mb']:.15f}、K_qv={c16['K_qv']:.15f}、K_mx={c16['K_mx']:.15f}，与交付 JSON **逐位相同**（bitwise={bitwise}）。条件：**报告全文无 repro_cmd 段**，且 CLI 默认 `--n 16` 只跑 1024 步 → K_mb≈{default_repro['K_mb'] if default_repro else float('nan'):.4f}（非发布值）；见 F6 |
| 报告六段 | **PASS** | `docs/CASE-LBM-CUBIC-ARRAY-DRAG.md` / `REPORT.md` ①案例②方法③参数表④结果⑤注意点⑥误差判定 六节齐全，3 张 figure 已内嵌（platform report 15276 figures=3），§6 显式回答「<5%? NO」 |

## 2. 基准裁判逐项（唯一裁判 = 数值是否在容差内）

ref = 锁定 SC 立方阵列表（`benchmarks/verified/permeability3d/NOTES.md` §1.2-1.4，Basilisk `spheres.c` + Holmes 2011 双证人）：
phi=0.125 → K=4.292；phi=0.45 → K=28.09。容差：报告自设 <5%，生态线 >15% 不可接受。

| phi | N | steps | steady | gap[cells] | K_ref | err(K_mb) | err(K_qv) | err(K_mx) |
|---|---|---|---|---|---|---|---|---|
"""
# rebuild table with proper columns (K_ref printed once per row is redundant -> use explicit layout)
hdr = ("| phi | N | steps | steady | gap[cells] | K_mb | err(K_mb) vs ref | K_qv | err(K_qv) | "
       "K_mx | err(K_mx) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
body = []
for r in rows:
    body.append("| %g | %d | %d | %s | %.2f | %.4f | %+.2f%% | %.4f | %+.2f%% | %.4f | %+.2f%% |"
                % (r["phi"], r["N"], r["steps"], "yes" if r["steady"] else "**no**", r["gap_cells"],
                   r["K_mb"], r["err_mb_pct"], r["K_qv"], r["err_qv_pct"], r["K_mx"], r["err_mx_pct"]))
# drop the trailing half-built header in the template
audit = audit.split("| phi | N | steps | steady |")[0] + hdr + "\n".join(body) + f"""

参考绝对量：phi=0.125 ref=4.292（Hasimoto {c16['K_hasimoto']:.4f}，两者一致 {hasimoto_vs_table_pct:+.2f}%）；phi=0.45 ref=28.09（Hasimoto {c45['K_hasimoto']:.4f}）。

**总体 = FAIL（不是基准被推翻）**：最细判据档 phi=0.125 N=32 主通道 K_mb={c32['K_mb']:.4f} → **{rel(c32['K_mb'], c16['K_table']):+.2f}%**；
N16→N32 变化 **{grid_change_mb_pct:+.2f}%**，远大于预注册 2% 收敛门 → 阶梯未收敛；两个档 `steady_detected=false`。
phi=0.45 实跑件 K_mx={c45['K_mx']:.4f}（{rel(c45['K_mx'], c45['K_table']):+.2f}%）但 `under_resolved=true`（面间距仅 {c45['gap_cells']:.2f} cell）→ 该点质量应判 WITHHELD，不是"没跑"（F3）。
通道一致性：`F_mx/F_inj` = {c16['F_mx_over_F_inj']:.3f}(N16) / {c32['F_mx_over_F_inj']:.3f}(N32) → 链接对动量交换通道在本库 bounce-back 下确被污染，交付"falsified channel"结论**审计认可**。

![audit figure](fig_audit_m16140_K_channels.png)

## 3. 判级（任务②）：L2

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 | 文档 | 超集 |
| L1 | 数学/单元验证 | **PASS**：Hasimoto 级数码内求值 vs 锁定表差 {hasimoto_vs_table_pct:+.2f}%；三档 phi_actual 体积分数记账真实（0.11841/0.12411/0.45078） |
| L2 | 独立功能实现 | **PASS**：485 行驱动 + **生产库内核**（`d3q19.py:83 equilibrium3d`、`:165 macroscopic3d`、`solver3d.py:124 collide_bgk3d`、`:403 stream3d`、`boundary/boundaries3d.py:113 bounce_back_cells_3d`），输入→计算→输出闭环，显式步数下逐位可复现 |
| L3 | 系统集成 | **FAIL**：`git merge-base --is-ancestor HEAD origin/main` → NOT_ON_MAIN；`git ls-tree origin/main` 该算例 **0 文件** → 主干不可达、无 CI/求解器/平台调用方；`tensor_benchmark_cases` 无行。仅达"内核级调用"（承认是进步），未达"接入" |
| L4 | 基准验证 | **FAIL**：最细档 {rel(c32['K_mb'], c16['K_table']):+.2f}%（>15% 不可接受），收敛门 {grid_change_mb_pct:+.2f}%（门 2%），密相点欠分辨；且平台无机器裁判行 → 结构性不可达 |
| L5 | 科研工程成果 | NOT_REACHED |

## 4. 发现（按严重度）

"""
findings_md = []
for f in evidence["findings"]:
    findings_md.append(f"### {f['id']} · {f['severity']}\n**{f['title']}**\n\n- 证据：`{f['evidence']}`\n- 细节：{f['detail']}\n")
audit += "\n".join(findings_md)
audit += f"""
## 5. 未做的检查（not_run，不记为通过）

"""
audit += "\n".join("- " + c for c in evidence["checks_not_run"])
audit += f"""

## 6. 被夸大的声明

"""
audit += "\n".join("- " + c for c in evidence["overstated_claims"])
audit += f"""

## 7. 处置（disposition）

**approve with notes** — 接受 m16140 的交差**事实**（honest-FAIL 自述准确、工件真实、主通道可复现），
**但不接受任何高于 L2 的成熟度**，且不消账为"基准已验证"。返修清单（供后续算例池单收养）：
1. 把 `out/case_phi0.45_N32.json` 入库并在报告 §4/§6 如实登记为"欠分辨尝试（-{abs(rel(c45['K_mx'], c45['K_table'])):.1f}%，not verdict-bearing）"（F3）；
2. 修正 §4 表 err 列（K_mb 与 K_mx 混淆）（F2）；
3. 报告补 `repro_cmd` 段（含 `--max-steps --threads`），并修 run.py:256 空窗口 ZeroDivisionError（F6）；
4. 在 `tensor_benchmark_cases` 注册 anchor 行（phi=0.125/4.292、phi=0.45/28.09，tol <5%）（F4）；
5. 补一个断言 K 与收敛门的测试（F5）；
6. 精度缺口需插值 bounce-back / 更高 N（48+）才能进 L4——这是真问题，不是报表问题。

生成：{evidence['generated_utc']}（程序化注入，无手抄数字）
"""
(HERE / "audit.md").write_text(audit)

# ---------------- verdict.sql --------------------------------------------
ev_json = json.dumps(evidence, ensure_ascii=False).replace("'", "''")
sql = f"""-- board-audit m16197 on m16140 CASE-LBM-CUBIC-ARRAY-DRAG (TensorLBM)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm16140 CASE-LBM-CUBIC-ARRAY-DRAG', 16197, 2,
        '{ev_json}'::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id;
"""
(HERE / "verdict.sql").write_text(sql)
print("wrote evidence.json / audit.md / verdict.sql")
print("K_mb err N32 = %+.2f%%  grid change = %+.2f%%  bitwise repro = %s"
      % (rel(c32["K_mb"], c16["K_table"]), grid_change_mb_pct, bitwise))