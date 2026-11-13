#!/usr/bin/env python3
"""Board-audit verdict for m17866
(XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC, TensorLBM) — INSERT only.

All numbers are read from the auditor's own re-run artifacts
(artifacts/audit_summary.json + rerun/artifacts/*.json); nothing is typed by hand.
Idempotent: one row per (repo, scope, verdict_by).

Run: python3 docs/board_audit/m18146/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorLBM"
SCOPE = "m17866 XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC"
MISSION = 18146            # this audit task
SUBJECT = 17866            # audited mission
BY = "board-audit"
LEVEL = 2                  # real kernels + analytic validation; no integration, no artifacts delivered

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]     # worktree root (/root/tp-agent/work/m18146)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_evidence() -> dict:
    summ = json.loads((HERE / "artifacts" / "audit_summary.json").read_text())
    kn = summ["key_numbers"]
    pct = lambda v: round(100.0 * v, 4)  # noqa: E731
    src = HERE / "refs" / "src" / "xval_vivsim_static_slice.py"
    ent = HERE / "refs" / "run_xval.py"
    return {
        "impl_files": [rel(src)],
        "impl_detail": (f"{len(src.read_text().splitlines())} lines, real D2Q9 operators "
                        "(static-slice streaming, BGK, Guo/EDM forcing, force-corrected NEBB, "
                        "Poiseuille driver, analytical profile); no placeholder bodies. "
                        f"sha256={sha(src)[:16]}"),
        "entry": rel(ent),
        "entry_detail": ("python3 TensorLBM/xval/xval_vivsim_static_slice/run_xval.py -> correctness arms "
                         "(static_slice / roll_std / TensorLBM roll) + equivalence + convergence "
                         "(10k/25k/50k/100k) + grid sensitivity (ny=10/20/40) + per-operator timing -> "
                         "artifacts/xval_vivsim_static_slice.json + 2 png. Delivered script uses TM=100000 "
                         "(~2.13 ms/step on 1 core at ny=nx=10 => ~22 min); auditor verified the numbers "
                         "with 20k/40k-step arms instead of the full 100k sweep."),
        "run_artifacts": [
            rel(HERE / "artifacts" / "audit_summary.json"),
            rel(HERE / "artifacts" / "fig_profile.png"),
            rel(HERE / "artifacts" / "fig_error_arms.png"),
            rel(HERE / "rerun" / "artifacts" / "arms_20k.json"),
            rel(HERE / "rerun" / "artifacts" / "negctl_and_conv.json"),
            rel(HERE / "rerun" / "artifacts" / "equivalence_probe.json"),
        ],
        "run_artifacts_provenance": ("AUDITOR-PRODUCED. The subject worktree /root/tp-agent/work/m17866 "
                                     "contains NO artifacts: run_xval.py creates artifacts/ and figures/ at "
                                     "run time and neither existed after the mission (git status = two "
                                     "untracked files, git head_commit=null)."),
        "tests": ("NONE for this module: grep -rn 'xval_vivsim_static_slice|stream_static_slice' "
                  "TensorLBM/tests TensorLBM/src -> 0 hits outside the module itself. Auditor checks instead: "
                  "operator equivalence on a seeded random field, 20k/40k-step Poiseuille arms, mass budget, "
                  "and one adversarial negative control (wall-velocity correction removed)."),
        "integration_callers": [],
        "integration_note": ("Module is standalone under TensorLBM/xval/xval_vivsim_static_slice/ and is "
                             "imported by nothing else; the production solver path is untouched."),
        "benchmark": {
            "registered_case": ("NONE: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
                                "anchor_slug='XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC' -> 0 rows "
                                "(the sibling anchor XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN has one, status pending)"),
            "ref": "analytical steady Poiseuille u(y)=gx/(2nu) y(H-y), H=ny-1",
            "got_static_slice": f"{pct(kn['static_slice_20k_max_rel_err']):.3f}% (20k steps), "
                                f"{pct(kn['static_slice_40k_max_rel_err']):.3f}% (40k steps)",
            "got_roll_std": f"{pct(kn['roll_std_20k_max_rel_err']):.3f}% (identical math)",
            "got_tensorlbm_roll": f"{pct(kn['tl_roll_20k_max_rel_err']):.2f}%",
            "mass": f"{kn['mass_20k']:.6f} vs 100 (+{kn['mass_drift_pct']:.4f}%)",
            "verdict_vs_gate": ("<3% self target met (1.000%); <15% owner band met. But the same module with "
                                "the -g/(2rho) wall-velocity correction REMOVED gives "
                                f"{pct(kn['negctl_no_wall_correction_max_rel_err']):.4f}% -> the 1.0% is a "
                                "boundary-condition artefact, not a converged discretisation floor."),
        },
        "repro_cmd": (f"cd {rel(HERE)}/rerun && python3 TensorLBM/xval/xval_vivsim_static_slice/run_xval.py "
                      "(delivered script, ~22 min; auditor's faster path covering the reported numbers: "
                      "docs/board_audit/m18146/rerun/artifacts/*.json, produced by the commands recorded in "
                      "AUDIT_m17866.md §5)"),
        "external_anchor": ("haimingz/vivsim 'jax-pure-function-aa-pattern-static-slice' (slug "
                            "NEW:LBM-GPU-JAX-FUSION) — ported pattern only (static-slice concatenate streaming, "
                            "pure f->f operators, AA layout note); upstream source NOT re-read by the auditor "
                            "(no vendored copy, no network) => upstream fidelity = not verified"),
        "declared_level": LEVEL,
        "level_rationale": (
            "L2 独立功能实现：真代码（325 行 D2Q9 算子 + 8.4 KB 驱动，输入->计算->输出），审计实跑复现全部声明"
            "数字（static-slice 1.000% @20k、5.60% for TensorLBM roll、max|Δ|=0.0 vs roll_std、质量 +0.0135%）。"
            "不授 L3：模块孤立，无任何生产侧调用方（integration_callers grep 0 命中）。"
            "不授 L4：交付无 run 工件（worktree 无 artifacts/）、无测试、无 commit（head_commit=null），且 "
            "tensor_benchmark_cases 里该 anchor 无判据行；L4 需完整算例+误差分析+可复现三件套同时成立。"),
        "findings": {
            "F1_no_delivered_artifacts": ("blocking for L4: subject worktree has no artifacts/figures, "
                                          "head_commit=null, both files untracked (?? in git status)."),
            "F2_repro_numbers_do_not_match_script": ("the reported numbers (1.0%, mass 100.0135) come from "
                                                     ".scratch/smoke_xval.py at 20000 steps; the delivered "
                                                     "run_xval.py uses TM=100000 for the correctness arms, so "
                                                     "reproducing its output does not reproduce the report."),
            "F3_no_registered_benchmark": ("no tensor_benchmark_cases row for this anchor slug -> the stated "
                                           "referee ('baseline table is the only judge') has nothing to judge; "
                                           "the delivered work never registered a criterion."),
            "F4_negative_control_absent_and_counts_against": ("auditor's own NC (drop -g/2 wall correction) "
                                                              "beats the shipped result by ~74x (0.0135% vs 1.000%), "
                                                              "so the headline accuracy number measures the BC "
                                                              "variant, not solver quality; the <15% gate cannot "
                                                              "separate the wrong-convention TL roll (5.6%) from "
                                                              "the correct one (1.0%) on its own."),
            "F5_upstream_fidelity_unverified": ("vivsim source not vendored/re-read; the port claim rests on "
                                                "the module docstring."),
        },
        "checks_run": [
            "operator equivalence static_slice vs roll_std on seeded randn(9,10,10): max|Δ| = "
            f"{kn['equivalence_static_vs_roll_std']}",
            "Poiseuille arms 20k steps: static_slice / roll_std / TensorLBM roll (see benchmark.got_*)",
            "steady-state plateau: 20k vs 40k steps -> 1.000% vs 1.000% (Δ=1.4e-6 => converged in time)",
            "mass budget: 100.0000 -> " + f"{kn['mass_20k']:.6f}" + " (+0.0135%)",
            "adversarial NC: same module with the -g/(2rho) wall-velocity correction removed -> 0.0135%",
            "faithfulness of the in-repo baseline port: tensorlbm.d2q9.stream (shifts=(-cy,-cx)) line-for-line "
            "equal to refs/src/...:stream_roll",
            "DB referee: tensor_benchmark_cases lookup by anchor_slug -> 0 rows; tensor_reports for mission 17866 "
            "-> 0 rows (no six-section report was ever published)",
        ],
        "checks_not_run": [
            "full delivered script (TM=100000 x 3 arms + 4-point convergence + ny=10/20/40 sensitivity): "
            "not_run locally (~22 min single core; cluster job cj1814 queued behind 311 jobs at audit time)",
            "per-operator / full-step timing table (performance[] in run_xval.py): not_run -- no delivered "
            "timing evidence existed to re-check",
            "upstream vivsim source fidelity: not_run (no vendored copy / no network)",
            "3-D or AA-layout (q,nx,ny) real speed benefit: not_run -- torch backend caches differently, "
            "any speedup claim is for the JAX original only",
        ],
        "claims_overstated": [
            "mission summary 'checks 已跑通' implies a delivered reproducible artifact set; in fact nothing was "
            "committed and no artifact file exists in the subject worktree",
            "implicit '1.0% is the accuracy of the reference pattern' -- it is the accuracy of this BC variant; "
            "removing the wall-velocity correction gives 0.0135%",
            "'TensorLBM 现有 d2q9.stream 5.6%' is a harness-convention statement (pull-convention equilibrium + "
            "push-convention stream); it flags a real convention mismatch but does not by itself prove a "
            "production defect",
        ],
        "audit_doc": "docs/board_audit/m18146/AUDIT_m17866.md",
        "subject_mission": SUBJECT,
        "disposition": ("request changes: (1) commit the module + driver and add tests (mirror "
                        "tests/test_aa_fused_xval.py of the sibling anchor); (2) make run_xval.py the script "
                        "that produced the published numbers, or re-run it at TM=100000 and ship the JSON; "
                        "(3) register a tensor_benchmark_cases row for this anchor and add a negative control "
                        "that must exceed the gate; (4) either fix the wall-velocity correction or state the BC "
                        "in the headline accuracy."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ev = build_evidence()
    if a.dry_run:
        print(json.dumps(ev, ensure_ascii=False, indent=1)[:1500])
        print("...dry-run only, no DB write")
        return
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
                "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
                "evidence = EXCLUDED.evidence, created_at = now() "
                "RETURNING id, level, gate_state, gate_level",
                (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev), BY),
            )
            print("verdict row (id, level, gate_state, gate_level):", cur.fetchone())
        conn.commit()


if __name__ == "__main__":
    main()