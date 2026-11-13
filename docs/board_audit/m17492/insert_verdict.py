#!/usr/bin/env python3
"""Insert the m17492 board-audit verdict for m17450 into tensor_maturity_verdicts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUD = "/root/tp-agent/work/m17492/docs/board_audit/m17492"
DEL = "/root/tp-agent/work/m17450/benchmarks/xval_abb_inamuro_wetnode"

evidence = {
    "impl_files": [
        f"{DEL}/run.py:96-126 apply_abb_halfway (post-streaming g[opp] at fluid node -> the 30% 'standard' arm)",
        f"{DEL}/run.py:128-152 apply_abb_inamuro_wetnode (g_q(wall) = -g_pre[opp_q](wall self) + 2 w_q C_w)",
        f"{DEL}/run.py:156-241 run_case / 242-253 oss_reference block (formula claim lives only in metadata, no OSS call)",
        f"{DEL}/out/summary.json",
        f"{DEL}/docs/xval_abb_inamuro_wetnode_report.md" if False else
        "/root/tp-agent/work/m17450/docs/xval_abb_inamuro_wetnode_report.md (branch agent/m17450 @5269be0c, NOT merged into main)",
    ],
    "entry": "python3 benchmarks/xval_abb_inamuro_wetnode/run.py single|scan <args> (standalone benchmark driver; it re-implements its own D2Q5 solver, it does not call src/tensorlbm)",
    "run_artifacts": [
        f"{DEL}/out/case_H20_standard.json",
        f"{DEL}/out/case_H40_standard.json",
        f"{DEL}/out/case_H20_inamuro_wetnode.json",
        f"{DEL}/out/case_H40_inamuro_wetnode.json",
        f"{DEL}/out/summary.json",
        f"{DEL}/out/profile_comparison_H40.png",
        f"{AUD}/audit_xval_abb_results.json (P1 replay + P2-P5 independent probes)",
        f"{AUD}/audit_p6_kernel_swap.json (single-kernel-swap inside the delivered loop)",
        f"{AUD}/figs/fig1_profile_H40.png",
        f"{AUD}/figs/fig2_convergence.png",
        f"{AUD}/figs/fig3_kernel_identity.png",
    ],
    "tests": [
        "NONE: grep -rl 'inamuro|xval_abb' tests/ in the audited tree -> 0 files; mission result.checks = '' (no test command was run)",
        f"{AUD}/audit_xval_abb.py::P1_replay (auditor re-ran the vendored run.py: committed L2 1.45436%/0.696392% replay delta 0.0, profile_max_diff 0.0 -> got reproducible)",
        f"{AUD}/audit_p6_kernel_swap.py (auditor: same loop, one kernel swapped: delivered vs equilibrium vs OSS-ABB)",
    ],
    "integration_callers": [
        "NONE: grep -rn 'def apply_abb' /root/tp-agent/work/m17450/src/ -> 0 hits; the deliverables live only in benchmarks/, src/tensorlbm has no entry point",
        "branch agent/m17450 @5269be0c is NOT an ancestor of main in the audited tree (git branch -a --contains -> only agent/m17450) -> not landed",
        "related pre-existing library idiom (not delivered here): /root/tp-agent/work/m17450/src/tensorlbm/convective_mass_transfer.py:441 g[2,0,:] = -g[4,0,:] + 2*w[2]*c_wall (same wall-self sampling pattern)",
    ],
    "benchmark": {
        "machine_referee": "ABSENT: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE' -> 0 rows (no registered case, metric or tolerance; referee idles)",
        "declared_criterion": "<15% relative L2 error vs analytic linear profile (owner precision standard); no tolerance row in DB",
        "ref_vs_got": {
            "delivered_kernel_vs_report_halfcell_ref": {"H20": 0.014544, "H40": 0.006964, "pass": True,
                                                        "note": "reproduced bit-exactly; but the half-cell reference belongs to the ABB convention while the kernel fixes the wall ON the node -> reference/kernel convention mismatch"},
            "delivered_kernel_vs_node_ref_Y_over_Hplus1": {"H20": 0.009348, "H40": 0.005273, "H80": 0.003885,
                                                           "pass": True, "note": "honest reference for a node Dirichlet; still <3% but ~1.5x the reported error and order decays 0.83 -> 0.44"},
            "OSS_ABB_rederivation_vs_report_halfcell_ref": {"H20": 0.0, "H40": 0.0, "H80": 0.0,
                                                            "pass": True, "measured_max_abs_err": 1.4e-07,
                                                            "note": "the ABB of film_antibb.cpp reproduces the report's own reference to machine zero -> the delivered kernel is NOT that method"},
            "kernel_identity_delivered_vs_equilibrium_wetnode": {"H20": 0.0, "H40": 0.0, "H80": 0.0,
                                                                 "unit": "max abs profile difference",
                                                                 "note": "bit-identical in all three grids (float32 AND float64) -> the delivered 'anti-bounce-back' kernel carries zero ABB content; its wall recursion has the fixed point g_q = w_q C_w"},
            "convergence_order_reported_vs_measured": {"reported": 1.062, "measured_vs_node_ref": [0.826, 0.441],
                                                       "verdict": "reported order is an artefact of the reference/kernel convention mismatch (mismatch scales as 1/H)"},
        },
        "why_not_L4": "(1) anchor method not implemented: delivered kernel == equilibrium wet node bit-exactly, while the OSS ABB (film_antibb.cpp:152-205, sampling the adjacent fluid node between collide and stream) yields machine-zero error in the same loop -> the 'cross-validation of ABB' compares two variants neither of which is the OSS ABB; (2) reference misaligned with the wall convention, so the error table and the claimed order 1.05 are both artefacts; (3) no OSS output ever produced (the oss_reference block at run.py:242-253 is a hard-coded dict of file:line strings, no call, no sha pin of the upstream tree); (4) no machine referee row, no test, no library integration, branch not merged",
    },
    "repro_cmd": "cd /root/tp-agent/work/m17492/docs/board_audit/m17492 && OMP_NUM_THREADS=1 python3 audit_xval_abb.py && OMP_NUM_THREADS=1 python3 audit_p6_kernel_swap.py && OMP_NUM_THREADS=1 python3 make_figs.py   # deliverable itself: python3 /root/tp-agent/work/m17450/benchmarks/xval_abb_inamuro_wetnode/run.py single 40 out.json --max-steps 1000",
    "external_anchor": [
        "lbm-principles-practice/code (Kruger textbook code, MIT) cloned at 6e2c592; verified file existence AND line ranges: chapter8/film_antibb.cpp 347 lines, 152-205 = update_bounce_back() with f2[off+3] = -f2[(adjacent fluid node)*NPOP+1] + 2*weights[3]*conc_wall  -> ABB sampled at the ADJACENT FLUID node",
        "chapter8/film_inamuro.cpp 414 lines, 157-215 = update_bounce_back() implementing the INAMURO mass-fitting BC: dense_temp=(conc_wall - sum(f_known))/sum(w_unknown_unknown); f2[off+k]=weights[k]*dense_temp -> NOT the -g_opp+2*w*C formula the deliverable attributes to it",
        "chapter5/couette_wetnode.m 163 lines, 111-146 = wet-node rho back-solve + wetnode==1/2/3 (equilibrium / non-equilibrium extrapolation / non-equilibrium bounce-back); contains no ABB either",
        "consequence: the deliverable's claim 'formula identical to film_inamuro.cpp, applied at wall node (Inamuro wet-node)' is falsified by the upstream source itself; the formula it does implement matches the film_antibb.cpp shape only on the wall-node side, and samples the wrong node",
    ],
    "db_state": {
        "tensor_dev_missions": "id=17450 status=succeeded finished_at=2026-11-13T06:09:33Z report_owed=f report_id=NULL checks=''",
        "tensor_reports": "0 rows for mission_id=17450 and 0 rows whose title/markdown mentions this anchor -> no paper-grade report was ever published (only a 116-line md inside the unmerged branch)",
        "tensor_benchmark_cases": "0 rows for the anchor",
    },
    "level_rationale": {
        "L1": "met: analytic linear-profile comparison, steady state confirmed (1000 vs 3000 steps max diff 0.0), numbers reproducible bit-exactly",
        "L2": "NOT met: the anchored functional capability (anti-bounce-back Dirichlet) is absent - the shipped kernel is bit-identical to the equilibrium wet node (max profile diff 0.0e+00 at H=20/40/80), a form the library already has at src/tensorlbm/convective_mass_transfer.py:441; the D2Q5 solver is a benchmark-local duplicate with no src/ entry point, so no new functional capability was added to TensorLBM. What exists is a verification driver (analytic linear-profile comparison) = L1 substance. Aligning with the m17085 precedent (verification driver whose claimed capability is absent -> L1).",
        "L3": "NOT met: src/tensorlbm has no such kernel/entry point, 0 callers, branch never merged to main",
        "L4": "NOT met: anchored method absent (kernel == equilibrium wet node), reference/convention mismatch invalidates the error table and the claimed order, no OSS output comparison, no machine referee row, no tests",
        "L5": "NOT met",
        "level": 1,
    },
    "findings": [
        {"sev": "blocking", "id": "F1-kernel-is-not-ABB", "claim": "delivered apply_abb_inamuro_wetnode implements anti-bounce-back", "evidence": "max|profile(delivered) - profile(equilibrium wet node)| = 0.000e+00 at H=20/40/80 both float32 and float64; analytic fixed point of the wall recursion g^{n+1}_q = -g^{post}_opp(wall) + 2 w_q C_w is g_q = w_q C_w", "fix": "sample the adjacent fluid node's post-collision population between collide and stream (film_antibb.cpp:152-205)"},
        {"sev": "blocking", "id": "F2-external-anchor-misattributed", "claim": "formula identical to lbm-principles-practice chapter8/film_inamuro.cpp", "evidence": "upstream @6e2c592: film_inamuro.cpp:157-215 is the Inamuro mass-fitting BC (dense_temp=(C_w - sum f_known)/sum w_unknown), the -g_opp+2wC form lives in film_antibb.cpp:152-205 and samples the adjacent fluid node", "fix": "cite film_antibb.cpp for ABB; implement film_inamuro.cpp separately if the Inamuro method is the target"},
        {"sev": "major", "id": "F3-reference-convention-mismatch", "claim": "L2 error 1.45%/0.70% and order 1.05 validate the wet-node ABB", "evidence": "kernel fixes the wall on the node (gap H+1) but the reference is C=(Y-0.5)/H; honest node reference gives 0.935%/0.527%/0.389% with order 0.83 -> 0.44; the OSS ABB (half-cell convention) matches the report's reference to machine zero", "fix": "pair each wall convention with its own analytic reference; report three grids in the same convention"},
        {"sev": "major", "id": "F4-standard-arm-misdiagnosed", "claim": "30% error of the 'standard half-way ABB' is a position-convention difference", "evidence": "apply_abb_halfway (run.py:96-126) uses post-streaming populations and writes the fluid node's same-direction slot; the correct ABB in the same loop gives 0.0000%", "fix": "use post-collision populations and write the slot that streaming pulls into the fluid, or delete the arm"},
        {"sev": "major", "id": "F5-no-machine-referee-no-report", "claim": "cross-validation completed", "evidence": "tensor_benchmark_cases 0 rows for the anchor; tensor_reports 0 rows for mission 17450 (report_id NULL) though status=succeeded; tests 0 references; branch unmerged", "fix": "register case_name/metric/tolerance; publish the six-section report; merge"},
    ],
    "disposition": "request_changes",
    "remediation": [
        "implement the ABB exactly as chapter8/film_antibb.cpp:152-205 (adjacent fluid node, collide->stream window) as a reusable kernel in src/tensorlbm/ with a unit test asserting machine-zero error for the linear scalar Couette case",
        "if the Inamuro method is what the anchor wants, implement film_inamuro.cpp:157-215 (mass fitting) as a separate kernel and tabulate ABB vs Inamuro differences",
        "pair reference with wall convention (half-cell <-> (Y-0.5)/H, node <-> Y/(H+1)); report convergence on H=20/40/80/160 within one convention",
        "register anchor row in tensor_benchmark_cases (case_name/metric/tolerance)",
        "fix or remove the 'standard' arm; publish the paper-grade report; merge agent/m17450 into main",
    ],
}

scope = "m17450 XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE"
sql = """
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', $json$%s$json$, 17492, 1, $json$%s$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;
""" % (scope, json.dumps(evidence, ensure_ascii=False))

(HERE / "verdict_m17492.sql").write_text(sql)
print(sql[:400])
print("...", len(sql), "bytes")
out = subprocess.run(
    ["psql", "dbname=tensorplatform user=foundation host=/var/run/postgresql", "-v", "ON_ERROR_STOP=1", "-f",
     str(HERE / "verdict_m17492.sql")], capture_output=True, text=True)
print("STDOUT:", out.stdout)
print("STDERR:", out.stderr[-800:])
print("rc:", out.returncode)