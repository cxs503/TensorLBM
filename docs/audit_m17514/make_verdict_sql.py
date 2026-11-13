#!/usr/bin/env python3
"""Emit verdict_m17514.sql — board-audit verdict row for m17514.

Evidence values under the machine-checked keys (impl_files / entry / run_artifacts
/ tests / integration_callers / benchmark / repro_cmd) are either existing file
paths (relative to this audit worktree) or the literal data the gate reads, so
maturity_gate.files_exist / impl_real / has_numbers can verify them.  Every path
points into docs/audit_m17514/evidence_snapshot/, a byte-identical frozen copy of
the m17514 delivery (sha256 in evidence_snapshot/SHA256SUMS.txt) — the audited
worktree itself is recyclable and its delivery was never committed (finding F1).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = "docs/audit_m17514/evidence_snapshot"

evidence = {
    "impl_files": [f"{SNAP}/aa_pattern.py"],
    "impl_files_original": [
        "/root/tp-agent/work/m17514/src/tensorlbm/aa_pattern.py (353 lines, real implementation: "
        "HALF_A/HALF_B partition assertion, per-direction row-permutation cache, in-place dual kernel, "
        "phase state machine, parity counters; NOT a placeholder)",
        "/root/tp-agent/work/m17514/src/tensorlbm/__init__.py:77-81,2961-2962 (public API export, "
        "import verified: tensorlbm.AAPatternStepper resolves)",
    ],
    "entry": [f"{SNAP}/xval_aa_pattern_even_odd.py"],
    "run_artifacts": [
        f"{SNAP}/result.json",
        f"{SNAP}/k_history_aa_tg_n32_re100.csv",
        f"{SNAP}/k_history_ab_tg_n48_re100.csv",
        f"{SNAP}/tg_decay_aa_vs_ab_vs_exact.png",
    ],
    "run_artifacts_original": [
        "/root/tp-agent/work/m17514/artifacts/xval_aa_pattern_even_odd/result.json (+6 CSV, 3 PNG); "
        "auditor re-ran the bench in an isolated copy: 10/10 criteria PASS and every non-timing field "
        "bit-identical to the shipped artifact"
    ],
    "tests": [f"{SNAP}/test_aa_pattern_parity.py"],
    "tests_detail": "tests/test_aa_pattern_parity.py: 15 invariant tests (allocation bytes, one kernel "
                    "pair == AB streaming, multi-step bitwise equivalence, probe non-perturbation, "
                    "mass/momentum conservation, half-streamed read refused, input contract, no aliasing) "
                    "-> 15 passed in 7.28 s, re-run by the auditor",
    "integration_callers": [],
    "integration_callers_note": "NONE. grep -rn 'aa_pattern|AAPattern' over the whole m17514 tree returns "
                                "only the module itself, its test, its benchmark and the lazy-export table in "
                                "src/tensorlbm/__init__.py:77-81 (API surface, not a caller). "
                                "src/tensorlbm/solver.py and app/backend/ contain 0 references -> L3 not met.",
    "benchmark": {
        "metric": "gamma_K (kinetic-energy decay rate) and velocity-field L2 vs the closed-form 2D "
                  "Taylor-Green solution; plus bitwise AA-vs-AB population equality",
        "gate_pct": 15.0,
        "machine_referee": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
                           "anchor_slug='XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER' -> 0 rows "
                           "(no platform referee row for this anchor; referee is the delivery's own "
                           "criteria list, re-run by the auditor)",
        "ref_vs_got": {
            "gamma_K_vs_exact": {"n32": {"ref": 0.0024674011, "got": 0.0024647956, "err_pct": -0.1056},
                                 "n48": {"ref": 0.0016449338, "got": 0.0016495503, "err_pct": 0.2809},
                                 "worst_err_pct": 0.280914, "declared_tol": "< 15 %", "pass": True},
            "field_l2_vs_exact": {"n32": 1.4485, "n48": 1.4226, "err_pct": 1.44848,
                                  "declared_tol": "< 15 %", "pass": True,
                                  "auditor_recomputed_n48": 1.4226039246715139},
            "aa_vs_ab_bitwise_max_abs": {"got": 0.0, "declared_tol": "< 1e-12", "pass": True,
                                         "auditor_independent_own_reference_50_steps": 0.0},
            "mass_drift_rel_worst": 4.05231e-14,
            "mom_drift_rel_worst": 4.46419e-16,
            "half_window_hazard_white_noise_pct": 115.5503,
            "parity_blind_field_l2_pct": 168.2663,
            "parity_blind_gamma_K_err_pct": 19.4069,
        },
        "shipped_config_is_ci_size": "--quick: N=32/48, 400 steps, Re=100 only. The default (full) config in "
                                     "benchmarks/xval_aa_pattern_even_odd.py:634-643 is N=64/128 Re=100 + "
                                     "N=128 Re=320, 1000 steps and was NOT the shipped artifact",
        "convergence_caveat": "field L2 1.4485 % (N=32) -> 1.4226 % (N=48): span -1.8 %, error does not "
                              "decrease; consistent with a Ma=0.0866 compressibility floor, so the doc's "
                              "implicit 'grid convergence' reading is not supported by two grids",
    },
    "repro_cmd": "cd /root/tp-agent/work/m17514 && PYTHONPATH=src python3 benchmarks/xval_aa_pattern_even_odd.py "
                 "&& PYTHONPATH=src python3 -m pytest tests/test_aa_pattern_parity.py -q   # auditor ran it in "
                 "an isolated copy: 10/10 criteria PASS, 15 tests passed, all non-timing fields bit-identical",
    "external_anchor": [f"{SNAP}/XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER.md"],
    "external_anchor_note": "Taylor (1923) Green-Taylor exact solution of 2-D incompressible Navier-Stokes "
                            "(closed form, K(t)=U0^2/4 exp(-4 nu k^2 t), gamma_K=4 nu k^2) plus the in-repo "
                            "AB two-field reference. Analytic/textbook anchor only: no experimental or "
                            "literature discrete dataset, and the upstream walberla aa-pattern-e/o source "
                            "could not be retrieved (sandbox egress blocked) -> L5 impossible; byte-level "
                            "isomorphism with upstream is NOT verified",
    "delivery_git_state": {
        "branch": "agent/m17514",
        "head": "c05d7af3 Merge #2000 (no mission commit)",
        "git_status": ["M src/tensorlbm/__init__.py", "?? src/tensorlbm/aa_pattern.py",
                       "?? benchmarks/xval_aa_pattern_even_odd.py", "?? tests/test_aa_pattern_parity.py",
                       "?? docs/XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER.md",
                       "?? artifacts/xval_aa_pattern_even_odd/", "?? figures/xval_aa_pattern_even_odd/"],
        "git_ls_files_aa_pattern": "empty -> the delivery exists only as untracked files in a recyclable "
                                   "worktree; mission result agrees (changed_files=[], head_commit=null, "
                                   "branch=null, server_quality_audit.gate_valid=false, key_present=false)",
        "auditor_action": f"frozen byte-identical copies under {SNAP}/ with SHA256SUMS.txt to keep the "
                          "evidence chain alive",
    },
    "findings": {
        "F1_blocking_integrity": "delivery never committed to git (untracked files in a recyclable worktree)",
        "F2_semantics": "the even/odd parity flip is NOT load bearing: removing it leaves all 10 criteria "
                        "PASS with values bit-identical to the control (mutation M1). Only the phase gate "
                        "_require_complete is load bearing (hazard 6.62 % smooth / 115.55 % noise field)",
        "F3_doc_contradiction": "doc §6 says the parity-blind gamma_K deviation stays below the 15 % gate; "
                                "the shipped data give +36.83 % (N=32) and +19.41 % (N=48), both above it; "
                                "a template placeholder '(无)' is still in the shipped doc",
        "F4_overstated_storage": "doc §7 claims a per-direction O(N) row buffer and 8*N indices; code "
                                 "materialises a full (ny,nx) plane per direction and caches 16*N int64 "
                                 "= 128*N bytes -> AA total 460800 B > AB 331776 B at N=48 (no memory win)",
        "F5_evidence_scope": "shipped artifacts are the CI-size --quick config; the declared full config "
                             "(N=64/128, Re=320, 1000 steps) was never run",
        "F6_platform_gap": "tensor_benchmark_cases has 0 rows for this anchor -> no independent machine "
                           "referee",
        "F7_report_debt": "tensor_reports rows for mission 17514 = 0 although status succeeded and "
                          "report_owed=false",
    },
    "checks_run": [
        "isolation re-run of the bench (probe A): 10/10 criteria PASS, payload identical except wall_s",
        "independent own D2Q9 reference (probe B): max|f_AA - f_own| = 0.0 over 50 steps at N=16 and N=32",
        "independent 400-step N=48 re-run: field L2 = 1.4226 % (matches shipped 1.4226 %)",
        "mutation probes (probe C): M1 parity flip removed -> still PASS bit-identical; "
        "M2 permutation off-by-one -> IndexError aa_pattern.py:206; M3 overlapping halves -> non-zero exit",
        "storage accounting (probe D) and doc-vs-data consistency (probe E)",
        "pytest tests/test_aa_pattern_parity.py: 15 passed",
        "public API import check: from tensorlbm import AAPatternStepper OK",
    ],
    "checks_not_run": [
        "byte/line comparison against upstream walberla aa-pattern-e/o (no network, source unavailable) "
        "-> 'isomorphic to upstream' remains UNVERIFIED, not passed",
        "non-periodic boundaries inside the half-streamed window, ND = SPEEDS + PAD dual-slot layout, "
        "AMR/multi-GPU (out of scope per the delivery, not verified by me either)",
        "Ma refinement study (single Ma = 0.0866) -> the 1.4 % field error is not decomposed into "
        "discretisation vs compressibility",
        "TensorBench scoring for this anchor (no referee row exists)",
    ],
    "level_rationale": "L1 met (15 invariant tests, re-run). L2 met: real non-placeholder implementation "
                       "(353 lines), an entry (bench driver + public API export) and real run artifacts "
                       "(result.json with 10 machine-checked criteria + CSVs + PNGs) that reproduce "
                       "bit-identically in an isolated re-run, and that I independently confirmed against "
                       "my own D2Q9 reference (0.0 difference). L3 NOT met: zero solver/platform call sites. "
                       "L4 substance (complete case + error analysis + reproducible) is in fact present "
                       "(gamma_K 0.28 %, field L2 1.45 %, bitwise 0) but the cumulative gate scheme requires "
                       "integration_callers, so the capability is recorded as L2 with L4-strength evidence "
                       "attached. L5 NOT met (analytic anchor only).",
    "overstated_claims": [
        "doc §6: 'the parity-blind gamma_K error is still below the 15 % gate' (measured +36.83 %/+19.41 %)",
        "doc §1/§6: 'losing the even/odd alternation = one half streamed twice' -> only true for the author's "
        "own parity_blind_step strawman; the natural mutation (no flip) is numerically inert",
        "doc §7: 'AA = one field + O(N) row buffer, 8*N indices' (measured: plane temp per direction, "
        "128*N bytes of int64 index cache, total memory larger than the AB reference)",
        "doc §5: 'two resolutions + two parameter points' -> both points are Re=100 with different N, i.e. one "
        "physical parameter point, and the run is the CI-size configuration",
    ],
    "disposition": "request_changes (bounded): numbers accepted (reproducible, independently reproduced, "
                   "<3 %), level L2 confirmed; four non-numerical items to close in a follow-up - commit the "
                   "delivery (F1), fix the self-contradicting §6 sentence + '(无)' placeholder (F3), correct "
                   "the §7 storage table / drop the memory-win claim (F4), scope the load-bearing claim to "
                   "the phase gate and label the CI-size config (F2/F5).",
    "independence_limitations": "auditor ran in a separate worktree (m17572) with its own D2Q9 reference and its "
                                "own mutations; the audited code was not modified; the upstream walberla source "
                                "and any experimental dataset were unavailable, so no external discrete anchor "
                                "was checked by anyone.",
}

sql = f"""INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17514 XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER', 17572, 2,
        $audit${json.dumps(evidence, ensure_ascii=False)}$audit$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;
"""
out = HERE.parent.parent / "verdict_m17514.sql"
out.write_text(sql)
print(out, len(sql))