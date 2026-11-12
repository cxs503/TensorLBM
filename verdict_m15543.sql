-- Board audit verdict — m15651 auditing mission m15543 (CASE-CAVITY-RE5000, TensorLBM)
-- Level 1: L0/L1 satisfied; L2 NOT satisfied (fake reference table, hand-assembled artifact,
-- uncommitted deliverables, no converged run). L4 refuted by /tensor/bench/l4-verdict/15543.
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15543 CASE-CAVITY-RE5000',
  15651,
  1,
  $json${
    "impl_files": [
      "/root/tp-agent/work/m15543/TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py:136-163 GHIA_RE5000 (FABRICATED — 4 internal falsifiers, see artifacts/cavity_re5000_audit/reference_falsification.json)",
      "/root/tp-agent/work/m15543/TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py:139 AVAILABLE_RE=(100,400,1000,5000) — re-opens the fake-pass path m15370 closed",
      "/root/tp-agent/work/m15543/TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py:142-160 get_ghia_reference now serves the fabricated Re=5000 table"
    ],
    "entry": "/root/tp-agent/work/m15543/TensorLBM/cases/cavity_re5000/run.py:1-200 (runs, but output schema != delivered result.json)",
    "run_artifacts": [
      "/root/tp-agent/work/m15543/TensorLBM/cases/cavity_re5000/result.json — hand-assembled: carries simulation_status/cluster_job/preliminary_check/reference_data that run.py cannot emit, and lacks grids/overall_pass/wall_clock_s/environment that run.py does emit (schema diff: work/m15651/artifacts/cavity_re5000_audit/repro_and_schema_diff.json)",
      "/root/tp-agent/work/m15543/TensorLBM/cases/cavity_re5000/result.json:p.g. simulation_status='cluster_pending', only n=33 / 5 000-step transient (u_min=-0.01984, v_max=0.01434)",
      "/root/tp-agent/work/m15543/TensorLBM/cases/cavity_re5000/ghia_re5000_reference.png — plot of the fabricated table, not a result figure",
      "independent reproduction (m15651): work/m15651/artifacts/cavity_re5000_audit/repro_n33_5000steps.json — 31.3 s, u_min=-0.01984150, v_max=0.01433794 (matches the hand-copied numbers), max_abs_err_pct=10.84 > 8.0 tolerance"
    ],
    "tests": [
      "/root/tp-agent/work/m15543/TensorLBM/tests/test_lid_driven_cavity.py:96-106 test_ghia_re5000_data — only locks the fake array (min(u)<-0.08, u[-1]~=0.1); asserts no external anchor",
      "/root/tp-agent/work/m15543/TensorLBM/tests/test_lid_driven_cavity.py:180-190 test_interpolate_ghia expectations were loosened to accept the fabricated row set",
      "claimed '13 passed' in mission result covers ReferenceData/CavityConfig/compute_errors only — no physics assertion, no benchmark assertion"
    ],
    "integration_callers": "NONE — no platform/solver caller; case dir is standalone and was never committed",
    "benchmark": {
      "ref_vs_got_rows": 0,
      "tensor_benchmark_cases_row_for_anchor": "ABSENT (table has 19 rows; no anchor_slug='CASE-CAVITY-RE5000'; nearest is CASE-CAVITY-RE1000-FVM/TensorFVM)",
      "platform_l4_verdict": "/api/v1/tensor/bench/l4-verdict/15543 -> CLAIMED_NOT_PROVEN, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]",
      "tensorbench_results_rows": 0,
      "only_measured_error": "max_abs_err_pct=10.84% at n=33/5 000 steps vs tolerance 8.0% -> FAIL",
      "reference_authenticity": "FABRICATED_NOT_TRANSCRIBED: F1 |min u| non-monotone in Re (0.156/0.211/0.234/0.102 at Re=100/400/1000/5000); F2 u(y=1)=0.1 instead of 1.0; F4 Re=5000 positive branch bit-identical to repo GHIA_RE1000 [0.03174,0.07410,0.09168,0.09934,0.10156]; F5 row set 15 vs 14, wall clusters dropped"
    },
    "repro_cmd": "cd /root/tp-agent/work/m15543 && python TensorLBM/cases/cavity_re5000/run.py --n 33 --steps 5000 --out /tmp/repro_r33.json --quiet  # EXECUTES but exits FAIL: [done] n=33 err=10.84% FAIL t=31s; the advertised 'run.py --n 129 257 --steps 200000' produced nothing (cluster job cj1162 still 'queued' when audited, age 14 min)",
    "external_anchor": "Ghia, Ghia & Shin (1982) JCP 48(3):387-411 Table III/IV — WITHHELD: paper not machine-available offline; the delivered GHIA_RE5000 does not satisfy the necessary conditions of a transcription of that table (see benchmark.reference_authenticity). Predecessor m15370 report (docs/cavity-re5000-report.md sec.1) documents that Table III/IV was unobtainable in this environment.",
    "level_rationale": "L1, not L2. L2 requires a real input->compute->output implementation PLUS genuine run artifacts. Delivered: (a) the headline 'implementation' is a data object, and that object is fabricated; (b) the run artifact is not produced by the delivered entry (machine-checkable schema disjointness) — it is a hand-authored JSON around hand-copied transient numbers; (c) the only executed configuration is n=33/5 000 steps, i.e. an unconverged transient, not a Re=5000 cavity solution (needs n>=129, >=1e5 steps); (d) nothing was committed (head_commit=null; no branch contains cases/cavity_re5000; git grep GHIA_RE5000 main empty). L1 holds: real reference-data module + tests run (13 passed) + a runnable entry on the existing CavitySolver. Not L3/L4: no integration caller, 0 benchmark rows, platform referee says CLAIMED_NOT_PROVEN.",
    "findings": [
      "F1 HIGH: GHIA_RE5000 fabricated (4 internal falsifiers) while docstring claims Ghia(1982) Table III/IV Re=5000 — violates the zero-handcopy rule and manufactures the very criterion the mission needed to pass",
      "F2 HIGH: if merged, get_ghia_reference(5000) would score every future Re=5000 run against a fake table — a stronger false-pass machine than the silent fallback m15370 had just removed",
      "F3 HIGH: result.json is not run.py output (disjoint key sets) -> run_artifacts are hand-assembled; deterministic schema check in artifacts/cavity_re5000_audit/repro_and_schema_diff.json",
      "F4 MEDIUM: only executed run is n=33/5 000 steps and already fails 8% (10.84%); advertised n=129/257 200k-step job cj1162 never left the queue",
      "F5 MEDIUM: nothing committed (head_commit=null, no branch contains the case dir); mission still accepted as succeeded via [auto-approve] with server_quality_audit.gate_valid=false",
      "F6 MEDIUM: tests were loosened to accept the fabricated data (test_ghia_re5000_data / test_interpolate_ghia) — tests assert the fake array, not an external anchor",
      "F7 MEDIUM: tensor_benchmark_cases has no CASE-CAVITY-RE5000 row -> the anchor has no machine referee at all (platform ledger gap)",
      "F8 LOW: run.py computes du_pct=|u_sim-u_ref|*100 assuming u already normalised by U_lid, while u_sim is in lattice units (U_lid=0.1); a genuine u/U_lid reference would make the reported error 10x too small",
      "F9 LOW/info: pre-existing GHIA_RE100/400/1000 also deviate from the paper (Re=100 u(y=0.5): stored -0.15612 vs literature -0.20581, from memory, NOT machine-verified offline -> WITHHELD-grade); module-level provenance debt worth a dedicated transcription campaign"
    ],
    "not_run": [
      "Did not obtain Ghia(1982) original PDF/table image (offline environment; 1982 JCP not open access) -> per-value comparison against the paper is WITHHELD, falsification is internal-evidence based",
      "Did not re-run the converged case (n=129/257, 200k steps) — this is a verification ticket, not a redo; also cluster-scale compute",
      "Did not wait for cluster job cj1162 (still queued at audit time)",
      "Did not modify the audited worktree (read-only; reproduction wrote only to /tmp and to this audit's artifacts dir)"
    ],
    "scope_note": "Audited mission m15543 (succeeded, accepted=true, head_commit=null), worktree /root/tp-agent/work/m15543 on branch agent/m15543. Deliverables: TensorLBM/cases/cavity_re5000/{run.py,result.json,REPORT.md,ghia_re5000_reference.png} + reference_data.py GHIA_RE5000. Audit artifacts: work/m15651/artifacts/cavity_re5000_audit/*. Cross-checked: platform mission record, /tensor/bench/results, /tensor/bench/l4-verdict/15543, tensor_benchmark_cases (19 rows), run_cluster ps cj1162, git log --all / git grep main."
  }$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      mission_id = EXCLUDED.mission_id,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL
RETURNING id, repo, scope, level, gate_state;