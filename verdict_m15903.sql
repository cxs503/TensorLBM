-- Board audit verdict — m15903 auditing mission m15658 (GAP-LBM-T4-PHYSICS-AI-LOOP, TensorLBM)
-- Declared level 2 = independent functional implementation (real code, input -> compute -> output),
-- proven by auditor reproduction. L3 (no caller anywhere) and L4 (no tensorbench_results row,
-- no benchmark case row for this anchor, and the run artifacts were NOT merged into main) are
-- REFUTED. The anchor's headline half — the AI closure — was never executed by the mission;
-- the auditor ran it and the shipped surrogate scores 45.6% one-step rel-L2 on held-out Re,
-- i.e. far outside any acceptable band, so the "AI loop" is NOT a delivered capability.
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15658 GAP-LBM-T4-PHYSICS-AI-LOOP',
  15903,
  2,
  $json${
 "impl_files": [
  "src/tensorlbm/ai/flow_field_loop.py:249 run_case (D2Q9 BGK, real collide/stream)",
  "src/tensorlbm/ai/flow_field_loop.py:292 verify_producer (analytic TGV ruler, criteria field<1%/decay<2%)",
  "src/tensorlbm/ai/flow_field_loop.py:393 produce_dataset (npz + manifest sha256 + sqlite catalog)",
  "src/tensorlbm/ai/flow_field_loop.py:641 train_flow_map (FNO, torch.save state_dict+arch)",
  "src/tensorlbm/ai/flow_field_loop.py:786 evaluate_holdout (fno vs viscous_prior vs persistence)",
  "src/tensorlbm/ai/flow_field_loop.py:887 feedback_acceleration (surrogate rollout -> LBM handback -> healing)",
  "bench_t4_physics_ai_loop.py (--quick/--stage all|produce|train driver)",
  "tools/t4_report_from_results.py (needs artifacts/t4_physics_ai_loop/results.json — that file does not exist)"
 ],
 "entry": "python bench_t4_physics_ai_loop.py --stage all  (or per-stage); library API: tensorlbm.ai.flow_field_loop",
 "run_artifacts": [
  "/root/tp-agent/work/m15658/artifacts/t4_physics_ai_loop/dataset/fields/*.npz — 24 files (6x n=64, 18x n=128), 504 snapshots, all 24 sha256 MATCH manifest.json (auditor recompute)",
  "/root/tp-agent/work/m15658/artifacts/t4_physics_ai_loop/dataset/manifest.json — totals.wall_time_s=263.89, split train_re=[50,100,200,400] / holdout_re=[75,300]",
  "/root/tp-agent/work/m15658/artifacts/t4_physics_ai_loop/model/train_summary.json — 12 epochs, best_val_mse=0.0548, train_wall_s=599.68",
  "/root/tp-agent/work/m15658/artifacts/t4_physics_ai_loop/model/fno_flow_map.pt — 97049 B, verified loadable; NOT in git (git cat-file -e 7e71cfab:...fno_flow_map.pt -> not in tree)",
  "/root/tp-agent/work/m15658/artifacts/t4_physics_ai_loop/producer_verification.json — verified=true, field 0.1954%, decay 0.0438%",
  "ABSENT: artifacts/t4_physics_ai_loop/results.json (the file tools/t4_report_from_results.py consumes)",
  "ABSENT: artifacts/t4_physics_ai_loop/figures/ is EMPTY; ABSENT: docs/T4_PHYSICS_AI_LOOP.md",
  "ABSENT from main: git ls-tree HEAD artifacts/t4_physics_ai_loop/ -> empty; merge 0898713d (PR #1846) carried 5 code/test files only, no dataset, no checkpoint, no verification json"
 ],
 "tests": [
  "python -m pytest tests/test_flow_field_loop.py -q -> 10 passed, 14.63 s (auditor run in /root/tp-agent/work/m15658)",
  "test names cover: analytic decay, non-analytic multi-mode dataset, restart determinism, viscous prior, manifest/hash/catalog/split, train+holdout end-to-end, feedback handback, load_dataset roundtrip, closed-form reference"
 ],
 "integration_callers": "NONE — grep -rn 'flow_field_loop' over src/ app/ tools/ returns only tools/t4_report_from_results.py docstrings and t4_send_report.py; tensorlbm/ai/__init__.py does not export it; no solver, service or platform module calls it. Repo-level merge only (PR #1846 -> merge 0898713d, ancestor of HEAD 53046bcc).",
 "benchmark": {
  "tensor_benchmark_cases_rows_for_anchor": 0,
  "tensor_benchmark_cases_repo_TensorLBM": ["GAP-LBM-MCMP-POISEUILLE", "GAP-LBM-SPHERE", "H2-FRAZIL-RHEOLOGY"],
  "note": "no machine referee exists for this anchor (table has 25 rows, none matching GAP-LBM-T4-PHYSICS-AI-LOOP or any T4 slug) -> L4 can never be confirmed for this mission as recorded",
  "tensorbench_results_rows_for_mission_15658": 0,
  "mission_self_criteria": {"u_rms_max_abs_rel_err": "<1%", "decay_rate_max_abs_rel_err": "<2%"},
  "auditor_re_certification_of_producer": {
   "n": 64, "re": [50.0, 200.0], "n_steps": 2000,
   "max_abs_rel_err_field": 0.0030956784038827854,
   "decay_rate_rel_err": [0.001243919116412051, 0.007122959843359473],
   "independent_ruler": "u_rms = u0/sqrt(2)*exp(-2 nu k^2 t) re-derived by the auditor, not read from analytic_single_mode()",
   "verdict": "producer sub-component PASSES its own declared criteria when re-run (n=128 mission value 0.1954% also reproduced in shape at n=64: 0.31% > n=128 because of resolution)"
  },
  "auditor_holdout_run_of_the_AI_half_shipped_checkpoint": {
   "clean_ref": "none — no held-out metric was ever produced by the mission",
   "one_step_rel_l2_fno": 0.45632018093019716,
   "one_step_rel_l2_viscous_prior": 0.952536417543888,
   "one_step_rel_l2_persistence": 0.9519115701317787,
   "rollout_final_rel_l2_fno": 1.0729878470301628,
   "rollout_final_rel_l2_persistence": 1.284396231174469,
   "feedback_handback_rel_l2_re300": [0.6567, 0.6687, 0.7047],
   "feedback_healed_rel_l2_re300": [0.7412, 0.7658, 0.7495],
   "speedup_vs_lbm_wall_clock": 33.1,
   "verdict": "surrogate is ~2x better than persistence one-step but 45.6% rel-L2 and >100% on rollout -> fails the >15% unacceptable threshold of the platform; a 33x speedup at 66% handback error is not an acceleration claim"
  }
 },
 "repro_cmd": [
  "cd /root/tp-agent/work/m15658 && python -m pytest tests/test_flow_field_loop.py -q   # 10 passed / 14.63 s",
  "cd /root/tp-agent/work/m15658 && python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py A   # 24/24 sha256 match",
  "cd /root/tp-agent/work/m15658 && python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py B   # re-run tgv_n128_re200_s0_t4, max|d|/peak = 0.0 in 12.33 s",
  "cd /root/tp-agent/work/m15658 && python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py C   # producer re-certified, independent ruler",
  "cd /root/tp-agent/work/m15658 && python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py D   # holdout + feedback with the shipped checkpoint (auditor side)"
 ],
 "external_anchor": "Taylor-Green vortex exact Navier-Stokes solution (G. I. Taylor, Phil. Mag. 46 (1923) 671) — single-mode u_rms=u0/sqrt(2)exp(-2 nu k^2 t); used by verify_producer() as the ruler and re-derived independently by the auditor",
 "report_six_sections": "ABSENT — no results.json, no docs/T4_PHYSICS_AI_LOOP.md rendered, no tensor_reports row for mission 15658 (SELECT count(*) FROM tensor_reports WHERE mission_id=15658 -> 0). The mission's own headline declares the held-out evaluation / feedback / results.json / platform report 'not produced (budget exhausted)'.",
 "level_rationale": "L2 because: real solver + real dataset + real (executable) surrogate/feedback code paths exist and reproduce (L2 satisfied); L1-only claims are exceeded. L3 withheld: zero callers outside the mission's own bench/tests. L4 withheld: no benchmark case row for the anchor, no tensorbench_results row, artifacts not merged into main, and the AI-loop quantitative claim is refuted by the auditor run (45.6%/107%). L5 impossible.",
 "honest_gap": "the anchor's second half ('AI 闭环') is a code-only deliverable: train ran to 12 epochs, but holdout / feedback / figures / results.json / docs / platform report were never produced; auditor's own run of those paths shows the accuracy is unusable, so the capability is NOT closed and should stay open on the ledger."
 }$json$,
  'board-audit'
);