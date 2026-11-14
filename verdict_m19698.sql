-- Cross-audit verdict for mission m19633 (XVAL-LBM-ROUND2-METHOD-SELECTION-3D), TensorLBM.
-- Level: 0  (L0 floor; the documentation artifact L0 itself requires never landed either)
-- Every pointer below is a *negative* pointer produced mechanically by
-- tools/audit_m19633_verify.py -> artifacts/m19633_audit/verify_result.json
-- (worktree + full git history 6021 commits + platform DB + real execution of repro_cmd).
-- Nothing here rests on the m19633 agent's narrative.

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by, gate_state)
VALUES (
  'TensorLBM',
  'm19633 XVAL-LBM-ROUND2-METHOD-SELECTION-3D',
  19698,
  0,
  '{"impl_files": ["NONE — 0/10 claimed artifacts reachable: git log --all over 6021 commits and `git rev-list --all | git grep -l round2_method_selection_3d` both return 0 hits; the only claimed file xval/round2_method_selection_3d/oss_pylbm_ref.py was untracked (mission result git_status=\"?? xval/round2_method_selection_3d/\", branch=null, head_commit=null) and the worktree /root/tp-agent/work/m19633 no longer exists; see artifacts/m19633_audit/verify_result.json::files"], "entry": "NONE — no callable entry point exists in the repo; the two entries the mission cited (kernel_xval_3d.py, run_matrix3d.py) do not exist on disk or in git", "run_artifacts": ["NONE — xval/round2_method_selection_3d/runs/kernel_xval_3d.json: absent", "NONE — xval/round2_method_selection_3d/runs/table_3d.csv: absent", "NONE — xval/round2_method_selection_3d/runs/summary_3d.json: absent", "artifacts/m19633_audit/verify_result.json (audit-side, regenerable)", "artifacts/m19633_audit/table_evidence.md (audit-side, regenerable)"], "tests": ["NONE — tests/test_xval_round2_method_selection_3d.py absent on disk and in git; mission self-reports it was never written (final shell blocked)"], "integration_callers": ["NONE — no caller: the deliverable directory has 0 blobs in the entire git history, so no module can import it"], "benchmark": {"anchor": "XVAL-LBM-ROUND2-METHOD-SELECTION-3D", "judge_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug = (anchor constant)", "rows_returned": 0, "ref_vs_got": "NOT VERIFIABLE — no benchmark case registered, so no ref/tolerance exists to compare any got value against; the 12-row table_3d.csv the mission cites is itself absent", "tolerance": "none registered (owner standard <15% cannot be applied without a case)", "verdict": "benchmark judge cannot confirm — 0 rows"}, "repro_cmd": {"claimed": ["PYTHONPATH=/root/oss-cache/pylbm python3 xval/round2_method_selection_3d/kernel_xval_3d.py", "PYTHONPATH=src python3 xval/round2_method_selection_3d/run_matrix3d.py"], "executed": "both run in the audit worktree m19698", "exit_codes": [2, 2], "stderr_first_line": "python3: cannot open file .../xval/round2_method_selection_3d/{kernel_xval_3d,run_matrix3d}.py: [Errno 2] No such file or directory", "executable": false}, "external_anchor": ["OSS pylbm present at /root/oss-cache/pylbm @ f9f5d1647ccc10aa2fc7a93690cc56450432ee24 (2026-11-09) — the anchor exists, only the TensorLBM-side deliverable is missing", "platform report for mission 19633: 0 rows in tensor_reports", "mission self-declared fail_stage=budget_exhausted+forced_finish; server_quality_audit.gate_valid=false, gate_missing_warning=true"], "level_rationale": "L1 needs re-runnable math/unit tests (absent), L2 needs real input->compute->output code (absent), L3/L4 need solver integration / full case with error analysis and reproducibility (absent) => level 0; L0 itself (a documentation artifact) is also not delivered", "audit_artifacts": ["tools/audit_m19633_verify.py", "tools/m19633_audit_report.py", "docs/AUDIT_M19633_XVAL_LBM_ROUND2_METHOD_SELECTION_3D.md", "artifacts/m19633_audit/evidence_coverage.png"]}',
  'board-audit',
  'pending'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL;