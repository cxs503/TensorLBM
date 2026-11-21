-- Verdict for m17842 XVAL-LBM-GITLAB-CI-STAGED-GATES
-- Audit mission m17887, board-audit
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17842 XVAL-LBM-GITLAB-CI-STAGED-GATES', 17887, 4,
'{
  "impl_files": [
    "xval/gitlab_ci_staged_gates/ci_spec.py:445 lines",
    "xval/gitlab_ci_staged_gates/gate_sim.py:276 lines",
    "xval/gitlab_ci_staged_gates/oss_refs.py:219 lines",
    "xval/gitlab_ci_staged_gates/run_xval.py:458 lines",
    "xval/gitlab_ci_staged_gates/measure_selections.py:82 lines",
    "xval/gitlab_ci_staged_gates/_collect_markers.py:16 lines"
  ],
  "entry": "xval/gitlab_ci_staged_gates/run_xval.py",
  "run_artifacts": [
    "xval/gitlab_ci_staged_gates/summary.json",
    "xval/gitlab_ci_staged_gates/table.csv",
    "xval/gitlab_ci_staged_gates/figures/fig1_guard_fraction.png",
    "xval/gitlab_ci_staged_gates/figures/fig2_ladder_structure.png",
    "xval/gitlab_ci_staged_gates/figures/fig3_simulator_fidelity.png",
    "xval/gitlab_ci_staged_gates/figures/fig4_vocabulary_leakage.png",
    "xval/gitlab_ci_staged_gates/data/corpus_markers.json",
    "xval/gitlab_ci_staged_gates/data/selection_measurements.json",
    "xval/gitlab_ci_staged_gates/oss_refs/manifest.json"
  ],
  "tests": [
    "tests/test_ci_staged_gates_xval.py (8/8 PASSED)"
  ],
  "integration_callers": ["tests/test_ci_staged_gates_xval.py"],
  "benchmark": {
    "H1_parity": "0.0-0.6% diff on all 4 events (<15%) PASS",
    "H2_leakage": "verbatim=100%>=15%, adapted=0%<15% PASS",
    "fidelity": "0.0% error on all 4 expressions (<15%) PASS",
    "all_pass": true
  },
  "repro_cmd": "PYTHONPATH=src python -m xval.gitlab_ci_staged_gates.run_xval",
  "external_anchor": "lbmpy GitLab CI (i10git.cs.fau.de/pycodegen/lbmpy)"
}',
'board-audit');
