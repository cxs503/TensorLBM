-- m18217 board audit verdict for m18160 XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS (TensorLBM)
-- Level 2: L1 (math/unit verification) and L2 (independent functional implementation) are
-- satisfied by the on-disk artefacts of the m18160 worktree (verified by this audit: the
-- registry.py fail-closed fix, the 9-test assertion gate, and the Poiseuille analytic run all
-- execute and pass). L3 is denied because the work was NEVER COMMITTED (head_commit=null,
-- changed_files=[], git status shows uncommitted changes; branch agent/m18160 does not carry
-- the work, so repo-of-record has no evidence). L4 is denied on two independent grounds
-- (no tensor_benchmark_cases row for the anchor + report sha256 stale vs on-disk results.json).
-- Every evidence value is a real pointer verified by this audit.

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18160 XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS',
  18217,
  2,
  $json${
    "impl_files": [
      "/root/tp-agent/work/m18160/TensorLBM/src/tensorlbm/verification/xval/registry.py:169-196 (fail-closed on non-finite metrics: 'if not np.isfinite(val)' -> explicit failure with recorded reason; 'if k in metrics and np.isfinite(metrics[k])' -> only finite metrics are threshold-checked; fixes the IEEE-754 'nan > 0.15 is False' -> PASS defect)",
      "/root/tp-agent/work/m18160/TensorLBM/tests/test_xval_assert_gate.py:1-160 (9 tests + 1 strict xfail: TestDeclaredContract structural asserts, TestNonFiniteIsFailure parametrized nan/inf/-inf regression, TestAnalyticAnchor real Poiseuille run, test_lid_cavity_re100_state_stays_finite strict xfail)",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/run_xval.py:1-120 (runner: FAST_TIER=['poiseuille_2d_H16'], writes results.json/kpi.csv/results.sha256/provenance.json, exit 0 only if every metric finite)",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/make_figures.py (generates fig1_poiseuille_vs_analytic.png + fig2_cavity_divergence.png)",
      "/root/tp-agent/work/m18160/.github/workflows/ci.yml:96-105 (adds 'Cross-validation assert gate' step: pytest -q tests/test_xval_assert_gate.py --timeout=900)"
    ],
    "entry": [
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/run_xval.py (CLI: --cases fast|all|comma-list --outdir)",
      "/root/tp-agent/work/m18160/TensorLBM/tests/test_xval_assert_gate.py (pytest entry, CI-wired)"
    ],
    "run_artifacts": [
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/results.json (poiseuille_2d_H16: eps_inf=0.003607843111901292, eps_L2=0.004920911776990591, mass_drift=3.0808688933348094e-13, passed=true, nonfinite_metrics=[]; sha256=78f5a4d43c0b320cb6c0689625185ee4a8e20d2aafe4e308d010ed9f78896480)",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/kpi.csv (case,metric,value,threshold,passed: poiseuille_2d_H16,eps_inf,0.0036078431119,0.02,True; eps_L2,0.00492091177699,0.02,True; mass_drift,3.08086889333e-13,1e-08,True)",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/provenance.json (git_sha=0a1cf0e20c45f242e3a23d6c5a5118ea1b7a4830, machine=loongarch64, torch=2.7.1, torch_threads=1, results_sha256=78f5a4d43c0b320cb6c0689625185ee4a8e20d2aafe4e308d010ed9f78896480)",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/figures/fig1_poiseuille_vs_analytic.png (1260x560, 99.61% non-white) + fig2_cavity_divergence.png (840x560, 99.41% non-white)",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/predeclared.json (predeclared decision/alternatives/metrics/acceptance/stopping_rule/tiers/confounders/external_anchors)"
    ],
    "tests": [
      "auditor re-run 2026-11-13: python -m pytest TensorLBM/tests/test_xval_assert_gate.py -q -> 9 passed, 1 xfailed, 1 warning in 8.81s (worktree /root/tp-agent/work/m18160)",
      "auditor re-run 2026-11-13: python TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/run_xval.py --cases fast --outdir /tmp/m18160_repro -> [PASS] poiseuille_2d_H16: eps_inf=3.6078e-03 eps_L2=4.9209e-03 mass_drift=3.0809e-13 [4.5s], exit 0"
    ],
    "integration_callers": [
      "/root/tp-agent/work/m18160/TensorLBM/tests/test_xval_assert_gate.py:20-24 (from tensorlbm.verification.xval.cases import register_default_cases; from tensorlbm.verification.xval.registry import XValRegistry) - the gate drives the shipped verification/xval module",
      "/root/tp-agent/work/m18160/TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/run_xval.py:30-33 (from tensorlbm.verification.xval.cases import register_default_cases; from tensorlbm.verification.xval.registry import XValRegistry)",
      "caveat: the ci.yml step and registry.py change are UNCOMMITTED (git status: ' M .github/workflows/ci.yml', ' M TensorLBM/src/tensorlbm/verification/xval/registry.py', '?? TensorLBM/tests/test_xval_assert_gate.py', '?? TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/'); head_commit=null, changed_files=[] -> repo-of-record has NO evidence for this mission"
    ],
    "benchmark": "NO machine judge exists for this anchor: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS' returns 0 rows (and 0 rows in tensor_xval_cases for the same anchor) -> L4 cannot be machine-adjudicated. Deliverable-internal criterion (predeclared owner 15% band, ref=analytical Poiseuille u(y)=g/(2 nu) y (H-y)): got eps_inf=0.36% / eps_L2=0.49% / mass_drift=3.08e-13 vs declared 2%/2%/1e-8 -> all PASS, well inside the owner 15% standard. The lid_cavity_re100_n65/n33 arms are honest FAIL (NaN divergence at step 36/39, kept as strict xfail) and womersley_alpha5 is untested (>110 s/run) - these are NOT in the fast-tier results.json (only poiseuille_2d_H16 is).",
    "repro_cmd": "cd /root/tp-agent/work/m18160 && python -m pytest TensorLBM/tests/test_xval_assert_gate.py -q   # auditor: 9 passed, 1 xfailed in 8.81s; python TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/run_xval.py --cases fast --outdir /tmp/m18160_repro   # auditor: exit 0, [PASS] poiseuille_2d_H16",
    "external_anchor": "Poiseuille: u(y)=g/(2 nu) y (H-y) (Schlichting, Boundary-Layer Theory); Guo, Zheng & Shi (2002) Phys. Rev. E 65:046308 (forcing); Ghia, Ghia & Shin (1982) J. Comput. Phys. 48:387-411 (lid-driven cavity DNS); Womersley (1955)/Sexl (1930) (oscillatory flow). Physical anchor for the cavity is WITHHELD: the shipped solver diverges to NaN (honest FAIL, strict xfail), so the Ghia <15% claim is unverifiable.",
    "level_reasoning": {
      "L1": "PASS - TestNonFiniteIsFailure parametrized nan/inf/-inf regression verifies the assertion semantics (non-finite metric is a failure, never a pass)",
      "L2": "PASS - real implementation + entry + real run artifacts: registry.py fail-closed fix (verified), run_xval.py produces real results (Poiseuille 0.36% < 2%), 9-test assertion gate (verified 9 passed, 1 xfailed)",
      "L3": "FAIL - the work was NEVER COMMITTED: head_commit=null, changed_files=[], git status shows ' M .github/workflows/ci.yml', ' M TensorLBM/src/tensorlbm/verification/xval/registry.py', '?? TensorLBM/tests/test_xval_assert_gate.py', '?? TensorLBM/xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/'; branch agent/m18160 does not carry the work, so repo-of-record has no evidence for this mission",
      "L4": "FAIL - two independent grounds: (1) anchor has no tensor_benchmark_cases row (0 rows) and no tensor_xval_cases row (0 rows) -> no machine judge; (2) report (tensor_reports id=17030) claims results.json sha256=0b578992a94b1b9c27334b7bd7f0a6245ea47ba4f279b83f5ae23a2df6207e24 but the on-disk results.json is 78f5a4d43c0b320cb6c0689625185ee4a8e20d2aafe4e308d010ed9f78896480 (README.md was updated to match, the report was not) -> stale provenance",
      "L5": "FAIL - no adjudicable external reference for a complete benchmark; the cavity divergence is an honest FAIL (strict xfail) and womersley is untested"
    },
    "findings": [
      "F1 blocking (L3): work never committed - head_commit=null, changed_files=[], git status shows uncommitted changes; branch agent/m18160 does not carry the work; repo-of-record has no evidence for this mission",
      "F2 blocking (L4): benchmark anchor unregistered - 0 rows in tensor_benchmark_cases AND 0 rows in tensor_xval_cases for anchor XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS",
      "F3 non-blocking (L4): report sha256 stale - tensor_reports id=17030 claims 0b578992a94b1b9c27334b7bd7f0a6245ea47ba4f279b83f5ae23a2df6207e24 but on-disk results.json is 78f5a4d43c0b320cb6c0689625185ee4a8e20d2aafe4e308d010ed9f78896480 (README.md updated to match, report not)",
      "F4 non-blocking: report 'Measured this round' table lists 4 cases (poiseuille + 2 lid_cavity + womersley) but the fast-tier results.json only contains poiseuille_2d_H16; the lid_cavity/womersley arms are described in the report but not in the results.json artifact",
      "F5 non-blocking: server_quality_audit in mission result shows gate_valid=false, key_present=false, gate_missing_warning=true - the auto-approve gate did NOT validate the work"
    ]
  }$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      mission_id = EXCLUDED.mission_id
RETURNING id, level, coalesce(gate_state, ''), coalesce(gate_level, -1), created_at;
