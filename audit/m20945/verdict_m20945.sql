-- Verdict for m20945: XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN
-- Audit date: 2026-11-15
-- Auditor: board-audit (m20975)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm20945 XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN',
  20945,
  2,
  '{
    "impl_files": [
      "src/tensorlbm/multiphase.py:534-750 (sc_two_component_force + collide_sc_two_component, D2Q9 two-component Shan-Chen with Guo/shift forcing, self/mixture/viscous u_eq, standard/balanceable sc_form) @m20945 worktree",
      "TensorLBM/xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py (480+ lines, MPLBM-UT reference re-implementation + TensorLBM import + droplet Laplace law cross-validation + coexistence density + spurious current analysis)"
    ],
    "entry": "TensorLBM/xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py (--quick for smoke, full protocol for Laplace law + coexistence + spurious currents)",
    "run_artifacts": [
      "TensorLBM/xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/artifacts/results.json (4444 B; droplet comparison at R=12,16; coexistence densities; Laplace fit; σ cross-error 0.00%)",
      "TensorLBM/xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/artifacts/fig1_laplace_law.png (63970 B)",
      "TensorLBM/xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/artifacts/fig2_sigma_comparison.png (45130 B)",
      "TensorLBM/xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/artifacts/fig3_spurious_currents.png (58470 B)",
      "xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/artifacts/summary.json (1958 B; stratified Poiseuille cells all NaN/converged=false)"
    ],
    "tests": "TensorLBM/tests/test_xval_mcmp_two_lattice_shan_chen.py — CANNOT RUN: SyntaxError (from __future__ import annotations not at beginning of file; line 1 is `import sys; sys.path.insert(...)`). Additionally, imports collide_sc_two_component and sc_two_component_force from tensorlbm.multiphase but these functions do NOT exist in TensorLBM/src/tensorlbm/multiphase/__init__.py (they exist only in m20945/src/tensorlbm/multiphase.py). Test is non-functional.",
    "integration_callers": [
      "src/tensorlbm/multiphase.py:687 collide_sc_two_component (called by xval runner via _get_tensorlbm_step)",
      "benchmarks/verified/mcmp_stratified_poiseuille/ (separate verified benchmark, err_Q_mix < 1% at ny=25/33)"
    ],
    "benchmark": {
      "registered_case": "tensor_benchmark_cases WHERE anchor_slug LIKE ''%MCMP%TWO%LATTICE%SHAN%'' → 0 rows (table exists in tensor_platform.db with 16 verdict rows but no benchmark_cases table found in any database)",
      "platform_rows": "No tensorbench_results rows for mission_id=20945",
      "judge": "No machine referee registered → L4 structurally unreachable",
      "ref_vs_got_internal": "TensorLBM xval droplet cross-validation vs MPLBM-UT: σ cross-error 0.00% (PASS), Δp cross-error 0.00% (PASS), u_ratio 1.000 — but this is cross-validation against another code, NOT against analytic solution. The stratified Poiseuille test (xval/oss_solutions.json cells) all show converged=false, Q_mix=NaN — the xval runner lacks wall bounce-back handling for the channel flow case. However, the separate verified benchmark (benchmarks/verified/mcmp_stratified_poiseuille/result.json) shows err_Q_mix_mixture = -0.0014 (ny=25) and +0.0062 (ny=33), both well within 15% tolerance.",
      "note": "Cross-validation (TensorLBM vs MPLBM-UT) passes at 0.00% error, but this is code-to-code comparison, not code-to-analytic. The analytic benchmark (stratified Poiseuille) in the xval runner fails (NaN), while the separate verified benchmark passes (<1%). This inconsistency indicates the xval runner has a setup defect (missing wall bounce-back), not a model defect."
    },
    "repro_cmd": "cd /srv/tensor/agent/work/m20945/TensorLBM && PYTHONPATH=../src python3 xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py --quick  # Audit verified: runs successfully, produces results.json with 0.00% cross-error",
    "external_anchor": "Shan & Chen (1993) Phys. Rev. E 47, 1815; Shan & Doolen (1995) J. Stat. Phys. 81, 379; Martys & Chen (1996) J. Comput. Phys. 126, 276; Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308; OSS: geoelements/mcmp-lbm (commit 015a26f)",
    "defects_found": [
      "D1 (blocking): TensorLBM/tests/test_xval_mcmp_two_lattice_shan_chen.py has SyntaxError — `import sys; sys.path.insert(...)` on line 1 before `from __future__ import annotations`. Test cannot be collected by pytest.",
      "D2 (blocking): Test imports collide_sc_two_component and sc_two_component_force from tensorlbm.multiphase, but these functions do not exist in TensorLBM/src/tensorlbm/multiphase/__init__.py. They exist only in m20945/src/tensorlbm/multiphase.py — path mismatch between TensorLBM/ tree and m20945/src/ tree.",
      "D3 (non-blocking): xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py stratified Poiseuille cells all produce NaN (converged=false) because the runner uses stream() without wall bounce-back handling. The oss_solutions.json correctly marks this as partial/honest-fail.",
      "D4 (structural): No tensor_benchmark_cases entry for this anchor → L4 cannot be granted regardless of result quality."
    ],
    "level_rationale": "L2 granted: real implementation exists (collide_sc_two_component in src/tensorlbm/multiphase.py, 534-750 lines), xval droplet cross-validation runs and produces correct results (0.00% cross-error vs MPLBM-UT). Not L3: test infrastructure is broken (SyntaxError + import mismatch), integration not verified through test suite. Not L4: no machine referee (tensor_benchmark_cases missing), and the xval runner''s analytic benchmark (stratified Poiseuille) fails with NaN. The separate verified benchmark (mcmp_stratified_poiseuille) passes but is a different artifact."
  }',
  'board-audit'
);
