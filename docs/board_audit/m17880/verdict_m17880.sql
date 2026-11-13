-- board audit m17880 : benchmark referee rows + maturity verdict for mission m17762
-- anchor XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE had ZERO rows in tensor_benchmark_cases before this audit.
BEGIN;

INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
SELECT 'XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE',
       'CH-CORE-TRANSPLANT-EQUIVALENCE (z-invariant single mode, N=64, Gamma=1, modes m=3/5/8/11/14)',
       'ebinan92/Fingering_dynamics@150a5739 lattice_boltzmann/fingering_periodic.py (sha256 d3789463b1b63ee1...) executed via importlib by scripts/xval_m17762_free_energy_ch_core.py; auditor m17880 verified HEAD+sha256',
       'max relative |omega_TL - omega_OSS|/|omega_OSS| over the 5 modes (lee_lin vs OSS)',
       '<15% (owner standard); measured max 0.0522% (m=3), 0.0007-0.0083% on m=5/8/11/14 => 5/5 pass',
       'TensorLBM', 'passed', now()
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases
                  WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE'
                    AND case_name LIKE 'CH-CORE-TRANSPLANT-EQUIVALENCE%');

INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE',
       'CH-ANALYTIC-DISPERSION-SINGLE-MODE (omega = Gamma(tau_g-1/2)k^2(A-kappa k^2), N=64, A=0.075, kappa=0.0375, k_c=1.4142)',
       'continuum Cahn-Hilliard linear dispersion relation (analytic), same protocol as the OSS run',
       'per-mode relative error of measured omega vs the analytic CH relation, modes m=3/5/8/11/14',
       '<15% per mode (owner standard); measured -4.56% / +2.00% / +7.26% / +23.69% / +301.75% => 3/5 pass. m=11 (k=1.0799, healthy mode) 23.69% FAIL, identical to the OSS arm => discretisation-vs-continuum, not a transplant bug; m=14 (k/k_c=0.972, near-neutral) 301.75% FAIL on a 2.27e-3 reference, absolute deviation 6.84e-3',
       'TensorLBM', 'failed'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases
                  WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE'
                    AND case_name LIKE 'CH-ANALYTIC-DISPERSION-SINGLE-MODE%');

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17762 XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE', 17880, 3,
$ev${
 "audit_mission": 17880,
 "impl_files": ["docs/board_audit/m17880/refs/src/tensorlbm/multiphase3d.py"],
 "impl_patch": "docs/board_audit/m17880/refs/src/tensorlbm/multiphase3d.patch",
 "impl_production_path": "src/tensorlbm/multiphase3d.py:775 free_energy_step_3d(..., geq_convention='swift'|'lee_lin') (+41/-2)",
 "entry": ["docs/board_audit/m17880/refs/scripts/xval_m17762_free_energy_ch_core.py"],
 "run_artifacts": ["docs/board_audit/m17880/refs/artifacts/xval_free_energy_ch_core.json",
                   "docs/board_audit/m17880/refs/artifacts/fig1_ch_dispersion.png",
                   "docs/board_audit/m17880/artifacts/ref_vs_got.json",
                   "docs/board_audit/m17880/artifacts/repro_compare.json"],
 "tests": ["tests/test_xval_m17762_free_energy_ch.py 6 passed in 9.37s (auditor m17880 re-run)",
           "tests/test_multiphase3d_d3q27_free_energy.py + tests/test_multiphase_extensions.py 35 passed, 1 xfailed in 22.25s (auditor re-run; default swift path unaffected)"],
 "integration_callers": ["free_energy_step_3d"],
 "integration_call_sites": ["src/tensorlbm/dam_break_3d.py:862 (solver call, default swift)",
                            "scripts/bench_fe_water_entry_3d.py:83 (benchmark call, default swift)",
                            "src/tensorlbm/__init__.py:1868 (re-export)",
                            "NOTE: the new 'lee_lin' mode itself has 0 adopted call sites; grep geq_convention in repo -> def + this mission's script/test only"],
 "benchmark": {
   "anchor_rows_before_audit": 0,
   "ref_equivalence": "ebinan92/Fingering_dynamics@150a5739 (fingering_periodic.py sha256 d3789463...)",
   "ref_analytic": "omega = Gamma(tau_g-1/2) k^2 (A - kappa k^2), A=0.075, kappa=0.0375, Gamma=1, tau_g=0.788675",
   "tolerance": "<15% (owner standard)",
   "got_vs_equivalence_max_rel_pct": 0.0522,
   "pass_vs_equivalence": "5/5 modes",
   "got_vs_analytic_rel_pct_by_mode": {"m3": -4.56, "m5": 2.00, "m8": 7.26, "m11": 23.69, "m14": 301.75},
   "pass_vs_analytic": "3/5 modes (m=11 and m=14 exceed 15%)",
   "other_kpis": {"sigma_identity_rel_err": 1.4e-16, "mu_operator_rel_l2_tl_vs_oss": 1.9e-13,
                  "M_eff_vs_predicted_pct": [1.94, 2.00, 2.12],
                  "profile_sigma_pct": -10.40,
                  "swift_spurious_decay_vs_model_pct_max": 9.76}
 },
 "repro_cmd": "cd /root/tp-agent/work/m17762 && PYTHONPATH=src python3 scripts/xval_m17762_free_energy_ch_core.py --parts ABCDEF --oss-root /tmp/Fingering_dynamics --out /tmp/m17880_repro",
 "repro_result": "bit-identical: JSON payload identical (all keys except timestamp/torch), 3 PNG sha256 identical (a494ff2d/ae733aee/9a2c9396)",
 "external_anchor": "ebinan92/Fingering_dynamics@150a5739e320498c6ecf22ba31ab9ebc57130462, lattice_boltzmann/fingering_periodic.py sha256 d3789463b1b63ee154d5a653ea5f0f605a89bec30feb443810461d5c0ac11f72 (auditor re-verified HEAD + sha256). It is a SIBLING implementation (code-to-code), NOT independent experimental/literature physical truth => L5 refused",
 "level_rationale": "L2 met: geq_convention branch is real input->compute->output code, 6 tests green on auditor re-run, artefact reproduced bit-identically. L3 met by board precedent m17515 (threading must be live): the modified function free_energy_step_3d sits on the main multiphase solver path (dam_break_3d.py:862, bench_fe_water_entry_3d.py:83, re-export __init__.py:1868) and the switch is live (changing the convention moves the measured growth rate from -0.0551 to +0.0099 at m=8), while the default swift path is unchanged (35 existing tests green). Caveat recorded: the new lee_lin mode has 0 adopted call sites.",
 "why_not_L4": ["no predeclared tolerance/protocol and 0 rows in tensor_benchmark_cases before this audit (the sole referee was empty; auditor registered 2 rows)",
                "against the analytic CH dispersion the referee FAILS at 2/5 modes: m=11 23.69% (healthy mode, k=1.0799) and m=14 301.75% (k/k_c=0.972 near-neutral, absolute deviation 6.84e-3) -- owner bar 15%",
                "the regression test asserts only the single most favourable mode (m=6, N=48, 7.26%); the 2 failing modes were never asserted",
                "all 4 deliverable paths are untracked (git cat-file -e HEAD -> ABSENT; geq_convention count in HEAD src = 0) => no provenance for a delivered benchmark"],
 "why_not_L5": ["external anchor is a sibling implementation, not an experiment or literature value",
                "scope is linear single-mode z-invariant only; 3D nonlinear fingering/Saffman-Taylor branch untouched"],
 "landing_defect_F1": {"finding": "mission m17762 status=succeeded but its branch agent/m17762 HEAD contains NONE of the deliverables",
   "machine_checks": {"scripts/xval_m17762_free_energy_ch_core.py": false,
                      "scripts/render_xval_m17762_report.py": false,
                      "tests/test_xval_m17762_free_energy_ch.py": false,
                      "docs/XVAL_M17762_LBM_FREE_ENERGY_CAHN_HILLIARD_CORE.md": false,
                      "src/tensorlbm/multiphase3d.py geq_convention_in_HEAD": 0,
                      "git_status": [" M src/tensorlbm/multiphase3d.py", "?? artifacts/xval_m17762/", "?? scripts/render_xval_m17762_report.py", "?? scripts/xval_m17762_free_energy_ch_core.py", "?? tests/test_xval_m17762_free_energy_ch.py"]},
   "preserved": "docs/board_audit/m17880/refs/** with sha256 parity to the m17762 worktree (artifacts/manifest.json)",
   "remediation": "git apply docs/board_audit/m17880/refs/src/tensorlbm/multiphase3d.patch + commit the 3 new files"},
 "report_defect_F3": {"finding": "the doc cited by both the source docstring and the test docstring does not exist; its renderer does not compile",
   "doc_cited": "docs/XVAL_M17762_LBM_FREE_ENERGY_CAHN_HILLIARD_CORE.md (ABSENT)",
   "renderer": "scripts/render_xval_m17762_report.py py_compile rc=1 SyntaxError line 48 (30 occurrences of '%%' that should be '%')",
   "tensor_reports_rows_for_mission": 0},
 "overstated_claim": "the delivered narrative says the ported core 'reproduces the analytic Cahn-Hilliard growth rate'; the artefact's own table shows this only for k<=0.785 (3/5 modes), with m=11 at 23.69%",
 "audit_artifacts": ["docs/board_audit/m17880/AUDIT.md",
                     "docs/board_audit/m17880/artifacts/ref_vs_got.json",
                     "docs/board_audit/m17880/artifacts/repro_compare.json",
                     "docs/board_audit/m17880/artifacts/landing_checks.json",
                     "docs/board_audit/m17880/artifacts/manifest.json",
                     "docs/board_audit/m17880/artifacts/fig_audit_dispersion.png",
                     "docs/board_audit/m17880/build_evidence.py",
                     "docs/board_audit/m17880/make_figures.py"],
 "not_checked": ["3D multi-mode nonlinear fingering (upstream fingering.py branch) unverified",
                 "Gamma sweep not independently re-run (only the 3 delivered points re-read)",
                 "origin of the -10.40% profile-measured sigma (attributed to discrete gradient-energy integration) not independently sourced",
                 "the 80-tool-call trajectory of the audited agent not inspected (terminal artefacts only)"],
 "disposition": "approve with required remediation: level L3 confirmed by machine checks, but F1 (nothing landed), F2 (referee row absent -> now registered, analytic KPI failed) and F3 (missing doc + broken renderer) must be closed by a follow-up; L4/L5 refused"
}$ev$::jsonb, 'board-audit');

COMMIT;

SELECT id, repo, scope, mission_id, level, verdict_by, gate_state FROM tensor_maturity_verdicts WHERE mission_id=17880;
SELECT anchor_slug, left(case_name,50), status FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE';