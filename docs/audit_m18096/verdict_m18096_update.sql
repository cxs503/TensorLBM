-- m18149 board-audit verdict for m18096 (XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY, TensorLBM)
-- Evidence uses ABSOLUTE pointers into the audited worktree because m18096 never committed
-- its deliverables (head_commit=null, git status shows ' M/??' for every artifact).
UPDATE tensor_maturity_verdicts SET level = 2, evidence = $json$
{
  "impl_files": [
    "/root/tp-agent/work/m18096/src/tensorlbm/turbulence/turbulence.py",
    "/root/tp-agent/work/m18096/src/tensorlbm/__init__.py"
  ],
  "impl_symbols": [
    "turbulence.py:2015 _abs_cc_weights = (sum_n |c_n|)^2",
    "turbulence.py:2020 les_scalar_q_abs_cc_proxy_2d (D2Q9)",
    "turbulence.py:2039 les_scalar_q_abs_cc_proxy_3d (D3Q19)",
    "turbulence.py:2058 tau_eff_q_abs_cc_proxy = 0.5(tau0+sqrt(tau0^2+6 Q Sc/rho))",
    "turbulence.py:2087 collide_smagorinsky_q_abs_cc_proxy_bgk",
    "turbulence.py:2118 collide_smagorinsky_q_abs_cc_proxy_bgk3d",
    "__init__.py:530-541 lazy-export + __all__",
    "IMPL IS WORKTREE-ONLY / UNCOMMITTED: git log --all -S les_scalar_q_abs_cc_proxy in the fresh checkout /root/tp-agent/work/m18149 = 0 commits; grep over its src/ = 0 hits"
  ],
  "entry": "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/run_xval.py",
  "run_artifacts": [
    "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/kpi_prim.json",
    "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/kpi_final.json",
    "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/fig1_proxy_isotropy.png",
    "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/fig2_port_fidelity.png",
    "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/fig3_field_identity.png",
    "/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/predeclared.json"
  ],
  "run_artifacts_missing": [
    "out/kpi_flow.json (E2 TGV arm equivalence) absent",
    "out/kpi_cavity.json (E4 Ghia) absent",
    "no script in the tree emits kpi_final.json / fig3 (grep field_identity over m18096 *.py = 0 hits)"
  ],
  "tests": [
    "tests/test_xval_les_qabs_cc_proxy.py (9 tests) at /root/tp-agent/work/m18096/tests/test_xval_les_qabs_cc_proxy.py - auditor re-ran isolated copy 2026-11-13: 9 passed in 6.31 s"
  ],
  "integration_callers": "none: only __init__ lazy-export + tests + the xval runner reach the new symbols; no solver/pipeline caller (contrast collide_smagorinsky_bgk at src/tensorlbm/ai/pipeline.py:82 and src/tensorlbm/apps/ai4s_flagship.py:52)",
  "benchmark": {
    "machine_referee": "0 rows: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY' -> 0 (ILIKE '%LES%' AND ('%PROXY%' OR '%Q%AB%' OR '%SCALAR%') also 0). No registered referee; judged from mission artifacts + verbatim OSS source anchor.",
    "K1_max_rel_err_Q_7_states": {"ref": "OSS transcription", "got": 0.0, "tol": "15%", "pass": true, "caveat": "exact by construction: same formula transcribed twice -> fidelity, not accuracy"},
    "K2_max_rel_err_post_collision_f": {"ref": "OSS transcription", "got": 0.0, "tol": "15%", "pass": true, "caveat": "same as K1"},
    "K2b_clip_arm_vs_ref": {"got": 0.0, "pass": true, "note": "limiter never triggered; tau_eff in [0.5050, 0.6619]"},
    "E3_a_Q_over_3rhotauS": {"ref": "sqrt(2)/9 = 0.1571348402636772", "got": [0.15713484026367674, 0.15713484026367788], "rel_err": 3.5e-15, "spread_pct": 7.24e-13, "tol": "15%", "pass": true, "independent_check": "docs/audit_m18096/independent_check.py (numpy, no tensorlbm import): 2000 random incompressible strains, max |S2/(|S|^2/81)-1| = 8.88e-16, max rel dev of a vs sqrt(2)/9 = 5.30e-16"},
    "E3_Cs_implied": {"got": [0.23804761428476165, 0.2380476142847617], "ratio": 1.0000000000000002, "pass": true},
    "field_identity_max_rel_dev": {"got": 58295512079842.125, "tol": "15%", "pass": false, "note": "report table 4 marks this OK; 5.83e13 cannot pass any tolerance - pointwise ratio blow-up at |S|->0, and no generating script exists"},
    "E4_ghia_re1000": {"ref": "Ghia 1982", "got": "not_run", "pass": false, "note": "the only physical-accuracy leg; honestly declared in report sec.6"}
  },
  "repro_cmd": "PYTHONPATH=/root/tp-agent/work/m18096/src python3 -m pytest /root/tp-agent/work/m18096/tests/test_xval_les_qabs_cc_proxy.py -q && PYTHONPATH=/root/tp-agent/work/m18096/src python3 /root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/run_xval.py prim  [auditor executed both 2026-11-13 from an isolated copy: 9 passed; prim reproduced kpi_prim.json with 0/241 leaf diffs]",
  "external_anchor": [
    "/tmp/lbtf/LatFlow/Domain.py",
    "/tmp/lbtf/LatFlow/D2Q9.py"
  ],
  "external_anchor_detail": [
    "Domain.py:57 self.Sc = 0.17 - verbatim",
    "Domain.py:90-93 self.EEk = sum_{n,m} |C[...,n]*C[...,m]| - verbatim",
    "Domain.py:121-126 NonEq=f-Feq; Q=sqrt(2 sum NonEq^2 EEk); tau=0.5(tau0+sqrt(tau0^2+6 Q Sc/rho)); f=f-NonEq/tau - verbatim",
    "D2Q9.py:3-6 WEIGHTS/LVELOC order identical to tensorlbm.d2q9 - verbatim",
    "Ghia, Ghia & Shin 1982 (in-repo GHIA_RE1000) - carried but NOT exercised (E4 not_run)"
  ],
  "level_rationale": "L1 met (9/9 tests re-run + algebra re-derived independently to 5e-16). L2 met (five real functions + package export, distribution -> Q -> tau_eff -> collided f, exercised on real TGV/cavity snapshots; not a stub). L3 not met (zero integration callers; and the code is uncommitted, so not even in the repo). L4 not met (0 registered benchmark rows; the physical anchor Ghia Re=1000 is not_run; the 0.0 errors are port-fidelity self-comparison, not accuracy against an external reference). L5 not met (faithful port of a known heuristic closure + its algebra). Verdict L2.",
  "overstated_claims": [
    "report table 4: 'field identity (fig3, 64^2 TGV) max relative deviation 5.8e+13 -> PASS' graded against a <15% criterion; the producing script is in no file of the tree",
    "report table 4 presents K1/K2 = 0.0 as accuracy passes; they are exact by construction (torch.equal against a line-by-line transcription)",
    "mission result JSON repeats those two rows as 'KPI met'",
    "no tensor_reports row for mission 18096 (report_id IS NULL); the six-element report exists only as markdown inside the worktree"
  ],
  "impl_landed_in_repo": false,
  "report_six_elements": "present but compressed (64 lines): problem+anchor, method, parameter table, ref-vs-got table, convergence/sensitivity, failure modes+error verdict. Figures fig1-3 exist as PNGs but are never embedded/captioned in the markdown; no measured-vs-analytic comparison curve.",
  "checks_not_run": ["E4 Ghia Re=1000 cavity", "E2 TGV arm equivalence", "D3Q19 3-D isotropy algebra", "OSS TF1 binary execution as black box"],
  "audit_artifacts": [
    "docs/audit_m18096/audit.json",
    "docs/audit_m18096/AUDIT_m18096_XVAL_LES_SCALAR_Q_ABS_CC_PROXY.md",
    "docs/audit_m18096/independent_check.py",
    "docs/audit_m18096/make_figure.py",
    "docs/audit_m18096/fig_audit_m18096.png",
    "docs/audit_m18096/verdict_m18096.sql"
  ]
}
$json$ WHERE id = 691;

SELECT id, repo, scope, mission_id, level, verdict_by, gate_state FROM tensor_maturity_verdicts WHERE id = 691;