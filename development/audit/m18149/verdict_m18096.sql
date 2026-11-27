-- m18149 board-audit verdict for m18096 (XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY, TensorLBM)
-- Auditor: board-audit (agent/m18149). Evidence bundle: docs/audit_m18096/audit.json
-- Machine referee: 0 rows in tensor_benchmark_cases for this anchor_slug (verified).
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18096 XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY', 18149, 2, $json$
{
  "impl_files": [
    "src/tensorlbm/turbulence/turbulence.py:2015 _abs_cc_weights",
    "src/tensorlbm/turbulence/turbulence.py:2020 les_scalar_q_abs_cc_proxy_2d",
    "src/tensorlbm/turbulence/turbulence.py:2039 les_scalar_q_abs_cc_proxy_3d",
    "src/tensorlbm/turbulence/turbulence.py:2058 tau_eff_q_abs_cc_proxy",
    "src/tensorlbm/turbulence/turbulence.py:2087 collide_smagorinsky_q_abs_cc_proxy_bgk",
    "src/tensorlbm/turbulence/turbulence.py:2118 collide_smagorinsky_q_abs_cc_proxy_bgk3d",
    "src/tensorlbm/__init__.py:530-541 lazy-export + __all__ (worktree-only, UNCOMMITTED)"
  ],
  "entry": "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/run_xval.py (422 lines, subcommands prim/flow/cavity/tgv)",
  "run_artifacts": [
    "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/kpi_prim.json (9535 B, sha256 40e104f6…dcaafc; reproduced bit-identical, 241/241 leaf keys)",
    "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/kpi_final.json (2659 B; provenance unresolved - field_identity_max_rel_dev is written by no script in the tree)",
    "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/fig1_proxy_isotropy.png (56944 B)",
    "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/fig2_port_fidelity.png (43660 B)",
    "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/fig3_field_identity.png (74652 B; no generating script)",
    "xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/predeclared.json",
    "MISSING: out/kpi_flow.json (E2), out/kpi_cavity.json (E4)"
  ],
  "tests": [
    "tests/test_xval_les_qabs_cc_proxy.py (9 tests) - auditor re-ran in isolated copy: 9 passed in 6.31 s (2026-11-13)"
  ],
  "integration_callers": "none: only __init__ lazy-export + tests + the xval runner; no solver/pipeline caller (contrast collide_smagorinsky_bgk used at src/tensorlbm/ai/pipeline.py:82, src/tensorlbm/apps/ai4s_flagship.py:52)",
  "benchmark": {
    "machine_referee": "0 rows: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY' -> 0 (ILIKE variants also 0). No registered referee; judged from mission artifacts + OSS source anchor.",
    "K1_max_rel_err_Q_7_states": {"ref": "OSS transcription", "got": 0.0, "tol": "15%", "pass": true, "caveat": "exact by construction (same formula twice) - fidelity, not accuracy"},
    "K2_max_rel_err_post_collision_f": {"ref": "OSS transcription", "got": 0.0, "tol": "15%", "pass": true, "caveat": "same"},
    "E3_a_Q_over_3rhotauS": {"ref": "sqrt(2)/9 = 0.1571348402636772", "got": [0.15713484026367674, 0.15713484026367788], "rel_err": 3.5e-15, "spread_pct": 7.24e-13, "tol": "15%", "pass": true, "independent_check": "docs/audit_m18096/independent_check.py: 2000 random strains, max |S2/(|S|^2/81)-1| = 8.88e-16, max rel dev vs sqrt(2)/9 = 5.30e-16"},
    "field_identity_max_rel_dev": {"got": 58295512079842.125, "pass": false, "note": "report marks this OK; 5.83e13 cannot pass <15% - pointwise ratio blow-up at |S|->0"},
    "E4_ghia_re1000": {"ref": "Ghia 1982", "got": "not_run", "pass": false, "note": "only physical-accuracy leg; honestly declared in report sec.6"}
  },
  "repro_cmd": "PYTHONPATH=<repo>/src python -m pytest tests/test_xval_les_qabs_cc_proxy.py -q ; PYTHONPATH=<repo>/src python xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/run_xval.py prim  [auditor executed both 2026-11-13 from /tmp/m18149_audit: 9 passed; prim reproduced kpi_prim.json with 0/241 diffs]",
  "external_anchor": [
    "/tmp/lbtf/LatFlow/Domain.py:57 (Sc=0.17), :90-93 (EEk), :121-126 (Q, tau, collision) - source-verified verbatim by the auditor",
    "/tmp/lbtf/LatFlow/D2Q9.py:3-6 WEIGHTS/LVELOC identical ordering",
    "Ghia, Ghia & Shin 1982 (in-repo GHIA_RE1000) - not exercised"
  ],
  "level_rationale": "L1 met (9/9 tests + algebra re-derived independently to 5e-16). L2 met (real 5-function implementation + package export, distribution->Q->tau_eff->f). L3 not met (zero integration callers; code uncommitted so not even in the repo). L4 not met (0 registered benchmark rows; physical anchor Ghia not_run; the 0.0 errors are port-fidelity self-comparison). L5 not met. Verdict L2.",
  "overstated_claims": [
    "table 4 'field identity 5.8e+13 ✅' against a <15% criterion",
    "table 4 presents K1/K2 = 0.0 as accuracy passes; they are exact by construction",
    "no tensor_reports row for mission 18096 (report_id NULL); six-element report exists only as markdown in the worktree"
  ],
  "impl_landed_in_repo": false,
  "checks_not_run": ["E4 Ghia cavity", "E2 TGV arm equivalence", "D3Q19 3-D isotropy algebra", "OSS TF1 binary execution"]
}
$json$, 'board-audit');