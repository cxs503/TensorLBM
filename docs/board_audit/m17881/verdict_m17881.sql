-- Board-audit verdict for m17763 (交差核验单 m17881)
-- Re-runnable / idempotent on (repo, scope, verdict_by).
WITH ev AS (
  SELECT $json${
  "impl_files": ["/root/tp-agent/work/m17763/src/tensorlbm/free_energy_stress.py"],
  "impl_detail": ["src/tensorlbm/free_energy_stress.py:1-217 — FreeEnergyParams(a=-A,b=B,kappa), chemical_potential (a*phi+b*phi^3-kappa*lap), bulk_pressure (a/2 phi^2+3b/4 phi^4), pressure_tensor (d*delta_ab+kappa d_a phi d_b phi, d=p0-kappa*phi*lap-kappa/2|grad|^2), stress_divergence_force (F=-div P), grad_mu_force (F=-phi grad mu), surface_tension_analytic (2sqrt2/3 sqrt(kappa A^3)/B), planar/droplet_profile; no NotImplementedError/stub"],
  "entry": ["/root/tp-agent/work/m17763/experiments/xval_ludwig_fe_stress/run_xval.py"],
  "run_artifacts": ["/root/tp-agent/work/m17763/experiments/xval_ludwig_fe_stress/results.json", "/root/tp-agent/work/m17763/experiments/xval_ludwig_fe_stress/figures/xval_fe_stress.png"],
  "tests": ["tests/test_free_energy_stress.py::test_surface_tension_from_pressure_tensor (analytic sigma, <15%, kappa=2/8/32)", "tests/test_free_energy_stress.py::test_force_vanishes_on_equilibrium_profile (both force arms <15% of sigma/xi^2)", "tests/test_free_energy_stress.py::test_curved_interface_surface_tension (droplet R=32 <15%) — auditor re-ran: 3 passed in 5.28s"],
  "integration_callers": [],
  "integration_note": "ZERO product-code callers: grep -rn 'free_energy_stress|stress_divergence_force|grad_mu_force' src/ (excluding the module itself) -> 0 hits; module not exported in src/tensorlbm/__init__.py; src/tensorlbm/multiphase.py:1223 free_energy_step body lines 66-69 still build Fx=-phi*grad_mu_x, Fy=-phi*grad_mu_y (+rho_eff*g). The deliverable documents this itself (差异清单: 体力入碰撞 = 未改求解器, 列入后续). => L3 NOT met by design of this mission; not an overclaim.",
  "benchmark": {
    "machine_judge": "ABSENT — SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE' -> 0 rows (anchor_slug ILIKE '%FREE-ENERGY%' -> 0 rows in 65); the declared sole judge does not exist, so L4 cannot be confirmed on the table basis (platform-side registration gap, same as m17713)",
    "ref_source": "analytic (2sqrt2/3)sqrt(kappa A^3)/B and 2D curvature balance dP=sigma/R; reference formula independently re-derived symbolically by the auditor",
    "ref_vs_got": [
      {"metric": "sigma flat xi=2 (kappa=2)", "ref": 1.3333333, "got": 1.2521403, "rel_err": 0.06089},
      {"metric": "sigma flat xi=4 (kappa=8)", "ref": 2.6666667, "got": 2.6232546, "rel_err": 0.01628},
      {"metric": "sigma flat xi=8 (kappa=32)", "ref": 5.3333333, "got": 5.3112430, "rel_err": 0.00414},
      {"metric": "sigma curved R=32 xi=4", "ref": 2.6666667, "got": 2.6228258, "rel_err": 0.01644},
      {"metric": "sigma curved R=64 xi=8", "ref": 5.3333333, "got": 5.3110483, "rel_err": 0.00418},
      {"metric": "curvature balance int(P_NN-P_TT)/r dr vs sigma_flat/R (R=32)", "ref": 0.08200, "got": 0.08244, "rel_err": 0.00534},
      {"metric": "curvature balance (R=64)", "ref": 0.08300, "got": 0.08343, "rel_err": 0.00518},
      {"metric": "equilibrium force residual max|F_stress|/(sigma/xi^2) xi=2/4/8", "ref": 0.0, "got": [0.05299, 0.02091, 0.00579], "rel_err": [0.05299, 0.02091, 0.00579]},
      {"metric": "pointwise force difference max|F_stress-F_gradmu|/max|F_gradmu| xi=4", "ref": 0.0, "got": 1.0422, "rel_err": 1.0422, "flag": "honest FAIL declared by author; auditor showed the denominator is itself an O(h^2) residual (ill-conditioned metric), the two forms are analytically identical"}
    ],
    "auditor_independent_recompute": "numpy re-derivation (no tensorlbm import) of sigma on the same analytic tanh field: 6.0895e-2 / 1.62795e-2 / 4.14202e-3 -> matches results.json to 15 digits; full-domain trapezoid (seam included) gives 144%-600% error, confirming the +-8xi window is a necessary control",
    "under_3pct_line": "xi>=4 PASS (1.63% / 0.41%); xi=2 FAIL (6.09%) — the deliverable adjudicates against the owner 15% band and does not state the <3% answer explicitly"
  },
  "repro_cmd": "cd /root/tp-agent/work/m17763 && python experiments/xval_ludwig_fe_stress/run_xval.py && python -m pytest tests/test_free_energy_stress.py -q   # auditor: exit 0, printed summary reproduces results.json to all printed digits, git status clean (deterministic); pytest 3 passed in 5.28s",
  "external_anchor": "ludwig-cf/ludwig build-artifact snapshot /tmp/ludwig_tar/ludwig-master (loongarch64; src/*.c count = 0, so no source text — author's claim confirmed). Auditor corroboration: nm symmetric.o -> fe_symm_str, fe_symm_str_v, fe_symm_interfacial_tension, fe_symm_interfacial_width; readelf decodedline has 404/405/410 in the claimed range; .rodata.cst8 holds IEEE-754 doubles 0.5, 0.75, -8.0, 9.0, -2.0, pi (the claimed .LC constants); nm phi_force_stress.o -> pth_kernel, pth_stress_set, pth_stress, pth_stress_compute (two-stage set-then-difference as claimed); strings fe_force_method.o -> FE_FORCE_METHOD_NO_FORCE/RELAXATION_SYMM/RELAXATION_ANTI + 'stress_divergence'. LIMIT: no instruction-level LC->source-line binding re-derived; the anchor is implementation-forensics only (no runnable OSS case), so it cannot adjudicate physics.",
  "level_reasoning": {
    "L1": "PASS — 3 unit tests assert invariants (analytic sigma, vanishing equilibrium force, curved sigma); analytic sigma formula and the div-P <-> -phi grad mu identity re-derived independently by the auditor",
    "L2": "PASS — real 217-line implementation (input phi -> pressure tensor -> force), runnable entry, 2 non-empty run artifacts, deterministic re-run",
    "L3": "FAIL — zero callers in product code; free_energy_step unchanged; module not exported",
    "L4": "FAIL — no tensor_benchmark_cases row (0/65); sigma measured on a prescribed analytic field, not on a solver-produced field; no time-stepped case",
    "L5": "FAIL — anchor is implementation forensics with no runnable OSS case, so no adjudicable external physical reference"
  },
  "severity_findings": [
    {"id": "F1", "severity": "blocking-L3", "finding": "force not wired into any solver path (free_energy_step still -phi grad mu)"},
    {"id": "F2", "severity": "blocking-L4", "finding": "anchor has no tensor_benchmark_cases row; sole declared judge absent"},
    {"id": "F3", "severity": "minor-metric", "finding": "force_diff_rel divides by an O(h^2) residual -> 104% is not a physical error measure"},
    {"id": "F4", "severity": "minor-scope", "finding": "5pt vs iso9 algebraically identical on a 1D planar profile (4/12+1/12+1/12=1/2); curved case uses iso9 only"},
    {"id": "F5", "severity": "minor-scope", "finding": "sigma from analytic profile = discretisation check, not solver benchmark"},
    {"id": "F6", "severity": "verified-good", "finding": "formula/identity correct; OSS constants and symbols corroborated"},
    {"id": "F7", "severity": "note", "finding": "<3% line: only xi>=4 passes; xi=2 = 6.09%"}
  ],
  "overstated_claims": "none found — headline numbers (6.09% / 1.64% / 0.41% / 0.58-5.3%) all reproduced; the 104% pointwise-difference row is self-labelled honest FAIL and is not used as a pass",
  "audit_artifacts": ["/root/tp-agent/work/m17881/docs/board_audit/m17881/AUDIT_m17763_FE_STRESS.md", "/root/tp-agent/work/m17881/docs/board_audit/m17881/audit_numbers.json", "/root/tp-agent/work/m17881/docs/board_audit/m17881/figures/m17881_audit_verification.png"],
  "merge_state": "module absent from /root/tensor-ecosystem/TensorLBM main (be78a87b); commit a53d734c lives on branch agent/m17763",
  "disposition": "approve with notes -> L2"
}$json$::jsonb AS ev
)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
SELECT 'TensorLBM', 'm17763 XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE', 17881, 2, ev, 'board-audit'
FROM ev
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL
RETURNING id, repo, level, gate_state, jsonb_array_length(evidence->'integration_callers') AS n_callers;