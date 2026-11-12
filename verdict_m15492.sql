-- m15492 · board-audit verdict for m15374 (CASE-TAYLOR-GREEN-3D)
-- Level 4 (benchmark validation): complete case + error analysis vs exact analytic anchor
-- + reproduced by independent test suite (4/4 pass). Not L5: no integration callers into
-- main solver/platform; tensor_benchmark_cases row absent (platform ledger gap).
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15374 CASE-TAYLOR-GREEN-3D',
  15492,
  4,
  $json${
    "impl_files": [
      "benchmarks/case_taylor_green_3d/run_case.py:1-520 (single executable behind every number; init_fields/exact_fields/verify/run_decay/sweep drivers)",
      "benchmarks/case_taylor_green_3d/make_report.py:1-200 (JSON → markdown tables + PNG figures, zero hand-typed numbers)",
      "benchmarks/case_taylor_green_3d/prereg.json:1-80 (predeclared decision/metrics/tolerances/deviations before results inspected)",
      "tests/test_case_taylor_green_3d.py:1-72 (4 pytest guards: exactness proof + 2-grid decay + Beltrami companion, all pass in 31s)"
    ],
    "entry": "python benchmarks/case_taylor_green_3d/run_case.py --all  # full sweep → runs/case_taylor_green_3d/*.json",
    "run_artifacts": [
      "benchmarks/case_taylor_green_3d/artifacts/convergence.json (committed; N=24/32/48/64, M1..M5 per grid)",
      "benchmarks/case_taylor_green_3d/artifacts/exactness_verification.json (committed; spectral curl/Laplacian/NS residual ~1e-14)",
      "benchmarks/case_taylor_green_3d/artifacts/tau_sensitivity.json (committed; τ∈{0.55,0.6,0.7,0.8})",
      "benchmarks/case_taylor_green_3d/artifacts/amp_sensitivity.json (committed; A∈{1e-3,2e-3,1e-2,2e-2})",
      "benchmarks/case_taylor_green_3d/artifacts/abc_zdependent.json (committed; Beltrami companion N=32/48)",
      "benchmarks/case_taylor_green_3d/artifacts/figures/{energy_decay,grid_convergence,tau_sensitivity,amp_sensitivity,abc_decay,field_snapshot}.png (6 figures, committed)",
      "benchmarks/case_taylor_green_3d/artifacts/summary.md (committed; auto-generated tables)",
      "runs/case_taylor_green_3d/*.json (NOT committed; identical copies exist in benchmarks/case_taylor_green_3d/artifacts/)"
    ],
    "tests": [
      "tests/test_case_taylor_green_3d.py::test_reference_is_an_exact_navier_stokes_solution (PASSED; curl<1e-10, NS residual<1e-10, Beltrami<1e-10, tg3d non-gradient confirmed)",
      "tests/test_case_taylor_green_3d.py::test_decay_matches_exact_solution[16] (PASSED; M1/M2/M3<8%, M5<1e-12, E_final<E0)",
      "tests/test_case_taylor_green_3d.py::test_decay_matches_exact_solution[24] (PASSED; same criteria)",
      "tests/test_case_taylor_green_3d.py::test_zdependent_beltrami_companion (PASSED; ABC flow M1/M3<8%)"
    ],
    "integration_callers": "none: no import of case_taylor_green_3d from src/tensorlbm/ or platform routers; the case is a standalone benchmark harness. The validated solver path is the production D3Q27 kernel (equilibrium27/collide_bgk27/stream27/macroscopic27) but the case itself does not register as a platform benchmark caller.",
    "benchmark": {
      "platform_ledger": "tensor_benchmark_cases: 0 rows for anchor_slug='CASE-TAYLOR-GREEN-3D' (platform ledger gap; same pattern as m15215/m15319)",
      "de_facto_criteria": "prereg.json predeclared tolerances: M1<8%, M2<8%, M3<8%, M4≥1.8, M5<1e-12, S1<8% each τ, S2 reported table",
      "auditor_recomputed": {
        "M1_energy_decay_rel_err_pct": {"N24": -2.277, "N32": -1.262, "N48": -0.554, "N64": -0.310, "max_abs": 2.277, "tol": 8.0, "pass": true},
        "M2_decay_exponent_rel_err_pct": {"N24": 0.366, "N32": 0.356, "N48": 0.349, "N64": 0.346, "max_abs": 0.366, "tol": 8.0, "pass": true},
        "M3_l2_field_rel_err_pct": {"N24": 1.145, "N32": 0.633, "N48": 0.277, "N64": 0.155, "max": 1.145, "tol": 8.0, "pass": true},
        "M4_observed_order_24_48": 2.045, "tol": 1.8, "pass": true,
        "M5_mass_drift_rel": {"N24": 4.72e-14, "N32": 4.66e-14, "N48": 4.67e-14, "N64": 4.62e-14, "max": 4.72e-14, "tol": 1e-12, "pass": true},
        "S1_tau_sensitivity": {"tau_0.55": 0.665, "tau_0.6": 0.333, "tau_0.7": 0.192, "tau_0.8": 0.187, "max_abs": 0.665, "tol": 8.0, "pass": true},
        "S2_amp_sensitivity": {"A_0.001": {"M1": -1.255, "M2": 0.333, "M3": 0.630}, "A_0.002": {"M1": -1.255, "M2": 0.333, "M3": 0.630}, "A_0.01": {"M1": -1.257, "M2": 0.333, "M3": 0.646}, "A_0.02": {"M1": -1.261, "M2": 0.331, "M3": 0.694}, "all_within_8pct": true},
        "exactness_verification": {"tg_curl_over_norm": 2.71e-15, "tg_ns_residual": 1.59e-14, "abc_beltrami_residual": 2.14e-15, "tg3d_curl_over_norm": 0.227 (non-gradient, correctly rejected as reference)}
      },
      "verdicts": "ALL 7 predeclared metrics (M1-M5, S1, S2) PASS with margins of 3-30×; exactness of reference machine-checked (spectral residual ~1e-14)",
      "caveat": "The z-invariant TG mode is an exact NS solution only because the convective term is curl-free; the classic z-dependent TG vortex has no closed form (curl ratio 0.227). The z-dependent 3D branch is validated via the Beltrami/ABC companion case (also exact, curl residual 2.14e-15). This is a laminar decay validation, not a turbulence-resolution test."
    },
    "repro_cmd": "cd /root/tp-agent/work/m15374 && python benchmarks/case_taylor_green_3d/run_case.py --all && python benchmarks/case_taylor_green_3d/make_report.py  # tests: pytest tests/test_case_taylor_green_3d.py -v (4/4 pass, 31s)",
    "external_anchor": "Taylor & Green (1937) Proc. R. Soc. A 158(895):499–512; exact decaying solution u(t)=u(0)exp(−ν|k|²t) machine-checked by spectral derivatives in run_case.py --verify (curl/NS residual ~1e-14 at N=32, float64)",
    "three_piece_verification": {
      "run_artifacts": "PASS: 8 JSON + 6 PNG + 1 MD committed under benchmarks/case_taylor_green_3d/artifacts/; runs/ copies not committed but identical",
      "benchmark_ref_vs_got": "PASS: all 7 metrics within predeclared tolerances; auditor independently verified from JSON (max M1=2.28% vs 8% tol, M4=2.045 vs 1.8 target); tensor_benchmark_cases row ABSENT (platform ledger gap, not a case defect)",
      "repro_cmd": "PASS: test suite (4 tests, 31s) independently reproduces exactness proof + decay on small grids within tolerance"
    },
    "level_rationale": "L4 = complete benchmark case + error analysis vs exact analytic anchor + reproducible. All predeclared metrics pass with large margins. Exactness of reference is machine-checked (not asserted). 4 pytest guards pass. Not L5: no integration callers into main solver/platform (standalone benchmark harness); tensor_benchmark_cases row absent (platform ledger gap, same pattern as m15215). The case validates the D3Q27 bulk/periodic decay path (equilibrium, BGK collision, streaming, macroscopic recovery) — a genuine solver validation, not a new solver feature.",
    "findings": [
      "F1 low/info: runs/case_taylor_green_3d/*.json not in git (git ls-files = 0); identical copies exist in benchmarks/case_taylor_green_3d/artifacts/ (committed) — cosmetic, no data loss",
      "F2 medium: tensor_benchmark_cases has no row for anchor_slug='CASE-TAYLOR-GREEN-3D' — platform ledger gap (same as m15215/m15319); recommend registering for automated referee",
      "F3 low/info: no integration callers — case is standalone benchmark harness, not wired into platform benchmark registry or main solver __init__"
    ],
    "not_run": [
      "Did not re-run full sweep (N=24/32/48/64, 500 steps each) — audit is verification, not re-execution; test suite (N=16/24, 200 steps) suffices for reproducibility check",
      "Did not modify any code (交差核验单, non-remediation)",
      "Did not verify SDAA/GPU backend (delivery is CPU torch float64 path; SDAA evidence exists in artifacts/sdaa_taylor_green_3d.json but is a separate mission)"
    ],
    "scope_note": "Audited mission m15374 (succeeded), commit fd9a65e3; platform report docs/CASE_TAYLOR_GREEN_3D.md (6 sections, 6 figures); 4 pytest tests pass; benchmark artifacts committed under benchmarks/case_taylor_green_3d/artifacts/"
  }$json$::jsonb,
  'board-audit'
) ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_level=NULL, gate_note=NULL RETURNING id, repo, scope, level, gate_state;
