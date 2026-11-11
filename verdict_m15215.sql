-- m15273 · board-audit verdict for m15215 (XVAL-LBM-TAYLORGREEN)
-- Level 4 (benchmark validation): complete case + error analysis vs analytic anchor
-- + reproduced bit-exactly by the auditor. Not L5: 2 of 3 solver axes (OpenFOAM
-- degraded, honestly declared), no tensor_benchmark_cases row, branch unmerged.
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15215 XVAL-LBM-TAYLORGREEN',
  15273,
  4,
  $json${
    "impl_files": [
      "benchmarks/xval_taylor_green_2d/case_spec.json:1-84 (single spec, sha256 957b951f...d3ce, reused verbatim by both legs)",
      "benchmarks/xval_taylor_green_2d/run_ours.py:1-108 (TensorLBM leg; :73 collide_bgk(stream(f)) -> production kernel)",
      "benchmarks/xval_taylor_green_2d/palabos/tg2d_palabos.cpp:1-114 (palabos leg, data-processor init/sampling)",
      "benchmarks/xval_taylor_green_2d/run_palabos.py:1-105 (build + run + provenance driver)",
      "benchmarks/xval_taylor_green_2d/xval_common.py:1-178 (:50 tau=0.5+3*nu; one shared reduction path)",
      "benchmarks/xval_taylor_green_2d/analyze.py:1-187",
      "benchmarks/xval_taylor_green_2d/make_report.py:1-271"
    ],
    "entry": "python3 benchmarks/xval_taylor_green_2d/run_ours.py --out runs/tensorlbm ; python3 benchmarks/xval_taylor_green_2d/run_palabos.py ; python3 benchmarks/xval_taylor_green_2d/analyze.py -> result.json",
    "run_artifacts": [
      "benchmarks/xval_taylor_green_2d/runs/tensorlbm/timeseries.csv (99 rows, committed, bit-exactly reproduced by auditor: max_rel 0.0 on t,K,umax,phase_re,phase_im)",
      "benchmarks/xval_taylor_green_2d/runs/palabos/timeseries.csv (99 rows, committed, bit-exactly reproduced by auditor)",
      "benchmarks/xval_taylor_green_2d/runs/tensorlbm/run_meta.json (float64, tau=0.692, wall 53.77s, mass_drift -3.9324099532223045e-13, mlups 1.493)",
      "benchmarks/xval_taylor_green_2d/result.json:1-343 (xval_table/criteria/bug_suspects/degradation/friction)",
      "benchmarks/xval_taylor_green_2d/figures/fig1_K_decay.png, fig2_phase.png, fig3_criteria.png",
      "docs/audit/m15215/audit_metrics.json (auditor independent re-reduction, numpy only)",
      "MISSING: benchmarks/xval_taylor_green_2d/runs/palabos/run_meta.json (claimed in result.checks, absent on disk and in git; result.tool_provenance.palabos={}) -> finding F4"
    ],
    "tests": [
      "tests/test_xval_taylor_green_2d.py:38-88 (6 passed in 6.66s, auditor rerun 2026-11-11; asserts |gamma_K err|<1%, |gamma_vel err|<1%, phase_err<1e-2%, R2>0.9999, two-impl cross diff<5e-3, result.json all PASS + degradation key present)"
    ],
    "integration_callers": "run_ours.py:27-29 imports tensorlbm + tensorlbm.d2q9/solver; the validated path is the production kernel src/tensorlbm/solver.py:68 (collide_bgk) and :405 (stream) with src/tensorlbm/d2q9.py:68 (equilibrium) / :90 (macroscopic) - no private reimplementation. No platform benchmark-registry caller yet (finding F2).",
    "benchmark": {
      "platform_ledger": "tensor_benchmark_cases: 0 rows for anchor_slug=XVAL-LBM-TAYLORGREEN (WHOLE table = 14 rows; TensorLBM has only MCMP-POISEUILLE/SPHERE/FRAZIL, all status=pending) -> no machine referee registered, ledger gap",
      "de_facto_criteria": "mission goal tolerances == case_spec.json.criteria, verbatim: cross-tool gamma_K <5%, vs Taylor-Green series solution <8%, peak-velocity phase <3%",
      "auditor_recomputed": {
        "gamma_K_tensorlbm": 6.166609e-4, "gamma_K_palabos": 6.166609e-4,
        "gamma_K_theory_4nuk2": 6.168503e-4, "gamma_K_err_pct": -0.03070,
        "gamma_vel_err_pct": -0.00534,
        "cross_tool_diff_pct": 8.720583e-12,
        "phase_err_pct_max_tensorlbm": 4.523456e-13,
        "phase_err_pct_max_palabos": 2.827160e-14,
        "lnK_fit_r2": 0.9999996673,
        "K_pointwise_max_rel_err_pct": 0.4135
      },
      "verdicts": "3/3 criteria PASS with margins of 3-14 orders of magnitude",
      "caveat": "two independent BGK D2Q9 codebases agree to 3.7e-14 (round-off), which certifies wiring/spec conformance only, not discrete-model error; the mission report states this limitation itself"
    },
    "repro_cmd": "PYTHONPATH=src python3 benchmarks/xval_taylor_green_2d/run_ours.py --out runs/tensorlbm && python3 benchmarks/xval_taylor_green_2d/run_palabos.py && python3 benchmarks/xval_taylor_green_2d/analyze.py ; auditor rerun: /tmp/tg2d_palabos --n 128 --u0 0.05 --re 100 --steps 4900 --every 50 --out <csv> (7.37s, bit-exact vs committed csv)",
    "external_anchor": "Analytic Taylor-Green (G.I. Taylor 1923): single Fourier mode is an exact NS solution, K(t)=(U0^2/4)exp(-4*nu*k^2*t), gamma_K=4*nu*k^2, phase=pi/2. Second implementation: palabos @ gitlab commit e04e58b40cb210df04de6bc99f5efe82b985185d (auditor verified via git -C /tmp/pb_src rev-parse HEAD), binary sha256 cb8cb31231db687dc617e826923267a568910ceeb9c470f7bbb11a0227c1e9c0",
    "audit": {
      "auditor_mission": 15273,
      "branch": "agent/m15215 @ 61345fbd (NOT an ancestor of origin/main; git merge-base --is-ancestor -> false)",
      "report_id": 14330,
      "report_sections": 8,
      "report_numbers_match_auditor_recompute": true,
      "determinism": "both legs rerun by auditor reproduce the committed timeseries bit-exactly (max_rel 0.0 on all columns)",
      "not_run": ["OpenFOAM FVM axis (loongarch64: no package, wmake unknown arch, budget) - degradation honestly declared with 5 friction entries",
                  "cross-resolution N=64/256 and cross-Re convergence sweep - outside this audit's scope"],
      "findings": [
        "F1 medium: branch unmerged, artifact reachable only on agent/m15215",
        "F2 medium: no tensor_benchmark_cases row for this anchor (ledger gap, recommend registering)",
        "F3 medium: 2 of 3 axes delivered (OpenFOAM degraded); tool-vs-tool criterion therefore checks wiring not model error",
        "F4 low: runs/palabos/run_meta.json claimed but absent; provenance re-established by auditor",
        "F5 low: palabos leg depends on out-of-repo /tmp/pb_src + /tmp/pb_build2 (documented in driver)",
        "F7 low: anchor slug mismatch - mission/ledger says XVAL-LBM-TAYLORGREEN while case_spec.json/result.json record XVAL-TAYLORGREEN; reconcile before registering the benchmark row"
      ],
      "disposition": "approve with notes; level 4 confirmed, no blocking defect"
    }
  }$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence;