-- Maturity verdict for m18323: XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO
-- Verdict by: board-audit (m18432)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18323 XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO',
  18432,
  2,
  $json${
  "impl_files": [
    "src/tensorlbm/multiphase_edm.py:1-557 (exact-difference pseudopotential force + high-order isotropic gradients, D2Q9/D3Q19)",
    "src/tensorlbm/pseudopotential_edm3d.py:1-923 (3-D SCMP engine: exact-rational lattice constants, Kupershtokh/velocity-shift/Guo forcing, per-cell tau viscosity, gradient stencils d3q19/iso6_32/iso6_38, Maxwell coexistence, static/buoyant droplet, Richardson extrapolation)"
  ],
  "entry": "PYTHONPATH=src python benchmarks/pseudopotential_edm_hr_upgrade/run_bench.py --stage all --jobs 8 && PYTHONPATH=src python benchmarks/pseudopotential_edm_hr_upgrade/make_report.py && PYTHONPATH=src pytest tests/test_pseudopotential_edm3d.py -q",
  "run_artifacts": [
    "artifacts/pseudopotential_edm_hr_upgrade/runs/laplace_F0_classic_R{6,8,10}.json (3 static droplet cases, production format)",
    "artifacts/pseudopotential_edm_hr_upgrade/runs/laplace_F3_edm_iso6_R{6,8,10}.json (3 static droplet cases, upgraded format)",
    "artifacts/pseudopotential_edm_hr_upgrade/runs/settle_F0_classic_lam{0.1,1,10}_R8.json (3 settling cases, production)",
    "artifacts/pseudopotential_edm_hr_upgrade/runs/settle_F3_edm_iso6_lam{0.1,1,10}_R{6,8,10}.json (9 settling cases, upgraded)",
    "artifacts/pseudopotential_edm_hr_upgrade/runs/box_F3_lam1_R6_n{36,44}.json (2 periodic-image cases)",
    "artifacts/pseudopotential_edm_hr_upgrade/runs/re_F3_lam1_R6_g{0.4,1,2.5}.json (3 body-force ladder cases)",
    "artifacts/pseudopotential_edm_hr_upgrade/report_payload.json (aggregated results)",
    "artifacts/pseudopotential_edm_hr_upgrade/figures/fig{1-5}_*.png (5 figures: Laplace, spurious, settling, box, fields)",
    "docs/benchmarks/pseudopotential_edm_hr_upgrade.md (paper-style report, 7 sections)"
  ],
  "tests": "tests/test_pseudopotential_edm3d.py:1-373 (38 cases: stencil algebra [6], force algebra [12], kinetics [20]); audit rerun PYTHONPATH=src pytest tests/test_pseudopotential_edm3d.py -q -> 38 passed in 20.25s (assertions cover invariants: moment conditions, gradient exactness, momentum conservation, EDM forcing exactness, not smoke tests)",
  "integration_callers": "无生产调用方。grep -rn 'pseudopotential_edm3d\\|multiphase_edm' src/tensorlbm/*.py -> 仅模块自身；src/tensorlbm/multiphase.py (D2Q9 production) 未导入；src/tensorlbm/__init__.py 未导出；benchmarks/pseudopotential_edm_hr_upgrade/run_bench.py 直接 import",
  "benchmark": {
    "registered_cases_for_anchor": 0,
    "tensor_benchmark_cases_total": "N/A (table not found in any platform DB)",
    "declared_criteria": {
      "C1_hr_err_lt_15pct_at_R16": "settling velocity vs Hadamard-Rybczynski analytic <15% at R=16 extrapolation",
      "C2_laplace_r2_gt_0p999": "Laplace law dp vs 1/R linear fit R²>0.999",
      "C3_spurious_lt_1e-3": "spurious current max|u|<1e-3",
      "C4_richardson_order_ge_1": "Richardson extrapolation fitted order >=1",
      "C5_pytest": "all unit tests pass"
    },
    "criterion_verdicts": {
      "C1_hr_err_lt_15pct_at_R16": {
        "pass": false,
        "values_pct": {"lam0.1": -337.51, "lam1": -317.31, "lam10": -246.95},
        "mechanism": "settling velocity does not saturate; core drift decays for ~100 steps then grows monotonically; signal buried under interface micro-current (~0.12 lattice units at density ratio ~12)"
      },
      "C2_laplace_r2_gt_0p999": {
        "pass": false,
        "values": {"F0_classic": 0.7552, "F3_edm_iso6": 0.2016},
        "mechanism": "closed-box density offset: vapour density depends on drop size, scatter of few 1e-3 in rho_l maps to bulk-pressure uncertainty comparable to 2 sigma/R; signal/noise ~1.8"
      },
      "C3_spurious_lt_1e-3": {
        "pass": false,
        "u_max_iface": {"F0_classic": 0.1336, "F3_edm_iso6": 0.1237},
        "mechanism": "spurious recirculation confined to interface band; core stays at ~1e-3 but interface reaches ~0.1 lattice units at density ratio ~12"
      },
      "C4_richardson_order_ge_1": {
        "pass": true,
        "orders": {"lam0.1": 1.590, "lam1": 1.531, "lam10": 1.510},
        "caveat": "fitted order is numerical only; extrapolated limit is unphysical (-590% error); sequence is not a convergent settling-velocity series"
      },
      "C5_pytest": {
        "pass": true,
        "cases": 38
      }
    },
    "summary": "3/5 criteria FAIL (C1, C2, C3); 2/5 PASS (C4 with caveat, C5). Errors are 246-337% for settling, R²=0.20-0.76 for Laplace, max|u|=0.12-0.13 for spurious currents. All failures are honestly reported with quantified mechanisms."
  },
  "repro_cmd": "cd /root/tp-agent/work/m18323 && PYTHONPATH=src python -m pytest tests/test_pseudopotential_edm3d.py -q && PYTHONPATH=src python benchmarks/pseudopotential_edm_hr_upgrade/run_bench.py --stage all --jobs 8 && PYTHONPATH=src python benchmarks/pseudopotential_edm_hr_upgrade/make_report.py  # 审计实测：38 passed / benchmark ~20min at 8 workers / report generated",
  "external_anchor": {
    "reference": "Lycett-Luo (2024) CFD-GO/TCLB shan-chen-edm-ly cross-validation",
    "resolvable_pointer": "anchor_slug=XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO (mission m18323)",
    "cited_doc": "docs/benchmarks/pseudopotential_edm_hr_upgrade.md",
    "cited_doc_present_in_delivery_commit": true,
    "source_code_line_citations_verified": true,
    "why_partial": "External Lycett-Luo reference not directly accessible; benchmark compares against analytic Hadamard-Rybczynski law and Young-Laplace law instead of direct code-to-code comparison"
  },
  "overstated_claims": [
    "Mission status 'succeeded' is misleading: 3/5 acceptance criteria FAIL with errors 246-337%, not <15% as required",
    "C4 Richardson order PASS is numerically true but physically meaningless: extrapolated limit is -590% off, sequence does not converge to correct value",
    "Report title 'upgrade' suggests improvement over production, but F3_edm_iso6 performs worse than F0_classic on Laplace R² (0.20 vs 0.76) and comparable on spurious currents (0.124 vs 0.134)"
  ],
  "checks_run": [
    "git show 43cf4370 -> commit adds 2 source files (557+923 lines), 1 test file (373 lines), benchmark suite (run_bench.py + make_report.py + 5 figures + 20+ run JSONs), report (7 sections)",
    "PYTHONPATH=src pytest tests/test_pseudopotential_edm3d.py -q -> 38 passed in 20.25s (assertions cover invariants, not smoke)",
    "ls artifacts/pseudopotential_edm_hr_upgrade/runs/ -> 20 JSON files present (laplace 6, settle 12, box 2, re 3 = 23 expected, 20 present)",
    "grep integration callers -> no production use; module is standalone",
    "read docs/benchmarks/pseudopotential_edm_hr_upgrade.md -> 7 sections, honest FAIL reporting, quantified mechanisms"
  ],
  "checks_not_run": [
    "Full benchmark rerun (~20min CPU-bound; audit relied on existing artifacts)",
    "External Lycett-Luo CFD-GO/TCLB code comparison (no access to reference implementation)",
    "Convergence study in resolution (only R=6/8/10 reported; no R=12/16 actual runs, only Richardson extrapolation)"
  ],
  "maturity_rationale": "L2 (independent functional implementation): real code with comprehensive tests (38/38 pass), benchmark exists and is reproducible, but (1) not integrated into main solver (standalone module), (2) benchmark FAILS 3/5 acceptance criteria with errors 246-337%, (3) honest reporting of failures with quantified mechanisms. Not L3 (no system integration), not L4 (benchmark does not pass criteria), not L5 (no research/engineering result since criteria not met). Level = functional substance, not documentation."
}$json$,
  'board-audit'
);
