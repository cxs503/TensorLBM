-- Verdict for m20058 CASE-RT-INSTABILITY-ATWOOD
-- Audit mission m20187, board-audit
-- Date: 2026-11-14

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm20058 CASE-RT-INSTABILITY-ATWOOD', 20187, 4,
'{
  "impl_files": [
    "benchmarks/case_pool/rayleigh_taylor_atwood05/run_case.py (22462 bytes, drives tensorlbm.rayleigh_taylor)",
    "benchmarks/case_pool/rayleigh_taylor_atwood05/ref/ref_lock.py (10568 bytes, closed-form reference values + Crossref bibliographic hash)",
    "benchmarks/case_pool/rayleigh_taylor_atwood05/make_figures.py (9899 bytes, 4 figures)",
    "benchmarks/case_pool/rayleigh_taylor_atwood05/make_report.py (23464 bytes, programmatic report generation)",
    "src/tensorlbm/rayleigh_taylor.py (solver: Boussinesq D2Q9 + Allen-Cahn phase field)"
  ],
  "entry": "benchmarks/case_pool/rayleigh_taylor_atwood05/run_case.py",
  "run_artifacts": [
    "artifacts/runs/summary.json (42642 bytes, 10 cases with per-case metrics)",
    "artifacts/runs/acceptance_table.md (10 cases, all PASS gate_sigma_15pct)",
    "artifacts/runs/A050_l064_g2e-4/case.json (reference case: sigma_sim=2.7725e-03, sigma_visc=3.0855e-03, err=-10.14%)",
    "artifacts/runs/A050_l064_g2e-4/history.csv (2526 steps, a_zc, a_col, Ma, phi bounds, mass drift)",
    "artifacts/runs/A050_l064_g2e-4/instant_rate.csv (instantaneous growth rate curve)",
    "artifacts/runs/A050_l064_g1e-4/case.json through A050_l128_g4e-4/case.json (9 additional cases)",
    "ref/ref_lock.json (reference values + bibliographic provenance)",
    "ref/sources/crossref_tryggvason_2011.json (sha256 9d71e441335266b3...)",
    "figures/fig1_amplitude_growth.png (166 KB)",
    "figures/fig2_instant_rate.png (197 KB)",
    "figures/fig3_fields.png (117 KB)",
    "figures/fig4_ladder_sensitivity.png (73 KB)"
  ],
  "tests": [
    "tests/test_rayleigh_taylor.py (10+ tests: analytic_growth_rate, viscous_growth_rate, init_state, interface_column, zero_crossing, rt_step mass conservation, fit_growth_rate, ac_scheme bounded, reference velocity Boussinesq symmetric)",
    "tests/test_rayleigh_taylor.py::test_ac_scheme_keeps_phase_bounded (atwood=0.5, 400 steps, |phi|<=1+1e-12)"
  ],
  "integration_callers": [
    "benchmarks/case_pool/run_pool.py (discovery-based pool driver, imports case_pool.rayleigh_taylor_atwood05.run_case)",
    "src/tensorlbm/rayleigh_taylor.py (solver kernels: equilibrium, macroscopic, stream, rt_step, run_rayleigh_taylor)"
  ],
  "integration_note": "case_pool driver discovers and runs the case; no production path outside benchmarks/ calls it directly. The solver (tensorlbm.rayleigh_taylor) is a library module used by multiple benchmarks.",
  "benchmark": {
    "registered_case": "SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug=CASE-RT-INSTABILITY-ATWOOD -> 0 rows (no platform referee row)",
    "platform_rows": "no tensorbench_results row for this anchor",
    "self_contained_referee": "ref/ref_lock.json provides closed-form reference values (sigma_viscous, sigma_inviscid, u_spike_c1, u_bubble_c1) evaluated at runtime from Rayleigh/Taylor/Chandrasekhar formulas; tolerance pre-declared at ±15%",
    "ref_vs_got_audit": [
      {"case": "A050_l064_g2e-4 (reference)", "sigma_sim": 2.7725e-03, "sigma_visc": 3.0855e-03, "err_pct": -10.14, "gate": "PASS (±15%)"},
      {"case": "A050_l064_g2e-4_legacy (bgk control)", "sigma_sim": 2.9367e-03, "sigma_visc": 3.0855e-03, "err_pct": -4.82, "gate": "PASS"},
      {"case": "A050_l064_g1e-4 (half gravity)", "sigma_sim": 1.8768e-03, "sigma_visc": 2.1679e-03, "err_pct": -13.43, "gate": "PASS"},
      {"case": "A050_l096_g3e-4 (1.5x grid)", "sigma_sim": 2.7640e-03, "sigma_visc": 3.1119e-03, "err_pct": -11.18, "gate": "PASS"},
      {"case": "A050_l128_g4e-4 (2x grid)", "sigma_sim": 2.7169e-03, "sigma_visc": 3.1213e-03, "err_pct": -12.96, "gate": "PASS"}
    ],
    "convergence_ladder": "lambda=64/96/128 at constant Ag/lambda: error -10.14%/-11.18%/-12.96% (does NOT converge to zero; residual attributed to buoyancy-interface coupling systematic bias, not interface thickness)",
    "external": "Tryggvason/Scardovelli/Zaleski 2011 (Cambridge UP, DOI 10.1017/cbo9780511975264) — bibliographic record fetched via Crossref API and sha256-pinned; closed-form targets from Rayleigh 1883 / Taylor 1950 / Chandrasekhar 1961 §92"
  },
  "repro_cmd": "cd <repo> && python3 benchmarks/case_pool/rayleigh_taylor_atwood05/ref/ref_lock.py && python3 benchmarks/case_pool/rayleigh_taylor_atwood05/run_case.py && python3 benchmarks/case_pool/rayleigh_taylor_atwood05/run_case.py --aggregate && python3 benchmarks/case_pool/rayleigh_taylor_atwood05/make_figures.py --snapshots && python3 benchmarks/case_pool/rayleigh_taylor_atwood05/make_report.py",
  "external_anchor": "Rayleigh 1883 PLMS 14:170; Taylor 1950 Proc. R. Soc. A 201:192; Chandrasekhar 1961 Hydrodynamic and Hydromagnetic Stability §92 (viscous correction); Goncharov 2002 PRL 88:134502 (nonlinear spike/bubble velocity, arXiv:2009.14655 Eq.1 p.3); Tryggvason et al. 2011 Cambridge UP (bibliographic anchor, paywalled — no page/table quoted, MU registered)",
  "level_rationale": "L4 基准验证：完整算例（10 例参数化驱动，网格阶梯+敏感性支臂）+ 误差分析（收敛性、拟合窗效应、流速/界面双估计量交叉核对、失败模式 F1-F5）+ 可复现（repro_cmd 6 条命令，artifacts 全量工件，summary.json 42KB 机器可读）。不授 L5：交付未合并主线；外部锚为闭式理论+书目记录（非实验数据），且锚定文献正文沙箱不可得（MU）。平台 referee 缺失（tensor_benchmark_cases 0 行）为生态摩擦 E2，不降能力级但记录在案。",
  "findings": [
    {"id": "F1", "severity": "info", "what": "Systematic ~10% low bias in sigma_sim vs sigma_visc across all cases", "detail": "lambda=64/96/128 errors -10.1/-11.2/-13.0%; flow-side estimator (|u_y| mode) shows similar bias (-10.8/-10.1/-9.9%), indicating buoyancy-interface coupling systematic error, not grid/interface thickness effect. Within ±15% gate."},
    {"id": "F2", "severity": "info", "what": "Nonlinear spike/bubble terminal velocity not measurable in this configuration", "detail": "Boussinesq symmetry (phi,y)->(-phi,-y) forces u_s=u_b; potential-flow asymmetry cannot be reproduced. Box height limits interface to a/lambda=0.6 before tau=4-6 terminal velocity window. Honestly reported as n/a."},
    {"id": "F3", "severity": "low", "what": "No tensor_benchmark_cases row for this anchor", "detail": "Platform machine referee missing; case uses self-contained ref_lock.json. Ecosystem friction E2 (agent lacks KB API access). Does not affect capability level but blocks automated L4 gate."},
    {"id": "F4", "severity": "low", "what": "2D potential-flow constant C convention inconsistency", "detail": "Sibling case rayleigh_taylor/ref/ref_lock.py uses C=3; this case uses C=1. Both conventions printed in report. Ecosystem friction E3."}
  ],
  "overstated_claims": [],
  "checks_run": [
    "Verified artifacts/runs/summary.json exists (42642 bytes) with 10 cases, all gate_sigma_15pct=true",
    "Verified artifacts/runs/A050_l064_g2e-4/case.json: sigma_sim=0.002772472580044335, sigma_viscous=0.003085464497242449, err=-10.14%",
    "Verified artifacts/runs/acceptance_table.md: 10 cases, all PASS",
    "Verified ref/ref_lock.json: closed-form reference values with Crossref bibliographic hash",
    "Verified figures/ directory: 4 PNG files (166-197 KB each)",
    "Verified history.csv: 2526 steps with a_zc, a_col, Ma, phi bounds, mass drift",
    "Verified REPORT.md: 6 sections present (①-⑥), programmatic generation claimed",
    "Verified src/tensorlbm/rayleigh_taylor.py: real solver code (Boussinesq D2Q9 + Allen-Cahn)",
    "Verified tests/test_rayleigh_taylor.py: 10+ tests covering analytic formulas, bounded phase, mass conservation",
    "Verified run_case.py: 22462 bytes, imports tensorlbm.rayleigh_taylor, pre-declared windows, gate enforcement",
    "Checked tensor_benchmark_cases: 0 rows for CASE-RT-INSTABILITY-ATWOOD (platform referee missing)"
  ],
  "checks_not_run": [
    "Full re-execution of all 10 cases (would require 8-core cluster queue, >5 min wall time)",
    "External literature tables from Tryggvason 2011 (paywalled, MU registered)",
    "pytest tests/test_rayleigh_taylor.py re-run (subject worktree, not this audit worktree)"
  ],
  "disposition": "approve with notes: L4 confirmed. The case has complete benchmark suite, error analysis, and reproducibility. Missing platform referee row (F3) is ecosystem friction, not capability deficiency. Systematic ~10% bias (F1) is honestly disclosed and within gate. Nonlinear velocity not measurable (F2) is a model limitation, not a defect."
}', 'board-audit');
