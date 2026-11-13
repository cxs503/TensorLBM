INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18070 XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL',
  18161,
  2,
  $json${
    "impl_files": [
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/run_xval.py (500+ lines, SCMP + phase-field 2D implementations)",
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/make_figures.py (figure generation)"
    ],
    "entry": "run_scmp_laplace() / run_phasefield_laplace() in run_xval.py",
    "run_artifacts": [
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/comparison.json (structured comparison data, 90 lines)",
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/figures/structural_comparison.png (144KB)",
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/figures/parameter_comparison.png (61KB)",
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/figures/scmp_laplace_verified.png (76KB)",
      "xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/REPORT.md (full six-element report)"
    ],
    "tests": "none (no pytest tests written for this xval)",
    "integration_callers": "none (code is standalone, not imported by any TensorLBM module)",
    "benchmark": {
      "metric": "NOT_APPLICABLE - structural comparison, not numerical benchmark",
      "machine_referee": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL' -> 0 rows (no registered benchmark case)",
      "ref_vs_got": {
        "SCMP_laplace_sigma_recovery": {
          "b1_R20": {"target": 0.01, "recovered": 0.009994, "err_pct": 0.06, "pass": true},
          "b1_R32": {"target": 0.01, "recovered": 0.010000, "err_pct": 0.004, "pass": true},
          "b1_R48": {"target": 0.01, "recovered": 0.010044, "err_pct": 0.44, "pass": true},
          "b3_rho10": {"target": 0.01, "recovered": 0.009996, "err_pct": 0.04, "pass": true},
          "b3_rho100": {"target": 0.01, "recovered": 0.009994, "err_pct": 0.06, "pass": true},
          "note": "Data sourced from benchmarks/verified/laplace_cac_phasefield/result.json (repo-verified, not re-run by m18070)"
        },
        "cross_model_comparison": "NOT_APPLICABLE - SCMP and phase-field solve different equations, direct number comparison is meaningless"
      },
      "negative_control": "none"
    },
    "repro_cmd": "cd /root/tp-agent/work/m18070 && python3 xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/make_figures.py && python3 xval/XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL/run_xval.py",
    "external_anchor": "AminZar69/CudaPhaseField (Fakhari family phase-field LBM, CUDA Fortran - not executable in CPU sandbox); Fakhari et al. (2017) Phys. Rev. E 96, 053301",
    "report_six_elements": {
      "case_introduction": "present (REPORT.md §1: 问题背景与参考实现)",
      "method": "present (REPORT.md §2: SCMP vs 相场关键实现差异 + 工具链)",
      "parameter_table": "present (REPORT.md §3: 完整参数表 + 无量纲数对照)",
      "comparison_table": "present (REPORT.md §4: 对照表 ref vs got + 为什么不能对标)",
      "convergence_sensitivity": "present (REPORT.md §5: 4 组敏感性分析)",
      "failure_modes_artifacts": "present (REPORT.md §6: NOT_APPLICABLE 失败模式 + 工件清单 + 复现命令 + 移植取舍)"
    },
    "report_state": "REPORT.md exists in work tree; tensor_reports submission NOT found for mission_id=18070",
    "git_status": "agent/m18070 branch at same commit as main (4903d86c); deliverables are UNTRACKED (not committed)",
    "runs_directory": "EMPTY - no actual run outputs produced",
    "overstated_claims": [
      "FINAL_REPORT.md references Poiseuille convergence (p=1.92) which is from a DIFFERENT mission, not m18070's phase-field xval",
      "Report cites benchmarks/verified/laplace_cac_phasefield/result.json data but did not re-run these benchmarks; data is from prior missions"
    ],
    "honest_failures_documented": [
      "NOT_APPLICABLE verdict: SCMP and phase-field are fundamentally different models - this is a structural negative result, not a numerical failure",
      "OSS CudaPhaseField (CUDA Fortran) cannot run in CPU sandbox; used simplified Python implementation instead",
      "runs/ directory empty - no actual simulation outputs from run_xval.py were captured"
    ],
    "level_rationale": "L0✓ 报告六段齐全；L1✓ 有数学分析（Laplace 定律验证、参数敏感性）；L2✓ 真实代码（run_xval.py 实现 SCMP + 相场双模型，500+ 行）；L3✗ 未集成到主求解器（代码未提交、未被任何模块 import）；L4✗ runs/ 目录为空，无实际运行输出，benchmark 数据引用自其他任务而非本任务重跑；L5✗ 本质是结构性不可对标分析，非新物理/新能力科研突破。",
    "disposition": "reopened",
    "remediation": [
      "Commit deliverables to agent/m18070 branch (currently untracked)",
      "Run run_xval.py and capture outputs to runs/ directory",
      "Submit tensor_report for mission_id=18070",
      "Add pytest tests for the comparison logic",
      "Clarify which benchmark data is from m18070 vs prior missions"
    ]
  }$json$,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;
