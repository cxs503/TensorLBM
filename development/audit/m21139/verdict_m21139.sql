-- Verdict for m21139: 交差审计 m20982 GAP-DIFFLBM-CAV-RES
-- Auditor: board-audit (m21139)
-- Date: 2026-11-15

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm20982 GAP-DIFFLBM-CAV-RES',
  21139,
  2,
  '{
    "impl_files": [
      "benchmarks/difflbm_cavity_resolution/run.py (driver, 7-case sweep harness)",
      "benchmarks/difflbm_cavity_resolution/make_report.py (report builder, NOT executed)",
      "benchmarks/verified/cavity/re100/run.py (solver hook, capture= centreline)",
      "benchmarks/verified/cavity/re400/run.py (solver hook, capture= centreline)"
    ],
    "entry": "benchmarks/difflbm_cavity_resolution/run.py --shard i --shards 7 (or --all --aggregate)",
    "run_artifacts": [
      "artifacts/gap_difflbm_cav_res/summary.json (aggregated, 3 of 7 cases)",
      "artifacts/gap_difflbm_cav_res/case_re100_nx33.json (Re=100, n=33, max_dev=2.896%)",
      "artifacts/gap_difflbm_cav_res/case_re100_nx65.json (Re=100, n=65, max_dev=1.518%)",
      "artifacts/gap_difflbm_cav_res/case_re100_nx97.json (Re=100, n=97, max_dev=0.899%)"
    ],
    "run_artifacts_missing": [
      "case_re100_nx129.json (Re=100, n=129 — REQUIRED for Richardson r=2 self-convergence)",
      "case_re400_nx65.json (Re=400, n=65)",
      "case_re400_nx97.json (Re=400, n=97)",
      "case_re400_nx129.json (Re=400, n=129)"
    ],
    "tests": "None specific to this sweep; relies on benchmarks/verified/cavity/re100 and re400 existing tests",
    "integration_callers": "None — standalone driver, not called by main solver or platform pipeline",
    "benchmark": {
      "reference": "Ghia, Ghia & Shin (1982) JCP 48:387-411, Tables I & II",
      "metric": "max_abs_dev_pct = 100 * max |u_sim - u_Ghia| over 34 centreline points",
      "got_values": {
        "Re100_n33": 2.8963,
        "Re100_n65": 1.5184,
        "Re100_n97": 0.8991
      },
      "threshold_max_dev_pct": 3.0,
      "threshold_n_at_least": 97,
      "n_ge_97_pass": true,
      "tensor_benchmark_cases_registered": false,
      "note": "No row in tensor_benchmark_cases for anchor_slug=GAP-DIFFLBM-CAV-RES; no machine-verifiable referee"
    },
    "repro_cmd": "cd <repo> && PYTHONPATH=src:benchmarks python3 benchmarks/difflbm_cavity_resolution/run.py --all --aggregate",
    "repro_cmd_status": "executable but incomplete — only 3/7 cases produced; n=129 and all Re=400 cases missing",
    "external_anchor": "Ghia et al. (1982) JCP 48:387-411 — standard lid-driven cavity reference",
    "criteria_results": {
      "order_p_self_l2_ge_1.8": false,
      "order_p_self_max_ge_1.8": false,
      "max_dev_lt_3pct_for_all_n_ge_97": true,
      "max_dev_monotone_decrease": true,
      "all_steady": true,
      "all_criteria_met": false
    },
    "report_generated": false,
    "figures_generated": false,
    "level_rationale": "L2 独立功能实现：真代码真计算（run.py 驱动 verified cavity 求解器，产出 3 个 case JSON + summary.json，数值可复现）。不授 L3：无生产调用方（standalone driver，未接入主求解器/平台管线）。不授 L4：①7 个预声明算例仅完成 3 个（缺 Re=100 n=129 及全部 Re=400 三档）；②Richardson 自收敛阶无法计算（缺 n=129，orderset incomplete）；③all_criteria_met=false；④报告/图未生成（make_report.py 未执行）；⑤tensor_benchmark_cases 无注册行，无机器裁判。不授 L5：交付未合并主线。"
  }',
  'board-audit'
);
