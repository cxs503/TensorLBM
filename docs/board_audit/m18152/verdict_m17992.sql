-- board-audit verdict for m17992 (XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE, TensorLBM)
-- audit mission: m18152 | 2026-11-13 | executed: INSERT ... ON CONFLICT DO UPDATE
-- result: id=704, level=2, gate_state=pending
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17992 XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE', 18152, 2, '{
  "impl_files": [
    "xval/model_matrix_ci_plus_coverage/run_matrix.py — 693 行，6 测试用例真调 tensorlbm.d2q9/d3q19 的 collide_bgk/stream/macroscopic/equilibrium（非桩）",
    "xval/model_matrix_ci_plus_coverage/model_matrix.json — 模型矩阵定义（lattices×collisions×physics×boundaries + TCLB 对照）",
    "xval/model_matrix_ci_plus_coverage/make_figures.py — 图生成",
    "docs/XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE.md — 165 行报告"
  ],
  "entry": "python xval/model_matrix_ci_plus_coverage/run_matrix.py --quick（本审计复跑 28.8s rc=0）",
  "run_artifacts": [
    "xval/model_matrix_ci_plus_coverage/artifacts/result.json — timestamp 2026-11-13T11:03:15Z, quick_mode=true, 6 results, summary.pass_rate=0.1667, all_pass=false",
    "xval/model_matrix_ci_plus_coverage/figures/fig1_model_matrix_comparison.png — 81922B, 2250x750 真 PNG",
    "xval/model_matrix_ci_plus_coverage/figures/fig2_coverage_radar.png — 198684B, 1200x1200 真 PNG",
    "xval/model_matrix_ci_plus_coverage/figures/fig3_test_results.png — 52233B, 1500x900 真 PNG"
  ],
  "tests": "无独立 pytest 套件；run_matrix.py 内 6 用例（TC01/02/03/04/06/09）即交付物本体；本审计复跑 6 项数值与 result.json 逐项相等（73.616/57.505/100/100/False/0.6435）",
  "integration_callers": "none: 独立 xval 脚本，未接入主求解器/CI/平台，仅消费 tensorlbm 模块 → L3 不成立",
  "benchmark": {
    "registered_case": "none: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug=''XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE'' → 0 行 → 唯一裁判缺位，L4 不予",
    "delivered_ref_vs_got": {
      "TC01_poiseuille_2d": {"ref": "解析 u_max=0.05", "got": "u_max_sim=0.0", "rel_err_pct": 73.616, "pass": false},
      "TC02_couette_2d": {"ref": "u_wall=0.05", "got": "u_wall_sim=0.0", "rel_err_pct": 57.505, "pass": false},
      "TC03_lid_cavity": {"ref": "Ghia1982 u_min=-0.2058 / v_max=0.1715", "got": "u_min_sim=0.0 / v_max_sim=0.0", "err_pct": "100/100", "pass": false},
      "TC04_poiseuille_3d": {"ref": "u_center=0.05", "got": "u_center_sim=-1.729e-08", "rel_err_pct": 100.0, "pass": false},
      "TC06_laplace_law": {"ref": "存在压力跳变", "got": "delta_p_sim=-0.0062, has_pressure_jump=false", "pass": false},
      "TC09_acoustic_pulse": {"ref": "能量比 0.1-1.0", "got": "energy_ratio=0.6435, c=0.5774", "pass": true}
    },
    "threshold": "<15%（owner 判据，mission 预声明）",
    "pass_rate": "1/6 = 16.7%（honest FAIL，报告未粉饰）"
  },
  "repro_cmd": "cd /root/tp-agent/work/m17992 && python xval/model_matrix_ci_plus_coverage/run_matrix.py --quick && python xval/model_matrix_ci_plus_coverage/make_figures.py — 本审计复跑通过，数值逐位复现",
  "external_anchor": {"anchor": "CFD-GO/TCLB model-matrix-ci-plus-coverage（GPL-3.0）", "source": "https://github.com/CFD-GO/TCLB", "durable": "仅 URL，无 commit 钉扎；TCLB 侧 lattice/collision/physics 清单为报告转述，无源码行号指针"},
  "report_sections": {"①案例介绍": true, "②方法": true, "③参数表": true, "④对照表": true, "⑤敏感性/差异清单": true, "⑥失败模式/工件": true},
  "git_delivery_state": {"worktree_head": "8cb29b59", "status": ["?? docs/XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE.md", "?? xval/model_matrix_ci_plus_coverage/"], "commits_touching_deliverable": "(none)", "remote_branch": "none", "mission_decision": "auto-approved (TP_AUTO_APPROVE_DEV_ACCEPTANCE, server_quality_audit.gate_valid=false)", "risk": "succeeded 但 0 commit、交付物 untracked、无远端分支 → worktree 回收即丢失"},
  "level_rationale": "L2：真代码 input→compute→output（run_matrix.py 真跑 tensorlbm 求解器、覆盖度计算真实、3 张真图、可复现）。L3 不予（独立脚本无集成点、无消费者）。L4 不予（benchmark 0 行 + 5/6 物理 FAIL 57-100% + 失败源于 harness 缺陷而非求解器）。L5 不予。",
  "findings": [
    "F1 (blocking-for-durability): 交付物 untracked，branch agent/m17992 无 commit、无远端分支；mission 系 auto-approve（gate_valid=false）",
    "F2 (基准裁判缺位): tensor_benchmark_cases 该 anchor 0 行",
    "F3 (harness 缺陷, 报告归因误导): TC01-TC04 sim 值恰为 0.0 — run_matrix.py:116-118 零速度初始化且无压力梯度驱动，流永不发展；零流加多少步都为零，报告⑤.1『步数不足』归因不成立（TC03『边界条件实现问题』部分诚实）",
    "F4 (报告④表): got 列填的是误差百分比而非实测值（实测值在 result.json）",
    "F5 (正面): honest FAIL 未粉饰；repro_cmd 可复现；真图非占位"
  ]
}'::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_note=NULL;
SELECT id, repo, scope, level, verdict_by, gate_state FROM tensor_maturity_verdicts WHERE scope='m17992 XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE' AND verdict_by='board-audit';
