-- board-audit verdict: m18237 XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR (TensorLBM)
-- audit mission: 18354 (交差审计, owner 2026-11-11 任务榜令)
-- level = 4 (L4 基准验证: 完整算例 + 误差分析 + 可复现)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18237 XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR',
  18354,
  4,
  '{
    "impl_files": [
      "src/tensorlbm/les_graph_unroll/selection.py:1-338 (SelectionCase + field/energy/enstrophy 可微目标 + BPTT 选型 + sweep/golden 无 autograd 对照 + 留出评估器)",
      "scripts/xval_r2_selection_bptt_tgv.py:1-384 (分阶段 driver: ref/arm×12/controls×6/judge)",
      "scripts/xval_r2_selection_bptt_tgv_figures.py:1-146 (figures.json 生成, 含 sha256)"
    ],
    "entry": "scripts/xval_r2_selection_bptt_tgv.py (PYTHONPATH=src, CPU float64, 无随机种子)",
    "run_artifacts": [
      "artifacts/xval_r2_selection_bptt_tgv/results.json (87943B: 12 arms 全 BPTT 历史 + 6 controls + 2 sweep 曲线 + verdict 块)",
      "artifacts/xval_r2_selection_bptt_tgv/reference.npz (745941B, 独立 numpy BGK 128²/256² + 4×4 滤波)",
      "artifacts/xval_r2_selection_bptt_tgv/fig1_objective_landscapes.png",
      "artifacts/xval_r2_selection_bptt_tgv/fig2_holdout_ke_vs_theta.png",
      "artifacts/xval_r2_selection_bptt_tgv/fig3_bptt_trajectories.png",
      "artifacts/xval_r2_selection_bptt_tgv/fig4_holdout_metrics.png",
      "artifacts/xval_r2_selection_bptt_tgv/figures.json (444475B, captions+sha256+base64)"
    ],
    "tests": "tests/test_les_graph_unroll_selection.py (8 passed, 9.39s; board-audit 2026-11-13 独立复跑确认)",
    "integration_callers": "src/tensorlbm/les_graph_unroll/ (round-1 可微 D2Q9 核; selection.py 为其上新增选型层, 非孤立脚本)",
    "benchmark": "tensor_benchmark_cases 对本锚点 0 行 (SELECT ... WHERE anchor_slug=XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR 空; 仅 round-1 锚 XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER 有 2 例) → 事实裁判=预注册 P1-P5 (docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR-PREREG.md §4) + owner 15% 标准 (feature_ledger #439); got: P1 0.074% pass, P2 loss_gap≤0.02% pass / θ 16.0% fail, P3 0.561% pass, P4 5.34e-5pp pass, P5 2.3% pass / 16.3% fail / 850-1050% fail (honest)",
    "repro_cmd": "cd /root/tp-agent/work/m18237 && PYTHONPATH=src python3 scripts/xval_r2_selection_bptt_tgv.py (分阶段, ~11min) && PYTHONPATH=src python3 -m pytest tests/test_les_graph_unroll_selection.py -q",
    "external_anchor": "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow (graph_unroll + in-graph τ + BPTT); 出处=本仓 round-1 源码行号表 docs/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER.md §1 (零手抄)",
    "audit_notes": "报告 17147 六段齐全, 数字与 results.json 程序化核对一致 (ref KE 2.12530e-3/2.13104e-3/2.11250e-3, 选中 oss_proxy|enstrophy θ=0.001632 J=1.1276e-4 ke_err 0.561%, regret 5.34e-5pp 均由原始 JSON 重算吻合); 文档笔误 2 处 (P1 0.007% 应为 0.074%; P3 5.6098e-4 应为 5.6098e-3), 百分比判据列无误; 全量 driver 复跑 not_run (11min 预算外); 提交 f95e1d63"
  }'::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET
  mission_id = EXCLUDED.mission_id,
  level = EXCLUDED.level,
  evidence = EXCLUDED.evidence;
