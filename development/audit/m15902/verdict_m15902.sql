-- m15902 交差审计落表：m15657 · GAP-LBM-T2-GPU-COREPATH（TensorLBM）
-- 判级 L1：入口不可构造 / 交付 repro 第 1 步即崩 / 零运行工件 / Triton 后端实为 eager /
--          多 GPU halo 切错轴且无 init_process_group；唯一可执行内核 fused_step_eager
--          对解析通道解 u_max 偏差 2.53%、剖面 rel L2 3.92%（审计实测）→ 仅达 L1。
-- 已执行版本见 tensor_maturity_verdicts id=403（evidence 全文见 JSON）。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm15657 GAP-LBM-T2-GPU-COREPATH', 15902, 1,
'{"impl_files":"src/tensorlbm/gpu_core_path.py (718 lines, commit 848966ba / PR #1841)",
  "entry":"src/tensorlbm/gpu_core_path.py:444 GPUCorePathSolver -- NOT constructible (AttributeError perturbation_amplitude at :497); only fused_step_eager (:224) is executable",
  "run_artifacts":[],
  "tests":["artifacts/audit_m15657/e3_stream_sign.py::channel_poiseuille (u_max 2.53%, profile rel L2 3.92%)"],
  "integration_callers":"NONE (not exported in src/tensorlbm/__init__.py, no callers in src/app/models/tests)",
  "benchmark":"auditor ref 0.10000 vs got 0.097473 = 2.53%; error grows 0.68/1.30/2.53% at 100/200/400 steps; mission side 0 numbers",
  "repro_cmd":"PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend eager -> IndexError at src/tensorlbm/gpu_core_path.py:260",
  "external_anchor":"analytic plane Poiseuille u=F*y*(2h-y)/(2*nu); Ghia 1982 / TG declared but never produced"}'::jsonb,
'board-audit');
