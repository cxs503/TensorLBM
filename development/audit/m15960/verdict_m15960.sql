-- m15960 返修轮1 交差 verdict: GAP-LBM-T2-GPU-COREPATH (TensorLBM)
-- 原单 m15657 -> m15920 (L1 返修) -> 本单 m15960 补齐 L2/L3/L4 证据链
-- 全部数字来源 artifacts/t2_gpu_core_path/bench.json (程序化生成, 零手抄)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'GAP-LBM-T2-GPU-COREPATH (T2 GPU 核心路径统一优化)',
  15960,
  4,
  $json${
    "impl_files": [
      "src/tensorlbm/gpu_core_path.py:1-898 (real implementation: GPUCorePathConfig:150-260 derived physics + validate; collide_guo_into:330-380 BGK+Guo force-corrected equilibrium; stream_into:383-395; apply_channel_walls half-way BB; GPUCorePathSolver:444-640 ping-pong buffers, zero per-step allocation, precision tiers FP64FP64/FP64FP32/FP32FP32/FP32FP16; DistributedGPUCorePathSolver:610-660 CUDA-gated; run_channel_case:690-770; run_taylor_green_case:773-820; CLI main:860-898)",
      "src/tensorlbm/general_sim.py:183-187 (SolverConfig.core_path flag), :574-578 (run() delegation hook), :555-590 (_run_core_path: main solver constructs and runs GPUCorePathSolver)"
    ],
    "entry": "PYTHONPATH=src python -m tensorlbm.gpu_core_path --ny 18 --tau 0.8 --steps 400 --out artifacts/t2_gpu_core_path/tests/entry_cli.json  (executed this mission: rc=0, channel row + 2 TG rows printed, entry_cli.json written)",
    "run_artifacts": [
      "artifacts/t2_gpu_core_path/bench.json (31959 B, generated 2026-11-12T06:19Z, 270.7 s wall)",
      "artifacts/t2_gpu_core_path/channel_profile.png (57203 B)",
      "artifacts/t2_gpu_core_path/taylor_green_decay.png (67679 B)",
      "artifacts/t2_gpu_core_path/channel_convergence.png (61182 B)",
      "artifacts/t2_gpu_core_path/tests/bench_full.log (full bench stdout)",
      "artifacts/t2_gpu_core_path/tests/pytest_full.log (14 tests: 13 passed + 1 skipped compile-parity, TL_TEST_COMPILE unset)",
      "artifacts/t2_gpu_core_path/tests/entry_cli.json (CLI entry run artifact)"
    ],
    "tests": [
      "tests/test_gpu_core_path.py:1-190 (14 tests: construction regression pin for the m15657 AttributeError, collision vs independent BGK+Guo reference atol=1e-10, streaming direction, half-way BB reflection, mass conservation <1e-10, TG decay <3%, channel <3%, precision tier honoured, forced-triton raises without CUDA, distributed requires CUDA, GeneralSimEngine core_path routing + solid-domain rejection)",
      "run: 13 passed, 1 skipped (test_compiled_backend_matches_eager, opt-in via TL_TEST_COMPILE=1) in 516 s on loong64 CPU host"
    ],
    "integration_callers": [
      "src/tensorlbm/__init__.py _LAZY_ATTRS + TYPE_CHECKING + __all__: GPUCorePathConfig, GPUCorePathSolver, DistributedGPUCorePathSolver, channel_analytic_velocity, run_channel_case, run_taylor_green_case (lazy export table; test_public_api_lazy_exports pre-existing 4 failures unchanged by this change, verified by stash A/B)",
      "src/tensorlbm/general_sim.py GeneralSimEngine.run() -> _run_core_path(): main solver really calls GPUCorePathSolver when solver.core_path=True (pinned by tests/test_gpu_core_path.py::test_general_sim_engine_routes_to_core_path, stats['via']=='tensorlbm.gpu_core_path.GPUCorePathSolver')",
      "benchmarks/t2_gpu_core_path/bench.py + benchmarks/validation_matrix/cases/channel_core_path/run.py (benchmark-side callers)"
    ],
    "benchmark": {
      "reference": "analytic Poiseuille u(y)=a/(2 nu) s (H-s), s=y-0.5, H=ny-2 (Batchelor 1967); Taylor-Green KE(t)=u0^2/4 exp(-4 nu k^2 t) (Taylor & Green 1937)",
      "channel_u_max": [
        {"ny": 18, "tau": 0.8, "ref": 0.04980468750000001, "got": 0.04962440419459965, "err_pct": 0.362, "profile_l2_pct": 0.493, "steps": 3000},
        {"ny": 18, "tau": 1.2, "ref": 0.04980468750000001, "got": 0.0499365234375, "err_pct": 0.266, "profile_l2_pct": 0.363, "steps": 1400},
        {"ny": 34, "tau": 0.8, "ref": 0.049951171875, "got": 0.0497479248046875, "err_pct": 0.407, "profile_l2_pct": 0.422, "steps": 6000},
        {"ny": 34, "tau": 1.2, "ref": 0.049951171875, "got": 0.04998291015625, "err_pct": 0.063, "profile_l2_pct": 0.088, "steps": 4600}
      ],
      "taylor_green_max_ke_rel_err_pct": [
        {"nx": 32, "tau": 0.6, "max_rel_err_pct": 1.272},
        {"nx": 32, "tau": 0.9, "max_rel_err_pct": 2.967},
        {"nx": 64, "tau": 0.6, "max_rel_err_pct": 0.345},
        {"nx": 64, "tau": 0.9, "max_rel_err_pct": 0.275}
      ],
      "backend_parity": [
        {"ny": 18, "eager_ms_per_step": 8.42, "compiled_ms_per_step": 3.26, "speedup": 2.58, "rel_field_diff": 2.33e-15},
        {"ny": 34, "eager_ms_per_step": 14.07, "compiled_ms_per_step": 5.89, "speedup": 2.39, "rel_field_diff": 1.67e-15}
      ],
      "precision_tiers": [
        {"precision": "FP64FP64", "u_max_err_pct": 0.361, "profile_l2_pct": 0.492, "mass_drift": 3.7e-08},
        {"precision": "FP32FP32", "u_max_err_pct": 0.357, "profile_l2_pct": 0.488, "mass_drift": 1.9e-04},
        {"precision": "FP32FP16", "u_max_err_pct": 0.357, "profile_l2_pct": 0.488, "mass_drift": 1.9e-04}
      ],
      "multi_gpu": "not_run (torch.cuda.is_available()==False on this loong64 CPU host; DistributedGPUCorePathSolver is CUDA-gated by design, construction raises instead of silently running serially)"
    },
    "repro_cmd": "PYTHONPATH=src python benchmarks/t2_gpu_core_path/bench.py --quick   (full: drop --quick; ~5 min CPU, 2 threads)",
    "external_anchor": "Batchelor 1967 ch.4 (Poiseuille); Taylor & Green 1937 (exact decay); Guo-Zheng-Shi PRE 65 046308 (2002) body force; Ziegler 1993 / Ladd 1994 (half-way BB)",
    "known_limitations": [
      "multi-GPU halo path: CUDA-gated, not_run on this host; local-slab periodic wrap means the distributed path is unverified (honest not_run, not a pass)",
      "Triton kernels: not_run without CUDA; backend='auto' resolves to torch_compile here (measured 2.4-2.6x, field parity 1.7e-15)"
    ]
  }$json$,
  'agent-m15960'
);
