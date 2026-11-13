-- m18155 交差核验 verdict: 对象 m17995 XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED (TensorLBM)
-- 判级 L2：独立功能实现（真代码 + 10 tests 绿 + 真实 2-rank gloo TGV 工件）；L3/L4 不予（未接入 shipped transport；基准表 0 行且物理判据 FAIL）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17995 XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED', 18155, 2, $json${
  "verdict_basis": "L2 授级：独立功能实现——真代码（1139 LOC：PackInfo 集合推导/pack-unpack/torch.distributed isend-irecv/D3Q19 BGK TGV），真实输入→计算→输出（2 rank gloo x-slab，5 臂含 serial，results.json/table.csv 工件），10/10 单测在本审计工作树重跑通过，外部锚 lbmpy._extend_dir 逐方向机器比对且测试实际执行（0 skipped）。L3 不予：shipped src/tensorlbm/multi_gpu.py:928 D3Q19GlooTransport 未被修改，产品/求解器无任何 import，3.8x 字节收益对库用户不可用。L4 不予：tensor_benchmark_cases 对 anchor 0 行（无注册容差），且预声明物理判据 tgv_rel_l2 24.887% > 15% FAIL，4 配置矩阵仅交付 1 个。L5 不适用。",
  "delivered_level": 2,
  "claimed_level_implicit": "任务叙述称『移植进 TensorLBM x-slab D3Q19 传输』+ 逐比特一致 + 15% 判据，隐含 L3/L4；实际为隔离 xval 包的 A/B 复刻",
  "gap_to_claim": "集成面（L3）与基准面（L4）均缺；物理判据 FAIL 属诚实申报，但归因错误（见 reference_defect）",
  "impl_files": [
    "src/tensorlbm/xval_mpi_ghost_fused/packinfo.py:1-121 (face_packinfos/face_index，方向选择集合)",
    "src/tensorlbm/xval_mpi_ghost_fused/kernels.py:1-59 (pack_face/unpack_face 单趟 index_select/index_write)",
    "src/tensorlbm/xval_mpi_ghost_fused/transport.py:28-158 (SlabExchange: isend/irecv 四条臂单变量对照 + replicate_shipped_exchange)",
    "src/tensorlbm/xval_mpi_ghost_fused/case.py:1-145 (D3Q19 TGV 初值/平衡/BGK/误差度量；缺陷见 reference_defect)",
    "src/tensorlbm/xval_mpi_ghost_fused/campaign.py:1-271 (serial 参考 + 分布式臂 + 工件写出)"
  ],
  "entry": "scripts/xval_mpi_ghost_fused.py:1-178 (MATRIX 4 配置；实际交付仅 n64_r2 完成)",
  "entry_note": "驱动脚本存在且可读；但交付的 results.json 只含 1/4 配置，用交付 results.json 重跑 make_figures 直接 KeyError 'n64_r4'（图阶段不可复现）",
  "run_artifacts": [
    "xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/results.json (1 配置 n64_r2；5 臂：bytes/step、exchange_ms、tgv_rel_l2、identity_max_abs_vs_serial)",
    "xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/table.csv (6 行 = serial + 4 臂)",
    "xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/predeclared.json (预声明阈值/矩阵/停止规则，含错误解析式)",
    "xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/figs/fig2_error_vs_analytic.png",
    "xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/figs/fig3_transport_cost.png",
    "xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/figs/fig4_field_identity.png",
    "NOT TRANSPLANTED (176 MB, 留在 /root/tp-agent/work/m17995/xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/): fields_n64_r2.pt (99.6MB), decomposed_n64_r2.pt (79.7MB)"
  ],
  "tests": [
    "tests/test_xval_mpi_ghost_fused.py::test_crossing_populations_match_oss_extend_dir (import lbmpy.advanced_streaming.communication._extend_dir @ /usr/lib/python3.11/site-packages/lbmpy/advanced_streaming/communication.py:233, lbmpy 2.0, 26 comm dirs)",
    "tests/test_xval_mpi_ghost_fused.py::test_crossing_counts_are_stencil_derived",
    "tests/test_xval_mpi_ghost_fused.py::test_extend_dir_is_the_oss_expansion",
    "tests/test_xval_mpi_ghost_fused.py::test_pack_unpack_is_a_lossless_face_permutation[selective=True/False]",
    "tests/test_xval_mpi_ghost_fused.py::test_equilibrium_matches_repository_implementation",
    "tests/test_xval_mpi_ghost_fused.py::test_tgv_initial_state_is_solenoidal_up_to_the_differencing_error",
    "tests/test_xval_mpi_ghost_fused.py::test_slab_exchange_requires_a_process_group",
    "tests/test_xval_mpi_ghost_fused.py::test_dtype_weight_selection_is_exact_for_float64",
    "tests/test_xval_mpi_ghost_fused.py::test_collide_bgk_preserves_mass_and_momentum_of_equilibrium",
    "tests_run: PYTHONPATH=src python -m pytest tests/test_xval_mpi_ghost_fused.py -q -> 10 passed, 1 warning, 7.0 s, 0 skipped (re-run in audit worktree m18155 after transplanting the untracked delivery)"
  ],
  "integration_callers": [
    "NONE — grep -rn 'xval_mpi_ghost_fused' src/ app/ examples/ benchmarks/ (excluding the package itself) = 0 hits",
    "src/tensorlbm/multi_gpu.py:928 D3Q19GlooTransport / :1147 exchange_ghosts 未被修改（git status 仅 ?? src/tensorlbm/xval_mpi_ghost_fused/）"
  ],
  "benchmark": {
    "machine_referee_row": "tensor_benchmark_cases: 0 rows for anchor_slug='XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED' (91 rows in table, checked 2026-11-13, psql -U postgres) -> 无注册 ref/tolerance，基准维度 WITHHELD",
    "declared_gates_from_predeclared_json": {
      "tgv_rel_l2": "gate <15% (owner band; 5%/3% stretch) vs got 24.887% (n64_r2, all 5 arms identical) -> FAIL",
      "identity_max_abs_vs_serial": "gate exactly 0 vs got 0.0 (all 4 arms, float32) -> PASS",
      "byte_ratio": "gate exact 5/19=0.263157894737 vs got 0.263157894737 (1245184 -> 327680 B/step) -> PASS",
      "exchange_ms": "reported 17.229 -> 12.351 ms/step (-30.4%), not a gate"
    },
    "matrix_completeness": "predeclared_run_matrix 4 配置 (n64_r2/n64_r4/n32_r2/n32_r2_f64) vs results.json+table.csv 仅 n64_r2 -> 1/4；图注却写『全 4 配置』",
    "verdict": "FAIL on the declared physics gate; benchmark not registered; L4 denied"
  },
  "reference_defect": {
    "defect": "case.py::TGV.decay = exp(-2*nu*k^2*t)（predeclared.json::baseline_control 预声明同式）用于 3D ABC 场 ux=U0 sin(kx)cos(ky)cos(kz), uy=-U0 cos(kx)sin(ky)cos(kz), uz=0；该场每分量波矢 |k|^2=3k^2，线性 NS 衰减应为 exp(-3 nu k^2 t)。2nu k^2 是 2D TGV 的衰减率。",
    "reproduction": "audits/m18155/verify_tgv_reference.py + audits/m18155/verify_tgv_reference.json（monolithic only，无 MPI）",
    "numbers": "n=64,t=400: rel-L2 24.887% (2nk^2, 交付值复现) -> 17.632% (3nk^2)；n=32,t=200: 34.706% -> 8.713%；RMS 拟合 nu_eff@3k^2 = 0.051697 (+3.39% vs nu=0.05) / 0.051326 (+2.65%)，独立印证 3k^2 指数",
    "impact": "报告的 FAIL 结论仍成立（17.63% > 15%，U0*k*t=1.57 使谐波确实 O(1)），但 24.887% 中 7.25pp 是可修的解析式缺陷；报告 ⑤/⑥ 的归因『解析基准只含线性衰减模态』点错根因；文档参数表『终点衰减比 e^{-2nk^2t}=0.8087』自相矛盾（e^{-2nk^2*400}=0.6801，0.8087 是错基准下的数值/解析幅值比）"
  },
  "repro_cmd": "cd <repo> && PYTHONPATH=src python -m pytest tests/test_xval_mpi_ghost_fused.py -q（10 passed 已验证）；完整战役 PYTHONPATH=src python scripts/xval_mpi_ghost_fused.py --out <dir>（需重跑 4 配置；对交付 results.json 仅跑图阶段会 KeyError 'n64_r4'）；审计复核 PYTHONPATH=src python audits/m18155/verify_tgv_reference.py",
  "external_anchor": "lssfau/lbmpy 2.0 —— lbmpy.advanced_streaming.communication._extend_dir (line 233) 与 lbmpy_walberla packinfo.generate_lb_pack_info / PackingKernels.tmpl.h（predeclared.json::oss_reference 记源：/tmp/lbmpy_src 检出）；限制：本机无 mpi4py/MPI 实现，未跑 OSS 二进制，仅语义移植 + 源码级机器比对（mission 自述，如实）",
  "report": "tensor_reports id=16918 (published, 3898 chars, 六段①-⑥齐全)；缺陷：figures[] 仅 {name,caption} 无 png_base64；④ 承诺图1 TGV 衰减曲线但 figs/fig1_tgv_decay.png 不存在；serial.series={} 且 serial.step_ms/mlups=null，图1 数据源为空",
  "landing_defect": "m17995 返回 head_commit=null，git_status 全部交付路径为 '??'（未提交）；本审计把交付文件原样移植进 agent/m18155（代码/测试/脚本/docs/xval 工件，排除 176MB .pt）使证据可从落地 commit 解析；内容未改",
  "audit_doc": "docs/AUDIT_m18155_XVAL_LBM_MPI_GHOST_FUSED.md",
  "checks_run": "pytest 10 passed；psql 查询 tensor_benchmark_cases=0 行；grep 集成调用=0；make_figures 对交付 results.json = KeyError 'n64_r4'；monolithic TGV 复算 n=32/64 双基准（2nk^2 vs 3nk^2）+ nu 拟合"
}$json$, 'board-audit');