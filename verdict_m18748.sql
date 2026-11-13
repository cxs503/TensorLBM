-- m18748 交差核验 verdict: 对象 m18377 XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST (TensorLBM)
-- L2：真实现 + 10/10 复跑测试 + 逐位可复现工件；影子树/零调用方/零登记基准/无 runner = L3/L4/L5 反证
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18377 XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST', 18748, 2, $json${
 "verdict_basis": "L2：真实现（345 行、0 占位符）+ 真测试（本次复跑 10 passed in 10.56 s）+ 真工件（审计者只用交付模块就把 k=10.654330145418363 逐位复算出来，相对差 0.0）。L3/L4/L5 有反证而非仅缺证：canonical import ModuleNotFoundError、grep 调用方 0、tensor_benchmark_cases 该 anchor 0 行、交付无 runner、OSS Fortran 未执行。",
 "delivered_level": 2,
 "gap_to_claim": "L3 不成立：交付写在旧嵌套影子树 TensorLBM/src/tensorlbm/（canonical src/tensorlbm 无 porous 子包），`python3 -c 'import tensorlbm.sparse_sublist_neighbor_list'` → ModuleNotFoundError；grep -rl 'sparse_sublist' --include=*.py 全树排除模块自身/测试 = 0 命中（无主求解器/平台调用方）。L4 不成立：tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST' = 0 行（全表 110 行）；交付无 runner（xval/<anchor>/ 只有 predeclared.json），工件生产链不可由第三方按交付复现；唯一算例是单分辨率 H=16 通道，无网格收敛研究。L5 不成立：listLBM 为 Fortran，未编译未执行，只做源码级语义移植（锚点 sha 由审计者补测）；外部锚只有解析 slab 精确解，不是实验/文献数据。",
 "impl_files": [
  "/root/tp-agent/work/m18377/TensorLBM/src/tensorlbm/sparse_sublist_neighbor_list.py",
  "/root/tp-agent/work/m18377/TensorLBM/src/tensorlbm/porous/fluid_node_table.py"
 ],
 "impl_files_note": "/root/tp-agent/work/m18377/TensorLBM/src/tensorlbm/sparse_sublist_neighbor_list.py（345 行；SparseSublistNeighborList / build_sparse_sublist_neighbor_list / listlbm_stream1 / listlbm_stream2 / listlbm_stream / listlbm_step / permeability）；占位符扫描 NotImplementedError|raise TODO|pass # placeholder = 0；依赖 {AUD}/TensorLBM/src/tensorlbm/porous/fluid_node_table.py（同一影子树内的 push 控制臂）。",
 "entry": "/root/tp-agent/work/m18377/TensorLBM/tests/test_sparse_sublist_neighbor_list.py",
 "entry_note": "交付**没有** CLI/runner（F3：/root/tp-agent/work/m18377/xval/XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST/ 只有 predeclared.json）。唯一随交付的可执行入口是测试节点：`cd /root/tp-agent/work/m18377/TensorLBM && python3 -m pytest tests/test_sparse_sublist_neighbor_list.py -q` （10 passed / 10.56 s），物理算例入口 = tests/test_sparse_sublist_neighbor_list.py::test_plane_channel_permeability_within_15pct（文件:170）。这是**弱入口**，不改变 L2 判定（功能实质由模块本身定义），但它是 L4 不成立的一部分。",
 "run_artifacts": [
  "/root/tp-agent/work/m18377/TensorLBM/artifacts/xval_sparse_sublist_neighbor_list/results.json",
  "/root/tp-agent/work/m18377/TensorLBM/artifacts/xval_sparse_sublist_neighbor_list/figures/fig1_channel_profile.png",
  "/root/tp-agent/work/m18377/xval/XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST/predeclared.json",
  "/root/tp-agent/work/m18748/artifacts/m18748_audit/audit_m18748.json",
  "/root/tp-agent/work/m18748/artifacts/m18748_audit/figures/fig_m18748_evidence.png",
  "/root/tp-agent/work/m18748/docs/audit_m18748/AUDIT_m18748.md"
 ],
 "run_artifacts_note": "results.json = 交付工件（k_darcy=10.654330145418363，k_ref=10.6666666，err 0.1157%，operator_equivalence max_abs_diff 声称 <1e-14，table_equivalence bit-identical）；fig1_channel_profile.png = 交付图（1540×560 RGBA）；predeclared.json = 写于运行前的预声明；audit_m18748.json = 本审计机读读数（R1–R5 + 静态 + DB + 外部锚）；fig_m18748_evidence.png = 收敛梯 + 剖面 vs 抛物线双面板；AUDIT_m18748.md = 六段审计报告。",
 "tests": [
  "tests/test_sparse_sublist_neighbor_list.py::test_table_equivalence_with_fluid_node_table",
  "tests/test_sparse_sublist_neighbor_list.py::test_halfqs_are_one_from_each_opposite_pair",
  "tests/test_sparse_sublist_neighbor_list.py::test_listlbm_stream_equals_fluid_node_table_push[1|5|13]",
  "tests/test_sparse_sublist_neighbor_list.py::test_listlbm_step_equals_compressed_step[1|5|13]",
  "tests/test_sparse_sublist_neighbor_list.py::test_plane_channel_permeability_within_15pct",
  "tests/test_sparse_sublist_neighbor_list.py::test_mass_conservation"
 ],
 "tests_rerun": "cd /root/tp-agent/work/m18377/TensorLBM && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_sparse_sublist_neighbor_list.py -q -p no:cacheprovider -> 10 passed, 1 warning in 10.56s（本次复跑）",
 "integration_callers": "none：grep -rl 'sparse_sublist' --include='*.py' /root/tp-agent/work/m18377（排除模块自身与测试）= 0 命中；cd /root/tp-agent/work/m18377 && python3 -c \"import sys;sys.path.insert(0,'src');import tensorlbm.sparse_sublist_neighbor_list\" -> ModuleNotFoundError（canonical src/tensorlbm 无 porous 子包）。⇒ 无任何主求解器/平台调用路径，L3 反证成立",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "judge_sql": "SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST'",
  "judge_result": "0 rows (table total = 110) ⇒ 唯一裁判无数据，本单不能凭基准判据晋 L4",
  "declared_criterion": "predeclared.json：table bit-identical / operator <1e-14 / physics <15%（owner 带）",
  "channel_k_super_ref": 10.666666666666666,
  "channel_k_super_got_fresh": 10.654330145418363,
  "err_pct_fresh_2500_steps": 0.11565,
  "err_pct_converged_20000_steps": 0.10937,
  "convergence_drift_2500_to_20000_pct": 0.00629,
  "operator_equivalence_maxabs_fresh": 0.0,
  "table_equivalence_vs_independent_numpy": "nodes/neighbour/index all identical (353 fluid nodes)",
  "profile": {
   "parabola_resid_max_rel": 1.6986e-06,
   "symmetry_rel": 2.079e-15,
   "u_max_over_u_mean": 1.492722
  },
  "mass_drift_rel_200_steps": 4.584e-16,
  "auditor_note": "解析 slab 锚在 0.1157%（2500 步）/0.1094%（20000 步）双口径均 <15% 且 <3%，但基准裁判 0 行 + 无 runner + 单分辨率 ⇒ L4 资格不成立"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18748 && python3 scripts/audit/m18748_sparse_sublist_board_audit.py   # ≈90 s 单线程；只读导入 /root/tp-agent/work/m18377 的交付模块，逐位复算 k=10.654330145418363（rel 0.0）",
 "external_anchor": "WITHHELD（仅源码级）：listLBM Fortran 未编译未执行。审计者核 /tmp/listlbm HEAD=9feb6e3592ba5ebbbba1f8850db41f85d61f1489（2020-07-07）：stream/stream_mod.f90:45 opposite_streamfluid、:64 reversepdfdirs_streamfluid、lattice/lattice_mod.f90:87 halfqs=(1,2,5,6)、lists/base/mainlist_mod.f90:110-111 allocate(pdf(0:nq_1,ist:ind))/allocate(neilist(1:nq_1,ist:ind))（确认 pdf/neilist 只在流体区间）——机制引用成立；但交付 docs 引用的 'mainlist_mod.f90:14-20' 实为 GPL 头（行号错，F12）。可核的唯一精确锚 = 解析 slab k_super=(H/ny)H²/12",
 "audit_independence": "被审树只读（未改 /root/tp-agent/work/m18377 任何文件）；复跑写本工作树 artifacts/m18748_audit/。数字来源=(a) 交付工件静态读取 (b) 交付代码原样执行 R1–R5 (c) 独立 numpy 重写邻居表（不复用仓内 builder）(d) psql 查 tensor_benchmark_cases / tensor_dev_missions / tensor_reports (e) /tmp/listlbm 源码行号。局限：OSS 二进制未执行；只覆盖 D2Q9 单相单子表；无分辨率收敛研究；本审计者与被审单同体制，只对机器可核事实负责。",
 "findings": {
  "F1_high": "交付 5 个文件全部 untracked（模块/测试/工件/文档/预声明无 commit）⇒ 平台推分支会整包丢失",
  "F2_high": "零集成调用方（grep 排除自身/测试 = 0）",
  "F2b_high": "canonical 包不可导入（影子树 TensorLBM/src，ModuleNotFoundError）",
  "F3_high": "无交付 runner：xval/<anchor>/ 只有 predeclared.json，results.json 生产脚本未随交付",
  "F5_high": "tensor_benchmark_cases 该 anchor 0 行（唯一裁判无数据）",
  "F6_medium": "tensor_reports 无 mission 18377 行（报告未发布，平台侧不可复核）",
  "F12_low": "外部锚行号引用错（mainlist_mod.f90:14-20 实为 GPL 头；真实 31/33/110-111）",
  "F13_low": "results.json.oss.commit 无 sha（审计者补测 9feb6e35…）",
  "F8_positive": "2500 步已收敛（→20000 步漂移 0.0063%），报告 0.1157% 是真误差非瞬态",
  "negatives_cleared": "无硬编码结果（模块/测试内无 10.65/21.30/0.1156 字面量）；质量漂移 4.58e-16；剖面抛物线残差 1.7e-6、上下对称 2e-15；稠密控制臂与 swap 臂逐位相同；predeclared 阈值与实测一致，未见事后调阈"
 },
 "verdict": "CONFIRMED at L2（L3/L4/L5 有反证）；返修方向：搬进 canonical src、交付 run_xval.py、登记 benchmark case、接主流程调用方、提交 commit",
 "audited_deliverable": {
  "mission": 18377,
  "tree": "/root/tp-agent/work/m18377",
  "branch": "agent/m18377",
  "status": "succeeded (accepted=true, finished 2026-11-13 15:31:16Z)",
  "git_identity": "HEAD f8e10fa6 (Merge #2139) — 交付文件全未 commit"
 }
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;
