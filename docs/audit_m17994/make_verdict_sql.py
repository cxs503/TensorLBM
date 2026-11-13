"""Emit verdict_m17994.sql for the m18154 board audit of m17994."""
import json
import pathlib

EV = "docs/audit_m17994/evidence"

ev = {
    "verdict_basis": (
        "L2 授级：独立功能实现（真代码输入→计算→输出）且远超精度标准——交付 617 行 D2Q9 y-slab 求解器"
        "（零拷贝 unpacked halo + Testall 进度轮询 + 4 种传输替身），审计独立复跑 9 臂 × 6 个确定性 KPI "
        "相对差全为 0（probe/场 L2 误差 0.013780% vs 解析剪切波、identity(NP>1 vs NP=1)=0.0、"
        "wire 32768/24576=8/6 闭式、staging 0 vs 12288 B/步、质量漂移 1.9e-14），10/10 pytest 通过，"
        "repro_cmd 全量重跑 ~65s 可复现。"
        "不授 L3：无任何生产求解器/平台侧调用方（grep 仅命中 parallel/__init__.py 自身），"
        "模块 docstring 自述 verification instrument，且与已被 blockforest.py 调用的 "
        "domain_decomposition.HaloExchange 功能重叠未归一。"
        "不授 L4：L3 连锁失败 + 平台无该锚基准裁判行（tensor_benchmark_cases/tensor_xval_cases/repro_pool "
        "对该 anchor_slug 均 0 行）+ 对标另一侧 OSS 二进制从未执行（本镜像无 mpi4py/mpirun，" \
        "ref 是台账做法+自写替身，属语义移植而非双实现数值对照）。"
    ),
    "claimed_level_implicit": "L4（XVAL 交叉对标 + 预声明阈值 + 完整误差分析 + 可复现工件）",
    "delivered_level": 2,
    "gap_to_claim": (
        "差 2 级。非造假：所有确定性 KPI 审计均可逐位复现；差距来自①未接入任何求解器（L3 缺）；"
        "②对标 ref 侧未执行、平台无判据行（L4 缺）。另有三条证据链/纪律缺陷见 findings F1/F3/F4。"
    ),
    "impl_files": [
        EV + "/SNAPSHOT_slab_unpacked_m17994.py",
        "/root/tp-agent/work/m17994/src/tensorlbm/parallel/slab_unpacked.py",
        "/root/tp-agent/work/m17994/src/tensorlbm/parallel/__init__.py",
    ],
    "impl_lines": {
        "solver_step": "src/tensorlbm/parallel/slab_unpacked.py:553-597 (collide → isend 零拷贝 → 分块 stream+Testall(583-591) → waitall → ghost 粘贴)",
        "unpacked_exchange": "src/tensorlbm/parallel/slab_unpacked.py:386-447 (post: f[1:, row, :] 直发, copies_bytes 恒 0)",
        "packed_control": "src/tensorlbm/parallel/slab_unpacked.py:449-484 (selective+staging 对照臂)",
        "transports": "src/tensorlbm/parallel/slab_unpacked.py:293 SerialComm / :161 QueueChannel / :228 ThreadPeers / :317 SocketPeers",
        "analytic_anchor": "src/tensorlbm/parallel/slab_unpacked.py:69-80 shear_wave_analytic",
        "public_exports": "src/tensorlbm/parallel/__init__.py:3-14",
        "sha256": "slab_unpacked.py f76649e50b88c6ef870265dfd3d0de8dd95a5b5e7dbbecaf044fd5a859cef477 @ m17994 commit 9ae15ea20918ff220210ae2958a2d5d4626b1018",
    },
    "entry": EV + "/run_xval.py",
    "entry_note": "原件 /root/tp-agent/work/m17994/xval/XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA/run_xval.py；无参数即全量 300 步 @128²（9 臂+收敛+轮询+overlap），--quick 为 smoke",
    "run_artifacts": [
        EV + "/RESULTS.md",
        EV + "/kpi.csv",
        EV + "/tables.md",
        EV + "/fig1_field_and_profile.png",
        EV + "/fig2_comm_accounting.png",
        EV + "/fig3_overlap_and_polling.png",
        EV + "/audit_rerun_results.json",
        EV + "/kpi_audit_rerun.csv",
        EV + "/audit_timing_spread.json",
        EV + "/audit_step_amendment_check.json",
        docs_fig := "docs/audit_m17994/figures/figA_kpi_parity.png",
        "docs/audit_m17994/figures/figB_field_vs_analytic.png",
        "docs/audit_m17994/figures/figC_timing_spread.png",
        "/root/tp-agent/work/m17994/artifacts/xval_mpi_slab_unpacked_halo_overlap/results.json",
    ],
    "tests": [
        "tests/test_parallel_slab_unpacked.py::test_decomposition_is_exact[unpacked-2]",
        "tests/test_parallel_slab_unpacked.py::test_decomposition_is_exact[unpacked-4]",
        "tests/test_parallel_slab_unpacked.py::test_decomposition_is_exact[unpacked-selective-2]",
        "tests/test_parallel_slab_unpacked.py::test_decomposition_is_exact[unpacked-selective-4]",
        "tests/test_parallel_slab_unpacked.py::test_decomposition_is_exact[packed-selective-2]",
        "tests/test_parallel_slab_unpacked.py::test_decomposition_is_exact[packed-selective-4]",
        "tests/test_parallel_slab_unpacked.py::test_overlap_matches_blocking[2]",
        "tests/test_parallel_slab_unpacked.py::test_overlap_matches_blocking[4]",
        "tests/test_parallel_slab_unpacked.py::test_unpacked_block_pays_no_staging_and_sends_8_of_6",
        "tests/test_parallel_slab_unpacked.py::test_shear_wave_matches_analytic",
        EV + "/test_parallel_slab_unpacked.py",
    ],
    "tests_result": "python -m pytest tests/test_parallel_slab_unpacked.py -q → 10 passed in 8.65s（两次独立运行一致）；断言为不变量型：np.testing.assert_array_equal 逐位相等 + 闭式 8/6 字节比 + 剪切波 <1% + |uy|max<1e-12",
    "integration_callers": [
        "NONE — grep -rn 'slab_unpacked|tensorlbm.parallel' --include=*.py 仅命中 src/tensorlbm/parallel/__init__.py 自身；"
        "既有 domain_decomposition.HaloExchange 的调用方是 src/tensorlbm/blockforest.py:7,101,779，新模块未接入任何主求解器"
    ],
    "benchmark": {
        "oracle_row": "tensor_benchmark_cases / tensor_xval_cases / repro_pool 对 anchor_slug='XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA' 均 0 行（锚无登记判据行）",
        "owner_band_pct": 15.0,
        "ref": "解析横向剪切波 ux=u0*cos(k*y)*exp(-nu*k^2*t), k=2*pi/128, tau=0.8, u0=0.04（散度为零，不激发声学模态）",
        "got_vs_ref_pct": {
            "probe_rel_err_pct": 0.0137799,
            "field_l2_rel_err_pct": 0.0137799,
            "n64x75_steps_pct": 0.055118,
            "t600_steps_pct": 0.014709,
            "pass_lt_15pct": True,
            "pass_lt_3pct": True,
        },
        "decomposition_identity_maxrel": {"np2": 0.0, "np4": 0.0, "arms": 9, "repro_runs": 2},
        "comm_accounting": {
            "wire_bytes_per_step_unpacked": 32768,
            "wire_bytes_per_step_selective": 24576,
            "ratio": 1.3333333333333333,
            "closed_form": "8/6",
            "staging_bytes_per_step_unpacked": 0.0,
            "staging_bytes_per_step_selective": 12288.0,
        },
        "mass_drift_rel": 1.88738e-14,
    },
    "repro_cmd": (
        "cd /root/tp-agent/work/m17994 && python xval/XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA/"
        "run_xval.py --out /tmp/audit17994/results.json  # 审计实跑 ~65s 落 COMPLETE"
    ),
    "external_anchor": (
        "解析解（剪切波精确解）+ OSS 台账做法（lbm-principles-practice/code, commit 6e2c592, oss_solutions "
        "n315/n330）；但 OSS 二进制未执行（无 mpi4py/mpirun），故外部锚仅限解析解，不构成双实现对照"
    ),
    "findings": [
        {"id": "F1", "severity": "medium", "claim": "predeclared 600 步 / results.json 300 步 / mission checks 写 600 步",
         "evidence": "predeclared.json stopping_rule '600 at N=128' vs results.json config.steps=300；审计补跑 600 步 NP=1 探针 0.014709% vs 300 步 0.013780%（evidence/audit_step_amendment_check.json）",
         "impact": "物理结论不变（均 ≪3%），属预声明纪律与交差文本口径缺陷", "action": "补 amendment"},
        {"id": "F2", "severity": "medium", "claim": "XVAL 交叉对标（OSS 做法语义移植）",
         "evidence": "镜像无 mpi4py（ModuleNotFoundError）、无 mpirun/mpiexec；对照臂为本仓自写 QueueChannel/SocketPeers 替身",
         "impact": "H1 证明的是本仓自洽而非与 OSS 实现一致；报告①已标 comparison limit，不判造假", "action": "证据分级记 ref=WITHHELD(未执行)"},
        {"id": "F3", "severity": "medium", "claim": "MLUPS / overlap speedup / 轮询墙钟表",
         "evidence": "同机复跑 MLUPS 最大比值 2.763×；overlap speedup delivered{1.161,0.475,0.456} vs 复跑{1.102,0.205,0.420}；轮询扫描 64 行 vs 8 行排序翻转（17.53/13.45/16.25 → 11.37/6.11/6.32 s）",
         "impact": "墙钟类结论 n=1 无噪声带，不可作跨臂性能结论；轮询次数（29.8/3.4/1.0 次/步）复现", "action": "报告加噪声带/重复"},
        {"id": "F4", "severity": "low-medium", "claim": "global_scalars() 的 max|u| 走求和归约",
         "evidence": "slab_unpacked.py:541-556 local_scalars 第 7 项 = max norm，经 :554 allreduce_sum → NP=4 报 0.258 而真值 0.064",
         "impact": "公开 API 语义错误值，后续调用方会踩；报告⑥已披露但代码未防", "action": "拆 allreduce_max 或改名"},
        {"id": "F5", "severity": "low", "claim": "新增 parallel 包与 domain_decomposition 重复",
         "evidence": "PackedHaloExchange:449 复刻 domain_decomposition 语义；后者被 blockforest.py:7,101,779 调用",
         "impact": "两份 halo 语义将分叉；L3 集成路径就是它", "action": "归一或把主求解器切到 parallel"},
        {"id": "F6", "severity": "low", "claim": "kpi.csv identity_maxrel 在 np1 行为空",
         "evidence": "evidence/kpi.csv 第 2 行该列空", "impact": "下游 CSV 消费者会当缺失", "action": "填 0.0 或 'ref'"},
    ],
    "checks_run": {
        "pytest": "10 passed in 8.65s（m17994 工作树）",
        "rerun_xval": "9 臂全量 300 步 @128²，确定性 KPI 相对差 0（figA）",
        "step_amendment": "NP=1 300/600 步误差 0.013780%/0.014709%",
        "oracle_query": "psql tensor_benchmark_cases WHERE anchor_slug=... → 0 rows",
        "integration_grep": "grep slab_unpacked/tensorlbm.parallel in src/ → 仅 __init__ 自身",
        "env": "python 3.11.6 numpy 2.4.6 loongarch64 128核；无 mpi4py/mpirun",
    },
    "checks_not_run": [
        "真实 MPI（无 mpi4py/mpirun）→ not_run",
        "OSS 二进制对照 → not_run",
        "全 9 臂 600 步（仅补 NP=1）→ not_run",
        "集群多进程 → 未递交（本单为核验，不需重算）",
    ],
    "audit_report": "docs/audit_m17994/AUDIT_REPORT.md",
    "disposition": "approve with notes → L2 confirmed（消账；F1/F3/F4 随下一同类节点修，F5 为 L3 路径）",
}

payload = json.dumps(ev, ensure_ascii=False, indent=1)
assert "$json$" not in payload
sql = (
    "-- m18154 交差核验 verdict: 对象 m17994 XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA (TensorLBM)\n"
    "-- 判级 L2：真代码+可复现+精度 0.0138%≪15%；无集成调用方、无登记基准行、OSS ref 未执行 => L3/L4 不成立\n"
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
    "('TensorLBM', 'm17994 XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA', 18154, 2, $json$"
    + payload + "$json$, 'board-audit');\n"
)
out = pathlib.Path("verdict_m17994.sql")
out.write_text(sql, encoding="utf-8")
print(out, len(sql), "bytes")