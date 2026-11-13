#!/usr/bin/env python3
"""Land the board-audit verdict for m17677 (XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE).

Writes verdict_m17677.sql (reproducible artefact) and applies it to
``tensor_maturity_verdicts``; then dry-runs ``/root/tp-agent/maturity_gate.judge``
so the auditor knows what the machine gate will say before the daemon rules.

Level 2: real implementation + real tests + byte-exact reproducible run artifacts
(L2 = impl_files + entry + run_artifacts all machine-verifiable).  L3 is withheld
because the module sits in the repo's legacy nested shadow tree (canonical
``src/tensorlbm/`` cannot import it) and has zero main-solver callers; L4 is
withheld because ``tensor_benchmark_cases`` has 0 rows for this anchor and no
repro command was shipped with the deliverable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

WS = "/root/tp-agent/work/m17825"
AUD = "/root/tp-agent/work/m17677"
SCOPE = "m17677 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE"
LEVEL = 2
MISSION = 17825

EVIDENCE = {
    "verdict_basis": (
        "L2：真实现（压缩流体节点邻居表 + push 流式 + 半程反弹 + Guo 体力，0 占位符）、"
        "真测试（12 passed 本次复跑 31.9 s）、真工件（6 算例 k 值把交付 runner 步数常量钉到工件值后"
        "逐位复现，最大相对差 0.0；k=nu<u>/g 可反算，相对差 0.0；邻居表由 numpy 独立重写逐项一致），"
        "解析锚 C1 k_super 交付 10.553348 vs 10.666667 = 1.062%，收敛探针 6000 步 10.655000 = 0.109%。"
        "无 m17636 式人工改写/清零。不授 L3/L4/L5 的硬理由见 gap_to_claim。"
    ),
    "claimed_level_implicit": "XVAL 对标（mission 自述“部分完成，如实申报”）：表示等价 0.0 + 物理锚 15% 带内",
    "delivered_level": 2,
    "gap_to_claim": (
        "L3 不成立：模块与测试写在仓内旧嵌套影子树 TensorLBM/src|tests（174 文件子副本），"
        "canonical src/tensorlbm/porous 不存在，仓根 `import tensorlbm.porous` 抛 ModuleNotFoundError；"
        "grep 'tensorlbm\\.porous([^A-Za-z0-9_]|$)' 全树 3 命中全为模块自身/测试/xval，主求解器调用方 0。"
        "L4 不成立：tensor_benchmark_cases 该 anchor_slug 0 行（无登记案例/容差，基准裁判无数据）；"
        "交付未附 repro_cmd；预声明梯（H=32/N=128,256/12000 步）未跑完，工件 6 算例全部 converged=false。"
        "L5 不成立：外部 OSS 锚（openLBMPM CUDA）未执行，交付自述为语义移植（诚实声明，但不可核）。"
    ),
    "impl_files": [
        f"{AUD}/TensorLBM/src/tensorlbm/porous/fluid_node_table.py",
        f"{AUD}/TensorLBM/src/tensorlbm/porous/__init__.py",
    ],
    "impl_files_note": (
        f"{AUD}/TensorLBM/src/tensorlbm/porous/fluid_node_table.py:149-360 "
        "(build_fluid_node_table / compressed_step push 流式 / dense_stream+dense_step 稠密对照 / Guo forcing)；"
        f":78-149 (FluidNodeTable 压实表 + state_bytes 足迹)；:235-262 (collide_with_guo_forcing)"
    ),
    "entry": f"{AUD}/TensorLBM/xval/XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE/run_xval_fast.py",
    "entry_note": (
        "入口与工件不一致（本次审计实测）：run_xval.py main() 跑 C1_channel_H32/N=128,256/12000 步，"
        "与工件 6 算例（H=16/N=64,96/1200 步）不同；run_xval_fast.py 现值为 STEPS=2500，"
        "即提交原样不可复现工件，须把常量钉回工件 steps_cap=1200（审计脚本已程序化做到，结果逐位相同）。"
    ),
    "run_artifacts": [
        f"{AUD}/TensorLBM/artifacts/xval_compressed_fluid_node_neighbour_table/results.json",
        f"{WS}/artifacts/m17677_audit/audit_m17677.json",
        f"{WS}/artifacts/m17677_audit/figures/fig_m17677_evidence.png",
        f"{WS}/docs/audit_m17677/AUDIT_m17677.md",
    ],
    "run_artifacts_note": (
        "results.json = 交付 6 算例 + perf（本次把 runner 步数钉到工件 steps_cap=1200 逐位复现，k 相对差 0.0）；"
        "audit_m17677.json = 本次审计机读读数（静态/数值/独立重算/收敛/复现）；png = C1 收敛-误差曲线 + 足迹比；"
        "AUDIT_m17677.md = 六段审计报告"
    ),
    "tests": [
        "tests/test_porous_fluid_node_table.py::test_compaction_order_and_inverse_map",
        "tests/test_porous_fluid_node_table.py::test_neighbour_table_matches_bruteforce_with_periodic_wrap",
        "tests/test_porous_fluid_node_table.py::test_table_reports_solid_sentinel_only_for_solid",
        "tests/test_porous_fluid_node_table.py::test_compressed_and_dense_steps_agree_to_roundoff",
        "tests/test_porous_fluid_node_table.py::test_dense_stream_matches_repo_kernel_under_mirrored_labelling",
        "tests/test_porous_fluid_node_table.py::test_bounce_back_is_half_way_on_a_closed_box",
        "tests/test_porous_fluid_node_table.py::test_mass_is_conserved_by_the_push_scheme",
        "tests/test_porous_fluid_node_table.py::test_channel_permeability_within_owner_15pct",
        "tests/test_porous_fluid_node_table.py::test_compressed_state_is_smaller_than_dense_for_porous_media",
        "tests/test_porous_fluid_node_table.py::test_opposite_mapping_used_by_push_streaming_is_an_involution",
    ],
    "tests_rerun": "cd /root/tp-agent/work/m17677/TensorLBM && python3 -m pytest tests/test_porous_fluid_node_table.py -q -> 12 passed in 31.86s（本次复跑）",
    "integration_callers": (
        "none：grep -rlE 'tensorlbm\\.porous([^A-Za-z0-9_]|$)' /root/tp-agent/work/m17677 → 3 命中，"
        "全部为模块自身/测试/xval，0 主求解器或平台调用方；canonical src/tensorlbm/porous 不存在"
    ),
    "benchmark": {
        "registered_cases_for_anchor": 0,
        "judge_sql": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE'",
        "declared_criterion": "owner 15% 带 + 表示等价 1e-8（predeclared.json）",
        "C1_k_super_ref": 10.666667,
        "C1_k_super_got": 10.553348,
        "C1_err_pct": 1.062,
        "C1_k_super_converged_probe_6000_steps": 10.655,
        "C1_err_pct_converged": 0.109,
        "representation_identity_ux_maxabs_rel": 0.0,
        "other_cases_have_reference": False,
        "artifact_converged_flag": False,
        "auditor_note": "唯一解析锚通过（1.062% 与收敛后 0.109% 均 <15% 且 <3%）；其余 5 算例无任何基准，故 L4 资格不成立",
    },
    "repro_cmd": (
        "cd /root/tp-agent/work/m17825 && python3 scripts/audit/m17677_compressed_node_table_audit.py --with-repro"
        "   # ≈3 min；钉 STEPS=1200 重跑交付 runner，6 算例 k 相对差 0.0（逐位相同）"
    ),
    "external_anchor": (
        "WITHHELD：openLBMPM（PorousMediaSimulation）provenance 以 file:line 声明"
        "(ShanChen2D/ShanChenD2Q9.py:589、OptimizedD2Q9GPU.py:22/452)，但磁盘无 OSS 源码、无 GPU 不可执行，"
        "交付自述为语义移植（predeclared.json 已声明为比较局限）；可核外部锚只有解析 slab k_super=(H/ny)H²/12=10.666667"
    ),
    "audit_independence": (
        "被审树只读：未改 /root/tp-agent/work/m17677 任何文件；复现运行写入 /root/tp-agent/work/m17825/artifacts/m17677_audit/repro/。"
        "数字来源=(a) 被审工件静态读取 (b) 被审代码原样运行（收敛探针 1200/3000/6000 步、逐位复现）"
        "(c) numpy 独立重写邻居表 (d) psql 查 tensor_benchmark_cases。"
        "局限：未跑预声明 H=32/N=128,256/12000 全梯（>30 min 单线程），其可行性只由 run.log 的部分输出背书。"
    ),
    "findings": {
        "F1_high": "交付落在仓内旧嵌套影子树 TensorLBM/src|tests，canonical 包不可导入 ⇒ 能力对用户不可用、无 L3",
        "F2_high": "入口/工件不一致：run_xval.py 算例与步数与工件不同；run_xval_fast.py 现值 2500≠工件 1200，提交原样不可复现",
        "F3_high": "tensor_benchmark_cases 该锚点 0 行；交付无 repro_cmd、无六段报告（tensor_reports 无 mission 17677 行）、"
                  "porous/__init__.py 引用的 docs/xval_compressed_fluid_node_neighbour_table.md 从未生成",
        "F4_medium": "工件 6 算例全部 converged=false；ladder_note 声称 KPI 在 HALF 步采样，但代码中 HALF 从未使用 ⇒ 步数自收敛检查未实现",
        "F5_medium": "预声明全梯被截断（run.log 停在 C2 N=128 表头），H=32 的 0.028% 结果未进工件",
        "F6_info": "p=0.90 时压缩布局足迹为稠密的 1.78 倍（int64 邻居表），工件如实记录、报告未粉饰，但 mission 摘要只引 ms 未引此反向结果",
        "F7_positive": "无造假：k 可反算、邻居表独立重算一致、工件逐位可复现（对比 m17636 的 10 倍压缩/清零）",
    },
    "verdict": "CONFIRMED at L2（L3/L4/L5 不成立，理由见 gap_to_claim）；返修方向：搬进 canonical 包 + 接主求解器调用方 + 跑完/改写预声明梯 + 登记 benchmark case + 补 repro_cmd 与六段报告",
}


def sql_literal(obj) -> str:
    return "$json$" + json.dumps(obj, ensure_ascii=False, indent=1) + "$json$"


SQL = f"""-- m17677 交差核验 verdict: 对象 m17677 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE (TensorLBM)
-- 判级 L2：真实现+真测试+逐位可复现工件；影子树放置/零调用方/零登记基准 = L3/L4 不成立
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', '{SCOPE}', {MISSION}, {LEVEL}, {sql_literal(EVIDENCE)}, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;
"""


def main() -> int:
    path = os.path.join(WS, "verdict_m17677.sql")
    open(path, "w").write(SQL)
    run = subprocess.run(["psql", "-U", "postgres", "-d", os.environ.get("TP_PG_DB", "tensorplatform"),
                          "-v", "ON_ERROR_STOP=1", "-f", path], capture_output=True, text=True)
    print("psql rc", run.returncode, run.stdout.strip()[:200], run.stderr.strip()[:300])
    if run.returncode != 0:
        return 1

    row = subprocess.run(["psql", "-U", "postgres", "-d", os.environ.get("TP_PG_DB", "tensorplatform"),
                          "-tA", "-F", "|", "-c",
                          f"SELECT id, repo, level, verdict_by, gate_state FROM tensor_maturity_verdicts "
                          f"WHERE repo='TensorLBM' AND scope='{SCOPE}'"], capture_output=True, text=True)
    print("landed:", row.stdout.strip())

    # dry-run the machine gate the daemon uses
    sys.path.insert(0, "/root/tp-agent")
    import maturity_gate  # noqa: E402
    lvl, reasons = maturity_gate.judge(LEVEL, EVIDENCE, "TensorLBM", MISSION)
    print(f"maturity_gate.judge dry-run -> gate_level={lvl} reasons={reasons}")
    return 0 if lvl == LEVEL else 2


if __name__ == "__main__":
    raise SystemExit(main())