#!/usr/bin/env python3
"""m18207 落表：交差核验结论写入 tensor_maturity_verdicts（board-audit 席位，幂等）。

evidence 的每个指针都来自 docs/board_audit/m18207/verify_output.json 的实测记录；
落表前用平台守护 /root/tp-agent/maturity_gate.py::judge 做**干跑**（只读，不写库），
把「守门机器人会把它裁成几级」先算出来，再与本次声明级别对照。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text())
A1, A4, A6 = V["A1_deliverables"], V["A4_repro"], V["A5_A6_benchmark"]["A6_replica"]
rows = V["A5_A6_benchmark"]["delivered_criteria_summary"]

EV = "docs/board_audit/m18207/evidence/"
SCOPE = "m18114 XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE"
MISSION = 18207
LEVEL = 2

evidence = {
    "impl_files": [
        EV + "impl__binary_equilibrium_pressure.py",
    ],
    "entry": EV + "entry__run_xval.py",
    "run_artifacts": [
        EV + "artifact_json__xval_binary_equilibrium_pressure.json",
        EV + "artifact_png__fig_convergence_and_lbm.png",
    ],
    "tests": [
        "tests/test_binary_equilibrium_pressure_xval.py 复跑 7 passed rc=0（本单实跑 "
        f"{V['A3_tests']['wall_s']}s）；快照 " + EV + "tests__test_binary_equilibrium_pressure_xval.py"
    ],
    "integration_callers": (
        "none（本单实测 grep：run_binary_equilibrium_pressure_benchmark 仅定义于 "
        "src/tensorlbm/multiphase_benchmarks.py:713，run_multiphase_benchmark_suite 未注册、"
        "benchmarks/bench_multiphase.py 未导入、平台侧无引用；phasefield/__init__.py 只是包级导出"
        "→ 无下游调用方，L3 不达）"
    ),
    "benchmark": {
        "registered_cases_psql": V["A5_A6_benchmark"]["registered_cases"],
        "delivered_ref_vs_got": [
            {"interface_cells": r["interface_cells"],
             "sigma_ref": r["sigma_analytic"], "sigma_got": r["sigma_integral"],
             "sigma_rel_err": r["sigma_relative_error"],
             "p_nn_uniformity_rel": r["p_nn_max_deviation_relative"],
             "coex_rel_err": r["coexistence_relative_error"],
             "lbm_p_nn_total_dev_rel": r["lbm_final_p_nn_total_deviation_relative"],
             "lbm_umax": r["lbm_final_max_velocity"]}
            for r in rows],
        "owner_band": 0.15,
        "max_abs_sigma_rel_err": max(abs(r["sigma_relative_error"]) for r in rows),
        "within_owner_band": max(abs(r["sigma_relative_error"]) for r in rows) < 0.15,
        "replica_of_case85": A6,
    },
    "repro_cmd": ("cd /root/tp-agent/work/m18114 && python "
                  "xval/XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE/run_xval.py --out /tmp/repro18114"),
    "external_anchor": (
        "Pooley, Furtado & Carr, PRE 78, 056709 (2008)（90 度接触角 free-energy LBM 的解析平衡锚）；"
        "σ=(2√2/3)√(κA³)/B 与 p_coex=A²/(4B) 为本单独立重推；登记基准行 tensor_benchmark_cases id=85/86/87"
    ),
    "repro_verified": {
        "rc": A4["rc"], "wall_s": A4["wall_s"],
        "checks_bitwise_equal": A4["repro"]["checks_bitwise_equal"],
        "rows_bitwise_equal": A4["repro"]["rows_bitwise_equal"],
        "rerun_sha256": A4["repro"]["rerun_sha256"],
        "verdict_line": A4["tail"].splitlines()[-2:] if A4["tail"] else [],
    },
    "provenance": {
        "subject_worktree": "/root/tp-agent/work/m18114", "subject_head": "c6ed86d2",
        "snapshots": {k: {"origin": v["path_in_subject_worktree"], "sha256": v["sha256"],
                          "bytes": v["bytes"]} for k, v in A1.items()},
        "in_ecosystem_main": V["A2_in_ecosystem_main"],
    },
    "findings": [
        {"id": "F1", "severity": "blocking-for-L3", "claim": "无下游调用方",
         "evidence": V["A7_integration"]["verdict"], "impact": "L3/L4 不达；交付不可被套件/平台消费"},
        {"id": "F2", "severity": "blocking-for-L4", "claim": "基准覆盖不全",
         "evidence": "登记行 86（droplet Laplace R16/R24）与 87（dropR36 稳定性，status=failed）"
                     "在交付模块中无对应能力（模块无 droplet/Laplace 臂）；行 87 为锚点级 honest-FAIL 未消",
         "impact": "锚点基准只被部分覆盖；已覆盖的平面族 σ 10.15/6.09/2.84% 全在 <15% 带内"},
        {"id": "F3", "severity": "material", "claim": "双梯度格式（iso9/5pt）在平界面配置下恒等",
         "evidence": f"grad_x 最大差 {A6['scheme_iso9_vs_5pt_grad_x_max_delta']:.3e}，σ 差 "
                     f"{A6['scheme_iso9_vs_5pt_sigma_delta']:.1e}；y 不变场使 Tomita 权重 "
                     "(4/12+1/12+1/12)=1/2 退化为中心差分",
         "impact": "文档所称『离散敏感性可测而非假设』在 1-D 变化场上未被真正行使，scheme 行是重复计数"},
        {"id": "F4", "severity": "material", "claim": "σ 口径与登记行不同",
         "evidence": f"交付 σ 取自解析平衡场的自由能压力张量（{A6['sigma_measured_static_tensor']:.6f}），"
                     "登记行 85 的 got 取 population 二阶矩；本单用交付模块复算 case85 配置得 "
                     f"{A6['sigma_relative_error']:.4%}（登记 got 2.0097e-02），同带内但不可逐位复现；"
                     "Pnn 均匀性交付 3.59e-02（LBM 总压）vs 登记 1.4799e-05（population 口径）",
         "impact": "基准行与交付工件不是同一测量口径，裁判可比性需补注；不构成造假，但报告应写明"},
        {"id": "F5", "severity": "minor", "claim": "交付物未进生态 main",
         "evidence": V["A2_in_ecosystem_main"]["git_cat_file_impl"],
         "impact": "当前只在 agent/m18114 本地分支，合并前下游不可用"},
        {"id": "F6", "severity": "minor", "claim": "文档/自述瑕疵",
         "evidence": "run_xval.py 文档串声明两张 png，实际只 savefig 一张；m18114 result.checks 把 PNG 的 "
                     f"105928 B 写成 JSON 大小（JSON 实为 {A1['artifact_json']['bytes']} B）",
         "impact": "不影响判据数字，属自述准确性瑕疵"},
    ],
    "why_not_L3": "L3 需主求解器/平台侧真调用方；本单 grep 实测无调用点（F1）。"
                  "phasefield/__init__.py 的导出是公共 API 暴露，不是调用者；"
                  "方向相反的『验证模块调用生产 free_energy_step/stream』是好事但不算 integration_callers。",
    "why_not_L4": "L4 需 L3 全过；另基准覆盖只到平面族（F2），登记行 86/87 未覆盖，行 87 仍 failed；"
                  "且 σ 口径与登记行不同（F4），ref vs got 不可逐位复现。",
    "not_checked": [
        "登记行 86 的 droplet Laplace 两臂未复算（交付模块无该能力）",
        "登记行 87 的 dropR36 稳定性未复算（交付模块无该能力，且登记为 failed）",
        "未审查 GPU/CUDA 路径（交付为 CPU torch float64）",
        "未做网格无关性以外的参数扫描（tau_f、Gamma 未扫描）",
    ],
}

# ---- 守门机器人干跑（只读） -------------------------------------------------
spec = importlib.util.spec_from_file_location("mg", "/root/tp-agent/maturity_gate.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)
dry = {}
for lv in (2, 3, 4):
    gl, why = mg.judge(lv, evidence, "TensorLBM", MISSION)
    dry[str(lv)] = {"gate_level": gl, "reasons": why}
print("gate dry-run:", json.dumps(dry, ensure_ascii=False, indent=1))


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ("
    f"'TensorLBM', {q(SCOPE)}, {MISSION}, {LEVEL}, {q(json.dumps(evidence, ensure_ascii=False))}, 'board-audit')\n"
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
    "mission_id=EXCLUDED.mission_id, evidence=EXCLUDED.evidence, gate_state='pending';\n"
)
(HERE / "verdict.sql").write_text(sql)
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1", "-f",
                    str(HERE / "verdict.sql")], capture_output=True, text=True)
print("insert rc", r.returncode, r.stdout.strip()[:300], r.stderr.strip()[:300])
print(subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-c",
                      f"SELECT id,repo,scope,level,mission_id,verdict_by,gate_state,"
                      f"jsonb_array_length(evidence->'findings') findings FROM tensor_maturity_verdicts "
                      f"WHERE repo='TensorLBM' AND verdict_by='board-audit' AND scope LIKE '%18114%';"],
                     capture_output=True, text=True).stdout)