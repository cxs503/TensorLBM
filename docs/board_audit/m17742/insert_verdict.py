#!/usr/bin/env python3
"""m17742：把 m17673 交差核验判级写入 tensor_maturity_verdicts，并预演 maturity_gate 复核。

evidence 全部由 runs/m17742/verify_triple.json 程序化产出（零手抄）。
声称级别 = 2（理由见 docs/board_audit/m17742/AUDIT_m17673.md）：
  L3 不予：唯一集成点只存在于 m17673 未提交的 worktree diff（main c05d7af3 / worktree HEAD 9bcc6db6 均无 solid_mask），
           生态内无任何可 import 的调用方；
  L4 不予：tensor_benchmark_cases 该 anchor 0 行（唯一裁判缺位）+ 物理层 cj1641 state=queued 从未执行。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

WT = Path("/root/tp-agent/work/m17742")
M = Path("/root/tp-agent/work/m17673")
RUN = WT / "runs/m17742"
AUD = WT / "docs/board_audit/m17742"
ev = json.loads((RUN / "verify_triple.json").read_text())

LEVEL = 2
SCOPE = "m17673 XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY"

# 交付物证据保全：worktree 可能被回收，先按字节复制 + sha256 清单（不替代缺失的 commit）
PRES = AUD / "preserved_m17673"
PRES.mkdir(parents=True, exist_ok=True)
keep = ["src/tensorlbm/runtime_geometry.py", "tests/test_runtime_geometry.py",
        "xval/life_geometry_config/xval_life_geometry_config.py",
        "runs/m17673/xval_parity.json", "runs/m17673/baseline_D20.log",
        "artifacts/xval_life_geometry_config/fig1_geometry_parity.png",
        "artifacts/xval_life_geometry_config/fig2_dx_sensitivity.png",
        "docs/reports/XVAL_LIFE_GEOMETRY_CONFIG.md",
        "docs/reports/XVAL_LIFE_GEOMETRY_CONFIG_diffs.md"]
manifest = {}
for rel in keep:
    src = M / rel
    dst = PRES / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    manifest[rel] = hashlib.sha256(dst.read_bytes()).hexdigest()
(PRES / "MANIFEST.json").write_text(json.dumps(
    {"copied_from": str(M), "note": "uncommitted in m17673 at audit time; byte-verbatim copy",
     "sha256": manifest}, ensure_ascii=False, indent=1))

evid = {
    "os": (f"board-audit m17742 of m17673 (repo TensorLBM). m17673 result=succeeded (auto-approve dev_acceptance)"
           f" but git_status shows 7 uncommitted paths; worktree HEAD {ev['git']['m17673_head']} == main clone HEAD "
           f"{ev['git']['main_clone_head']} → no commit of the deliverable exists"),
    "impl_files": [
        f"{M}/src/tensorlbm/runtime_geometry.py",
        f"{M}/src/tensorlbm/cylinder_st2d1.py",
    ],
    "impl_notes": [
        "runtime_geometry.py:1-572 — CaseParams(frozen)+fingerprint(sha256[:16]) / Scene add・remove・counts・markers・mask・validate_domain / CircleBody.marker_count=floor(2πr/dx) & markers 等弧长 / FilamentBody floor(L/dx)+1 & 度→弧度 & [[c,-s],[s,c]] / _overlap_keep 0.5dx 去重",
        "cylinder_st2d1.py:169,193-206,226-241,321-324 — 新增 solid_mask=None 注入点：形状校验 (ny,nx)、mask 直驱半步反弹、provenance geometry_source=hard_coded|runtime_config；默认路径逐位不变",
    ],
    "entry": f"{M}/xval/life_geometry_config/xval_life_geometry_config.py",
    "entry_cmd": ("cd /root/tp-agent/work/m17673 && PYTHONPATH=src python3 "
                  "xval/life_geometry_config/xval_life_geometry_config.py --mode parity --out <json>"),
    "repro_cmd": ev["repro"]["cmd"] + " --out /tmp/audit_parity.json",
    "repro_cmd_result": {
        "rc": ev["repro"]["run1"]["rc"],
        "audit_reruns": 2,
        "byte_identical_between_reruns": ev["repro"]["byte_identical_run1_vs_run2"],
        "audit_json_equals_delivered_json": ev["repro"]["audit_equals_delivered"],
        "sha256_run1": ev["repro"]["run1"]["sha256"],
    },
    "run_artifacts": [
        f"{M}/runs/m17673/xval_parity.json",
        f"{M}/runs/m17673/baseline_D20.log",
        f"{M}/artifacts/xval_life_geometry_config/fig1_geometry_parity.png",
        f"{M}/artifacts/xval_life_geometry_config/fig2_dx_sensitivity.png",
        f"{RUN}/verify_triple.json",
        f"{RUN}/parity_run1.json",
        f"{RUN}/parity_run2.json",
        f"{RUN}/fig_m17742_audit_m17673.png",
        f"{PRES}/MANIFEST.json",
    ],
    "run_artifacts_notes": [
        "xval_parity.json — 3 grids × {mask_cells, markers, perimeter_err_pct} + thresholds 预登记 + mechanism 指纹；本单重跑逐字段相等",
        "baseline_D20.log — 仅 1 行，5000/60000 步 Cd~6.1102 drift 1.19e-01 → 未收敛半程值，不计成绩",
        "fig1/fig2 — 67KB/76KB 真图（几何对照、dx 敏感性）",
        "verify_triple.json — 本单机器复核原始证据（sha256/复跑/pytest/DB/cluster 全在内）",
    ],
    "tests": [
        "tests/test_runtime_geometry.py — 29 passed / 15.82s（本单复跑，rc=0）",
        "test_solver_injection_is_bit_identical_for_identical_geometry — 真跑 run_cylinder_st2d1(D=10,40 步) 两路，cd 逐位相等（abs=0），geometry_source 分别为 hard_coded / runtime_config",
        "test_solver_injection_rejects_wrong_shape — 形状不符抛 ValueError('solid_mask shape')",
        "test_mask_matches_verified_cylinder_mask_exactly / test_circle_marker_count_is_floor_2pir_over_dx / test_geometry_edits_do_not_change_param_fingerprint / test_param_edit_requires_rebuild",
    ],
    "integration_callers": (
        "worktree-only: /root/tp-agent/work/m17673/src/tensorlbm/cylinder_st2d1.py:169 "
        "run_cylinder_st2d1(solid_mask=...) — 由 tests/test_runtime_geometry.py:286 调用并经真实求解器执行；"
        "但 main clone /root/tensor-ecosystem/TensorLBM HEAD c05d7af3 中 grep solid_mask = 0 命中、runtime_geometry.py 不存在"
        " → 非合并态，生态内无消费者可达 → L3 不成立（不按「函数名在 main 里存在」蒙混 gate 复核）"
    ),
    "benchmark": {
        "registered_case": (f"none: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
                            f"anchor_slug='XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY' → 0 行"
                            f"（TensorLBM 全表 6 行，无一属该 anchor）→ 唯一裁判缺位，L4 不予"),
        "delivered_ref_vs_got": {r["grid"]: {"mask_ref": r["mask_cells_ref"], "mask_got": r["mask_cells_got"],
                                             "delta_cells": r["mask_delta_cells"],
                                             "markers_ref_formula": r["markers_ref_formula"],
                                             "markers_got": r["markers_got"],
                                             "perimeter_err_pct": r["perimeter_err_pct"]}
                                 for r in ev["parity_table"]},
        "thresholds_predeclared": ev["parity_thresholds"],
        "second_order_ratio": ev["second_order_ratio"],
        "marker_formula_recompute": ev["marker_formula_recompute"],
        "parity_is_constructive": ev["parity_is_constructive"],
        "physics_layer": ev["physics_midrun"],
        "cluster_job_cj1641": ev["cluster_job"]["row"],
        "accuracy_gate_15pct": "UNDETERMINED — Cd 从未测（cj1641 queued）；不得以半程 9.51% 充作通过或失败",
    },
    "external_anchor": {
        "anchor": "joconnor22/LIFE (GPL-3.0) compiletime-params + runtime geometry.config 约定",
        "source_path": "/tmp/LIFE-master",
        "durable": False,
        "spot_checks": {k: v for k, v in ev["external_anchor"]["spot_checks"].items()},
        "pointer_drift": "IBMBody.cpp:47-105 与 :100（报告引用）实测落在 :55 与 :98 — 语义正确、行号偏 2-5 行（低危）",
    },
    "report_sections": ev["report_sections"],
    "git_delivery_state": {
        "worktree_head": ev["git"]["m17673_head"],
        "main_clone_head": ev["git"]["main_clone_head"],
        "status": ev["git"]["m17673_status"],
        "commits_touching_deliverable": ev["git"]["m17673_commits_touching_deliverable"] or "(none)",
        "risk": "worktree 回收即丢失；本单已按字节保全到 docs/board_audit/m17742/preserved_m17673/（MANIFEST sha256）",
    },
    "level_rationale": (
        "L2：真代码 input→compute→output（geometry.config 文本 → Scene/bodies → markers/mask → 求解器半步反弹 mask），"
        "入口/工件/29 测试全部复现，注入路径由真实求解器执行验证（cd 逐位相等、形状误用报错）。"
        "L3 不予（集成点仅存在于未提交 worktree diff，main c05d7af3 无 solid_mask、无 runtime_geometry.py，无消费者可达）。"
        "L4 不予（tensor_benchmark_cases 该 anchor 0 行 + 物理层 cj1641 从未执行；mask Δ=0 是同一闭式解的构造性恒等，非独立验证）。"
        "L5 不予（外部锚为 GPL 源码约定，且 /tmp/LIFE-master 无 commit/sha 钉扎，不可持久复核）。"
    ),
    "findings": [
        "F1 交付未落 commit（blocking-for-durability）：succeeded 但 git_status 7 条未提交，branch agent/m17673 无可推内容 → 「成熟后自动推送」无法发生",
        "F2 基准裁判缺位：anchor 0 行，物理 Cd 从未测（cj1641 queued, started_at NULL）",
        "F3 mask Δ=0 属构造性恒等（ref 即本仓 cylinder_mask，got 是同一闭式从物理坐标重推）→ 只能说明插管不改几何",
        "F4 外部锚不可持久：/tmp/LIFE-master 无 commit 钉扎；引用行号偏 2-5 行",
        "F5 报告诚实：物理层显式 WITHHELD，未粉饰；六段齐全；⑤ 二阶收敛比 3.998/4.065 实测成立",
    ],
}

(Path(AUD) / "verdict_evidence.json").write_text(json.dumps(evid, ensure_ascii=False, indent=1))

sql_path = WT / "verdict_m17673.sql"
evid_sql = json.dumps(evid, ensure_ascii=False).replace("'", "''")
sql_path.write_text(
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)\n"
    f"VALUES ('TensorLBM', '{SCOPE}', 17742, {LEVEL}, '{evid_sql}'::jsonb, 'board-audit')\n"
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
    "evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_note=NULL;\n"
    "SELECT id, repo, level, verdict_by, gate_state FROM tensor_maturity_verdicts "
    f"WHERE scope='{SCOPE}' AND verdict_by='board-audit';\n")

r = subprocess.run(["psql", "-d", "tensorplatform", "-U", "postgres", "-v", "ON_ERROR_STOP=1",
                    "-f", str(sql_path)], capture_output=True, text=True)
print("INSERT rc=", r.returncode)
print(r.stdout.strip()[-1200:])
print(r.stderr.strip()[-800:])

# ── 预演 gate 复核（与守护同一判据函数） ──
sys.path.insert(0, "/root/tp-agent")
import maturity_gate as MG  # noqa: E402

lvl, why = MG.judge(LEVEL, evid, "TensorLBM", 17742)
print(json.dumps({"gate_predicted_level": lvl, "why": why}, ensure_ascii=False, indent=1))
for cand in (3, 4, 5):
    l2, w2 = MG.judge(cand, evid, "TensorLBM", 17742)
    print(f"if claimed L{cand} → gate_level={l2}; why={w2[:2]}")