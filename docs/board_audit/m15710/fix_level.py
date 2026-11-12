#!/usr/bin/env python3
"""m15710 陈级修正：按生态先例（m15432=同族 CYLINDER-DRAG-RE100 判 L3：主求解器原语 + tensor_case_pool 注册 + 已报报告）
把 L2 修正为 L3，并补入 case_pool id=10 实证与 F9（池内状态仍 dispatched，未结算）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PSQL = ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-c"]


def q(sql: str) -> str:
    return subprocess.run(PSQL + [sql], capture_output=True, text=True).stdout.strip()


ev = json.loads(q("SELECT evidence::text FROM tensor_maturity_verdicts WHERE id=368"))
pool = q("SELECT id||' case_code='||case_code||' repo='||repo||' tol='||tol_policy||' status='||status||"
         " ' mission_id='||mission_id FROM tensor_case_pool WHERE id=10")

ev["integration_callers"] = (
    "平台注册（先例口径 L3）：tensor_case_pool " + pool + "；"
    "主求解器原语真实调用 src/tensorlbm/solver.py(collide_mrt/stream)+boundaries.py"
    "(zou_he_inlet_velocity/zou_he_outlet_pressure/bounce_back_cells/cylinder_mask/"
    "make_channel_wall_mask/compute_obstacle_forces)（本单 import 实测）；"
    "报告已入 tensor_reports id=14508 scenario_key=CASE-CYLINDER-DRAG-RE20。"
    "无自动化调用者：grep -rn 're20_st' tests/ .github/ = 0 命中 → 属“注册+复用主求解器”，非 CI 守护的集成。"
)
ev["benchmark"]["pool_tolerance"] = "<5%（tensor_case_pool id=10 tol_policy）；本单实测两档 |err_Cd|=0.5575%/2.8749% 均在池容差内"
ev["benchmark"]["quality_findings"]["F9_pool_not_settled"] = (
    "tensor_case_pool id=10 status 仍 'dispatched'（任务已 succeeded、交付已 PASS）→ 池结算缺口："
    "算例池锚点未关账；同时 tensor_benchmark_cases 无对应行 → 双重裁判面缺席"
)
ev["level_rationale"] = (
    "L3（对齐生态先例 m15432=同族 RE100 判 L3 的口径）：(a) 真代码 输入→计算→输出，本单实跑降档复算 rc=0，"
    "无硬编码结果/无读 JSON 冒充计算；(b) 完全构建在主求解器原语之上（tensorlbm.solver/boundaries 实测调用）；"
    "(c) 平台注册 tensor_case_pool id=10（tol <5%）+ 报告 tensor_reports id=14508。"
    "不达 L4：(1) tensor_benchmark_cases 无 CASE-CYLINDER-DRAG-RE20 行 → 唯一机器裁判缺席；"
    "(2) 误差随网格加密放大 ×5.16（0.5575%→2.8749%）、非单调收敛，D=80 距 3% 线仅 0.125%，报告自述 D=160 预期超线；"
    "(3) C_L 符号反、相对误差 -232.7%/-114.2%，二阶量未验证；(4) 本单零新计算——result_verified_baseline.json 与 "
    "benchmarks/verified/cylinder/re20_st/result.json 逐键相等，300k 步 bit-exact 系历史跑，复现只能 GPU（本机无 CUDA）。"
)
ev["verdict_summary"] = (
    "三件套：run_artifacts ✅（6 文件入 commit 2a72237d，sha256 已钉）；ref vs got ✅ 在 <5%(<3%) 容差内"
    "（但网格趋势反向、C_L 失败，如实披露）；repro_cmd ⚠️ 语法正确但仅 GPU 可执行（本机 torch cuda_avail=False）。"
    "基准裁判：tensor_benchmark_cases 无此案号 ❌；算例池 id=10 仍 dispatched ❌。报告六段 ✅ 齐全（授权 ①—⑥+显式误差判定），"
    "但 figures 3 条 png_base64 为空。判级 L3（能力真、集成面有注册、L4 未达）。"
)

sql = ("UPDATE tensor_maturity_verdicts SET level=3, evidence=$json$%s$json$::jsonb, gate_state='pending', "
       "gate_level=NULL, gate_note=NULL WHERE id=368 RETURNING id, level, gate_state;" % json.dumps(ev, ensure_ascii=False))
p = Path(__file__).resolve().parent
(p / "update_verdict_l3.sql").write_text(sql, encoding="utf-8")
(p / "evidence_final.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
print(subprocess.run(PSQL + [sql], capture_output=True, text=True).stdout.strip())
print(q("SELECT id||'|'||repo||'|'||scope||'|L'||level||'|'||verdict_by||'|'||gate_state FROM "
        "tensor_maturity_verdicts WHERE id=368"))