#!/usr/bin/env python3
"""Publish the m17677 board-audit report to the platform (paper-grade six sections).

markdown = docs/audit_m17677/AUDIT_m17677.md (numbers injected by the audit harness)
figures  = artifacts/m17677_audit/figures/fig_m17677_evidence.png (inline base64)
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

WS = "/root/tp-agent/work/m17825"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(API + path, method="POST",
                                 data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]

md = open(os.path.join(WS, "docs/audit_m17677/AUDIT_m17677.md")).read()
audit = json.load(open(os.path.join(WS, "artifacts/m17677_audit/audit_m17677.json")))
png = os.path.join(WS, audit["figure"])

md += (
    "\n\n## 附：机器读数与工件\n\n"
    f"- 审计 harness：`scripts/audit/m17677_compressed_node_table_audit.py --with-repro`（≈3 min，"
    "S1 静态三件套 / S2 数值反算+邻居表独立重算 / S3 收敛探针 / S4 逐位复现）\n"
    "- 机读读数：`artifacts/m17677_audit/audit_m17677.json`；复现产物：`artifacts/m17677_audit/repro/`\n"
    "- verdict：`tensor_maturity_verdicts`(repo='TensorLBM', scope='m17677 "
    "XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE', level=2, verdict_by='board-audit')；"
    "`maturity_gate.judge` 预演 = gate_level 2（无 fail reason）\n"
    "- 基准裁判 SQL：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
    "anchor_slug='XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE'` → 0 行\n"
)

payload = {
    "title": "交差审计 m17677 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE：工件逐位可复现，判级 L2（影子树放置/零登记基准封住 L3-L4）",
    "summary": (
        "对 m17677 交付做独立复核：12 项测试复跑通过；把交付 runner 步数常量钉回工件值后 6 算例 k 值逐位复现"
        "（相对差 0.0），k=nu<u>/g 可反算，邻居表由 numpy 独立重写逐项一致 ⇒ 工件真实、无造假；"
        "C1 槽道解析锚 k_super 10.553348 vs 10.666667 = 1.062%（收敛探针 6000 步 0.109%），两项均 <3%。"
        "但模块写在仓内旧嵌套影子树（canonical src/tensorlbm 不可导入）、主求解器调用方 0、"
        "tensor_benchmark_cases 该锚点 0 行、交付无 repro_cmd 与六段报告 ⇒ 判级 L2，"
        "maturity_gate 预演 gate_level=2 confirmed。"
    ),
    "scenario_key": "xval/lbm-compressed-fluid-node-neighbour-table",
    "mission_id": 17825,
    "tags": ["xval", "board-audit", "TensorLBM", "porous-media", "maturity-verdict", "L2"],
    "markdown": md,
    "figures": [{
        "name": "fig_m17677_evidence.png",
        "caption": "左：C1 平面槽道 k_super 相对解析锚误差 vs 步数（工件 1200 步瞬态点 1.06%，收敛 0.11%，灰带=±15%）"
                   "；右：交付工件状态足迹比 dense/compressed（p=0.90 时 <1，即高孔隙率下压缩布局更占内存）",
        "png_base64": base64.b64encode(open(png, "rb").read()).decode(),
    }],
}

res = post("/tensor/reports", payload, tok)
print("report id:", res.get("id"), "mission:", res.get("mission_id"))