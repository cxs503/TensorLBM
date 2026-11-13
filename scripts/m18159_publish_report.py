#!/usr/bin/env python3
"""Publish the m18159 board-audit report to the platform report lake.

Markdown = docs/board_audit/m18159/M18159_BOARD_AUDIT.md (rendered from
evidence.json with assert-bound numbers), figures embedded as base64.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 18159
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/board_audit/m18159"
EV = json.loads((ART / "evidence.json").read_text())

assert EV["subject_mission"] == 18068
assert EV["verdict"]["level"] == 0


def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main() -> None:
    md = (ART / "M18159_BOARD_AUDIT.md").read_text()
    m, v = EV["mission"], EV["verdict"]
    figs = [
        {"name": "fig1_evidence_matrix.png",
         "caption": "左：10 项预声明证据检查（仅 external_anchor 通过）；右：L2–L4 前置条件全未满足，判级 L%d。" % v["level"],
         "png_base64": b64(ART / "figures/fig1_evidence_matrix.png")},
        {"name": "fig2_budget_utilization.png",
         "caption": "失败归因「预算耗尽」与实测用量对照：tool_calls %.0f%%、wall %.0f%%。"
                    % (100 * m["budget_used_frac"]["tool_calls"], 100 * m["budget_used_frac"]["wall_seconds"]),
         "png_base64": b64(ART / "figures/fig2_budget_utilization.png")},
    ]
    body = {
        "title": "m18159 交差核验 — m18068 XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY 判 L0（reopened）",
        "summary": ("m18068 runner 状态 succeeded，但交付实质为空：0 提交/0 工件/0 测试/0 基准算例/0 报告，"
                    "唯一产物是 2 个未跟踪接口探针；判级 L0，锚点能力缺口未闭合。另发现三条平台级问题："
                    "「预算耗尽」归因不实（tool_calls %.0f%%、wall %.0f%%）、auto-approve 放行 no-progress、"
                    "该锚点在 tensor_benchmark_cases 无裁判行（平台共 %d 条）。"
                    % (100 * m["budget_used_frac"]["tool_calls"], 100 * m["budget_used_frac"]["wall_seconds"],
                       EV["anchor_evidence"]["platform_benchmark_case_total"])),
        "scenario_key": "board_audit_xval_lbm_no_multiphase_layout_pattern_only",
        "mission_id": MISSION_ID,
        "tags": ["TensorLBM", "board-audit", "XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY",
                 "m18068", "maturity-verdict", "L0", "reopened"],
        "markdown": md,
        "figures": figs,
    }
    token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    res = post("/tensor/reports", body, token)
    print("published:", json.dumps(res)[:400])


if __name__ == "__main__":
    main()