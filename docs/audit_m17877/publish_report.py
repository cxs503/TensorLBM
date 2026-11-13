#!/usr/bin/env python3
"""Publish the m17877 board-audit report to the platform report ledger."""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
API = "http://127.0.0.1:8010/api/v1"
TOKEN = Path("/tmp/tok").read_text().strip()

md = (HERE / "AUDIT_m17877.md").read_text()
fig = (HERE / "fig_m17877_evidence.png").read_bytes()
S = json.loads((HERE / "audit_summary.json").read_text())

body = {
    "title": "m17877 交差核验：m17724 XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION → 判级 L2（压力定义不实 + 收敛为单相位采样）",
    "summary": (
        "受审 m17724（SOY 自由能 D2Q9 LBM 的 PyTorch 移植）：主算例 Δp 误差 0.0708%<15%，审计逐位复现，"
        "R=10 复现 0.6434%，毛细能量独立路线 2.90% 互洽 ⇒ 判 L2。但压力公式与 OSS 不符而报告称 identical（OSS 定义下 0.911%）、"
        "收敛为单相位采样（时间带 0.2%-6.8%）、TensorLBM 侧与 OSS 侧均无对照数值、零 commit/无测试/无裁判行 ⇒ 不授 L3/L4。"
    ),
    "scenario_key": "board_audit",
    "mission_id": 17877,
    "tags": ["board-audit", "TensorLBM", "XVAL", "LBM", "free-energy", "Laplace", "m17724", "L2"],
    "markdown": md,
    "figures": [
        {
            "name": "fig_m17877_evidence.png",
            "caption": (
                "左：Laplace 误差 —— 交付 4 点采样（黑方块，末点 0.0708%）vs 审计 1200-1999 步每 25 步重采样的时间带"
                f"（{S['probes']['convergence_band_err_pct'][0]:.2f}-{S['probes']['convergence_band_err_pct'][1]:.2f}%）；"
                "15% 判据红线。右：压力定义敏感性（移植 ρ/3+ψμ/3 = 0.0708%；OSS ρ/3+ψμ = 0.911%；R=10 = 0.6434%）"
                "与时间带误差棒。"
            ),
            "png_base64": base64.b64encode(fig).decode(),
        }
    ],
}

req = urllib.request.Request(
    f"{API}/tensor/reports",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    print("POST ok:", r.status, {k: out.get(k) for k in ("id", "title", "mission_id", "status", "qa_status")})
    Path("/tmp/report_posted.json").write_text(json.dumps({"id": out.get("id")}))
except urllib.error.HTTPError as e:
    print("HTTPError", e.code, e.read().decode()[:800])
    raise