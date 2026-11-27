#!/usr/bin/env python3
"""发布 m17876 交差核验报告到平台 tensor_reports（六要素 + 内嵌图）。

凭据取自环境变量（TENSOR_AGENT_USER / TENSOR_AGENT_PASS / TENSOR_API_BASE），不落盘。
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path,
                                data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json",
                                         **({"Authorization": f"Bearer {token}"} if token else {})},
                                method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")
print("login ok:", bool(token))

fig_path = os.path.join(ROOT, "docs", "audit_m17835", "fig_m17835_evidence.png")
png = base64.b64encode(open(fig_path, "rb").read()).decode()

md = open(os.path.join(ROOT, "docs", "audit_m17835", "AUDIT_m17835.md")).read()

report = {
    "title": "交差审计 · XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL（m17835）· 判级 L2：真功能实现、判据 FAIL、零入库",
    "summary": ("对 m17835 的交差核验：三件套存在且工件可逐位复现（54.64955859108693/18.550358759828203），"
                "但 Poiseuille 判据 FAIL（内部 max 54.65%、L2 18.55%，38/62 结点越界；u_max 偏低 15.66%）；"
                "conv-as-streaming 路径边界零填充致质量外泄 18.56%/步（doc 'exact TF parity' 自证伪，求解器实际只走 roll）；"
                "LDC 半边无工件；交付 0 commit、3 文件 untracked、runs/ 被 gitignore、平台报告 0 行。判级 L2，处置 request changes。"),
    "scenario_key": "anchor_audit",
    "mission_id": 17876,
    "stage": "board-audit",
    "tags": ["TensorLBM", "XVAL", "audit", "LBM", "conv-as-streaming", "m17835", "L2"],
    "markdown": md,
    "figures": [{
        "name": "fig_m17835_evidence.png",
        "caption": ("m17835 证据三图：①Poiseuille got vs 闭式解剖面；②相对误差沿 y 单调增长（10.7%→54.7%，15%/3% 判据线）；"
                    "③conv 路径质量损失 18.56% vs roll 0%，内部 |conv−roll|=0、边界≠0。"),
        "png_base64": png,
    }],
}

res = post("/tensor/reports", report, token)
print(json.dumps(res, ensure_ascii=False)[:600])