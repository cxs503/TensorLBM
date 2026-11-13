#!/usr/bin/env python3
"""Publish the m18218 audit report to tensor_reports (figures embedded base64)."""
import base64
import json
import os
import sys
from pathlib import Path

import requests

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = Path("/root/tp-agent/work/m18218/docs/audit_m18218")

MD = (HERE / "AUDIT_m18218_XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS.md").read_text()

FIGS = [
    ("fig_m18218_front_vs_mm.png",
     "图1 左：作者工件的溃坝前沿 Z(T) 三臂 vs Martin&Moyce 1952，灰带为预声明窗口 T∈[1.5,4.0] 未执行段"
     "（审计以 T=3.75 重跑，max err 不变）；右：三臂相对误差随 T 的走位与 15% 门（绿带）。"),
    ("fig_m18218_diagnostics.png",
     "图2 左：液量守恒（TensorLBM 全程 <0.06%，OSS 臂 207.8%/19.1%，破 1% 门）；"
     "右：审计独立重跑三点 —— tensorlbm tmax=1.5/4.0 均为 13.806%（过），OSS quirk 修正臂 15.961%（差 0.96pp 未过）。"),
]

figs = []
for name, cap in FIGS:
    p = HERE / name
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

payload = {
    "title": "m18218 交差核验：XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS（对象 m18073）判级 L2，gate=reopened",
    "summary": ("独立复核 m18073：三件套中 run_artifacts 存在且审计重跑逐位复现（Z max err 13.806% / 15.961%），"
                "但 tensor_benchmark_cases 对该锚点 0 行、报告 0 行、全部交付物 untracked、测试 5 例 1 红且失败原因被错报、"
                "两臂互洽 15% 门全破（22.7/63.6/53.8%）。判级 L2，不授 L3-L5。"),
    "scenario_key": "board_audit",
    "mission_id": 18218,
    "tags": ["board-audit", "XVAL", "TensorLBM", "D2Q9", "free-surface", "dam-break",
             "maturity-verdict", "m18073"],
    "markdown": MD,
    "figures": figs,
}

s = requests.Session()
r = s.post(f"{BASE}/auth/login", json={"username": USER, "password": PASS}, timeout=30)
if r.status_code >= 400:
    print("login failed", r.status_code, r.text[:200])
    sys.exit(1)
d = r.json()
tok = d.get("access_token") or d.get("token") or d.get("data", {}).get("access_token")
r = s.post(f"{BASE}/tensor/reports", json=payload,
           headers={"Authorization": f"Bearer {tok}"}, timeout=120)
print(r.status_code, r.text[:300])