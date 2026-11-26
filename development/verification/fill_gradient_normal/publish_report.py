#!/usr/bin/env python3
"""Publish the XVAL report for mission 17757 to the platform."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "")
pw = os.environ.get("TENSOR_AGENT_PASS", "")


def req(method, path, body=None, tok=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


st, d = req("POST", "/auth/login", {"username": user, "password": pw})
assert st == 200, (st, d)
tok = d["access_token"]

md = (HERE / "REPORT.md").read_text()
figs_b64 = json.loads((HERE / "figures_b64.json").read_text())
captions = {
    "fig1_circle_normals.png": "锐圆 R=32 (N=128)：A1 TensorLBM（红）与 A2 OSS（橙，幅值x2 以便可见）法向重合；A4 解析（绿）。",
    "fig2_convergence.png": "收敛/敏感性：左=锐 fill 分辨率梯（~5% 轴偏置底）；右=抹平宽度梯（w=8 时 0.71%）。",
    "fig3_sphere_normals.png": "3D 球 R=16 (N=64)：A1 fill-gradient 法向（红），dir_err vs 解析 8.40%。",
}
figures = [{"name": n, "caption": captions[n], "png_base64": b}
           for n, b in figs_b64.items()]

payload = {
    "title": "XVAL-LBM-FILL-GRADIENT-NORMAL: TensorLBM vs Maarten-vd-Sande/lbm fill-gradient 界面法向（对标误差 0.00% < 15% 通过）",
    "summary": "四臂对照（A1 TensorLBM as-is / A2 OSS clean / A3 OSS w/ GAS=0.5 bug / A4 解析），同一 fill 场只变法向公式：交叉实现方向误差 A1 vs A2 = 0.0（7/7 案例，<15% 通过）；vs 解析解锐界面 4.5-10.2%（方法固有轴偏置，两实现共有），抹平界面 w=8 时 0.71%；复现知识库记载的 OSS GAS=0.5 优先级 bug（幅值 2x 低估，mag_ratio=0.500）。差异清单 7 项+移植取舍见报告。",
    "scenario_key": "XVAL-LBM-FILL-GRADIENT-NORMAL",
    "mission_id": 17757,
    "tags": ["xval", "tensorlbm", "free-surface", "fill-gradient-normal", "Maarten-vd-Sande/lbm", "oss-id-85"],
    "markdown": md,
    "figures": figures,
}
st, d = req("POST", "/tensor/reports", payload, tok)
print("POST /tensor/reports ->", st, json.dumps(d, ensure_ascii=False)[:300])
