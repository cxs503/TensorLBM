#!/usr/bin/env python
"""Publish the P233W2-N06 report to the platform report API.

Logs in with ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS`` against
``TENSOR_API_BASE/auth/login``, embeds the campaign figures as base64 PNG and
posts the markdown report to ``/tensor/reports``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "P233W2_N06_SPHERICAL_FORCE_METHOD.md"
FIGS = ROOT / "artifacts" / "spherical_force_method" / "figures"


def post(path: str, payload: dict, token: str | None = None) -> dict:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers=headers,
                                method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    tok = post("/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })["access_token"]

    figures = []
    captions = {
        "planar_channel.png": "Plane-channel (planar leg) validation: wall-shear force error vs "
                              "the analytic mu*U/H and velocity-profile error for three wall offsets "
                              "(q = 1/2 and interpolated branches).",
        "cd_vs_re.png": "Cylinder drag coefficient vs Re: spherical force method compared with the "
                        "2D free-stream numerical cluster (Dennis & Chang 1970; Braza et al. 1986).",
        "grid_convergence.png": "Grid convergence of Cd at Re = 100 (three lattice diameters).",
        "field_re100.png": "Velocity magnitude and vorticity field around the cylinder "
                           "(spherical force method, BFL boundary).",
    }
    for name, cap in captions.items():
        p = FIGS / name
        if p.exists():
            figures.append({
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            })

    summary = json.loads((ROOT / "artifacts" / "spherical_force_method" / "summary.json").read_text())
    ac = summary.get("acceptance", {})
    headline = {
        "AC1_planar_pass": ac.get("AC1_planar", {}).get("pass"),
        "AC2_cd_pass": ac.get("AC2", {}).get("pass"),
        "AC4_grid_pass": ac.get("AC4", {}).get("pass"),
        "AC5_ibm_pass": ac.get("AC5", {}).get("pass"),
    }

    payload = {
        "title": "P233W2-N06 球面（曲壁）力法：曲面边界力积分 + 圆柱绕流 Cd 验证",
        "summary": (
            "TensorLBM D2Q9 球面（曲壁）力法实现（BFL 插值反弹 + 动量交换曲面力积分）："
            "几何 q/法向量解析精确（1e-15）；离散动量收支残差 2.6e-17（AC3 达标）；"
            "平面通道壁面剪力积分误差 0.00%（q=1/2）/3.5–4.1%（q≠1/2），总动量平衡 0.00%（AC1 平面腿达标）；"
            "曲壁静止态稳定（max|u|~1e-7）。但圆柱绕流暂态在 Re=20/D=10 下 <150 步发散为 NaN，"
            "故 AC2（Cd vs 参考 ≤10%）、AC4（网格收敛）、AC5（与 IBM 交叉≤5%）为 honest FAIL/未测；"
            "负结果已登记：把 q 全部钳到 ≥1/2（消除小 q 插值）并不能消除发散。速率受限于本机负载 300+ 与集群排队 240+。"
        ),
        "scenario_key": "ice/cfd/platform",
        "mission_id": 10684,
        "tags": ["TensorLBM", "LBM", "spherical-force-method", "BFL", "curved-boundary",
                 "cylinder-drag", "P233W2-N06", "honest-pending"],
        "markdown": DOC.read_text(),
        "figures": figures,
        "metadata": {"acceptance": headline, "node": "P233W2-N06", "repo": "TensorLBM"},
    }
    out = post("/tensor/reports", payload, tok)
    print(json.dumps(out, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()