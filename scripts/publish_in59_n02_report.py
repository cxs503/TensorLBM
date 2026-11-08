#!/usr/bin/env python3
"""Publish the IN59-N02 report to the TensorPlatform report API.

Auth and payload construction use the environment only:

    TENSOR_API_BASE, TENSOR_AGENT_USER, TENSOR_AGENT_PASS

The markdown body is the node report (``docs/IN59_N02_KCS_MANEUVERING_CFD.md``),
the figures are embedded as base64 PNGs straight from ``artifacts/``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "maneuvering_kcs_cfd"
DOC = ROOT / "docs" / "IN59_N02_KCS_MANEUVERING_CFD.md"
MISSION_ID = 6606

FIGURES = [
    ("fig1_geometry.png", "图1 体素几何：设计水线俯视 + 中横剖面（Cb 校验数字见报告 §1）"),
    ("fig2_force_traces.png", "图2 各斜航角 X(t)/Y(t)/N(t) 时程（收敛证据）"),
    ("fig3_drift_primes.png", "图3 X'/Y'/N' vs 斜航角：LBM 算例点 vs MMG 拟合式"),
    ("fig4_rudder_circular.png", "图4 舵角族与回转族（船体固连系）的力/力矩"),
    ("fig5_derivatives.png", "图5 辨识出的 MMG 导数一览（绝对值见 result.json）"),
    ("fig6_turning_circle.png", "图6 用辨识线性导数跑出的 δ=35° 回转轨迹与首摇率收敛"),
]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> None:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = _post(
        f"{base}/auth/login",
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
    )["access_token"]

    result = json.loads((ART / "result.json").read_text())
    crit = result["criteria"]
    summary = (
        "KCS 斜航/舵/回转 13 例 D3Q19-LBM 扫描 → SNAME 无因次 → 最小二乘辨识 MMG 导数；"
        f"几何 Cb 误差 {100*result['geometry']['cb_rel_error']:.2f}%，"
        f"合成回代 {result['synthetic_roundtrip']['max_rel_error_noiseless']:.2e}，"
        f"vs SIMMAN 实验精度判据 WITHHELD（Re_lat={result['lattice_setup']['re_lattice']:.0f}）"
    )
    figures = []
    for name, caption in FIGURES:
        path = ART / name
        if not path.exists():
            continue
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    payload = {
        "title": "IN59-N02 KCS 斜航/回转 LBM 力 → MMG 水动力导数辨识闭环（TensorLBM）",
        "summary": summary,
        "scenario_key": "maneuvering_mmg_cfd_identification",
        "mission_id": MISSION_ID,
        "tags": [
            "maneuvering", "MMG", "KCS", "LBM", "system-identification",
            "TensorLBM", "IN59-N02", "honest-FAIL",
        ],
        "markdown": DOC.read_text(),
        "figures": figures,
        "metadata": {
            "criteria": {k: v.get("pass") for k, v in crit.items()},
            "verdict": result["verdict"],
            "result_json": "artifacts/maneuvering_kcs_cfd/result.json",
            "derivatives_prime": result["derivatives_prime"],
        },
    }
    out = _post(f"{base}/tensor/reports", payload, token)
    print(json.dumps({k: out.get(k) for k in ("id", "title", "mission_id")}, ensure_ascii=False))


if __name__ == "__main__":
    main()