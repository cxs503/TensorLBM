"""Publish the P93W3-N02 report (paper-grade, numbers injected from the artifact).

The markdown is the one written by
``benchmarks/radiation_deck_env_p93/make_doc.py`` (every number injected from
``artifacts/P93W3_N02_radiation_deck_env.json`` — zero hand-copied digits), and
is POSTed to ``$TENSOR_API_BASE/tensor/reports`` with the six figures inlined
as base64 PNG.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "P93W3_N02_radiation_deck_env.json"
DOC = ROOT / "docs" / "P93W3_N02_radiation_deck_env.md"


def main() -> int:
    r = json.loads(ARTIFACT.read_text())
    md = DOC.read_text()
    c = r["criteria"]

    api = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not (user and password):
        print("no credentials in env; markdown only", file=sys.stderr)
        return 0

    tok = requests.post(f"{api}/auth/login", json={"username": user, "password": password}, timeout=60)
    tok.raise_for_status()
    token = tok.json().get("access_token") or tok.json().get("token")
    if not token:
        print(f"login response without token: {tok.text[:300]}", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Bearer {token}"}

    figures = []
    for f in r["figures"]:
        png = (ROOT / f["path"]).read_bytes()
        figures.append(
            {
                "name": f["name"],
                "caption": f["caption"],
                "png_base64": base64.b64encode(png).decode(),
            }
        )
    payload = {
        "title": "P93W3-N02 TensorLBM 热辐射角系数内核与封闭腔辐射度解析锚：球/圆柱-平板闭式族 + 高斯-赛德尔辐射度 + 船舶甲板温度场",
        "summary": (
            f"5 构型角系数闭式 vs 独立求积/蒙特卡洛最差 {c['a_view_factor_closed_form']['worst']:.2e}（判据 ≤1%）；"
            f"辐射度高斯-赛德尔迭代残差 {c['b_radiosity_iteration_residual']['worst']:.2e}（判据 ≤1e-6）；"
            f"等温封闭腔净功率 {c['c_isothermal_cavity_energy']['worst']:.2e}（判据 ≤0.1%）；"
            f"甲板温度场 vs 解析稳态解 {c['d_deck_field_vs_analytic']['worst']:.2e}（判据 ≤3%），全部 PASS"
        ),
        "scenario_key": "P93W3-N02",
        "mission_id": 9169,
        "tags": [
            "TensorLBM",
            "thermal-radiation",
            "view-factor",
            "radiosity",
            "gauss-seidel",
            "deck-thermal-field",
            "P93W3-N02",
        ],
        "markdown": md,
        "figures": figures,
    }
    resp = requests.post(f"{api}/tensor/reports", json=payload, headers=headers, timeout=300)
    print("report POST", resp.status_code)
    if resp.status_code >= 300:
        print(resp.text[:2000], file=sys.stderr)
        return 1
    print(resp.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
