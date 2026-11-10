"""Publish the P399W2-N04 report to the Tensor platform.

Logs in with ``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``, embeds the rendered
markdown and the generated figures (base64 PNG), and posts to
``$TENSOR_API_BASE/tensor/reports``.

    python scripts/p399w2_n04_curved_force/post_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/P399W2_N04_IBM_CURVED_FORCE.md"
FIG = ROOT / "figures/p399w2_n04"
ART = ROOT / "artifacts/p399w2_n04"

MISSION_ID = 10703
TITLE = (
    "P399W2-N04 曲面边界力法：浸没边界曲面力精确积分 + 动量/质量守恒闭环（TensorLBM）"
)
SUMMARY = (
    "在 IBM 框架内实现曲面边界力的精确积分（Gauss–Legendre 精确测度 + traction 重构 + "
    "法向投影质量闭合 + 预算守恒修正），以圆柱绕流验证：动量守恒误差、质量泄漏率、"
    "Re=100 阻力系数对标文献，并给出 Hasimoto 周期阵列 Stokes 解析锚点。"
)

FIGURES = [
    ("fig1_surface_quadrature.png", "Fig.1 曲面离散测度精确性与非均匀标记点的求积偏差"),
    ("fig2_cd_marker_convergence.png", "Fig.2 Re=100 阻力系数随标记点数收敛（求积 vs 旧法）"),
    ("fig3_conservation_audit.png", "Fig.3 逐算例守恒判据（动量/质量/预算闭合/局部不可穿透）"),
    ("fig4_vorticity_traction.png", "Fig.4 涡量场与壁面法向 traction 分布"),
    ("fig5_conservation_history.png", "Fig.5 阻力时程与质量守恒审计时程"),
    ("fig6_hasimoto_stokes.png", "Fig.6 Stokes 区阻力 vs Hasimoto 周期阵列解析解"),
]


def login(base: str) -> str:
    r = requests.post(
        f"{base}/auth/login",
        json={
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    tok = data.get("access_token") or data.get("token")
    if not tok:
        raise SystemExit(f"login returned no token: {data}")
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    markdown = DOC.read_text()
    figs = []
    for name, caption in FIGURES:
        p = FIG / name
        if p.exists():
            figs.append(
                {
                    "name": name,
                    "caption": caption,
                    "png_base64": base64.b64encode(p.read_bytes()).decode(),
                }
            )
    payload = {
        "title": TITLE,
        "summary": SUMMARY,
        "scenario_key": "lbm_curved_boundary_force",
        "mission_id": MISSION_ID,
        "tags": ["TensorLBM", "IBM", "curved-boundary", "force-integration", "conservation"],
        "markdown": markdown,
        "figures": figs,
    }
    if args.dry_run:
        out = ART / "report_payload.json"
        out.write_text(json.dumps({**payload, "figures": [f["name"] for f in figs]}, indent=2))
        print(f"[dry-run] {out} ({len(figs)} figures, {len(markdown)} chars)")
        return

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = login(base)
    r = requests.post(
        f"{base}/tensor/reports",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=300,
    )
    print(r.status_code, r.text[:500])
    r.raise_for_status()
    print("[report] published")


if __name__ == "__main__":
    sys.exit(main())