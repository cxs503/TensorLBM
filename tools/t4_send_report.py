#!/usr/bin/env python3
"""Post the T4 progress report to the platform report API.

Logs in with ``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS`` against
``$TENSOR_API_BASE``, then POSTs ``/tensor/reports`` with the rendered
markdown (``docs/T4_PHYSICS_AI_LOOP.md``) and the figures as base64 PNGs.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION = "15658"


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def login() -> str:
    for path, payload in (
        ("/auth/login", {"username": USER, "password": PASS}),
        ("/auth/token", {"username": USER, "password": PASS}),
        ("/login", {"username": USER, "password": PASS}),
    ):
        try:
            out = _post(path, payload)
        except urllib.error.HTTPError as exc:
            print(f"{path} -> HTTP {exc.code}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{path} -> {exc}")
            continue
        token = out.get("access_token") or out.get("token") or out.get("data", {}).get(
            "access_token"
        )
        if token:
            print(f"login ok via {path}")
            return token
    raise SystemExit("login failed")


def main() -> int:
    token = login()
    md = (ROOT / "docs/T4_PHYSICS_AI_LOOP.md").read_text()
    figures = []
    for png in sorted((ROOT / "artifacts/t4_physics_ai_loop/figures").glob("*.png")):
        figures.append(
            {
                "name": png.stem,
                "caption": {
                    "fig1_producer_verification": "生产端认证：单模 Taylor-Green 衰减 u_rms(t) 与解析 NS 解对比（左）及相对误差（右）",
                    "fig2_error_vs_re": "误差-Re：单步（左）与自回归（右）相对 L2，虚线=留出 Re",
                    "fig3_rollout_re75": "留出 Re=75 自回归曲线：FNO vs 精确黏性先验 vs persistence",
                    "fig3_rollout_re300": "留出 Re=300 自回归曲线：FNO vs 精确黏性先验 vs persistence",
                    "fig4_field_maps": "留出工况 |u| 云图：真 LBM / FNO 两步滚动 / 黏性先验 / 误差场",
                    "fig5_feedback": "反哺 LBM：交回误差与纯代理误差随替代步数 k 的变化",
                }.get(png.stem, png.stem),
                "png_base64": base64.b64encode(png.read_bytes()).decode(),
            }
        )
    payload = {
        "title": "T4 物理-AI 闭环：真 LBM 流场生产 → 资产 → FNO（留出 Re）→ 反哺求解器",
        "summary": "自研 D2Q9 LBM 自动生产 24 个工况/504 个真流场快照并登记 TD 侧 catalog；"
        "FNO 在训练未见过的 Re 上做流映射与自回归，对精确黏性先验与 persistence 双基线打分；"
        "代理作为时间粗步进器交回真求解器，量化交回误差与 healing。生产端用单模 Taylor-Green "
        "解析 NS 解认证（u_rms 误差 0.20%，衰减率误差 0.04%）。",
        "scenario_key": "GAP-LBM-T4-PHYSICS-AI-LOOP",
        "mission_id": MISSION,
        "tags": ["TensorLBM", "T4", "physics-AI-loop", "FNO", "real-LBM-data", "holdout-Re"],
        "markdown": md,
        "figures": figures,
    }
    out = _post("/tensor/reports", payload, token)
    print(json.dumps(out)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())