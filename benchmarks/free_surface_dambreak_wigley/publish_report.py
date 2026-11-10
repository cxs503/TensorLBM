"""Publish the p351w1_n07 progress report to the platform report API."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
ROOT = Path(__file__).resolve().parents[2]


def post(path: str, payload: dict, token: str = "") -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE.rstrip("/") + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token", "")
    md = (ROOT / "docs" / "free_surface_dambreak_vof_report.md").read_text()
    figures = []
    for name, caption in [
        ("fig_front_vs_reference.png", "波前位置 vs Martin & Moyce (1952) 14 点实验表（两套壁面工况）"),
        ("fig_error_vs_reference.png", "逐点波前误差 %，绿带为 ±10% 验收带"),
        ("fig_mass_ledger.png", "各工况全程质量漂移与 2% 门（结构性守恒，未启用全局修正）"),
        ("fig_convergence.png", "界面输运 L2 收敛（含 1.5 阶目标斜率与拟合 R²）"),
        ("fig_fields.png", "6 个时刻 VOF 液分数（上）与压力场（下），真实运行快照"),
    ]:
        path = ROOT / "figures" / name
        if not path.exists():
            continue
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})
    payload = {
        "title": "自由面算法独立验证：2D NS+VOF 溃坝对 Martin & Moyce (1952)（RMSE 7.84%，质量 0.12%，收敛阶 0.94 honest-FAIL）",
        "summary": (
            "TensorLBM 新增 src/tensorlbm/free_surface.py：交错 MAC 不可压两相 NS + 守恒 VOF"
            "（superbee/MUSCL）+ 变密度多重网格投影 + 质量账本。溃坝 H/a=2 波前对 M&M 1952 14 点"
            "相对 RMSE 7.84%（判据 ≤10% PASS），全程质量漂移 0.12%（≤2% PASS），"
            "界面输运 L2 收敛阶 0.94（R²=0.997，目标 ≥1.5 → honest FAIL，诊断=界面增厚主导）；"
            "Wigley 兴波判据未交付（2D 剖面无法表示 Kelvin 兴波，理由与路线已写入报告）；"
            "surface tension 已实现但未验证（虚假毛细流）。pytest 28 项含负控。"
        ),
        "scenario_key": "p351w1_n07",
        "mission_id": 10726,
        "tags": ["TensorLBM", "free-surface", "VOF", "dam-break", "Martin&Moyce",
                 "verification", "honest-FAIL", "p351w1_n07"],
        "markdown": md,
        "figures": figures,
    }
    res = post("/tensor/reports", payload, token)
    print(json.dumps(res)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())