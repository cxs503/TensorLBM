#!/usr/bin/env python3
"""Publish the IN13W2-N06 progress report to the platform report store.

Reads the judged numbers (`artifacts/icebow_lbm_r2/report.json`), the generated
markdown and the base64 figures (`figures.json`) and posts one report bound to
the node's mission id.  Kept in the repo so the publication step is auditable
and repeatable instead of a one-off shell command.

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \\
        python3 scripts/publish_icebow_report.py --dir artifacts/icebow_lbm_r2
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict, token: str | None = None) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as fh:
            return fh.status, fh.read().decode()
    except Exception as exc:  # noqa: BLE001 - surface the server body
        body = ""
        if hasattr(exc, "read"):
            try:
                body = exc.read().decode()[:600]
            except Exception:  # noqa: BLE001
                body = ""
        return getattr(exc, "code", 0) or 0, f"{exc} {body}"[:600]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/icebow_lbm_r2")
    ap.add_argument("--doc", default="docs/icebow_bow_resistance_lbm.md")
    ap.add_argument("--mission", type=int, default=11301)
    ap.add_argument("--scenario-key", default="ship_ice_bow_resistance")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    judged = json.loads((d / "report.json").read_text())
    figures = json.loads((d / "figures.json").read_text()) if (d / "figures.json").exists() else []
    markdown = Path(args.doc).read_text()
    ch = judged["checks"]
    title = "破冰艏部船型族：几何先验证 → SDF 一致性 → LBM 黏性阻力 × GCI（IN13W2-N06）"
    summary = (
        f"几何闸门 {len(ch.get('alpha_angles_deg') or [])} 档艏角全 PASS；SDF 三档体素体积误差 "
        f"{ch['sdf_worst_volume_error_percent']:.3f}%；LBM 最细网格 GCI "
        f"{ch['lbm_gci_fine_percent']:.2f}%；球体 Cd 基准误差 {ch['sphere_error_percent']:+.2f}%；"
        f"艏角敏感度合成幅值 {100 * (ch.get('hybrid_max_abs_delta_rt') or 0):.2f}%。"
    )
    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": args.scenario_key,
        "mission_id": args.mission,
        "tags": ["TensorLBM", "ice-class", "bow-geometry", "LBM", "GCI", "resistance", "IN13W2-N06"],
        "markdown": markdown,
        "figures": figures,
    }
    if args.dry_run:
        print(json.dumps({"title": title, "summary": summary, "figures": [f["name"] for f in figures]}, indent=1, ensure_ascii=False))
        return 0
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    status, body = _post(
        f"{base}/auth/login",
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]},
    )
    if status != 200:
        print("login failed", status, body)
        return 1
    token = json.loads(body)["access_token"]
    status, body = _post(f"{base}/tensor/reports", payload, token)
    print("report", status, body[:400])
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())