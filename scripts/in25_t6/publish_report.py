#!/usr/bin/env python3
"""Publish the IN25-T6 mission report to the platform API.

Figures are downscaled before base64 encoding: the report endpoint silently
fails above ~250 KB of base64 payload (known platform trap), so each PNG is
resized/optimised to a fixed budget first.

Token: ``TENSOR_REPORT_TOKEN`` env var, else ``/tmp/tok.txt`` written by the
login helper.  Credentials are never embedded here.

Usage
-----
    python scripts/in25_t6/publish_report.py            # POST
    python scripts/in25_t6/publish_report.py --dry-run  # size check only
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "survey" / "figs"

FIG_CAPTIONS = (
    "图 1 SUBOFF 阻力锚车道 vs ITTC-1957：AFF-1/AFF-8 六点阶梯、湿面积区间带、"
    "公开锚 0.0031 复现（+1.46%）与常引 0.004 的落空位置。",
    "图 2 ITTC-1957 与 Schoenherr 的相对差随 Re：Re≥1e7 内最大 2.61%，落在 ±3% 门内；"
    "低 Re 发散是相关线/摩擦线的构造性差异。",
    "图 3 约定敏感性：湿面积 ±5.8%、密度 −2.6%，任一组合都够不到 0.004。",
)


def load_doc():
    spec = importlib.util.spec_from_file_location(
        "survey_doc", ROOT / "src/tensorlbm/external_survey/survey_doc.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(mod)
    return mod


def shrink(name: str, width: int = 820, budget_kb: int = 42) -> tuple[str, str]:
    from PIL import Image

    img = Image.open(FIGS / name).convert("RGB")
    buf = io.BytesIO()
    for scale in (1.0, 0.85, 0.7, 0.55, 0.45):
        w = int(width * scale)
        h = int(img.height * w / img.width)
        buf = io.BytesIO()
        small = img.resize((w, h), Image.LANCZOS)
        small.quantize(colors=160, method=Image.MEDIANCUT).save(buf, format="PNG", optimize=True)
        if buf.tell() / 1024 <= budget_kb:
            break
    return name, base64.b64encode(buf.getvalue()).decode()


def build_body() -> dict:
    doc = load_doc()
    md = doc.survey_markdown()
    # keep the platform report readable: drop the long automation appendix and
    # keep everything that carries a number/criterion
    figs = [
        shrink("in25_t6_ct_ladder.png"),
        shrink("in25_t6_friction_lines.png"),
        shrink("in25_t6_convention_sensitivity.png"),
    ]
    return {
        "title": "IN25-T6 外部成熟技术调研：五措施吸收 + SUBOFF 第二锚 + 五条 PLAYBOOK",
        "summary": (
            "三席真跑的外部成熟技术调研：16 条带离线可读出处的文献锚覆盖五措施；由 SUBOFF Table 14 "
            "重建独立阻力阶梯，AFF-1 复现公开锚 0.0031（+1.46%），常引 AFF-8 0.004 因需湿面积 "
            "5.50 m²（小于裸体）而判不可复现并剔除出打分器；五条 PLAYBOOK 与 24 项 pytest "
            "契约测试落库 TensorLBM。"
        ),
        "scenario_key": "tensor_lbm_survey",
        "mission_id": 5219,
        "tags": [
            "IN25-T6", "survey", "SUBOFF", "literature-anchors", "TensorLBM",
            "five-measures", "honest-fail",
        ],
        "markdown": md,
        "figures": [
            {"name": name, "caption": cap, "png_base64": b64}
            for (name, b64), cap in zip(figs, FIG_CAPTIONS)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="GET back the stored report")
    args = ap.parse_args()

    body = build_body()
    data = json.dumps(body).encode()
    print(f"payload {len(data)/1024:.1f} KB "
          f"(figures {sum(len(f['png_base64']) for f in body['figures'])/1024:.1f} KB b64)")
    if args.dry_run:
        return 0
    if len(data) / 1024 > 250:
        print("refusing: payload above the 250 KB platform limit")
        return 2

    token = os.environ.get("TENSOR_REPORT_TOKEN") or Path("/tmp/tok.txt").read_text().strip()
    base = os.environ["TENSOR_API_BASE"]
    req = urllib.request.Request(f"{base}/tensor/reports", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            print("POST", resp.status, resp.read()[:300].decode())
    except Exception as exc:
        print("POST failed:", exc, getattr(exc, "read", lambda: b"")()[:500])
        return 1

    if args.verify:
        vreq = urllib.request.Request(f"{base}/tensor/reports?mission_id=5219")
        vreq.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(vreq, timeout=60) as resp:
                payload = json.loads(resp.read())
            rows = payload if isinstance(payload, list) else payload.get("items", payload)
            print("stored reports:", len(rows) if hasattr(rows, "__len__") else rows)
        except Exception as exc:
            print("verify failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())