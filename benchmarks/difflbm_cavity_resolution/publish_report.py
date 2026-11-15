#!/usr/bin/env python3
"""Publish the GAP-DIFFLBM-CAV-RES report to the platform.

``make_report.py`` writes ``artifacts/gap_difflbm_cav_res/report_payload.json``
(title / markdown / figures).  This script adds the dispatch metadata
(mission id, scenario key, tags, summary line) and POSTs it to
``$TENSOR_API_BASE/tensor/reports`` with agent credentials taken from the
environment (``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``).  Nothing is
hard-coded: without the environment variables the script refuses to run.

Usage
-----
    PYTHONPATH=.:src:benchmarks python3 benchmarks/difflbm_cavity_resolution/publish_report.py \
        [--mission-id 21807] [--dry-run]

``--dry-run`` prints the payload size and the first lines of the markdown so
the content can be inspected before publishing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "gap_difflbm_cav_res"
PAYLOAD = ART / "report_payload.json"
SUMMARY = ART / "summary.json"

SCENARIO_KEY = "lbm.cavity.resolution-convergence"
TAGS = ["TensorLBM", "GAP-DIFFLBM-CAV-RES", "lid-driven-cavity", "Ghia1982",
        "grid-convergence", "observed-order", "Richardson", "verification"]


def _request(url: str, data: dict | None = None, token: str | None = None) -> dict:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as fh:
        return json.loads(fh.read().decode())


def build_payload(mission_id: int) -> dict:
    payload = json.loads(PAYLOAD.read_text())
    summary = json.loads(SUMMARY.read_text())
    verdict = []
    for re_key, blk in sorted(summary["by_re"].items()):
        crit = blk["criteria"]
        verdict.append(
            f"Re={re_key}: max-point " +
            "/".join(f"{v:.2f}%" for v in blk["max_abs_dev_pct"]) +
            f" @n={'/'.join(str(n) for n in blk['nx'])}; " +
            f"p_self_l2="
            f"{(blk['richardson'].get('vertical_centreline_u', {}).get('p') or {}).get('l2')}"
            f"; all_criteria_met={crit['all_criteria_met']}")
    payload["mission_id"] = mission_id
    payload["scenario_key"] = SCENARIO_KEY
    payload["tags"] = TAGS
    payload["summary"] = ("方腔锚（Ghia 1982, Re=100/400）网格分辨率收敛研究："
                          "n=33…129 扫描、max-point 误差与 L2 误差双对数曲线、"
                          "Richardson r=2 自收敛观测阶；判据 max-point<3% (n≥97) 与 p≥1.8。"
                          + " | ".join(verdict))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission-id", type=int,
                    default=int(os.environ.get("MISSION_ID", "21807")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = build_payload(args.mission_id)
    md = payload["markdown"]
    print(f"payload: {len(md)} chars markdown, "
          f"{len(payload.get('figures', []))} figures, title={payload['title']!r}")
    if args.dry_run:
        print("--- markdown head ---")
        print("\n".join(md.splitlines()[:25]))
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    pw = os.environ["TENSOR_AGENT_PASS"]
    tok = _request(f"{base}/auth/login", {"username": user, "password": pw})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        print("login failed:", list(tok)[:8], file=sys.stderr)
        return 1
    try:
        res = _request(f"{base}/tensor/reports", payload, token)
    except urllib.error.HTTPError as exc:
        print("POST failed", exc.code, exc.read().decode()[:800], file=sys.stderr)
        return 1
    print("posted:", json.dumps(res)[:400])
    rid = res.get("id") or res.get("report_id")
    print(f"report id = {rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())