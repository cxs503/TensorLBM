#!/usr/bin/env python3
"""Publish the m18462 progress report to the platform (paper-grade, 6 sections).

Reads the report markdown from docs/ and embeds the four committed figures as
base64 PNGs, so the platform copy is generated from artefacts (no hand copy).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_r2_selection_bptt_tgv"
DOC = ROOT / "docs" / "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR.md"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

CAPTIONS = {
    "fig1_objective_landscapes.png": "Training loss vs the sub-grid constant for both closures and all three objectives; dashed lines are the BPTT-selected constants (flat objective = unidentifiable theta).",
    "fig2_holdout_ke_vs_theta.png": "Held-out kinetic-energy error vs theta (25-point sweep) with the no-model and OSS-as-shipped markers and the 15% owner bar.",
    "fig3_bptt_trajectories.png": "BPTT trajectories of theta for the 12 arms (two starts each), showing clamp-bound integral objectives.",
    "fig4_holdout_metrics.png": "Held-out metrics (field / KE / enstrophy) per (closure, objective) combination.",
}


def req(path, data=None, token=None, method=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(BASE + path, data=body,
                               method=method or ("POST" if body is not None else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.loads(f.read().decode())


def main() -> None:
    tok = req("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
    bench = json.loads((ART / "benchmark.json").read_text())
    figs = []
    for name, cap in CAPTIONS.items():
        p = ART / name
        if not p.is_file():
            continue
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR (m18462): exact-TGV anchored BPTT closure/objective selection",
        "summary": ("Round-2 cross-benchmark of the OSS graph-unroll/BPTT mechanism: the fine-grid "
                    "reference is now anchored to the exact 2-D Taylor-Green solution (0.237 % KE, "
                    "0.032 % at 256^2); BPTT selection gradient 0.007 % vs FD, held-out KE 0.561 % "
                    "(<15 % bar). Verdict A2-with-a-caveat: the sub-grid constant is not identifiable, "
                    "the honest selection is theta=0."),
        "scenario_key": "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR",
        "mission_id": 18462,
        "tags": ["TensorLBM", "XVAL", "BPTT", "TaylorGreen", "LES", "selection", "open-source"],
        "markdown": DOC.read_text(),
        "figures": figs,
        "metadata": {"benchmark_all_pass": bench.get("all_pass"),
                     "benchmark_rows": len(bench.get("rows", [])),
                     "repro_cmd": bench.get("repro_cmd")},
    }
    out = req("/tensor/reports", payload, token=token)
    print(json.dumps(out, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()