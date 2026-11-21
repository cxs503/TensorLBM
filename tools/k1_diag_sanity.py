#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke check for the K1-DIAG-B diagnosis: independent depth measurement must
reproduce the benchmark's own ``depth_over_D`` bit-for-bit.

Input : runs/k1diag_smoke60/result.json (60-step SDAA smoke arm, --diag-cavity)
Output: runs/k1diag_smoke60/sanity.json
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "runs", "k1diag_smoke60", "result.json")
result = json.load(open(path))
history = {row["step"]: row for row in result["history"]}
rows = result["cavity_diag"]["rows"]
worst = 0.0
for row in rows:
    a = row["d_cav_open"]
    b = history[row["step"]]["depth_over_D"]
    worst = max(worst, abs(a - b))
out = {
    "arm": "k1diag_smoke60",
    "steps": result["history"][-1]["step"],
    "stop_reason": result["stop_reason"],
    "contact_step": result["contact_step"],
    "impact_speed_m_s": result["impact_speed_m_s"],
    "diag_rows": len(rows),
    "max_abs_delta_d_cav_open_vs_depth_over_D": worst,
    "d_cav_open_reproduces_benchmark_depth": worst == 0.0,
    "note": "cavity_diag.d_cav_open is an independent re-implementation of run.open_depth; "
            "bit-identical on every recorded step proves the diagnosis measures the same quantity.",
}
with open(os.path.join(ROOT, "runs", "k1diag_smoke60", "sanity.json"), "w") as handle:
    json.dump(out, handle, indent=1)
print(json.dumps(out, indent=1))
