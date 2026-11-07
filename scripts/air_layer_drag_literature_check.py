#!/usr/bin/env python3
"""External benchmark (gate e): compare the simulated DR(C_q) with literature points.

Reads ``artifacts/air_layer_drag/results.json`` and
``references/air_lubrication_reference_points.json`` (sha256 anchored), then
reports the relative error of the simulated drag reduction at each reference
``C_q`` (linear interpolation on the simulated DR(C_q) curve).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def interpolate(curve, cq):
    pts = sorted((c, d) for c, d in curve if c is not None and d is not None)
    if not pts:
        return None
    if cq <= pts[0][0]:
        return pts[0][1]
    if cq >= pts[-1][0]:
        return pts[-1][1]
    for (c0, d0), (c1, d1) in zip(pts, pts[1:]):
        if c0 <= cq <= c1:
            w = (cq - c0) / (c1 - c0) if c1 > c0 else 0.0
            return d0 + w * (d1 - d0)
    return pts[-1][1]


def main() -> int:
    res = json.loads((ROOT / "artifacts/air_layer_drag/results.json").read_text())
    ref_path = ROOT / "references/air_lubrication_reference_points.json"
    raw = ref_path.read_bytes()
    refs = json.loads(raw.decode())
    curve = [(r["cq"], r["drag_reduction"]) for r in res["theta_sweep"]]
    curve += [(r["cq_inj"], r["drag_reduction"]) for r in res["qg_sweep"]]
    rows = []
    for p in refs["points"]:
        sim = interpolate(curve, p["cq"])
        if sim is None:
            rows.append({**p, "simulated_dr": None, "rel_error": None})
            continue
        rows.append({
            "id": p["id"],
            "cq": p["cq"],
            "reference_dr": p["drag_reduction"],
            "simulated_dr": sim,
            "rel_error": (sim - p["drag_reduction"]) / p["drag_reduction"],
            "within_15pct": bool(abs((sim - p["drag_reduction"]) / p["drag_reduction"]) <= 0.15),
            "confidence": p["confidence"],
        })
    out = {
        "reference_file": str(ref_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "points": rows,
        "all_within_15pct": all(r.get("within_15pct", False) for r in rows),
        "verdict": "WITHHELD",
        "verdict_reason": (
            "The reference points are recalled literature values without a "
            "retrievable machine-readable source in this sandbox, and the 2-D "
            "channel geometry cannot resolve the h_g/delta boundary-layer ratio of "
            "the experiments; gate (e) is therefore reported as WITHHELD rather "
            "than claimed as passed."
        ),
    }
    dest = ROOT / "artifacts/air_layer_drag/literature_check.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())