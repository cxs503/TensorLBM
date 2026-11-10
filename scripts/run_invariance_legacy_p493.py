#!/usr/bin/env python
"""Legacy (velocity/pressure driven) inclined-channel comparison, P493-N02.

The historical design of criterion C1 imposed the exact developed profile at
both ends and let the streamwise pressure gradient develop.  In a weakly
compressible lattice gas that gradient is a *density* gradient, which drags an
O(Delta rho) velocity deviation along with it.  This script measures the same
channel with both drives so the report can quantify why the certificate's C1
uses the zero-pressure-gradient (body force) drive.

    python scripts/run_invariance_legacy_p493.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hashlib  # noqa: E402
import time  # noqa: E402

from tensorlbm.invariance import observed_order, poiseuille_case  # noqa: E402

ART = ROOT / "artifacts" / "invariance" / "P493-N02"


def write_json(name: str, payload: dict) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / name
    blob = json.dumps(payload, indent=2, default=float)
    path.write_text(blob)
    (ART / (name + ".sha256")).write_text(hashlib.sha256(blob.encode()).hexdigest() + "\n")
    print(f"[write] {path.relative_to(ROOT)}", flush=True)
    return path
WIDTHS = (8, 16)
ANGLES = (0.0, 45.0)


def main() -> int:
    rows = []
    for theta in ANGLES:
        for width in WIDTHS:
            for drive in ("body_force", "pressure"):
                res = poiseuille_case(theta, width, u_mean=0.01, nu=0.1,
                                      steps_factor=30.0, drive=drive)
                rows.append(res)
                print(f"theta={theta:g} H={width} drive={drive:11s} "
                      f"l2={res['l2_error']:.3e} drho={res['density']['max_abs']:.2e}",
                      flush=True)
    summary = {}
    for theta in ANGLES:
        for drive in ("body_force", "pressure"):
            sel = [r for r in rows if r["theta_deg"] == theta and r["drive"] == drive]
            sel.sort(key=lambda r: r["width"])
            summary[f"{theta:g}|{drive}"] = {
                "widths": [r["width"] for r in sel],
                "errors": [r["l2_error"] for r in sel],
                "drho": [r["density"]["max_abs"] for r in sel],
                "orders": observed_order([r["l2_error"] for r in sel],
                                         [r["width"] for r in sel]),
                "runs": sel,
            }
    write_json("poiseuille_legacy.json", {
        "provenance": {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "comparison": summary,
                                          "note": "drive=pressure reproduces the "
                                                  "P170W2-N08 design"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())