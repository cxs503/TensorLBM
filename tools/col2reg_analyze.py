#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-COL2-REG: reduce the SDAA arm results to the card table + figure.

Reads ``runs/<arm>/result.json`` for every arm, extracts the depth-peak
(depth_over_D, time_ms absolute and post-contact), the stop reason, the
contact step, the max lattice mach and the final volume drift, and compares
against the archived ctrl anchor and the Santini case-3 experimental peak.

Usage::

    python3 tools/col2reg_analyze.py --runs-dir runs --out artifacts/COL2-REG/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Anchors (17-digit, from the wave charter / DE_A1_FACTS; do not retype).
EXP_PEAK_D = 1.857434
EXP_PEAK_MS = 10.39446
CTRL_PEAK_D = 1.4352204748231814
CTRL_PEAK_MS_ABS = 11.090571428571428
CTRL_PEAK_MS_PC = 10.543339285714286
CTRL_CONTACT_STEP = 45
CTRL_WINDOW_MS = 24.005
CTRL_MACH = 0.2598
CTRL_DRIFT = 4.8e-6


def _peak(history):
    best = None
    for row in history:
        d = row.get("depth_over_D")
        if d is None:
            continue
        if best is None or d > best[0]:
            best = (d, row)
    if best is None:
        return None
    d, row = best
    return {
        "depth_over_D": d,
        "time_ms_abs": row["time_ms"],
        "step": row["step"],
    }


def reduce_arm(path: Path):
    res = json.loads(path.read_text())
    hist = res.get("history") or []
    pk = _peak(hist)
    contact = res.get("contact_step")
    out = {
        "arm": path.parent.name,
        "stop_reason": res.get("stop_reason"),
        "rr_tau": res.get("rr_tau"),
        "contact_step": contact,
        "peak": pk,
        "peak_time_ms_pc": (
            pk["time_ms_abs"] - contact * 1000 * res["scales"]["dt_s"]
            if pk and contact is not None
            else None
        ),
        "mach_max": max((r.get("mach_max") or 0.0) for r in hist) if hist else None,
        "volume_drift_final": hist[-1].get("volume_drift") if hist else None,
        "steps": len(hist),
    }
    if pk and contact is not None:
        out["peak_err_vs_ctrl_pct"] = 100.0 * (pk["depth_over_D"] - CTRL_PEAK_D) / CTRL_PEAK_D
        out["peak_err_vs_exp_pct"] = 100.0 * (pk["depth_over_D"] - EXP_PEAK_D) / EXP_PEAK_D
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--arms", default="rr_base,rr_t49,rr_t48,rr_t45,rr_t42,ctrl_bgk")
    ap.add_argument("--out", default="artifacts/COL2-REG/summary.json")
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    summary = {"arms": [], "missing": []}
    for arm in [s for s in a.arms.split(",") if s]:
        p = runs / arm / "result.json"
        if not p.exists():
            summary["missing"].append(arm)
            continue
        summary["arms"].append(reduce_arm(p))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
