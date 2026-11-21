#!/usr/bin/env python3
"""K3-MOMCONV git-tracked artefacts (trimmed summaries of the arm runs).

The raw SDAA artefacts stay in ``runs/k3_*/`` (git-ignored, as the other
campaign arms do).  This script writes the trimmed, reviewable summaries to
``artifacts/k3_momconv/`` so the numbers of the card are traceable in git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("ctrl", "inherit", "redistribute", "inherit_fix", "redistribute_liq")
ROOT = Path(__file__).resolve().parents[1]


def summarise(result: dict, stride: int = 10) -> dict:
    history = result.get("history") or []
    peak = None
    for row in history:
        value = row.get("depth_over_D")
        if value is None:
            continue
        if peak is None or value > peak.get("depth_over_D", -1.0):
            peak = row
    rows = (result.get("budget_diag") or {}).get("rows") or []
    keys = ("step", "time_ms", "ke_liq", "ke_axial", "ke_lat", "ke_drop", "conv_loss",
            "conv_gain", "p_liq_z", "p_drop_z", "area_total", "cav_volume", "diss_rate")
    budget = [
        {key: row.get(key) for key in keys}
        for row in rows
        if row["step"] % stride == 0 or (peak and row["step"] == peak["step"])
    ]
    return {
        "solver": result.get("solver"),
        "case_id": result.get("case_id"),
        "cells_per_D": result.get("cells_per_D"),
        "gap_over_D": result.get("gap_over_D"),
        "lattice_speed": result.get("lattice_speed"),
        "scales": result.get("scales"),
        "momconv": result.get("momconv", "off"),
        "stop_reason": result.get("stop_reason"),
        "contact_step": result.get("contact_step"),
        "u_limit": result.get("u_limit"),
        "peak_row": peak,
        "last_row": history[-1] if history else None,
        "history_stride": stride,
        "history": [
            {k: row.get(k) for k in ("step", "time_ms", "depth_over_D", "mach_max",
                                     "rho_min", "volume_drift", "geometric_volume_drift",
                                     "represented_mass_drift")}
            for row in history
            if row["step"] % stride == 0 or (peak and row["step"] == peak["step"])
        ],
        "budget_stride": stride,
        "budget_rows": budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default=str(ROOT))
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    out_dir = tree / "artifacts" / "k3_momconv"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm in ARMS:
        path = tree / "runs" / f"k3_{arm}" / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        out = out_dir / f"k3_{arm}_summary.json"
        out.write_text(json.dumps(summarise(result), indent=1) + "\n")
        written.append(out.name)
    verdict = tree / "runs" / "k3_momconv_verdict" / "verdict.json"
    if verdict.exists():
        (out_dir / "verdict.json").write_text(verdict.read_text())
        written.append("verdict.json")
    print("artifacts written:", out_dir, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
