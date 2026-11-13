#!/usr/bin/env python
"""Run ONE XVAL cell and append the record to a JSONL file.

Fallback driver for the case where the cluster queue is deep: each invocation
fits comfortably inside a single short shell slot, and the campaign can be
resumed cell by cell (the JSONL is append-only and re-runs of the same tag are
recorded, never overwritten).

Usage:
    python run_one.py --tag planar_L64_ratio10 --geometry planar --L 64 \
        --tr 0.8266 --steps 20000 --threads 4 --out artifacts/cells.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xval_driver as X  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--geometry", default="planar", choices=["planar", "droplet"])
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--tr", type=float, default=0.8266)
    ap.add_argument("--tier", default="ratio10")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "artifacts" / "cells.jsonl"))
    a = ap.parse_args()

    mw = X.P.maxwell_at_tr("pr", a.tr, G=1.0)
    rec = X.run_cell(a.tag, mw, geometry=a.geometry, L=a.L, R=a.R, steps=a.steps,
                     threads=a.threads, sample=a.sample, T_r=a.tr)
    rec["reference_tier"] = a.tier
    rec["errors"] = X.errors_against_maxwell(rec, mw) if rec.get("rho_l") else None
    if rec.get("rho_l") and a.geometry == "planar":
        rec["eos_readout"] = X.eos_readout(rec, mw)
    rec["maxwell"] = {k: mw[k] for k in ("T_r", "T", "rho_liquid", "rho_vapour", "ratio",
                                        "p_sat", "rho_cross") if k in mw}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    e = rec["errors"] or {}
    print(f"{a.tag}: stable={rec['stable']} ratio={rec.get('ratio')} "
          f"err={e.get('worst')} ma={rec.get('ma_max')} wall={rec['wall_s']}s -> {out}")


if __name__ == "__main__":
    main()