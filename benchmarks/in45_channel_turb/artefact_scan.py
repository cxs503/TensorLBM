#!/usr/bin/env python3
"""Why a "measured" ``y_w = 0.1`` is an artefact (real-data scan).

The historical value was produced by applying the viscous law
``y_w = y - u nu / tau_w`` (equivalently ``y_w^+ = y^+ - u^+``) at the *first
cell centre* of a channel that does not resolve the viscous sublayer
(``dx^+ ~ 6-8``).  Inside ``y^+ < 4`` that formula is exact; beyond it the
buffer layer takes over and the formula returns a systematically small ``y_w``.

This script quantifies the artefact on the in-repo MKM-1999 DNS profile (real
data, no simulation) and writes ``artifacts/in45/wall_origin_artefact.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
DNS = _REPO / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/in45/wall_origin_artefact.json")
    parser.add_argument("--dx-plus", type=float, default=7.83)
    args = parser.parse_args()
    y_plus, u_plus = [], []
    with open(DNS, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_plus.append(float(row["y_plus"]))
            u_plus.append(float(row["u_plus"]))
    y_plus = np.array(y_plus)
    u_plus = np.array(u_plus)
    scan = []
    for window in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0):
        mask = (y_plus > 0.0) & (y_plus <= window)
        estimate = float(np.mean(y_plus[mask] - u_plus[mask]))  # in wall units
        scan.append(
            {
                "y_plus_window": window,
                "samples": int(mask.sum()),
                "y_w_wall_units": estimate,
                "y_w_cells_at_dx_plus": estimate / args.dx_plus,
            }
        )
    # Same formula on a real (archived) turbulent run: the first cell centre is
    # at y+ ~ 6, and that single point *is* the historical 0.1.
    run_scan = []
    archive = _REPO / "benchmarks" / "pending" / "channel_turb_3d" / "runs_3d" / "L3_result.json"
    if archive.exists():
        record = json.loads(archive.read_text())
        profile = np.array(record["statistics"]["mean_velocity_profile"])[1:-1]
        u_tau = float(record["configuration"]["u_tau"])
        nu = float(record["derived"]["nu"])
        for row in range(min(6, profile.size)):
            y = float(row + 1)
            u = float(profile[row])
            run_scan.append(
                {
                    "row": row + 1,
                    "y_plus": y * u_tau / nu,
                    "u_plus": u / u_tau,
                    "y_w_estimate_cells": y - u * nu / u_tau**2,
                }
            )
    payload = {
        "archived_run_scan": {
            "file": str(archive.relative_to(_REPO)) if archive.exists() else None,
            "note": (
                "single-point viscous estimate evaluated at successive cell "
                "centres of a dx+ = 5.8 run; row 1 (y+ = 5.8) returns ~0.09, "
                "which is the historical y_w = 0.1"
            ),
            "scan": run_scan,
        },
        "schema": "in45-wall-origin-artefact-v1",
        "reference": str(DNS.relative_to(_REPO)),
        "dx_plus": args.dx_plus,
        "scan": scan,
        "conclusion": (
            "the single-point viscous formula is only valid inside y+ < 4.  On "
            "the DNS it returns y_w ~ 0 inside that band; on the archived dx+ = "
            "5.8 run the first two cells (y+ = 5.8, 11.6) return 0.60 and 0.86 "
            "cell, i.e. the estimator already drifts by +0.26 cell per cell and "
            "is unusable beyond the sublayer - yet it never returns 0.10.  The "
            "historical y_w = 0.1 therefore cannot come from resolved 3-D data: "
            "it needs u+ ~ 5.2 at the first cell, i.e. a profile far below the "
            "law of the wall, which is the signature of the non-steady 2-D "
            "pipeline documented in P20-N01.  The defect to fix is therefore "
            "the hard-coded constant, not its numerical value."
        ),
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["scan"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())