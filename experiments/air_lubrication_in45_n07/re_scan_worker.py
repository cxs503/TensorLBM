#!/usr/bin/env python3
"""IN45-N07 C7: 3 Re x 5 alpha lattice scan, run on the loong64 cluster.

Usage
-----
``python3 experiments/air_lubrication_in45_n07/re_scan_worker.py --out re_scan.json``

Writes ``{"cases": [...], "meta": {...}}`` and prints the JSON on stdout so the
result can also be recovered from the cluster job log.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.air_lubrication_frontier import (  # noqa: E402
    LAMBDA_LBM,
    RE_SCAN,
    re_scan_case,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="re_scan.json")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--ny", type=int, default=32)
    args = ap.parse_args()

    cases = []
    for re_val in RE_SCAN["re"]:
        for a in RE_SCAN["alpha_wall"]:
            case = re_scan_case(re_val, a, steps=args.steps, ny=args.ny)
            cases.append(case)
            print(
                f"Re={re_val:6.0f} alpha={a:4.2f} "
                f"dr_num={case['dr_num']:+.4f} dr_ref={case['dr_ref']:+.4f} "
                f"rel_err={case['dr_rel_err']:.3%}",
                flush=True,
            )
    payload = {
        "cases": cases,
        "meta": {
            "grid": {
                "re": list(RE_SCAN["re"]),
                "alpha_wall": list(RE_SCAN["alpha_wall"]),
            },
            "lambda_lbm": LAMBDA_LBM,
            "steps": args.steps,
            "ny": args.ny,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()