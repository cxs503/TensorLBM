#!/usr/bin/env python3
"""Regenerate ``summary.json`` (and a compact ``aggregates.json``) from a raw
campaign ``results.json`` - no physics is recomputed, so the summary is a pure
function of the recorded results (audit chain: results -> summary).

    python scripts/summarize_lbm_von_neumann_campaign.py \
        --results artifacts/lbm_von_neumann_p155_n08/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.stability.campaign import build_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None, help="output directory (default: alongside results)")
    ap.add_argument(
        "--supplement",
        default=None,
        help="optional results_supplement.json to merge into the criteria block",
    )
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text())
    sup_path = Path(args.supplement) if args.supplement else (
        Path(args.results).parent / "results_supplement.json"
    )
    if sup_path.exists():
        sup = json.loads(sup_path.read_text())
        results["criteria"].update(sup.get("criterion_block", {}))
    summary = build_summary(results)
    out = Path(args.out) if args.out else Path(args.results).parent
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "C5"}, indent=2)[:4000])
    print("... wrote", out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
