#!/usr/bin/env python3
"""Round-2 entry point for XVAL-LBM-ROUND2-METHOD-SELECTION.

This is the *entry* of the round-2 method-selection cross-validation: it runs
the predeclared arm x resolution matrix, aggregates results, and produces the
summary + figures.

Usage
-----
  # smoke (control-node budget, < 2 min): A1 + A6 at D=6
  python xval/benchmark_method_selection/run_round2_entry.py --smoke

  # rung D=10 (all arms): ~15 min on 8 cores
  python xval/benchmark_method_selection/run_round2_entry.py --rung 10

  # rung D=14 (all arms): ~30 min on 8 cores
  python xval/benchmark_method_selection/run_round2_entry.py --rung 14

  # rung D=20 (all arms): ~60 min on 8 cores
  python xval/benchmark_method_selection/run_round2_entry.py --rung 20

  # analyze + report (after all rungs finished)
  python xval/benchmark_method_selection/run_round2_entry.py --analyze

  # full pipeline (smoke → rung 10 → analyze)
  python xval/benchmark_method_selection/run_round2_entry.py --full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
XVAL = REPO / "xval" / "benchmark_method_selection"
sys.path.insert(0, str(XVAL))
sys.path.insert(0, str(REPO / "src"))

from run_arm import ARMS, run_arm  # noqa: E402

SMOKE_ARMS = ("A1", "A6")
SMOKE_D = 6
SMOKE_STEPS = 1200

# Round-2 arm set (excludes A3/A5 which are known to blow up at low lattice viscosity)
ROUND2_ARMS = ["A1", "A2", "A4", "A6"]
ROUND2_ARMS_ALL = ["A1", "A2", "A3", "A4", "A5", "A6"]
RESOLUTIONS = [10, 14, 20]


def smoke(arms: tuple[str, ...] = SMOKE_ARMS, out_root: Path | None = None) -> dict:
    """Short-budget smoke run; asserts invariants."""
    root = Path(out_root) if out_root else Path(tempfile.mkdtemp(prefix="xval_r2_smoke_"))
    records = {}
    for arm in arms:
        res = run_arm(arm, out=root / f"{arm}_D{SMOKE_D}", keep_fields=False,
                      D=SMOKE_D, steps=SMOKE_STEPS, periods=4.0, st_estimate=0.165,
                      warmup_frac=0.3, threads=4)
        rec = res["record"]
        assert rec["status"] == "ok", f"{arm}: stability failure {rec}"
        assert rec["cd_mean"] > 0.0, f"{arm}: non-positive mean drag {rec['cd_mean']}"
        assert 0.1 < rec["cd_mean"] < 10.0, f"{arm}: Cd not O(1) {rec['cd_mean']}"
        records[arm] = rec
    if len(records) > 1:
        vals = [r["cd_mean"] for r in records.values()]
        spread = 100.0 * (max(vals) - min(vals)) / min(vals)
        assert spread < 50.0, f"smoke arms disagree by {spread:.1f} % (> 50 %)"
        for r in records.values():
            r["smoke_cd_spread_pct"] = spread
    return {"smoke_params": {"D": SMOKE_D, "steps": SMOKE_STEPS, "arms": arms},
            "records": records, "out_root": str(root)}


def rung(D: int, arms: list[str] | None = None, threads: int = 8) -> dict:
    """Run one resolution rung; returns the list of arm tags that completed."""
    arms = arms or ROUND2_ARMS
    results = {}
    for arm in arms:
        out = XVAL / "runs" / f"{arm}_D{D}"
        res = run_arm(arm, out=out, keep_fields=True, D=D, threads=threads)
        results[arm] = res["record"].get("status", "unknown")
    return {"D": D, "arms": arms, "results": results}


def analyze() -> dict:
    """Run the round-2 aggregation and return the summary."""
    subprocess.run([sys.executable, str(XVAL / "analyze_round2.py")], check=True)
    summary_path = XVAL / "summary_round2.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Round-2 method-selection entry point")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="Smoke run (A1+A6 at D=6)")
    g.add_argument("--rung", type=int, choices=RESOLUTIONS, help="Run one resolution rung")
    g.add_argument("--analyze", action="store_true", help="Aggregate + report")
    g.add_argument("--full", action="store_true", help="Smoke → rung 10 → analyze")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="", help="Output root for smoke")
    args = ap.parse_args(argv)

    if args.smoke:
        out = smoke(out_root=Path(args.out) if args.out else None)
        print(json.dumps(out, indent=2))
        return 0
    if args.rung:
        result = rung(args.rung, threads=args.threads)
        print(json.dumps(result, indent=2))
        return 0
    if args.analyze:
        summary = analyze()
        print(json.dumps(summary.get("verdict", {}), indent=2))
        return 0
    if args.full:
        smoke_out = smoke()
        print(f"Smoke OK: {list(smoke_out['records'].keys())}")
        rung_result = rung(10, threads=args.threads)
        print(f"Rung 10: {rung_result['results']}")
        summary = analyze()
        print(json.dumps(summary.get("verdict", {}), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
