#!/usr/bin/env python3
"""Benchmark entry for XVAL-LBM-BENCHMARK-METHOD-SELECTION.

This is the *entry* of the method-selection cross-validation: it calls the same
arm runner that produced the published matrix (``xval/benchmark_method_selection/
run_arm.py::run_arm``), so a smoke run and a published run are the same code
path with different parameters.

Usage
-----
  # smoke (control-node budget, < 2 min): two arms at D=6
  python benchmarks/xval_method_selection_bench.py --smoke

  # full matrix (hours): arms x resolutions, then aggregate + report
  python benchmarks/xval_method_selection_bench.py --full

  # aggregate only (after the matrix finished)
  python benchmarks/xval_method_selection_bench.py --analyze

The smoke run asserts, on real numbers, the invariants the published matrix
relies on:
  * both arms integrate to a finite state,
  * the drag coefficient is positive and O(1) (the recorded sign defect that
    motivated the post-stream force sampler would show up as Cd < 0),
  * the two selections (MRT+sponge vs BGK) stay within a loose 50 % bracket at
    D=6, i.e. the harness itself does not manufacture a method difference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "xval" / "benchmark_method_selection"
sys.path.insert(0, str(XVAL))
sys.path.insert(0, str(REPO / "src"))

from run_arm import ARMS, run_arm  # noqa: E402

SMOKE = dict(D=6, steps=1200, periods=4.0, st_estimate=0.165, warmup_frac=0.3, threads=4)


def smoke(arms: tuple[str, ...] = ("A1", "A2"), out_root: Path | None = None) -> dict:
    """Short-budget run of *arms*; returns the KPI records (also usable by pytest)."""
    root = Path(out_root) if out_root else Path(tempfile.mkdtemp(prefix="xval_smoke_"))
    records = {}
    for arm in arms:
        res = run_arm(arm, out=root / f"{arm}_D{SMOKE['D']}", keep_fields=False, **SMOKE)
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
    return {"smoke_params": SMOKE, "records": records, "out_root": str(root)}


def full() -> dict:
    """Run the full matrix (D=10 all arms, D=14 A1/A2/A4) and aggregate."""
    subprocess.run(["bash", str(XVAL / "run_matrix.sh")], check=True)
    return analyze()


def analyze() -> dict:
    subprocess.run([sys.executable, str(XVAL / "analyze.py")], check=True)
    return json.loads((XVAL / "summary.json").read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.smoke:
        out = smoke(out_root=Path(args.out) if args.out else None)
        print(json.dumps(out, indent=2))
        return 0
    if args.full:
        print(json.dumps(full().get("verdict", {}), indent=2))
        return 0
    print(json.dumps(analyze().get("verdict", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())