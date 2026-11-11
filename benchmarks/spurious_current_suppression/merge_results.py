#!/usr/bin/env python3
"""Merge phase JSON files into the final ``results.json`` of node p362w2-n06.

The campaign is expensive enough that it is run in phases (the MRT collision is
~3x the cost per step of BGK on the loong32/loong64 worker).  Each phase writes
its own ``runs`` list; this script concatenates them, recomputes the summary and
the acceptance verdicts with exactly the code paths used by ``run.py``, and
writes the single artifact that ``build_report.py`` consumes.

Usage::

    python benchmarks/spurious_current_suppression/merge_results.py \\
        --runs phase1.json phase2.json --out artifacts/... --n-pytest 29
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import run as campaign  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/spurious_current_p362w2_n06")
    ap.add_argument("--n-pytest", type=int, default=0)
    ap.add_argument("--protocol", default=None, help="json file with the protocol block")
    args = ap.parse_args()

    runs: list[dict] = []
    protocol = {}
    for path in args.runs:
        d = json.loads(Path(path).read_text())
        runs.extend(d["runs"] if "runs" in d else d.get("results", []))
        protocol.update(d.get("protocol", {}))
    if args.protocol:
        protocol.update(json.loads(Path(args.protocol).read_text()))

    control = campaign._job_apply(
        dict(kind="control", method="baseline_bgk_lattice", n=64, radius=16.0,
             width=2.0, steps=500, stop_tol=1e-12, check_every=100,
             G12=0.0, uniform=True)
    )
    summary = campaign._summarise(runs, protocol)
    verdicts = campaign._verdicts(summary, control, args.n_pytest, protocol)

    out = Path(__file__).resolve().parents[2] / args.out
    payload = {
        "node": "p362w2-n06-lbm-spurious-current",
        "repo": "TensorLBM",
        "module": "tensorlbm.verification.spurious_current",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol,
        "n_runs": len(runs),
        "runs": runs,
        "control_sigma_zero": control,
        "summary": summary,
        "verdicts": verdicts,
        "field_artifacts": {
            p.stem.replace("field_", ""): str(p.relative_to(HERE.parents[1]))
            for p in out.glob("field_*.npz")
        },
    }
    prior = out / "results.json"
    wall = 0.0
    if prior.exists():
        try:
            wall = float(json.loads(prior.read_text()).get("wall_seconds", 0.0))
        except Exception:
            wall = 0.0
    # Sum of the individual run wall times (core-seconds of solver time); the
    # runs were executed in parallel phases, so this is not the campaign wall.
    payload["wall_seconds"] = sum(r["wall_s"] for r in runs)
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    print("wrote", out / "results.json", "runs:", len(runs))
    for k, val in verdicts.items():
        if isinstance(val, dict):
            print(f"  {k}: {val.get('pass')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
