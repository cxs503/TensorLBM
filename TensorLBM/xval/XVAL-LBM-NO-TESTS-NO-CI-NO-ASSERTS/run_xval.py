#!/usr/bin/env python3
"""XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS runner.

Executes the registered cross-validation cases of
``tensorlbm.verification.xval`` against their declared external anchors and
writes machine-derived artifacts (no hand-copied numbers):

    results.json   one record per case: anchor, params, metrics, thresholds,
                   pass/fail, wall time, state hash
    kpi.csv        flat KPI table (case, metric, value, threshold, passed)
    results.sha256 sha256 of results.json (provenance anchor)
    provenance.json interpreter/torch/git/thread count/environment

Exit code is 0 only if every executed case passed *and* every metric is
finite; a diverged run (NaN/inf) therefore fails the runner, which is exactly
what the pre-fix registry did not do.

Usage
-----
    python run_xval.py                       # fast tier (CI default)
    python run_xval.py --cases all           # full tier (slow, >4 min)
    python run_xval.py --cases poiseuille_2d_H16 --outdir /tmp/xv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.verification.xval.cases import register_default_cases  # noqa: E402
from tensorlbm.verification.xval.registry import XValRegistry  # noqa: E402

FAST_TIER = ["poiseuille_2d_H16"]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="fast", help="'fast', 'all' or a comma separated list")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Predeclared: single-threaded execution (see predeclared.json confounders).
    torch.set_num_threads(1)

    reg = XValRegistry()
    register_default_cases(reg)
    if args.cases == "fast":
        names = FAST_TIER
    elif args.cases == "all":
        names = reg.list_cases()
    else:
        names = [n.strip() for n in args.cases.split(",") if n.strip()]

    records = []
    failures = []
    for name in names:
        case = reg.get(name)
        t0 = time.time()
        res = reg.run_case(name, compute_hash=True)
        seconds = time.time() - t0
        nonfinite = [k for k, v in res.metrics.items() if not np.isfinite(v)]
        rec = {
            "case": name,
            "category": case.category,
            "external_anchor": case.external_anchor,
            "params": case.params,
            "metrics": {k: float(v) for k, v in res.metrics.items()},
            "thresholds": case.thresholds,
            "passed": bool(res.passed),
            "nonfinite_metrics": nonfinite,
            "seconds": round(seconds, 2),
            "state_hash": res.state_hash,
        }
        records.append(rec)
        status = "PASS" if rec["passed"] and not nonfinite else "FAIL"
        if status == "FAIL":
            failures.append(name)
        kpi = " ".join(f"{k}={v:.4e}" for k, v in rec["metrics"].items())
        print(f"[{status}] {name}: {kpi} [{seconds:.1f}s]", flush=True)

    results = {"feature_slug": "XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS", "cases": records}
    rpath = os.path.join(args.outdir, "results.json")
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.outdir, "kpi.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "metric", "value", "threshold", "passed"])
        for rec in records:
            for metric, value in rec["metrics"].items():
                writer.writerow(
                    [
                        rec["case"],
                        metric,
                        f"{value:.12g}",
                        rec["thresholds"].get(metric, ""),
                        rec["passed"],
                    ]
                )
    digest = hashlib.sha256(open(rpath, "rb").read()).hexdigest()
    with open(os.path.join(args.outdir, "results.sha256"), "w") as fh:
        fh.write(digest + "  results.json\n")
    provenance = {
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "results_sha256": digest,
        "executed_cases": names,
        "failed_cases": failures,
    }
    with open(os.path.join(args.outdir, "provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
    print(f"results.json sha256={digest}")
    print(f"failed: {failures if failures else 'none'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
