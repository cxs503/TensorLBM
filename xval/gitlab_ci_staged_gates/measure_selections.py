"""Measure real pytest selection sizes for a list of ``-m`` expressions.

Writes ``data/selection_measurements.json``: one record per expression with the
count pytest itself reports (zero hand copying), the wall clock of the
collection, and the process CPU time (wall clock on the shared measurement host
carries co-tenant load, CPU time does not).

Usage::

    PYTHONPATH=src python xval/gitlab_ci_staged_gates/measure_selections.py \
        --exprs "not slow" slow unit "not longrun"
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def measure(expr: str | None, extra: list[str]) -> dict:
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests",
        "app/tests",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--continue-on-collection-errors",
    ]
    if expr:
        cmd += ["-m", expr]
    cmd += extra
    t0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    # pytest prints either "N tests collected" or "N/M tests collected (K deselected)";
    # the selected count is the first number in both forms.
    m = re.search(r"(\d+)(?:/\d+)? tests collected", tail)
    n = int(m.group(1)) if m else None
    n_err = len(re.findall(r"^ERROR ", proc.stdout, re.M))
    return {
        "expr": expr if expr else "<all>",
        "n_collected": n,
        "n_collection_errors": n_err,
        "wall_s": round(wall, 3),
        "cpu_s": round(cpu, 3),
        "tail": tail,
        "returncode": proc.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exprs", nargs="+", default=["not slow", "slow", "unit", "not longrun"])
    ap.add_argument("--out", default=str(HERE / "data" / "selection_measurements.json"))
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()
    rows = [measure(None, args.extra)] + [measure(e, args.extra) for e in args.exprs]
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    for r in rows:
        print(
            f"{r['expr']:<14} n={r['n_collected']} wall={r['wall_s']}s cpu={r['cpu_s']}s rc={r['returncode']}"
        )


if __name__ == "__main__":
    main()
