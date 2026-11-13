#!/usr/bin/env python3
"""Case-pool driver: discover and execute pool cases (one sub-package per case).

Each pool case is a sub-package of ``benchmarks/case_pool/`` that exposes a
``run_case.py`` with a ``main()``.  This driver discovers those entries and runs
them, so a new capability becomes part of the pool by dropping a sub-package --
no hard-coded case list here (zero domain semantics in the driver).

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --list
    PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --case xval_method_selection --smoke
    PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --all --smoke
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

POOL = Path(__file__).resolve().parent
REPO = POOL.parents[1]


def discover() -> list[str]:
    """Return the names of pool cases (sub-packages with a run_case.py)."""
    out = []
    for d in sorted(POOL.iterdir()):
        if d.is_dir() and (d / "run_case.py").is_file():
            out.append(d.name)
    return out


def run_case(name: str, extra: list[str]) -> int:
    """Run one pool case by importing its run_case.main (in-process)."""
    mod = importlib.import_module(f"case_pool.{name}.run_case")
    return mod.main(extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list pool cases")
    ap.add_argument("--case", default="", help="run one case by name")
    ap.add_argument("--all", action="store_true", help="run every pool case")
    ap.add_argument("--smoke", action="store_true",
                    help="forward --smoke to each case")
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(discover()))
        return 0
    if args.case:
        extra = ["--smoke"] if args.smoke else []
        return run_case(args.case, extra)
    if args.all:
        rc = 0
        for name in discover():
            extra = ["--smoke"] if args.smoke else []
            print(f"== pool case: {name} ==")
            rc |= run_case(name, extra)
        return rc
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
