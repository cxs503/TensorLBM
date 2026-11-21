#!/usr/bin/env python3
"""Lid-driven square cavity, Re = 3200 -- benchmark entry point (anchor REPRO-LBM-HOU1995).

Thin wrapper: the implementation lives in the library
(:mod:`tensorlbm.verification.cavity_re3200`) so that the platform and any other
caller exercise exactly the same code path as this script.

Reproduce
---------
::

    # from the repository root
    python benchmarks/verified/cavity/re3200/run.py --n 128 --steps 2000 --verbose

    # the published grid set (records land in ./out, then score + figures)
    python benchmarks/verified/cavity/re3200/run.py \
        --n 128 192 256 --steps 150000 --monitor-every 5000 --resid-tol 1e-6
    python benchmarks/verified/cavity/re3200/analyze.py
    python benchmarks/verified/cavity/re3200/benchmark.py

Exit codes: ``0`` all grids finite, ``2`` at least one grid failed (see
``out/run_manifest.json``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.verification.cavity_re3200.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))