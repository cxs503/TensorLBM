#!/usr/bin/env python
"""Benchmark-side entry point for XVAL-LBM-TAYLORGREEN.

Thin wrapper around :mod:`tensorlbm.xval` so the campaign can be reproduced from
the benchmark tree as well as from the API::

    python benchmarks/xval_taylor_green/run_xval.py --out runs/xval_taylor_green
    python benchmarks/xval_taylor_green/run_xval.py --out runs/xval_tg_boosted --boost

The heavy lifting (case spec, tool adapters, metrics, criteria, bug attribution)
lives in ``src/tensorlbm/xval`` and is shared with the platform endpoint
``POST /api/v1/xval/taylor-green``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.xval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
