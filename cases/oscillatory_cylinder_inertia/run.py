#!/usr/bin/env python3
"""Case entry: cylinder inertia (added-mass) coefficient in oscillatory flow.

Wraps :mod:`tensorlbm.oscillatory_cylinder_inertia`; see
``docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md`` for the reference/convention audit.

    python cases/oscillatory_cylinder_inertia/run.py --case quick
    python cases/oscillatory_cylinder_inertia/run.py --case base --out runs/xval_horizontal
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.oscillatory_cylinder_inertia import CASES, main  # noqa: E402


def run_case(name: str = "quick", out: str | None = None, threads: int = 4) -> int:
    """Run one predeclared case through the library entry point."""
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; known: {sorted(CASES)}")
    argv = ["--case", name, "--threads", str(threads)]
    if out:
        argv += ["--out", out]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
