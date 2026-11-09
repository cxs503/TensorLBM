#!/usr/bin/env python
"""Thin CLI wrapper for the cross-solver verification bench (node P78-N01)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.cross_solver_verification.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
