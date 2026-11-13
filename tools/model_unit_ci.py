#!/usr/bin/env python
"""CLI/TensorLBM adapter: run the model-directory build + CI units.

Thin wrapper over :mod:`tensorlbm.models.ci_unit` so CI scripts and the
Makefile never have to know where the model directories live::

    python tools/model_unit_ci.py                 # build + CI + artefacts
    python tools/model_unit_ci.py list            # discovered units
    python tools/model_unit_ci.py ci --only d2q9_trt_channel_poiseuille

Exit code is non-zero when any unit fails its build or one of its gates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.models.ci_unit import main as unit_main  # noqa: E402


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="?", default="all")
    known, rest = parser.parse_known_args(argv)
    return unit_main([known.command, *rest])


if __name__ == "__main__":
    raise SystemExit(cli())
