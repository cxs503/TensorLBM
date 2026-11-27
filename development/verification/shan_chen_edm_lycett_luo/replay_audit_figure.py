#!/usr/bin/env python3
"""Render the frozen m21400 audit figure from the relocated tracked evidence.

Only file locations are adapted; the historical plotting implementation and
recorded data remain unchanged. No simulation or external publishing is run.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / "development/audit/m21400"
COMPACT = HERE / "round2/artifacts/compact.json"
PROBE = AUDIT / "probe_m21378_results.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=AUDIT)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "archived_m21400_figure", AUDIT / "make_figure_m21378.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the original m21400 figure renderer")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.COMPACT = COMPACT
    renderer.PROBE = PROBE
    renderer.HERE = output
    renderer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
