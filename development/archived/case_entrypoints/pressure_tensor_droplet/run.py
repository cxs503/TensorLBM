#!/usr/bin/env python3
"""Compatibility entry for the pressure-tensor droplet benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.case_pool.pressure_tensor_droplet.run import main as main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
