#!/usr/bin/env python3
"""Compatibility entry for the oscillatory-cylinder benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.case_pool.oscillatory_cylinder_inertia.run import (  # noqa: E402
    main as main,
    run_case as run_case,
)


if __name__ == "__main__":
    raise SystemExit(main())
