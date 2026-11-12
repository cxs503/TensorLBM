"""``python -m tensorlbm.audit`` -- run the international benchmark audit.

Usage::

    python -m tensorlbm.audit --repo-root . --out artifacts/intl_bench_v1/run1
"""

from __future__ import annotations

import sys

from .intl_bench import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))