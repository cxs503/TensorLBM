"""``python -m tensorlbm.audit`` -- machine audits for TensorLBM.

Usage::

    python -m tensorlbm.audit --repo-root . --out artifacts/intl_bench_v1/run1
    python -m tensorlbm.audit example-antipattern --root examples --out /tmp/ap.json

The first form (no sub-command) keeps the historical behaviour and runs the
international-benchmark audit.  ``example-antipattern`` measures the
example-per-main anti-pattern (see :mod:`tensorlbm.audit.example_antipattern`).
"""

from __future__ import annotations

import sys

from . import example_antipattern
from .intl_bench import main

_SUBCOMMANDS = {"example-antipattern": example_antipattern.main}


def dispatch(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in _SUBCOMMANDS:
        return _SUBCOMMANDS[args[0]](args[1:])
    return main(args)


if __name__ == "__main__":
    sys.exit(dispatch())
