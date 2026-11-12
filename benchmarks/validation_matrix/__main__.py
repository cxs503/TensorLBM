"""Package entry point: ``python -m benchmarks.validation_matrix``.

Thin shim so the matrix is reachable as a module (the same CLI as
``runner.py``)::

    python -m benchmarks.validation_matrix audit
    python -m benchmarks.validation_matrix check
    python -m benchmarks.validation_matrix run --tier ci
    python -m benchmarks.validation_matrix status --write
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())