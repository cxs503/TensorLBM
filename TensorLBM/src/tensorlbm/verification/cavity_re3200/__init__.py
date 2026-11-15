"""Re = 3200 lid-driven cavity reproduction (anchor ``REPRO-LBM-HOU1995``).

Public surface
--------------
``runner.run_grid``          run one ``n x n`` grid, returns a JSON-able record
``runner.write_record``      write ``re3200_n<n>.json`` + sha256 sidecar
``scoring.score``            pointwise ref-vs-got rows, both coordinate conventions
``reference.load_reference`` the admitted Ghia (1982) Re = 3200 column

Command line::

    python -m tensorlbm.verification.cavity_re3200 --n 128 --steps 2000
    python -m tensorlbm.verification.cavity_re3200 --n 128 192 256 \
        --steps 120000 --resid-tol 1e-6 --out-dir <dir>
"""

from __future__ import annotations

from . import reference, runner, scoring  # noqa: F401
from .runner import (  # noqa: F401
    ANCHOR,
    CASE,
    RE,
    U_LID,
    centerline,
    run_grid,
    write_record,
)
from .scoring import TOL_PCT, score  # noqa: F401

__all__ = [
    "ANCHOR", "CASE", "RE", "U_LID", "TOL_PCT",
    "centerline", "run_grid", "write_record", "score",
    "reference", "runner", "scoring",
]

__version__ = "1.0.0"