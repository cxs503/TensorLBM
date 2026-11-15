"""Re-export shim: Re = 3200 cavity reproduction (anchor ``REPRO-LBM-HOU1995``).

Why this file exists
--------------------
The repository carries **two** trees that both provide a top-level ``tensorlbm``
package:

* ``src/tensorlbm``            -- the tree on pytest's ``pythonpath = ["src"]``
  (``pyproject.toml``), i.e. the one a plain ``pytest tests/`` binds first;
* ``TensorLBM/src/tensorlbm``  -- the tree that holds the verification subpackages
  used by the cavity benchmarks (``verification/lid_driven_cavity``,
  ``verification/layered_poiseuille`` ...) and the one the benchmark scripts put
  first on ``sys.path``.

The Re = 3200 implementation lives in the second tree, next to the sibling
Re = 5000 benchmark and the solver it drives.  Without this shim
``import tensorlbm.verification.cavity_re3200`` would raise
``ModuleNotFoundError`` whenever ``tensorlbm`` was bound to the first tree -- the
usual case under ``pytest``.  The shim extends the *import search path* of the
already-imported packages (append only, never prepend, so existing resolution is
untouched) and re-exports the public surface, so both callers see one
implementation and one set of numbers.
"""

from __future__ import annotations

from pathlib import Path

import tensorlbm
from tensorlbm import verification as _verification

_ALT_ROOT = Path(__file__).resolve().parents[4] / "TensorLBM" / "src" / "tensorlbm"
_ALT_VERIFICATION = _ALT_ROOT / "verification"

for _pkg, _extra in ((tensorlbm, _ALT_ROOT), (_verification, _ALT_VERIFICATION)):
    if _extra.is_dir() and str(_extra) not in _pkg.__path__:
        _pkg.__path__.append(str(_extra))

# our own submodules live in the library tree
__path__ = [str(_ALT_VERIFICATION / "cavity_re3200")]

from . import reference, runner, scoring  # noqa: E402,F401
from .runner import (  # noqa: E402,F401
    ANCHOR,
    CASE,
    RE,
    U_LID,
    centerline,
    run_grid,
    write_record,
)
from .scoring import TOL_PCT, score  # noqa: E402,F401

__all__ = [
    "ANCHOR", "CASE", "RE", "U_LID", "TOL_PCT",
    "centerline", "run_grid", "write_record", "score",
    "reference", "runner", "scoring",
]

__version__ = "1.0.0"