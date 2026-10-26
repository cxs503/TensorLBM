"""Backward-compatible alias for :mod:`tensorlbm.turbulence.rans_ke`.

Keep one module object (including private helpers and mutable state) under both paths.
"""

import sys
from importlib import import_module

_module = import_module("tensorlbm.turbulence.rans_ke")
sys.modules[__name__] = _module
