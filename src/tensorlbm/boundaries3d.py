"""Backward-compatible alias for :mod:`tensorlbm.boundary.boundaries3d`.

Keep one module object (including private helpers and mutable state) under both paths.
"""

import sys
from importlib import import_module

_module = import_module("tensorlbm.boundary.boundaries3d")
sys.modules[__name__] = _module
