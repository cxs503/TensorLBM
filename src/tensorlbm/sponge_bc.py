"""Backward-compatible alias for :mod:`tensorlbm.boundary.sponge_bc`.

Keep one module object (including private helpers and mutable state) under both paths.
"""

import sys
from importlib import import_module

_module = import_module("tensorlbm.boundary.sponge_bc")
sys.modules[__name__] = _module
