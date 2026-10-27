"""Backward-compatible alias for :mod:`tensorlbm.boundary.boundaries`.

Keep one module object (including private helpers and mutable state) under both paths.
"""

import sys
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .boundary.boundaries import *  # noqa: F403

_module = import_module("tensorlbm.boundary.boundaries")
sys.modules[__name__] = _module
