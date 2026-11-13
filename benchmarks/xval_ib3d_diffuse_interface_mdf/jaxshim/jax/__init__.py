"""JAX→NumPy compatibility shim entry point (xval harness only).

See :mod:`jax._array` for the semantics reproduced.  This is **not** a JAX
replacement: no JIT, no autodiff, no XLA, no multi-device.
"""

from __future__ import annotations

from ._array import _Array, wrap
from . import lax  # noqa: F401
from . import numpy  # noqa: F401

Array = _Array

__all__ = ["Array", "lax", "numpy", "wrap"]