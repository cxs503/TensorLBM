"""``jax.numpy`` on top of NumPy: identical elementwise/einsum/indexing semantics.

Attribute access is dynamic (PEP 562 module ``__getattr__``): every NumPy callable
is exposed, results are viewed as :class:`jax._array._Array` so that the JAX
``.at[]`` API keeps working, and JAX-only keyword arguments that NumPy rejects
(``precision=``) are dropped (they are exactness hints, not numerics).
"""

from __future__ import annotations

import types

import numpy as _np

from ._array import _Array, wrap

#: Names whose values are plain scalars/objects rather than arrays.
_SCALARS = {"pi", "e", "euler_gamma", "inf", "nan", "newaxis", "NAN", "INF"}
_DROP_KWARGS = {"precision", "preferred_element_type", "unique_indices", "mode", "out_sharding"}


class _ModuleProxy:
    """Lazily wrap a NumPy submodule (``jnp.linalg``, ``jnp.fft``, ...)."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return _wrap_attr(getattr(self._module, name))


def _wrap_callable(fn):
    def call(*args, **kwargs):
        for key in list(kwargs):
            if key in _DROP_KWARGS:
                kwargs.pop(key)
        return wrap(fn(*args, **kwargs))

    call.__name__ = getattr(fn, "__name__", "wrapped")
    call.__doc__ = getattr(fn, "__doc__", None)
    return call


def _wrap_attr(obj):
    if isinstance(obj, types.ModuleType):
        return _ModuleProxy(obj)
    if isinstance(obj, _np.dtype) or isinstance(obj, (_np.generic, str, int, float)):
        return obj
    # dtype constructors (``jnp.int32``, ``jnp.float64``, ...) must stay types
    if isinstance(obj, type) and issubclass(obj, (_np.generic, _np.dtype)):
        return obj
    if callable(obj):
        return _wrap_callable(obj)
    return obj


def __getattr__(name):  # noqa: D105 - module-level dynamic attribute
    if name in _SCALARS:
        return getattr(_np, name)
    if not hasattr(_np, name):
        raise AttributeError(f"jax.numpy shim has no attribute {name!r}")
    return _wrap_attr(getattr(_np, name))


def asarray(x, dtype=None, **kwargs):  # type: ignore[override]
    """``jnp.asarray`` returning the shim array type."""
    return wrap(_np.asarray(x, dtype=dtype))


def array(x, dtype=None, **kwargs):  # type: ignore[override]
    """``jnp.array`` returning the shim array type."""
    return wrap(_np.array(x, dtype=dtype))


Array = _Array