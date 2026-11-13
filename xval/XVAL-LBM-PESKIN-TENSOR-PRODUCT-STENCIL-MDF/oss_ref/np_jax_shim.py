"""Minimal numpy stand-in for the parts of ``jax.numpy`` / ``jax.lax`` that the
vendored haimingz/vivsim immersed-boundary modules use.

Why this file exists
--------------------
The upstream OSS implementation (``vivsim/ib/{kernels,stencil,mdf,geometry}.py``)
is written against JAX.  This environment has no JAX install and no GPU, so a
*shim* is used instead of a re-implementation: the upstream function bodies stay
**byte-identical** and only their import lines are redirected here (see
``PROVENANCE.json``, field ``shim``).  Any change of semantics is therefore
confined to this file, which is exactly what the provenance record exposes.

Semantics reproduced
--------------------
* ``jnp.abs/sqrt/where/floor/arange/meshgrid/asarray/sum/roll/pad/einsum/norm``
  map to the obvious numpy calls; ``einsum`` swallows JAX's ``precision=`` kwarg.
* ``x.at[key].add(v)`` (JAX's *unbuffered* scatter-add) maps to ``np.add.at``,
  so duplicate indices accumulate -- required for force spreading with
  overlapping marker stencils.
* ``lax.fori_loop(lo, hi, body, xs)`` maps to a Python loop.
* Arrays carry an ``_Arr`` ndarray subclass so that ``.at`` survives slicing,
  reshaping and arithmetic, exactly where JAX would return a ``jax.Array``.

Deliberate difference (recorded as a comparison limit in the report)
-------------------------------------------------------------------
JAX's *default* dtype is float32, numpy's is float64.  The shim keeps float64
throughout so that both arms run in float64 (TensorLBM's IBM path is float64).
The study compares spatial discretisation operators, not float precision, so
making both arms float64 removes dtype as a confounder; upstream float32
behaviour is not reproduced and is listed as a limit.
"""

from __future__ import annotations

import numpy as np

float64 = np.float64
int32 = np.int32
pi = np.pi
Array = np.ndarray

__all__ = [
    "Array",
    "abs",
    "arange",
    "asarray",
    "cos",
    "einsum",
    "float64",
    "floor",
    "int32",
    "lax",
    "linalg",
    "meshgrid",
    "pad",
    "roll",
    "sqrt",
    "sum",
    "where",
    "zeros",
    "zeros_like",
]


def _arr(x):
    """Return ``x`` as an ``_Arr`` view (JAX returns arrays everywhere)."""
    a = np.asanyarray(x)
    return a if isinstance(a, _Arr) else a.view(_Arr)


class _AtIndexer:
    """Result of ``a.at[key]``; only ``.add`` and ``.get`` are supported."""

    __slots__ = ("_arr", "_key")

    def __init__(self, arr, key):
        self._arr = arr
        self._key = key

    def add(self, values):
        out = np.array(self._arr, copy=True)
        np.add.at(out, self._key, np.asarray(values))
        return out.view(_Arr)

    def get(self):
        return np.array(self._arr, copy=True).view(_Arr)


class _AtRoot:
    """``a.at`` proxy: ``__getitem__`` records the index and returns a setter."""

    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, key):
        return _AtIndexer(self._arr, key)


class _Arr(np.ndarray):
    """ndarray subclass carrying JAX's ``.at`` accessor."""

    __slots__ = ()

    @property
    def at(self):
        return _AtRoot(np.asarray(self))


# --------------------------------------------------------------------------- #
# jnp surface used upstream
# --------------------------------------------------------------------------- #


def asarray(x, dtype=None):
    a = np.asanyarray(x)
    if dtype is not None:
        a = a.astype(dtype)
    return a.view(_Arr)


def abs(x):  # noqa: A001 - mirrors jnp.abs
    return _arr(np.abs(np.asarray(x)))


def sqrt(x):
    return _arr(np.sqrt(np.asarray(x)))


def where(cond, a, b):
    return _arr(np.where(np.asarray(cond), np.asarray(a), np.asarray(b)))


def cos(x):
    return _arr(np.cos(np.asarray(x)))


def floor(x):
    return _arr(np.floor(np.asarray(x)))


def arange(*args, dtype=None):
    return _arr(np.arange(*args, dtype=dtype))


def meshgrid(*args, **kwargs):
    return tuple(_arr(a) for a in np.meshgrid(*args, **kwargs))


def einsum(subscripts, *operands, **kwargs):
    kwargs.pop("precision", None)
    kwargs.pop("optimize", None)
    return _arr(np.einsum(subscripts, *(np.asarray(o) for o in operands)))


def zeros(shape, dtype=np.float64):
    return _arr(np.zeros(shape, dtype=dtype))


def zeros_like(x, dtype=None):
    a = np.asarray(x)
    return _arr(np.zeros_like(a, dtype=dtype or a.dtype))


def sum(x, axis=None, **kwargs):  # noqa: A001 - mirrors jnp.sum
    return np.sum(np.asarray(x), axis=axis, **kwargs)


def roll(x, shift, axis=None):
    return _arr(np.roll(np.asarray(x), shift, axis=axis))


def pad(x, pad_width, mode="constant", **kwargs):
    return _arr(np.pad(np.asarray(x), pad_width, mode=mode, **kwargs))


class linalg:  # noqa: N801 - mirrors jnp.linalg
    """Namespace for the few ``jnp.linalg`` calls used upstream."""

    @staticmethod
    def norm(x, axis=None, keepdims=False):
        return np.linalg.norm(np.asarray(x), axis=axis, keepdims=keepdims)


class lax:  # noqa: N801 - mirrors jax.lax
    """Namespace for the ``jax.lax`` primitives used upstream."""

    @staticmethod
    def fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val