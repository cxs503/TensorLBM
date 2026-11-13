"""Minimal ``jax.numpy`` stand-in (NumPy) for executing the vendored OSS source.

The vendored file ``vivsim_multigrid.py`` is the upstream ``haimingz/vivsim``
``vivsim/multigrid.py`` with a single import block replaced (see ``PROVENANCE``
in ``__init__.py`` / ``PROVENANCE.json``).  The functions themselves are pure
array algebra; the only JAX-specific API they touch is

* ``jnp.zeros(shape, dtype=jnp.float32)`` and
* the functional update ``x.at[idx].set(value)``.

Both are provided here on top of :mod:`numpy` with identical value semantics, so
the *upstream expressions* execute unchanged on a machine without JAX (the
loong64 workers have no JAX build).  ``.set`` always returns a new array, like
JAX, so accidental in-place aliasing cannot silently change the OSS semantics.
"""

from __future__ import annotations

import numpy as np

float32 = np.float32
float64 = np.float64


class _AtIndex:
    __slots__ = ("_array", "_index")

    def __init__(self, array: np.ndarray, index) -> None:
        self._array = array
        self._index = index

    def set(self, value) -> "_Array":
        out = np.array(self._array, copy=True)
        out[self._index] = np.asarray(value)
        return _Array(out)


class _Array:
    """``ndarray`` + ``.at[idx].set(...)`` (JAX functional-update subset)."""

    __slots__ = ("_data",)

    def __init__(self, data) -> None:
        self._data = np.asarray(data)

    # -- functional update -------------------------------------------------
    @property
    def at(self) -> "_AtProxy":
        return _AtProxy(self._data)

    # -- array protocol ----------------------------------------------------
    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value) -> None:  # pragma: no cover - guard
        raise TypeError("shim arrays are immutable; use .at[idx].set(value)")

    @property
    def shape(self):
        return self._data.shape

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"_Array(shape={self._data.shape})"


class _AtProxy:
    __slots__ = ("_array",)

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __getitem__(self, index) -> _AtIndex:
        return _AtIndex(self._array, index)


def zeros(shape, dtype=float32) -> _Array:
    return _Array(np.zeros(shape, dtype=dtype))


def ones(shape, dtype=float32) -> _Array:
    return _Array(np.ones(shape, dtype=dtype))


def asarray(x, dtype=None) -> _Array:
    return _Array(np.asarray(x, dtype=dtype))


def array(x, dtype=None) -> _Array:
    return _Array(np.array(x, dtype=dtype))