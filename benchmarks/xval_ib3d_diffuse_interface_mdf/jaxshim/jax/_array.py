"""ndarray subclass with JAX's ``.at[]`` scatter API plus result wrapping."""

from __future__ import annotations

import numpy as _np


def wrap(value):
    """Recursively view ndarray results as :class:`_Array` (JAX-like returns)."""
    if isinstance(value, _Array):
        return value
    if isinstance(value, _np.ndarray):
        return value.view(_Array)
    if isinstance(value, (list, tuple)):
        converted = [wrap(v) for v in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return value


class _AtIndexer:
    """``arr.at[idx]`` proxy implementing JAX's functional ``.add`` / ``.set``."""

    __slots__ = ("_base", "_idx")

    def __init__(self, base: _np.ndarray, idx=None):
        self._base = base
        self._idx = idx

    def __getitem__(self, key) -> "_AtIndexer":
        if self._idx is not None:
            raise NotImplementedError("chained .at[...][...] not supported by the shim")
        return _AtIndexer(self._base, key)

    def _scatter_add(self, values) -> _np.ndarray:
        base = _np.asarray(self._base)
        idx = self._idx
        values = _np.asarray(values)
        out = _np.array(base, copy=True)
        if idx is None:
            _np.add.at(out, Ellipsis, values)
            return out

        def accumulate(dst, key, vals):
            # NumPy's ufunc.at is UNBUFFERED: duplicate indices accumulate, which
            # is exactly JAX's documented .at[].add semantics.
            _np.add.at(dst, key, vals)

        if isinstance(idx, tuple):
            head, tail = idx[:-1], idx[-1]
            if all(isinstance(k, slice) for k in head) and isinstance(tail, _np.ndarray):
                view = out
                for k in head:
                    view = view[k]
                if head and all(k == slice(None) for k in head):
                    # leading axis is independent: vectorise over it
                    tail = _np.asarray(tail)
                    for comp in range(view.shape[0]):
                        vals = values[comp] if values.ndim >= 1 else values
                        accumulate(view[comp], tail, vals)
                    return out
                accumulate(view, tail, values)
                return out
            raise NotImplementedError(f"index tuple {idx!r} not supported by the shim")
        accumulate(out, idx, values)
        return out

    def add(self, values):
        """Unbuffered scatter-add (``out[idx] += values`` with duplicate summing)."""
        return wrap(self._scatter_add(values))

    def set(self, values):
        """Functional assignment."""
        out = _np.array(_np.asarray(self._base), copy=True)
        out[self._idx if self._idx is not None else Ellipsis] = values
        return wrap(out)


class _Array(_np.ndarray):
    """``ndarray`` carrying the JAX ``.at[]`` accessor.

    ``__getitem__`` reproduces JAX's *clamped* out-of-bounds gather semantics
    (JAX clamps rather than raising, which upstream ``vivsim`` relies on when a
    marker stencil pokes outside the local IBM block).  NumPy would raise
    ``IndexError`` here, so the clamp is applied to integer index arrays only.
    """

    __slots__ = ()

    @property
    def at(self) -> _AtIndexer:
        """Functional scatter accessor (``arr.at[idx].add(v)``)."""
        return _AtIndexer(_np.asarray(self))

    def __getitem__(self, key):
        return super().__getitem__(_clamp_key(key, self.shape))


def _clamp_key(key, shape):
    """Clamp integer index arrays to ``[0, n-1]`` per axis (JAX gather semantics)."""
    if isinstance(key, tuple):
        clamped = []
        for axis, item in enumerate(key):
            clamped.append(_clamp_key_axis(item, shape[axis] if axis < len(shape) else None))
        return tuple(clamped)
    return _clamp_key_axis(key, shape[0] if len(shape) else None)


def _clamp_key_axis(item, n):
    if isinstance(item, _np.ndarray) and item.dtype.kind in "iu" and n is not None:
        return _np.clip(item, 0, n - 1)
    if isinstance(item, (list, tuple)) and n is not None and item:
        arr = _np.asarray(item)
        if arr.dtype.kind in "iu":
            return _np.clip(arr, 0, n - 1)
    if isinstance(item, (int, _np.integer)) and not isinstance(item, bool) and n is not None:
        return int(min(max(int(item), 0), n - 1))
    return item