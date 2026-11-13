"""``jax.lax`` subset: ``fori_loop`` and the two dynamic slice helpers."""

from __future__ import annotations

import numpy as _np

from ._array import wrap


def fori_loop(lower, upper, body_fun, init_val):
    """Run ``body_fun(i, carry)`` for ``i = lower … upper-1`` chaining the carry.

    Matches :func:`jax.lax.fori_loop` semantics for this use: fixed trip count,
    sequential carry chaining, no early exit.
    """
    carry = init_val
    for i in range(int(lower), int(upper)):
        carry = body_fun(i, carry)
    return carry


def dynamic_slice(operand, start_indices, slice_sizes):
    """``lax.dynamic_slice``: static-shape slice starting at runtime indices."""
    operand = _np.asarray(operand)
    start = [int(s) for s in _np.asarray(start_indices).reshape(-1)]
    sizes = [int(s) for s in _np.asarray(slice_sizes).reshape(-1)]
    key = tuple(slice(s, s + n) for s, n in zip(start, sizes))
    return wrap(_np.array(operand[key], copy=True))


def dynamic_update_slice(operand, update, start_indices):
    """``lax.dynamic_update_slice``: functional in-place-style update."""
    operand = _np.array(_np.asarray(operand), copy=True)
    update = _np.asarray(update)
    start = [int(s) for s in _np.asarray(start_indices).reshape(-1)]
    key = tuple(slice(s, s + n) for s, n in zip(start, update.shape))
    operand[key] = update
    return wrap(operand)