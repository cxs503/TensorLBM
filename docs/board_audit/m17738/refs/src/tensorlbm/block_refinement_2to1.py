"""Buffer-layer 2:1 block refinement transfers, cross-validated against vivsim.

Context
-------
``haimingz/vivsim`` (MIT, commit ``fb84b12``, ``vivsim/multigrid.py``) implements a
block-structured 2:1 local refinement where the *coarser* side carries one
buffer layer of cells; data crosses the coarse/fine face with

* ``fine_to_coarse``: a 4-point average of the populations that cross the face
  (``0.25 * (f[., i, 0::2] + f[., i, 1::2] + f[., i+1, 0::2] + f[., i+1, 1::2])``),
* ``coarse_to_fine``: a plain copy of the coarse face populations onto both
  fine rows,
* ``get_omega(nu, level)``: the relaxation recomputed per level,

.. math:: \\omega_l = \\frac{2\\omega_0}{2^{l+1} + (1 - 2^l)\\omega_0},
          \\qquad \\omega_0 = \\frac{1}{3\\nu + 1/2},

which preserves the lattice viscosity under ``dx -> dx/2``, ``dt -> dt/2``.
Neither transfer rescales the non-equilibrium part of the distribution; the
upstream repository ships no example that calls the multigrid module.

This module ports exactly those three semantics to NumPy with the array layout
used by :mod:`tensorlbm.amr_cavity2d` -- ``(q, ny, nx)`` of **full** populations,
whereas upstream stores ``(q, nx, ny)`` of **shifted** populations
(``f_i = f_i^full - w_i``).  Neither mismatch changes the operators:

* the shift is a constant per population, and both ``mean`` over four fine cells
  and ``copy`` reproduce that constant, so shifted and full populations give the
  same transferred *full* field;
* the ``(q, ny, nx)`` / ``(q, nx, ny)`` difference is only a transpose, handled
  explicitly in ``fine_to_coarse_face`` / ``coarse_to_fine_face`` and pinned by
  ``tests/test_xval_buffer_layer_2to1.py`` against the vendored upstream code
  (executed verbatim, see ``benchmarks/xval_buffer_layer_2to1/oss_ref``).

What is *not* adopted
---------------------
``tensorlbm.amr_cavity2d`` keeps the Filippova-Hanel non-equilibrium rescaling
(``tau_f / (2 tau_c)``) on the coarse->fine transfer and the reversed factor on
fine->coarse.  Copying the non-equilibrium part without that factor changes the
interface's effective viscosity by O(1) -- quantified in the cross-validation
artifact -- so the OSS copy is offered only as a comparison arm
(``interface_scheme="vivsim_copy"``), never as the default.

References
----------
haimingz/vivsim, ``vivsim/multigrid.py`` @ fb84b12 (MIT).
Filippova & Haenel, *J. Comput. Phys.* 147 (1998) 219-228 (grid refinement).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "LEFT_DIRS",
    "RIGHT_DIRS",
    "UP_DIRS",
    "DOWN_DIRS",
    "SIDES",
    "get_omega",
    "fh_non_equilibrium_scale",
    "block_average",
    "block_repeat",
    "fine_to_coarse_face",
    "coarse_to_fine_face",
]

# D2Q9 direction numbers of tensorlbm.d2q9 == vivsim/lbm/lattice.py
# 0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE
LEFT_DIRS = np.array([3, 7, 6], dtype=np.int64)
RIGHT_DIRS = np.array([1, 5, 8], dtype=np.int64)
UP_DIRS = np.array([2, 5, 6], dtype=np.int64)
DOWN_DIRS = np.array([4, 7, 8], dtype=np.int64)

SIDES = ("left", "right", "up", "down")

Side = Literal["left", "right", "up", "down"]


def get_omega(nu: float, level: int = 0) -> float:
    """Relaxation parameter of level ``level`` for lattice viscosity ``nu``.

    Exact port of ``vivsim.multigrid.get_omega`` (body-equivalent, float64).
    ``level=0`` is the coarse grid, ``level=1`` the 2:1 refined block, and
    ``level=-1`` a block that is twice as coarse as level 0.
    """
    omega_0 = 1.0 / (3.0 * nu + 0.5)
    return 2.0 * omega_0 / (2.0 ** (level + 1) + (1 - 2.0 ** level) * omega_0)


def fh_non_equilibrium_scale(tau_coarse: float, tau_fine: float, ratio: int = 2) -> float:
    """Filippova-Hanel coarse->fine non-equilibrium factor ``tau_f/(r tau_c)``."""
    return tau_fine / (ratio * tau_coarse)


def block_average(f_fine: np.ndarray, ratio: int = 2) -> np.ndarray:
    """Plain volume average of ``ratio x ratio`` fine cells (no rescaling).

    ``(q, ratio*ny, ratio*nx) -> (q, ny, nx)``.  This is the restriction used by
    the OSS for the populations crossing the face, evaluated over the whole
    field (used by the strip solver's shadow-row restriction).
    """
    q, ny, nx = f_fine.shape
    if ny % ratio or nx % ratio:
        raise ValueError(f"fine shape {f_fine.shape} is not divisible by ratio={ratio}")
    return f_fine.reshape(q, ny // ratio, ratio, nx // ratio, ratio).mean(axis=(2, 4))


def block_repeat(f_coarse: np.ndarray, ratio: int = 2) -> np.ndarray:
    """Plain cell-block copy ``(q, ny, nx) -> (q, ratio*ny, ratio*nx)``, no rescaling."""
    return np.repeat(np.repeat(f_coarse, ratio, axis=1), ratio, axis=2)


def fine_to_coarse_face(
    f_fine: np.ndarray,
    f_coarse: np.ndarray,
    direction: Side,
) -> np.ndarray:
    """Restrict the face-crossing populations of a fine block onto a coarse face.

    Exact port of ``vivsim.multigrid.fine_to_coarse`` -- a 4-point average of a
    2x2 fine block, **no non-equilibrium rescaling** -- transposed from the
    upstream ``(q, nx, ny)`` layout to the ``(q, ny, nx)`` layout of
    :mod:`tensorlbm.amr_cavity2d`.  ``direction`` is the upstream argument:
    ``"left"`` writes the coarse face at ``x = -1``, ``"right"`` at ``x = 0``,
    ``"up"`` at ``y = 0`` and ``"down"`` at ``y = -1``; only the populations
    pointing across that face are written, every other population is returned
    unchanged (the buffer-layer convention: the rest streams in from the
    neighbouring block).
    """
    if f_fine.ndim != 3 or f_coarse.ndim != 3:
        raise ValueError("fields must be (q, ny, nx)")
    if f_coarse.shape[0] != f_fine.shape[0]:
        raise ValueError("fine and coarse must have the same q")
    out = np.array(f_coarse, copy=True)
    if direction == "left":
        out[LEFT_DIRS, :, -1] = 0.25 * (
            f_fine[LEFT_DIRS, 0::2, 0]
            + f_fine[LEFT_DIRS, 1::2, 0]
            + f_fine[LEFT_DIRS, 0::2, 1]
            + f_fine[LEFT_DIRS, 1::2, 1]
        )
    elif direction == "right":
        out[RIGHT_DIRS, :, 0] = 0.25 * (
            f_fine[RIGHT_DIRS, 0::2, -1]
            + f_fine[RIGHT_DIRS, 1::2, -1]
            + f_fine[RIGHT_DIRS, 0::2, -2]
            + f_fine[RIGHT_DIRS, 1::2, -2]
        )
    elif direction == "up":
        out[UP_DIRS, 0, :] = 0.25 * (
            f_fine[UP_DIRS, -1, 0::2]
            + f_fine[UP_DIRS, -1, 1::2]
            + f_fine[UP_DIRS, -2, 0::2]
            + f_fine[UP_DIRS, -2, 1::2]
        )
    elif direction == "down":
        out[DOWN_DIRS, -1, :] = 0.25 * (
            f_fine[DOWN_DIRS, 0, 0::2]
            + f_fine[DOWN_DIRS, 0, 1::2]
            + f_fine[DOWN_DIRS, 1, 0::2]
            + f_fine[DOWN_DIRS, 1, 1::2]
        )
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown direction {direction!r}; expected left/right/up/down")
    return out


def coarse_to_fine_face(
    f_coarse: np.ndarray,
    f_fine: np.ndarray,
    direction: Side,
) -> np.ndarray:
    """Prolongate a coarse face onto the fine buffer plane, by plain copy.

    Exact port of ``vivsim.multigrid.coarse_to_fine`` (no rescaling, one fine
    plane per face), transposed to ``(q, ny, nx)``.

    * ``"left"``  copies coarse ``x = 0`` onto fine ``x = -1`` (even and odd ``y``);
    * ``"right"`` copies coarse ``x = -1`` onto fine ``x = 0``;
    * ``"up"``    copies coarse ``y = -1`` onto fine ``y = 0`` (even and odd ``x``);
    * ``"down"``  copies coarse ``y = 0`` onto fine ``y = -1``.

    The upstream operator is hard-wired to a refinement ratio of 2.
    """
    if f_fine.ndim != 3 or f_coarse.ndim != 3:
        raise ValueError("fields must be (q, ny, nx)")
    out = np.array(f_fine, copy=True)
    if direction == "left":
        out[LEFT_DIRS, 0::2, -1] = f_coarse[LEFT_DIRS, :, 0]
        out[LEFT_DIRS, 1::2, -1] = f_coarse[LEFT_DIRS, :, 0]
    elif direction == "right":
        out[RIGHT_DIRS, 0::2, 0] = f_coarse[RIGHT_DIRS, :, -1]
        out[RIGHT_DIRS, 1::2, 0] = f_coarse[RIGHT_DIRS, :, -1]
    elif direction == "up":
        out[UP_DIRS, 0, 0::2] = f_coarse[UP_DIRS, -1, :]
        out[UP_DIRS, 0, 1::2] = f_coarse[UP_DIRS, -1, :]
    elif direction == "down":
        out[DOWN_DIRS, -1, 0::2] = f_coarse[DOWN_DIRS, 0, :]
        out[DOWN_DIRS, -1, 1::2] = f_coarse[DOWN_DIRS, 0, :]
    else:  # pragma: no cover
        raise ValueError(f"unknown direction {direction!r}; expected left/right/up/down")
    return out
