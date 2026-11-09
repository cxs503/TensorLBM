"""Explicit time-step bounds from the *actual* spectra of the 1-D operators.

Guessing a CFL constant for a diffusion term is how you get an "order
verification" that silently diverges at the finer grids.  Instead of guessing,
the spectral radius of the assembled 1-D DG operators is measured once on a
tiny periodic grid (``n_meas=6`` cells, ``n_node <= 6`` degrees of freedom per
cell -> an ``eigvals`` of a <= 36 x 36 matrix, microseconds) and cached.

For an explicit SSP-RK3 integrator the stability region along the negative real
axis reaches ``~2.5``; ``safety`` defaults to ``1.5`` which leaves a wide margin
and is verified empirically by the convergence benchmarks (they would blow up,
not converge, if it were too large).

The measured scalings are reported by :func:`spectral_report` and asserted in
``tests/test_dg_verification.py``:

    |lambda|_advection ~ c_adv(p) / dx        (c_adv = 6.0, 11.84, 19.16 for p=1,2,3)
    |lambda|_diffusion ~ kappa(p) / dx^2      (kappa  = 36, 148.3, 438.9 for p=1,2,3)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .element import ref_element
from .scalar import apply_axis, face_term

__all__ = ["operator_spectra_1d", "stable_dt", "spectral_report"]


def _adv_matrix(p: int, dx: float, n_cells: int = 6) -> np.ndarray:
    el = ref_element(p)
    ndof = n_cells * el.n_node
    A = np.zeros((ndof, ndof))
    for j in range(ndof):
        u = np.zeros((n_cells, el.n_node))
        u.reshape(-1)[j] = 1.0
        f = u  # unit advective speed
        dvol = apply_axis(f, el.Gr, -1)
        u_r = u.take(-1, axis=-1)
        u_l = np.roll(u.take(0, axis=-1), -1, axis=-1)
        f_star = np.where(1.0 >= 0.0, u_r, u_l)
        f_star_left = np.roll(f_star, 1, axis=-1)
        surf = face_term(f_star_left, f_star, el, -1, u.ndim)
        A[:, j] = (apply_axis(dvol - surf, el.Minv, -1) * (2.0 / dx)).reshape(-1)
    return A


def _diff_matrix(p: int, dx: float, n_cells: int = 6) -> np.ndarray:
    el = ref_element(p)
    ndof = n_cells * el.n_node
    D = np.zeros((ndof, ndof))
    for j in range(ndof):
        u = np.zeros((n_cells, el.n_node))
        u.reshape(-1)[j] = 1.0
        u_hat = u.take(-1, axis=-1)
        u_hat_l = np.roll(u_hat, 1, axis=-1)
        q = -apply_axis(u, el.Gr, -1) + face_term(u_hat_l, u_hat, el, -1, u.ndim)
        q = apply_axis(q, el.Minv, -1) * (2.0 / dx)
        q_hat = np.roll(q.take(0, axis=-1), -1, axis=-1)
        q_hat_l = np.roll(q_hat, 1, axis=-1)
        du = -apply_axis(q, el.Gr, -1) + face_term(q_hat_l, q_hat, el, -1, u.ndim)
        D[:, j] = (apply_axis(du, el.Minv, -1) * (2.0 / dx)).reshape(-1)
    return D


@lru_cache(maxsize=None)
def operator_spectra_1d(p: int) -> tuple[float, float]:
    """``(c_adv, kappa)``: ``|lambda|_adv = c_adv/dx``, ``|lambda|_diff = kappa/dx^2``.

    Measured on a 6-cell periodic grid with ``dx = 1/6``; the ``1/dx`` and
    ``1/dx^2`` scalings are exact for the scaled operators (verified by the
    ``dx``-independence of the returned constants).
    """
    dx = 1.0 / 6.0
    c_adv = float(np.abs(np.linalg.eigvals(_adv_matrix(p, dx))).max() * dx)
    kappa = float(np.abs(np.linalg.eigvals(_diff_matrix(p, dx))).max() * dx * dx)
    return c_adv, kappa


def stable_dt(
    p: int,
    dx: float,
    advective_speed: float = 0.0,
    nu: float = 0.0,
    safety: float = 1.5,
    ndim: int = 1,
) -> float:
    """Explicit step bound ``dt = safety / (ndim * (|lambda|_adv + |lambda|_diff))``.

    The spectral radius is measured on the 1-D operator; a tensor-product 2-D
    grid accumulates the per-direction contributions, so the radii are summed
    over ``ndim`` dimensions (the conservative choice).  ``safety`` keeps
    ``dt * |lambda|`` well inside the SSP-RK3 stability interval (~2.5 on the
    negative real axis); the convergence benchmarks are the proof that it is
    not too large -- they would diverge instead of converging.
    """
    c_adv, kappa = operator_spectra_1d(p)
    lam_adv = c_adv * abs(advective_speed) / dx if advective_speed else 0.0
    lam_diff = kappa * abs(nu) / (dx * dx) if nu else 0.0
    lam = ndim * (lam_adv + lam_diff)
    if lam <= 0.0:
        return float("inf")
    return float(safety / lam)


def spectral_report(degrees=(1, 2, 3)) -> dict:
    """Tabulate the measured spectral constants (used by the report + tests)."""
    return {
        f"P{p}": {"c_adv": operator_spectra_1d(p)[0], "kappa_diff": operator_spectra_1d(p)[1]}
        for p in degrees
    }