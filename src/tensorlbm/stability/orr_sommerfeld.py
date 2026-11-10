"""Chebyshev spectral Orr-Sommerfeld solver for plane Poiseuille flow.

Why this module exists
----------------------
Node ``P269W2-N04`` audits the *lattice* linear stability of plane Poiseuille
flow: does the D2Q9 operator (BGK / MRT / regularised) reproduce the
Orr-Sommerfeld (OS) growth rates that a spectral discretisation of the
Navier-Stokes equations gives?  That question needs an independent, high
accuracy reference spectrum, not a second lattice model.  This module provides
it: a classical Chebyshev collocation discretisation of the OS equation in the
``(v, omega)`` (wall-normal velocity / wall-normal vorticity) form, verified
against the standard benchmark of Orszag (1971).

Non-dimensionalisation (Orszag 1971 convention)
-----------------------------------------------
* wall-normal coordinate ``y in [-1, 1]``, channel half-width ``h = 1``;
* base flow ``U(y) = 1 - y^2`` (centreline velocity ``U_max = 1``);
* ``Re = U_max h / nu``;
* disturbance ``exp(i alpha (x - c t))`` so that ``c_i > 0`` is unstable and the
  growth rate in units of ``U_max / h`` is ``omega_i = alpha c_i``.

Benchmark: ``alpha = 1``, ``Re = 10000`` -> ``c = 0.23752649 + 0.00373967 i``
(Orszag 1971, table 1; reproduced in Schmid & Henningson 2001, table 3.1).
Plane Poiseuille flow is linearly stable for ``Re < Re_c = 5772.22``.

Discretisation
--------------
``(D^2 - a^2) v = omega`` and
``(D^2 - a^2) omega = i a Re [ (U - c)(D^2 - a^2) v - U'' v ]``
are enforced at the ``N-1`` interior Chebyshev-Gauss-Lobatto points, with the
four no-slip rows ``v(+-1) = v'(+-1) = 0`` replacing the rows of the two blocks
closest to the walls.  The result is a square generalised eigenvalue problem of
size ``2(N+1)`` whose mass matrix is singular; QZ returns a few infinite
eigenvalues which are filtered out.  This "matrix method" (Gottlieb & Orszag
1977, ch. 8; Trefethen 2000, ch. 7) keeps the 4th-order operator free of the
spurious modes produced by a single 4th-order polynomial ansatz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "ORSZAG_RE10000_ALPHA1",
    "ORSZAG_RE5772",
    "OSSpectrum",
    "cheb",
    "cheb_interp",
    "os_spectrum",
    "os_growth_rate",
    "orszag_benchmark_check",
]

#: Orszag (1971) benchmark eigenvalue at ``alpha = 1``, ``Re = 10000``.
ORSZAG_RE10000_ALPHA1 = complex(0.23752648882034, 0.0037396706229)

#: Critical Reynolds number of plane Poiseuille flow (Orszag 1971).
ORSZAG_RE5772 = 5772.2218


def cheb(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev differentiation matrix ``D`` and Gauss-Lobatto points ``x``.

    ``x = cos(pi j / n)`` for ``j = 0..n`` (so ``x[0] = +1``, ``x[n] = -1``);
    ``D`` maps nodal values to nodal derivatives (Trefethen 2000, ``cheb.m``).
    """
    if n == 0:
        return np.zeros((1, 1)), np.ones(1)
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def cheb_interp(
    values: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray
) -> np.ndarray:
    """Barycentric Lagrange interpolation from Lobatto nodes onto ``x_dst``.

    Used to move an OS eigenfunction from its Chebyshev grid onto the uniform
    lattice nodes so the two solutions can be compared point-wise.  Weights are
    the standard Lobatto weights ``(-1)^j * delta_j`` (delta_0 = delta_n = 1/2).
    """
    values = np.asarray(values, dtype=complex).ravel()
    x_src = np.asarray(x_src, dtype=float).ravel()
    n = x_src.size - 1
    w = (-1.0) ** np.arange(n + 1)
    w[0] *= 0.5
    w[n] *= 0.5
    x_dst = np.asarray(x_dst, dtype=float).ravel()
    out = np.empty(x_dst.shape, dtype=complex)
    for k, xd in enumerate(x_dst):
        diff = xd - x_src
        hit = np.nonzero(np.abs(diff) < 1e-14)[0]
        if hit.size:
            out[k] = values[hit[0]]
            continue
        num = w / diff
        out[k] = np.dot(num, values) / num.sum()
    return out


@dataclass
class OSSpectrum:
    """Spectrum of the Orr-Sommerfeld operator at one ``(alpha, Re)``."""

    alpha: float
    Re: float
    n_cheb: int
    c: np.ndarray  #: eigenvalues (complex phase speeds)
    v: np.ndarray  #: eigenfunctions, shape ``(n_c, phi.ndim)``
    y: np.ndarray  #: Chebyshev nodes the eigenfunctions live on
    note: dict = None

    @property
    def omega(self) -> np.ndarray:
        """Complex frequency ``omega = alpha * c`` (units of ``U_max / h``)."""
        return self.alpha * self.c

    @property
    def growth(self) -> np.ndarray:
        """Growth rate ``omega_i`` (units of ``U_max / h``)."""
        return self.omega.imag

    @property
    def index_most_unstable(self) -> int:
        return int(np.argmax(self.c.imag))

    @property
    def c_leading(self) -> complex:
        return complex(self.c[self.index_most_unstable])

    @property
    def v_leading(self) -> np.ndarray:
        return self.v[self.index_most_unstable]

    @property
    def sigma_leading(self) -> float:
        """Leading growth rate ``alpha * c_i`` in units of ``U_max / h``."""
        return float(self.alpha * self.c_leading.imag)

    def summary(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "Re": float(self.Re),
            "n_cheb": int(self.n_cheb),
            "c_leading": [float(self.c_leading.real), float(self.c_leading.imag)],
            "sigma_leading": float(self.sigma_leading),
            "n_eigenvalues": int(self.c.size),
        }


def os_spectrum(
    alpha: float,
    Re: float,
    n_cheb: int = 96,
    *,
    base: Callable[[np.ndarray], np.ndarray] | None = None,
    base_second: Callable[[np.ndarray], np.ndarray] | None = None,
) -> OSSpectrum:
    """Chebyshev spectral Orr-Sommerfeld eigenvalue problem.

    Args:
        alpha: streamwise wavenumber (units of ``1/h``).
        Re: Reynolds number ``U_max h / nu``.
        n_cheb: number of Chebyshev intervals (``n_cheb + 1`` Gauss-Lobatto
            nodes).  ``n_cheb = 96`` reproduces the Orszag benchmark to better
            than ``1e-8`` (see :func:`orszag_benchmark_check`).
        base: optional base flow ``U(y)`` on ``[-1, 1]`` (default ``1 - y^2``).
        base_second: optional ``U''(y)`` (default ``-2``).

    Returns:
        :class:`OSSpectrum` holding the finite spectrum, the corresponding
        wall-normal velocity eigenfunctions and the node set.
    """
    alpha = float(alpha)
    Re = float(Re)
    D, x = cheb(int(n_cheb))
    n1 = x.size
    I = np.eye(n1)
    D2 = D @ D
    U = (1.0 - x**2) if base is None else np.asarray(base(x), dtype=float)
    Upp = (-2.0 * np.ones(n1)) if base_second is None else np.asarray(
        base_second(x), dtype=float
    )
    L = D2 - alpha**2 * I
    k = 1j * alpha * Re

    A = np.zeros((2 * n1, 2 * n1), dtype=complex)
    B = np.zeros((2 * n1, 2 * n1), dtype=complex)

    interior = np.arange(1, n1 - 1)
    # (D^2 - a^2) v - omega = 0
    A[interior, :n1] = L[interior, :]
    A[interior, n1:] = -I[interior, :]
    # (D^2 - a^2) omega - k (U (D^2 - a^2) v - U'' v) = -c k (D^2 - a^2) v
    A[n1 + interior, :n1] = -k * (np.diag(U) @ L - np.diag(Upp))[interior, :]
    A[n1 + interior, n1:] = L[interior, :]
    B[n1 + interior, :n1] = -k * L[interior, :]

    # no-slip: v(+1) = 0 (row 0), v(-1) = 0 (row n1-1)
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[n1 - 1, :] = 0.0
    A[n1 - 1, n1 - 1] = 1.0
    # no-slip derivative: v'(+1) = 0, v'(-1) = 0 (rows of the second block)
    A[n1, :] = 0.0
    A[n1, :n1] = D[0, :]
    A[2 * n1 - 1, :] = 0.0
    A[2 * n1 - 1, :n1] = D[n1 - 1, :]

    from scipy import linalg  # noqa: PLC0415  (optional heavy import kept local)

    c_all, vec = linalg.eig(A, B, right=True)
    finite = np.isfinite(c_all) & (np.abs(c_all) < 1e6)
    c = c_all[finite]
    vec = vec[:, finite]
    v = vec[:n1, :].T  # eigenfunctions of v (unknown #1 block)
    # normalise each eigenfunction for reproducible overlaps
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    v = v / norms
    order = np.argsort(-c.imag)
    return OSSpectrum(
        alpha=alpha, Re=Re, n_cheb=int(n_cheb), c=c[order], v=v[order], y=x,
        note={"n_infinite": int(np.count_nonzero(~finite))},
    )


def os_growth_rate(alpha: float, Re: float, n_cheb: int = 96) -> float:
    """Leading OS growth rate ``alpha * c_i`` in units of ``U_max / h``."""
    return os_spectrum(alpha, Re, n_cheb).sigma_leading


def orszag_benchmark_check(n_cheb: int = 96) -> dict:
    """Reproduce the Orszag (1971) eigenvalue ``alpha = 1``, ``Re = 10000``.

    Returns a dict with the computed eigenvalue, the reference value and the
    relative error on the real and imaginary parts.
    """
    spec = os_spectrum(1.0, 10000.0, n_cheb)
    c = spec.c_leading
    ref = ORSZAG_RE10000_ALPHA1
    return {
        "n_cheb": int(n_cheb),
        "c_computed": [float(c.real), float(c.imag)],
        "c_reference": [float(ref.real), float(ref.imag)],
        "abs_error": float(abs(c - ref)),
        "rel_error_complex": float(abs(c - ref) / abs(ref)),
        "rel_error_real": float(abs(c.real - ref.real) / abs(ref.real)),
        "rel_error_imag": float(abs(c.imag - ref.imag) / abs(ref.imag)),
    }