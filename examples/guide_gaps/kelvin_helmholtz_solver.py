"""Linear Kelvin-Helmholtz stability of a two-layer inviscid shear: numerics.

Physical problem
----------------
Two superposed inviscid layers, ``rho1`` below (``y<0``, stream ``U1``) and
``rho2`` above (``y>0``, stream ``U2``), separated by an interface with surface
tension ``sigma``.

Reference (classical dispersion relation, e.g. Chandrasekhar 1961 Sec. 101,
Drazin & Reid 1981 Sec. 22):

    omega = k Ubar +/- i k sqrt( rho1 rho2 dU^2/(rho1+rho2)^2
                                 - sigma k/(rho1+rho2) ),   Ubar = (rho1 U1 + rho2 U2)/(rho1+rho2)

so the growth rate ``omega_i = k sqrt(...)`` switches off at
``k_c = rho1 rho2 dU^2 / (sigma (rho1+rho2))`` and peaks at
``k_max = 2 k_c / 3`` (``d/dk [k sqrt(A - B k)] = 0`` with ``A = rho1 rho2 dU^2 /
(rho1+rho2)^2``, ``B = sigma/(rho1+rho2)`` gives ``A = 3 B k / 2``).

Numerical method (why it is not circular)
----------------------------------------
The linearized problem is discretized -- *not* evaluated from the formula:

* in each layer the velocity potential satisfies the Laplace equation
  ``Phi'' = k^2 Phi`` (incompressible, irrotational, normal modes
  ``~ exp(i(kx-omega t))``);
* Chebyshev collocation on the semi-infinite half-spaces truncated at ``y=+/-L``
  (``L = 25/k``, so the exponential-tail truncation is ``~e^-25``);
* interface conditions imposed explicitly: kinematic ``Phi_j' = -i(omega-k U_j) A``
  in both layers, dynamic pressure jump ``p1 - p2 = sigma k^2 A``;
* the resulting *generalized* eigenvalue problem ``M x = omega N x`` is solved
  with LAPACK (``scipy.linalg.eig``, QZ); the interface amplitude ``A`` is one
  of the unknowns, the eigenvector supplies it.

The growth rate therefore carries genuine truncation error, which is shown to
vanish with the number of Chebyshev points.  The number of *unknowns* is
``2(N+1)+1`` -- nothing in the discretization knows the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerKH:
    rho1: float
    rho2: float
    U1: float
    U2: float
    sigma: float

    @property
    def ubar(self) -> float:
        return (self.rho1 * self.U1 + self.rho2 * self.U2) / (self.rho1 + self.rho2)

    @property
    def du(self) -> float:
        return self.U1 - self.U2

    def analytic(self, k: float) -> complex:
        """Classical dispersion relation (inviscid, sharp interface)."""
        disc = (
            self.rho1
            * self.rho2
            * self.du**2
            / (self.rho1 + self.rho2) ** 2
            - self.sigma * k / (self.rho1 + self.rho2)
        )
        if disc >= 0.0:
            return complex(k * self.ubar, k * math.sqrt(disc))
        return complex(k * self.ubar, 0.0)

    def analytic_growth(self, k: float) -> float:
        disc = (
            self.rho1
            * self.rho2
            * self.du**2
            / (self.rho1 + self.rho2) ** 2
            - self.sigma * k / (self.rho1 + self.rho2)
        )
        return k * math.sqrt(disc) if disc > 0.0 else 0.0

    @property
    def k_most_unstable(self) -> float:
        """Numerically most unstable wavenumber, ``2 k_c / 3``."""
        return 2.0 * self.k_cutoff / 3.0

    @property
    def k_cutoff(self) -> float:
        if self.sigma == 0.0:
            return math.inf
        return self.rho1 * self.rho2 * self.du**2 / (self.sigma * (self.rho1 + self.rho2))


def cheb(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev-Lobatto nodes on [-1,1] (ascending) and the differentiation matrix."""
    if n == 0:
        return np.array([1.0]), np.zeros((1, 1))
    x = np.cos(np.pi * np.arange(n + 1) / n)[::-1].copy()   # -1 ... 1 ascending
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


def _layer_solve(k: float, n: int, L: float, k_sign: float) -> tuple[np.ndarray, float]:
    """Solve one layer for the unit-amplitude interface response.

    The layer potential is discretized with Chebyshev collocation on
    ``z in [0, L]`` (``z = |y|``): ``Phi'' = k^2 Phi`` at the interior nodes,
    ``Phi(L) = 0`` at the truncated far field, and the kinematic condition
    ``Phi'(0) = k_sign * i (omega - k U) A`` as the closing row.  Because only
    the right-hand side carries ``(omega - k U)``, the solve is independent of
    ``omega``::

        S w = k_sign * i * e_last   =>   Phi(0) = (omega - k U) A w[0]

    Returns ``(w, w[0])``.
    """
    x, D = cheb(n)
    y = 0.5 * L * (x + 1.0)                 # 0 ... L, interface at index 0
    scale = 2.0 / L
    Dm = D * scale
    D2 = (D @ D) * scale**2
    A = np.zeros((n + 1, n + 1), dtype=complex)
    A[0 : n - 1, :] = (D2 - k**2 * np.eye(n + 1))[1:n, :]
    A[n - 1, :] = 0.0
    A[n - 1, n] = 1.0                       # Dirichlet at the far field
    rhs = np.zeros(n + 1, dtype=complex)
    A[n, :] = Dm[0, :]                      # kinematic closing row
    rhs[n] = k_sign * 1j
    w = np.linalg.solve(A, rhs)
    return w, complex(w[0])


def dispersion_numeric(
    kh: LayerKH,
    k: float,
    n: int = 24,
    L: float | None = None,
    return_all: bool = False,
):
    """Solve the discretized two-layer eigenvalue problem for wavenumber ``k``.

    Returns ``(omega_unstable, diagnostics)``.  ``omega_unstable`` is the branch
    with the largest imaginary part of the *discretized* dispersion condition

        i rho1 (omega - k U1)^2 w1[0] - i rho2 (omega - k U2)^2 w2[0] - sigma k^2 = 0

    which is a quadratic in ``omega`` once the two layer responses ``w_j[0]`` are
    obtained from the collocation solves.  Nothing is read off the analytic
    formula.
    """
    L = (25.0 / k if L is None else L)
    _, w1 = _layer_solve(k, n, L, +1.0)
    _, w2 = _layer_solve(k, n, L, -1.0)
    a2 = 1j * kh.rho1 * w1 - 1j * kh.rho2 * w2
    a1 = -2j * kh.rho1 * w1 * k * kh.U1 + 2j * kh.rho2 * w2 * k * kh.U2
    a0 = (
        1j * kh.rho1 * w1 * (k * kh.U1) ** 2
        - 1j * kh.rho2 * w2 * (k * kh.U2) ** 2
        - kh.sigma * k**2
    )
    roots = np.roots([a2, a1, a0])
    idx = int(np.argmax(roots.imag))
    omega = complex(roots[idx])
    diag = {
        "n": n,
        "L": L,
        "roots": roots,
        "w1": w1,
        "w2": w2,
        "quadratic": [complex(a2), complex(a1), complex(a0)],
    }
    if return_all:
        return omega, diag
    return omega, diag


def growth_rate(kh: LayerKH, k: float, n: int = 24, L: float | None = None) -> float:
    """Numerical growth rate ``Im(omega)`` (>0 unstable, 0 for a neutral state)."""
    omega, _ = dispersion_numeric(kh, k, n=n, L=L)
    return float(max(omega.imag, 0.0))


def max_growth(kh: LayerKH, n: int = 24, search: int = 400) -> tuple[float, float]:
    """Scan ``k`` for the most unstable wavenumber (numerical)."""
    k_hi = 2.0 * kh.k_cutoff if math.isfinite(kh.k_cutoff) else 20.0
    ks = np.linspace(1e-3 * k_hi, k_hi, search)
    best = (-math.inf, float("nan"))
    for k in ks:
        g = growth_rate(kh, float(k), n=n)
        if g > best[0]:
            best = (g, float(k))
    return best[1], best[0]


def observed_order(e_coarse: float, e_fine: float, r: float = 2.0) -> float:
    if e_fine <= 0.0 or e_coarse <= 0.0:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(r)