"""Nodal Gauss-Lobatto reference element for the DG verification kernels.

Only the 1-D building blocks are built here; multi-dimensional operators are
assembled as tensor products applied axis by axis (see :mod:`scalar` and
:mod:`euler1d`).

Notation (reference element ``xi in [-1, 1]``, physical cell width ``dx``):

* ``V[q, j]  = l_j(xi_q)``   -- Lagrange basis at the ``nq`` Gauss points
* ``Vr[q, j] = l_j'(xi_q)``
* ``M  = V^T W V``                              -- mass matrix
* ``Gr[k, j] = sum_q Vr[q, k] w_q V[q, j]``      -- ``int l_k' l_j dxi``
* ``S[k, 0] = -l_k(-1)``, ``S[k, 1] = +l_k(+1)`` -- face selection

The weak form of ``u_t + f(u)_x = 0`` on one element then reads

    du/dt = (2/dx) M^-1 ( -Gr f + S [f*_left, f*_right] )

where ``f`` are the nodal flux values and ``f*`` the numerical face fluxes.
With Lobatto nodes the element endpoints *are* degrees of freedom, so no L2
projection is needed to evaluate the face traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = ["RefElement", "lobatto_nodes", "ref_element"]


def lobatto_nodes(degree: int) -> np.ndarray:
    """Gauss-Lobatto-Legendre nodes on ``[-1, 1]`` (``degree + 1`` nodes)."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if degree == 0:
        return np.array([0.0])
    if degree == 1:
        return np.array([-1.0, 1.0])
    from numpy.polynomial import legendre as _leg

    interior = np.sort(_leg.Legendre.basis(degree).deriv().roots())
    return np.concatenate(([-1.0], interior, [1.0]))


@dataclass(frozen=True)
class RefElement:
    """Pre-computed 1-D operators for one polynomial degree."""

    degree: int
    nodes: np.ndarray  # (n,) Lobatto nodes
    wq: np.ndarray  # (nq,) Gauss quadrature weights
    V: np.ndarray  # (nq, n)
    Vr: np.ndarray  # (nq, n)
    M: np.ndarray  # (n, n)
    Minv: np.ndarray  # (n, n)
    Gr: np.ndarray  # (n, n)  int l_k' l_j
    S: np.ndarray  # (n, 2)  face selection (-left, +right)
    D: np.ndarray  # (n, n)  nodal differentiation matrix
    Ven_inv: np.ndarray  # (n, n) nodal -> Legendre modal transform

    @property
    def n_node(self) -> int:
        return int(self.nodes.size)

    def to_quadrature(self, u: np.ndarray) -> np.ndarray:
        """Nodal values ``(..., n)`` -> values at the Gauss points ``(..., nq)``."""
        return u @ self.V.T

    def integral(self, u: np.ndarray, dx: float) -> np.ndarray:
        """Exact (to quadrature) cell integral of ``u`` over a width-``dx`` cell.

        ``u`` has shape ``(..., n)``; the node axis is reduced.
        """
        uq = self.to_quadrature(u)
        return 0.5 * dx * (uq @ self.wq)


@lru_cache(maxsize=None)
def _build(degree: int, nq_extra: int) -> RefElement:
    nodes = lobatto_nodes(degree)
    n = nodes.size
    nq = max(degree + nq_extra, 2)
    xq, wq = np.polynomial.legendre.leggauss(nq)

    # Lagrange basis via the monomial Vandermonde (exact for low degree).
    powers = np.arange(n, dtype=np.float64)
    vmono = nodes[:, None] ** powers[None, :]  # (n, n)
    vmono_inv = np.linalg.inv(vmono)
    vq = xq[:, None] ** powers[None, :]  # (nq, n) monomials at quad points
    V = vq @ vmono_inv
    dq = np.zeros_like(vq)
    dq[:, 1:] = powers[1:][None, :] * vq[:, :-1]
    Vr = dq @ vmono_inv

    W = np.diag(wq)
    M = V.T @ W @ V
    Minv = np.linalg.inv(M)
    Gr = Vr.T @ W @ V
    S = np.zeros((n, 2))
    S[0, 0] = -1.0  # -l_0(-1)
    S[-1, 1] = 1.0  # +l_{n-1}(+1)

    D = (Gr @ Minv).T  # nodal differentiation matrix l_j'(x_i), exact for deg <= n-1

    # Legendre modal basis evaluated at the nodes -> nodal/mean conversion.
    Ven = np.stack(
        [np.polynomial.legendre.Legendre.basis(k)(nodes) for k in range(n)], axis=1
    )
    Ven_inv = np.linalg.inv(Ven)

    return RefElement(
        degree=degree,
        nodes=nodes,
        wq=wq,
        V=V,
        Vr=Vr,
        M=M,
        Minv=Minv,
        Gr=Gr,
        S=S,
        D=D,
        Ven_inv=Ven_inv,
    )


def ref_element(degree: int, nq_extra: int = 3) -> RefElement:
    """Return the (cached) reference element for ``degree``."""
    return _build(int(degree), int(nq_extra))