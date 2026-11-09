"""Method of manufactured solutions (MMS) for steady incompressible Navier-Stokes.

Design constraint
-----------------
The *same* manufactured field must be representable by three completely different
discretisations (lattice Boltzmann, cell-centred finite volume, Q2-Q1 finite
element), so the exact solution is built from a stream function::

    psi(x, y) = A sin^2(k x) sin^2(k y),      k = 2*pi

giving

    u =  psi_y  =  B s^2 S C
    v = -psi_x  = -B s c S^2,          B = 2 A k

with ``s = sin(kx), c = cos(kx), S = sin(ky), C = cos(ky)``.  The field

* is divergence free **identically** (it comes from a stream function),
* vanishes on all four walls of the unit square (``u = v = 0`` on ``x,y in {0,1}``)
  and is ``k``-periodic, so it can be used with wall BCs *and* with periodic
  boxes -- no boundary layer, no singularity, no BC/discretisation aliasing,
* has smooth bounded derivatives of every order, so the observed convergence
  order is the order of the interior scheme (this is the whole point).

Pressure is chosen independently (any smooth function is admissible because the
source term is *defined* from it)::

    p = Q s S.

The source term then follows from the momentum residual

    f = (u.grad)u + grad p - nu lap(u)

whose components are derived in closed form below (``mms_source``).  With this
source the pair ``(u, p)`` is an exact steady solution of

    (u.grad)u = -grad p + nu lap(u) + f,     div u = 0.

Verified numerically in ``tests/test_cross_solver_verification.py``
(``test_mms_source_satisfies_momentum``) by substituting the closed forms back
into the residual with high-order finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ManufacturedSolution", "TAYLOR_GREEN_MMS"]

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class ManufacturedSolution:
    """Steady MMS for incompressible Navier-Stokes on the unit square.

    Parameters
    ----------
    nu:
        Kinematic viscosity (lattice/consistent units, rho = 1).
    amplitude:
        ``A`` in ``psi = A sin^2(kx) sin^2(ky)``; the peak velocity scale is
        ``B = 2 A k``.
    pressure_scale:
        ``Q`` in ``p = Q sin(kx) sin(ky)``.
    k:
        Wavenumber, ``2*pi`` by default (one wavelength per unit box).
    """

    nu: float = 0.02
    amplitude: float = 0.25 / (2.0 * TWOPI)  # -> B = 2*A*k = 0.25
    pressure_scale: float = 0.1
    k: float = TWOPI

    # ------------------------------------------------------------------ fields
    @property
    def B(self) -> float:
        """Velocity amplitude ``B = 2 A k``."""
        return 2.0 * self.amplitude * self.k

    def _trig(self, x, y):
        k = self.k
        s, c = np.sin(k * x), np.cos(k * x)
        S, C = np.sin(k * y), np.cos(k * y)
        return s, c, S, C

    def velocity(self, x, y):
        """Exact velocity ``(u, v)``."""
        s, c, S, C = self._trig(x, y)
        B = self.B
        return B * s * s * S * C, -B * s * c * S * S

    def pressure(self, x, y):
        """Exact pressure (gauge arbitrary, ``rho = 1``)."""
        s, c, S, C = self._trig(x, y)
        return self.pressure_scale * s * S

    def vorticity(self, x, y):
        """Exact vorticity ``omega = v_x - u_y = -lap(psi)``.

        ``psi = A s^2 S^2`` is *not* a Laplacian eigenfunction, so the
        curvature form is used explicitly::

            omega = -B k [cos(2kx) sin^2(ky) + sin^2(kx) cos(2ky)],  B = 2Ak

        which has zero spatial mean (required by the periodic Poisson problem
        ``lap(psi) = -omega``).
        """
        s, c, S, C = self._trig(x, y)
        k, B = self.k, self.B
        return -B * k * (np.cos(2 * k * x) * S * S + s * s * np.cos(2 * k * y))

    # ----------------------------------------------------------------- sources
    def source(self, x, y):
        """Body force ``f = (u.grad)u + grad p - nu lap(u)`` (closed form).

        Derivation (all terms below are the analytic expressions; the algebra
        uses ``c^2 + s^2 = 1`` and ``C^2 + S^2 = 1``):

        ``(u.grad)u|_x = B^2 k s^3 c S^2``,  ``(u.grad)u|_y = B^2 k s^2 S^3 C``
        ``lap(u) = 2 B k^2 S C (1 - 4 s^2)``,  ``lap(v) = 2 B k^2 s c (4 S^2 - 1)``
        ``p_x = Q k c S``,  ``p_y = Q k s C``.
        """
        s, c, S, C = self._trig(x, y)
        k, B, nu, Q = self.k, self.B, self.nu, self.pressure_scale
        nl_x = B * B * k * s**3 * c * S * S
        nl_y = B * B * k * s * s * S**3 * C
        p_x = Q * k * c * S
        p_y = Q * k * s * C
        lap_u = 2.0 * B * k * k * S * C * (1.0 - 4.0 * s * s)
        lap_v = 2.0 * B * k * k * s * c * (4.0 * S * S - 1.0)
        return nl_x + p_x - nu * lap_u, nl_y + p_y - nu * lap_v

    def curl_source(self, x, y):
        """``d f_y/dx - d f_x/dy`` in closed form (used by the psi-omega FVM).

        With ``s=sin kx, c=cos kx, S=sin ky, C=cos ky`` and ``B = 2Ak``::

            curl(f) = 2 B^2 k^2 s c S C (S^2 - s^2)
                      - 2 nu B k^3 [ (c^2-s^2)(4S^2-1) - (C^2-S^2)(1-4s^2) ]

        The pressure contribution cancels identically (curl grad p = 0), which is
        itself a useful self-check.
        """
        s, c, S, C = self._trig(x, y)
        k, B, nu = self.k, self.B, self.nu
        adv = 2.0 * B * B * k * k * s * c * S * C * (S * S - s * s)
        vis = -2.0 * nu * B * k**3 * (
            (c * c - s * s) * (4.0 * S * S - 1.0) - (C * C - S * S) * (1.0 - 4.0 * s * s)
        )
        return adv + vis
