"""Kovasznay (1948) 2D steady incompressible Navier-Stokes exact solution.

Reference
---------
Kovasznay, L. I. G. (1948). *Laminar flow behind a two-dimensional grid.*
Proc. Cambridge Philos. Soc. **44**, 58-62.  DOI 10.1017/S0305004100023999.

Non-dimensional form (characteristic velocity ``U0``, characteristic length
``L`` = one period in ``y``), with ``x' = x/L``, ``y' = y/L``::

    u(x',y') = U0 * (1 - exp(lam x') cos(2 pi y'))
    v(x',y') = U0 * (lam / 2 pi) exp(lam x') sin(2 pi y')
    p(x',y') = U0^2 * 0.5 * (1 - exp(2 lam x'))      (+ arbitrary constant)

    lam = Re/2 - sqrt(Re^2/4 + 4 pi^2)   (negative root of lam^2 - Re lam - 4 pi^2 = 0)
    Re  = U0 L / nu

Derivation of the compatibility condition (re-done here from scratch, because the
"lam ~= -0.5 at Re=40" figure quoted in older task sheets is wrong -- lam = -0.5
corresponds to Re ~= 78.5):

* continuity: ``u_x + v_y = -lam E c + (lam/2pi)(2 pi) E c = 0`` identically for
  every ``(x,y)`` -- the field is divergence free independent of ``lam``.
* y-momentum: ``u v_x + v v_y = -p_y + nu lap(v)`` with ``p_y = 0`` collapses to
  ``lam = nu (lam^2 - 4 pi^2)``, i.e. ``lam^2 - Re lam - 4 pi^2 = 0``.
* x-momentum then forces ``-p_x = lam E^2`` (all ``E c`` terms cancel *only* if
  the compatibility condition above holds), which integrates to
  ``p = -E^2/2 + const`` -- hence ``p = 0.5 (1 - E^2)`` up to a constant.

The functions below evaluate the exact **continuous** residual with analytic
derivatives (not finite differences), so ``ns_residual()`` returning ~1e-16
machine precision is a proof that the reference field is an exact solution and
that any measured LBM error is purely numerical.  (The pre-existing benchmark
checked the residual with finite differences and reported ~1e-3, which reads
like "the reference is only approximately a solution" -- it is not.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWOPI = 2.0 * math.pi


def kovasznay_lambda(re: float) -> float:
    """Decay rate ``lam`` = negative root of ``lam^2 - Re lam - 4 pi^2 = 0``."""
    if re <= 0:
        raise ValueError(f"re must be positive, got {re}")
    return re / 2.0 - math.sqrt(re * re / 4.0 + 4.0 * math.pi * math.pi)


@dataclass(frozen=True)
class KovasznayField:
    """Exact solution sampled on a uniform lattice across ``x' in [0, xmax]``.

    ``x[i] = i / ny`` (``ny`` cells per unit length -> ``L = ny`` lattice units,
    which is what makes ``Re = U0 * ny / nu``), ``y[j] = j / ny`` with the
    ``y`` direction 1-periodic.
    """

    re: float
    u0: float
    lam: float
    nu: float
    xmax: float
    ny: int

    @property
    def nx(self) -> int:
        return int(round(self.xmax * self.ny))

    @property
    def tau(self) -> float:
        """BGK relaxation time ``tau = 1/2 + 3 nu`` (lattice units)."""
        return 0.5 + 3.0 * self.nu

    @property
    def ma_max(self) -> float:
        return 2.0 * self.u0 / math.sqrt(1.0 / 3.0)

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.nx, dtype=np.float64) / self.ny

    @property
    def y(self) -> np.ndarray:
        return np.arange(self.ny, dtype=np.float64) / self.ny

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        xx, yy = np.meshgrid(self.x, self.y, indexing="xy")  # (ny, nx)
        return xx, yy

    def u(self, xx=None, yy=None) -> np.ndarray:
        if xx is None:
            xx, yy = self._grid()
        return self.u0 * (1.0 - np.exp(self.lam * xx) * np.cos(TWOPI * yy))

    def v(self, xx=None, yy=None) -> np.ndarray:
        if xx is None:
            xx, yy = self._grid()
        return self.u0 * (self.lam / TWOPI) * np.exp(self.lam * xx) * np.sin(TWOPI * yy)

    def p(self, xx=None, yy=None) -> np.ndarray:
        if xx is None:
            xx, yy = self._grid()
        return 0.5 * self.u0**2 * (1.0 - np.exp(2.0 * self.lam * xx))

    # -- exact continuous residual (analytic derivatives) -------------------

    def ns_residual(self) -> dict:
        """Max-norm residual of the steady incompressible N-S equations.

        Evaluated on the **non-dimensional** field (velocity scaled by ``U0``,
        pressure by ``U0^2``, ``nu = 1/Re``) because ``U.grad U = -grad P +
        (1/Re) lap U`` is the only form in which the two terms balance at the
        same order.  Derivatives are closed-form, so a value near machine
        epsilon certifies that the sampled field is an *exact* solution; the
        only error left is floating point.
        """
        xx, yy = self._grid()
        lam, nu = self.lam, 1.0 / self.re
        E = np.exp(lam * xx)
        c, s = np.cos(TWOPI * yy), np.sin(TWOPI * yy)
        a = lam / TWOPI

        # dimensionless velocity (amplitude 1) and pressure
        u = 1.0 - E * c
        v = a * E * s
        p = 0.5 * (1.0 - E * E)

        ux = -lam * E * c
        uy = TWOPI * E * s
        uxx = -lam * lam * E * c
        uyy = TWOPI * TWOPI * E * c
        vx = a * lam * E * s
        vy = lam * E * c
        vxx = a * lam * lam * E * s
        vyy = -a * TWOPI * TWOPI * E * s
        px = -lam * np.exp(2.0 * lam * xx)
        py = np.zeros_like(px)

        rx = u * ux + v * uy + px - nu * (uxx + uyy)
        ry = u * vx + v * vy + py - nu * (vxx + vyy)
        div = ux + vy
        return {
            "res_mom_x_max": float(np.max(np.abs(rx))),
            "res_mom_y_max": float(np.max(np.abs(ry))),
            "res_div_max": float(np.max(np.abs(div))),
            "compat_lhs": float(lam * lam - self.re * lam - 4.0 * math.pi**2),
            "nu_dimensionless": nu,
            "nu_from_compat": float(lam / (lam * lam - 4.0 * math.pi**2)),
            "form": "dimensionless: U.grad U = -grad P + (1/Re) lap U, |U| <= 1",
        }


def make_field(
    ny: int,
    re: float = 40.0,
    u0: float = 0.03,
    xmax: float = 3.0,
) -> KovasznayField:
    """Build the reference field; ``nu = U0 * ny / Re`` (``L`` = ``ny`` cells)."""
    if ny < 4:
        raise ValueError(f"ny must be >= 4, got {ny}")
    if xmax <= 0:
        raise ValueError(f"xmax must be positive, got {xmax}")
    lam = kovasznay_lambda(re)
    nu = u0 * ny / re
    return KovasznayField(re=re, u0=u0, lam=lam, nu=nu, xmax=xmax, ny=ny)