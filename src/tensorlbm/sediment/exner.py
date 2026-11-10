"""Bed-load flux (Meyer-Peter-Mueller) and the Exner bed-evolution equation.

Physics
-------
Bed-load flux, Meyer-Peter & Mueller (1948) in the dimensionless form used by
Soulsby & Whitehouse (2005)::

    q_s = 8 (theta - theta_cr)^{3/2} sqrt( g (s-1) d^3 )       [m^2 s^-1]
          (exactly zero when theta <= theta_cr)

Exner (1925) equation for the bed elevation ``z_b`` (positive upwards)::

    (1 - n) dz_b/dt = - dq_s/dx                    [n = porosity]

Discretisation
--------------
``exner_step`` is written in *conservative flux form*: the cell-face fluxes are
``q_face = 0.5 (q_i + q_{i+1})`` with **zero** flux at both domain ends, so the
divergence telescopes and the total bed volume ``sum(z_b) dx`` is conserved to
floating-point round-off for **any** ``dt`` (criterion C3).  The zero end fluxes
are the physical no-flux boundary condition of a periodic/closed reach; they are
what makes the discrete statement of mass conservation exact rather than
approximate.

``stable_dt`` returns the CFL-limited time step for the explicit scheme,
``dt <= C dx / max|dq_s/dz_b|``; without it the coupled run can drive the water
depth negative (that is exactly the failure mode the previous version of this
module hit in the C3 test: ``h_ref - z_b < 0`` produced ``nan``).

``exner_step(..., enabled=False)`` is the "Exner off" switch required by
negative control C4(c): with the bed update disabled the bed elevation must
remain **bit-identical** to its initial state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ExnerConfig",
    "bed_volume",
    "exner_step",
    "mpm_flux",
    "stable_dt",
]


@dataclass(frozen=True)
class ExnerConfig:
    """Sediment / bed parameters (SI units)."""

    d: float  # grain diameter (m)
    s: float = 1.65  # rho_s / rho
    g: float = 9.81  # m s^-2
    porosity: float = 0.4  # n
    theta_cr: float = 0.047  # critical Shields stress (incipient-motion gate)

    def __post_init__(self) -> None:
        if self.d <= 0.0:
            raise ValueError("require d > 0")
        if self.s <= 1.0:
            raise ValueError("require s > 1")
        if not (0.0 < self.porosity < 1.0):
            raise ValueError("require 0 < porosity < 1")

    @property
    def flux_scale(self) -> float:
        """sqrt(g (s-1) d^3) -- the MPM flux scale (m^2 s^-1)."""
        return math.sqrt(self.g * (self.s - 1.0) * self.d**3)


def mpm_flux(theta: np.ndarray | float, cfg: ExnerConfig) -> np.ndarray | float:
    """Meyer-Peter-Mueller bed-load flux (volume per unit width, m^2 s^-1).

    Zero for ``theta <= theta_cr`` (incipient-motion gate).  Accepts a scalar
    or an array of Shields numbers.
    """
    theta = np.asarray(theta, dtype=float)
    excess = np.clip(theta - cfg.theta_cr, 0.0, None)
    q = 8.0 * excess**1.5 * cfg.flux_scale
    return float(q) if q.ndim == 0 else q


def exner_step(
    z_b: np.ndarray,
    q_s: np.ndarray,
    dx: float,
    dt: float,
    cfg: ExnerConfig,
    enabled: bool = True,
) -> np.ndarray:
    """One conservative Exner update (new bed elevation).

    ``z_b`` and ``q_s`` are 1-D arrays on the same grid (``q_s`` at the cell
    centres; face fluxes are averaged).  Zero flux at both ends means the
    total bed volume is conserved exactly.  With ``enabled=False`` the bed is
    returned unchanged (bit-identical), which is the C4(c) negative control.
    """
    z_b = np.asarray(z_b, dtype=float)
    if not enabled:
        return z_b.copy()
    q_s = np.asarray(q_s, dtype=float)
    if z_b.shape != q_s.shape:
        raise ValueError("z_b and q_s must share a shape")
    n = z_b.size
    if n < 3:
        raise ValueError("need at least 3 cells")
    q_face = np.empty(n + 1)
    q_face[0] = 0.0
    q_face[-1] = 0.0
    q_face[1:-1] = 0.5 * (q_s[:-1] + q_s[1:])
    div = (q_face[1:] - q_face[:-1]) / dx
    return z_b - (dt / (1.0 - cfg.porosity)) * div


def bed_volume(z_b: np.ndarray, dx: float, z_ref: float = 0.0) -> float:
    """Total bed volume relative to ``z_ref`` (m^2 per unit width)."""
    return float(np.sum(np.asarray(z_b, dtype=float) - z_ref) * dx)


def stable_dt(
    z_b: np.ndarray,
    dx: float,
    cfg: ExnerConfig,
    theta_fn,
    cfl: float = 0.2,
) -> float:
    """CFL-limited explicit time step for the coupled Exner update.

    ``theta_fn(z_b)`` returns the cell-centre Shields array for a bed shape.
    The bed-wave speed is ``dq_s/dz_b / (1-n)``; a finite-difference estimate
    is used and the step is limited so the bed wave cannot cross a cell.
    """
    z_b = np.asarray(z_b, dtype=float)
    q0 = np.asarray(mpm_flux(np.asarray(theta_fn(z_b)), cfg), dtype=float)
    eps = 1.0e-3 * cfg.d
    dq = np.asarray(mpm_flux(np.asarray(theta_fn(z_b + eps)), cfg), dtype=float) - q0
    speed = float(np.max(np.abs(dq / eps))) / (1.0 - cfg.porosity)
    if speed <= 0.0:
        return 1.0e30
    return cfl * dx / speed