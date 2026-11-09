"""Exner bed-evolution equation with a Meyer-Peter-Muler type bedload flux.

Model
-----
Net bedload flux per unit width (Meyer-Peter & Muler form, volume flux)::

    q_s(theta) = 8 (theta - theta_cr)^{3/2} sqrt(g (s-1) d^3)   [theta > theta_cr]
    q_s(theta) = 0                                              [theta <= theta_cr]

with the Shields number ``theta = tau_b / ((rho_s - rho) g d)``.  The bed
elevation evolves by the Exner equation::

    dz_b/dt = - d q_s / dx / (1 - n)

discretised with a *conservative* finite difference so that the total bed
volume is preserved exactly (telescoping sum) up to floating-point round-off
-- this is what criterion C3 (mass drift < 1e-6) measures.

The hard threshold at ``theta_cr`` is what makes criterion C4 (negative
control: ``u* < 0.8 u*_cr`` implies ``q_s < 1e-12``) a genuine test of the
model's incipient-motion gate rather than a tautology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ExnerConfig", "mpm_flux", "exner_step", "bed_volume"]


@dataclass(frozen=True)
class ExnerConfig:
    """Physical parameters of the sediment/bed system (SI units)."""

    d: float  # grain diameter (m)
    s: float = 1.65  # rho_s / rho
    g: float = 9.81  # m s^-2
    porosity: float = 0.4  # bed porosity n
    theta_cr: float = 0.045  # critical Shields stress (model parameter)

    def __post_init__(self) -> None:
        if self.d <= 0.0 or self.s <= 1.0:
            raise ValueError("require d > 0, s > 1")
        if not (0.0 < self.porosity < 1.0):
            raise ValueError("require 0 < porosity < 1")

    @property
    def flux_scale(self) -> float:
        """sqrt(g (s-1) d^3) -- the MPM flux scale (m^2 s^-1)."""
        return math.sqrt(self.g * (self.s - 1.0) * self.d**3)


def mpm_flux(theta: np.ndarray | float, cfg: ExnerConfig) -> np.ndarray | float:
    """Meyer-Peter-Muler bedload flux (volume per unit width, m^2 s^-1).

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
) -> np.ndarray:
    """One conservative Exner update.

    ``z_b`` and ``q_s`` are 1-D arrays on the same grid (``q_s`` is the flux
    at the cell centres; face fluxes are averaged).  Zero-flux (no-slip)
    boundaries are implied, so the total bed volume is conserved exactly.
    """
    if z_b.shape != q_s.shape:
        raise ValueError("z_b and q_s must share a shape")
    n = z_b.size
    if n < 3:
        raise ValueError("need at least 3 cells")
    # Face fluxes: average of adjacent cell-centre fluxes; zero at the ends.
    q_face = np.empty(n + 1)
    q_face[0] = 0.0
    q_face[-1] = 0.0
    q_face[1:-1] = 0.5 * (q_s[:-1] + q_s[1:])
    div = (q_face[1:] - q_face[:-1]) / dx
    return z_b - (dt / (1.0 - cfg.porosity)) * div


def bed_volume(z_b: np.ndarray, dx: float, z_ref: float = 0.0) -> float:
    """Total bed volume relative to ``z_ref`` (m^2 per unit width)."""
    return float(np.sum(z_b - z_ref) * dx)
