"""Spherical-bubble cavitation dynamics kernel and its analytic anchors.

The kernel used by the mixture cavitation solvers of this repository is the
Schnerr-Sauer vapour source term

    S_alpha = 3 C alpha (1 - alpha) / R_b * sqrt(2 |p - p_v| / (3 rho_l)) * sgn(p_v - p)

with the bubble radius tied to the vapour fraction through
``alpha = n (4/3) pi R_b^3 / (1 + n (4/3) pi R_b^3)``.  Before that source term
can be trusted for a sheet/cloud cavity it has to reproduce the *single bubble*
dynamics it is derived from, which is what this module checks:

* Rayleigh collapse time of a vapour bubble in an infinite pressure field,
  ``T_c = 0.9147 R0 sqrt(rho/dp)`` (Rayleigh 1917) -- integrated with the full
  Rayleigh-Plesset equation including gas, surface tension and viscosity;
* Minnaert natural frequency ``f = (1/2 pi R0) sqrt(3 gamma p0 / rho)``;
* consistency of :func:`schnerr_sauer_source` with the volume-equivalent
  ``d alpha / dt`` obtained from the same Rayleigh-Plesset radius history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RayleighPlesset",
    "rayleigh_collapse_time",
    "minnaert_frequency",
    "schnerr_sauer_source",
    "alpha_from_radius",
    "radius_from_alpha",
]


@dataclass
class RayleighPlesset:
    """Rayleigh-Plesset bubble in a possibly time-dependent ambient pressure.

    ``R R'' + 3/2 R'^2 = (p_b - p_inf(t)) / rho`` with the bubble pressure

        p_b = p_v + p_g0 (R0/R)^(3 kappa) - 2 sigma / R - 4 mu R'/R
    """

    R0: float = 1.0e-4
    p_inf: float = 1.0e5
    p_v: float = 2.3e3
    rho: float = 1.0e3
    sigma: float = 0.0728
    mu: float = 1.0e-3
    kappa: float = 1.4
    p_g0: float = 0.0
    accel: float | None = None      # driving dp; None -> p_inf - p_v
    R_init_frac: float = 0.0        # initial radial perturbation (for spectra)
    history: dict = field(default_factory=dict, repr=False)

    @property
    def dp(self) -> float:
        """Driving pressure difference used for the collapse time reference."""
        if self.accel is not None:
            return abs(float(self.accel))
        return max(abs(self.p_inf - self.p_v), 1e-30)

    def bubble_pressure(self, R: float, Rdot: float) -> float:
        p = self.p_v - 2.0 * self.sigma / max(R, 1e-30)
        if self.p_g0:
            p += self.p_g0 * (self.R0 / max(R, 1e-30)) ** (3.0 * self.kappa)
        p -= 4.0 * self.mu * Rdot / max(R, 1e-30)
        return p

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        R, Rdot = y
        R = max(R, 1e-12)
        p_b = self.bubble_pressure(R, Rdot)
        Rddot = (p_b - self.p_inf) / self.rho - 1.5 * Rdot * Rdot
        return np.array([Rdot, Rddot / R])

    def initial_state(self) -> np.ndarray:
        """Radius ``R0 (1 + R_init_frac)``, zero radial velocity."""
        return np.array([self.R0 * (1.0 + self.R_init_frac), 0.0])

    # ------------------------------------------------------------------
    def integrate(self, t_end: float, dt: float | None = None,
                  n_out: int = 4000) -> dict:
        """RK4 integration; the step is auto-sized from the Rayleigh time."""
        T_ref = rayleigh_collapse_time(self.R0, self.dp, self.rho)
        if dt is None:
            dt = T_ref / 20000.0
        n = int(np.ceil(t_end / dt))
        y = self.initial_state()
        ts = np.empty(n + 1)
        Rs = np.empty(n + 1)
        Rds = np.empty(n + 1)
        ts[0], Rs[0], Rds[0] = 0.0, y[0], y[1]
        t = 0.0
        for i in range(n):
            k1 = self.rhs(t, y)
            k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.rhs(t + dt, y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            ts[i + 1], Rs[i + 1], Rds[i + 1] = t, y[0], y[1]
        return {"t": ts, "R": Rs, "Rdot": Rds, "dt": dt, "T_ref": T_ref}

    def collapse_time(self, dt: float | None = None) -> dict:
        """Integrate until the bubble radius reaches ``R0/100``."""
        R_stop = self.R0 * 0.01
        T_ref = rayleigh_collapse_time(self.R0, self.dp, self.rho)
        if dt is None:
            dt = T_ref / 40000.0
        y = self.initial_state()
        t = 0.0
        n_max = int(4.0 * T_ref / dt)
        for _ in range(n_max):
            k1 = self.rhs(t, y)
            k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.rhs(t + dt, y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            if y[0] <= R_stop:
                break
        return {"T_c": t, "T_rayleigh": T_ref, "dt": dt, "R_stop": R_stop,
                "R_final": float(y[0]), "Rdot_final": float(y[1])}


def rayleigh_collapse_time(R0: float, dp: float, rho: float) -> float:
    """Rayleigh (1917) collapse time of an empty cavity: ``0.9147 R0 sqrt(rho/dp)``."""
    return 0.9147 * R0 * np.sqrt(rho / max(dp, 1e-30))


def minnaert_frequency(R0: float, p0: float, rho: float, gamma: float = 1.4) -> float:
    """Minnaert natural frequency of a gas bubble, in Hz."""
    return float(np.sqrt(3.0 * gamma * p0 / rho) / (2.0 * np.pi * R0))


def alpha_from_radius(R, n_bub: float) -> np.ndarray:
    """Vapour volume fraction from the bubble radius and the bubble density."""
    V = n_bub * (4.0 / 3.0) * np.pi * np.asarray(R, dtype=float) ** 3
    return V / (1.0 + V)


def radius_from_alpha(alpha, n_bub: float):
    """Inverse of :func:`alpha_from_radius` (the Schnerr-Sauer closure)."""
    a = np.clip(np.asarray(alpha, dtype=float), 1e-12, 1.0 - 1e-12)
    return (a / (1.0 - a) * 3.0 / (4.0 * np.pi * n_bub)) ** (1.0 / 3.0)


def schnerr_sauer_source(alpha, p, p_v: float, rho_l: float, R_b, C: float = 1.0,
                         rho_v: float = 0.0):
    """Schnerr-Sauer net vapour source term ``S_alpha`` (production if p < p_v)."""
    a = np.asarray(alpha, dtype=float)
    dp = np.abs(np.asarray(p, dtype=float) - p_v)
    rate = np.sqrt(np.maximum(2.0 * dp / (3.0 * rho_l), 0.0))
    pref = 3.0 * C * a * (1.0 - a) / np.maximum(np.asarray(R_b, dtype=float), 1e-30)
    sign = np.where(np.asarray(p, dtype=float) < p_v, 1.0, -1.0)
    return pref * rate * sign