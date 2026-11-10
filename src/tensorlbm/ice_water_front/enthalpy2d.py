"""Genuine 2-D Cartesian enthalpy (diffuse-interface) solver and metrics.

The purpose of this module inside the verification surface is *contrastive*:
it solves the same circular ice-melting configuration as
:mod:`tensorlbm.ice_water_front.radial` with the classical enthalpy method, on
a 2-D Cartesian grid, and measures how many cells the phase-change interface
occupies.  The sharp tracker resolves the front inside one cut cell; the
enthalpy method spreads it over a band whose width grows with the Stefan
number.  That measurement is the evidence behind criterion C4.

The run is deliberately performed in **dimensionless units** (``rho = c = k =
alpha = 1``, melting temperature ``0``, superheat ``1``, latent heat
``L = 1/Ste``): the quantity of interest is the number of interface cells,
which is scale-free, and the reduced problem is cheap enough to run inside the
test suite.  A quantitative accuracy comparison against the literature anchor
is done with the sharp solver (criterion C2), not with this diffuse run.

Enthalpy formulation
--------------------
``H = T + L f`` with the liquid fraction ``f`` ramped linearly over a nominal
band ``|T| <= band/2``.  The energy equation ``dH/dt = lap(T)`` is advanced
explicitly and inverted analytically each step:

``T = H``                        for ``H <= -band/2``     (solid)
``T = H - L``                    for ``H >= L + band/2``  (liquid)
``T = (H - L/2)/(1 + L/band)``   in the mushy band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .audit import interface_thickness_cells, vof_circle_field

__all__ = ["Enthalpy2D", "Enthalpy2DResult", "vof_circle_thickness"]


@dataclass
class Enthalpy2DResult:
    liquid_fraction: np.ndarray
    temperature: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    front_radius: float
    interface_thickness_cells: float
    steps: int
    dt: float
    t_end: float
    ste: float
    r0: float
    half_width: float
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "ste": self.ste,
            "r0": self.r0,
            "t_end": self.t_end,
            "front_radius": self.front_radius,
            "interface_thickness_cells": self.interface_thickness_cells,
            "steps": int(self.steps),
            "dt": self.dt,
            "grid_cells": int(self.liquid_fraction.shape[0]),
        }


def _invert_enthalpy(h: np.ndarray, latent: float, band: float) -> np.ndarray:
    t = np.empty_like(h)
    solid = h <= -0.5 * band
    liquid = h >= latent + 0.5 * band
    mushy = ~(solid | liquid)
    t[solid] = h[solid]
    t[liquid] = h[liquid] - latent
    t[mushy] = (h[mushy] - 0.5 * latent) / (1.0 + latent / band)
    return t


def _liquid_fraction(t: np.ndarray, band: float) -> np.ndarray:
    return np.clip((t + 0.5 * band) / band, 0.0, 1.0)


class Enthalpy2D:
    """Explicit enthalpy solver on a Cartesian grid (dimensionless units)."""

    def __init__(
        self,
        ste: float = 0.1,
        r0: float = 2.0,
        half_width: float = 6.0,
        n: int = 72,
        t_end: float = 6.6,
        band: float = 0.05,
        cfl: float = 0.25,
    ) -> None:
        self.ste = float(ste)
        self.latent = 1.0 / ste
        self.r0 = float(r0)
        self.half = float(half_width)
        self.n = int(n)
        self.band = float(band)
        self.t_end = float(t_end)
        self.h = 2.0 * self.half / self.n
        edges = np.linspace(-self.half, self.half, self.n + 1)
        self.xc = 0.5 * (edges[:-1] + edges[1:])
        self.yc = self.xc.copy()
        self.X, self.Y = np.meshgrid(self.xc, self.yc, indexing="xy")
        self.R = np.hypot(self.X, self.Y)

        # initial: ice at T = -band/2 (fully solid), water at T = +1
        t0 = np.where(self.R <= self.r0, -0.5 * self.band, 1.0)
        self.temperature = t0
        self.enthalpy = t0 + self.latent * _liquid_fraction(t0, self.band)
        self.dt = cfl * self.h ** 2 / 4.0

    # -- numerics -------------------------------------------------------
    def _lap(self, t: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(t)
        lap[1:-1, 1:-1] = (
            t[1:-1, 2:] + t[1:-1, :-2] + t[2:, 1:-1] + t[:-2, 1:-1]
            - 4.0 * t[1:-1, 1:-1]
        ) / self.h ** 2
        return lap

    def _front_radius(self, f: np.ndarray) -> float:
        """Radius of the ``f = 0.5`` contour along the +x ray (linear interp)."""
        j = int(np.argmin(np.abs(self.yc)))
        row = f[j, :]
        k = int(np.argmin(np.abs(self.xc)))
        radii = []
        for direction in (1, -1):
            # walk outward from the centre until the liquid fraction crosses 0.5
            idx = k
            while 0 <= idx + direction < self.n and row[idx + direction] <= 0.5:
                idx += direction
            nxt = idx + direction
            if 0 <= nxt < self.n:
                f0, f1 = row[idx], row[nxt]
                x0, x1 = self.xc[idx], self.xc[nxt]
                if f1 != f0:
                    frac = (0.5 - f0) / (f1 - f0)
                    radii.append(abs(x0 + frac * (x1 - x0)))
        return float(np.mean(radii)) if radii else 0.0

    def run(self) -> Enthalpy2DResult:
        n_steps = int(math.ceil(self.t_end / self.dt))
        dt = self.t_end / n_steps if n_steps else self.dt
        temperature = self.temperature.copy()
        enthalpy = self.enthalpy.copy()
        for _ in range(n_steps):
            enthalpy[1:-1, 1:-1] += dt * self._lap(temperature)[1:-1, 1:-1]
            temperature = _invert_enthalpy(enthalpy, self.latent, self.band)
            # Dirichlet superheat on the box boundary
            temperature[0, :] = temperature[-1, :] = 1.0
            temperature[:, 0] = temperature[:, -1] = 1.0
            enthalpy[0, :] = enthalpy[-1, :] = 1.0 + 0.0
            enthalpy[:, 0] = enthalpy[:, -1] = 1.0
        f = _liquid_fraction(temperature, self.band)
        rf = self._front_radius(f)
        thick = interface_thickness_cells(f, low=0.05, high=0.95)
        return Enthalpy2DResult(
            liquid_fraction=f,
            temperature=temperature,
            xc=self.xc,
            yc=self.yc,
            front_radius=rf,
            interface_thickness_cells=thick["max_cells"],
            steps=n_steps,
            dt=dt,
            t_end=self.t_end,
            ste=self.ste,
            r0=self.r0,
            half_width=self.half,
            metadata={
                "band": self.band,
                "grid_n": self.n,
                "cell_size": self.h,
                "thickness_detail": thick,
            },
        )


def vof_circle_thickness(
    rf: float,
    half_width: float = 6.0,
    n: int = 72,
    low: float = 0.05,
    high: float = 0.95,
) -> dict:
    """Interface thickness of the *sharp* circular front on the same grid.

    The sharp representation is a sub-cell circle of radius ``rf``; its cell
    volume fractions are evaluated analytically (sub-cell sampling), which is
    exactly how a VOF/level-set visualisation of the tracked front would look.
    """
    h = 2.0 * half_width / n
    edges = np.linspace(-half_width, half_width, n + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    X, Y = np.meshgrid(xc, xc, indexing="xy")
    frac = vof_circle_field(X, Y, rf, h)
    detail = interface_thickness_cells(frac, low=low, high=high)
    detail["front_radius"] = rf
    detail["cell_size"] = h
    return detail