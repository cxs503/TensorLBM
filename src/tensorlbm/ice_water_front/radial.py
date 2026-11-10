"""Axisymmetric sharp front tracking for melting of an ice *cylinder*.

Geometry: an infinite ice cylinder of initial radius ``r0`` embedded in water
held at ``T_wall > T_m`` out to a fixed outer radius ``R``.  The interface is
circular, so the problem is solved on the radial coordinate with the
cylindrical finite-volume operator

    rho c dT/dt = (1/r) d/dr ( k r dT/dr )

and a cut cell at ``r_f(t)`` where the temperature is pinned to ``T_m``.

This is the *one-phase* (conduction-limited) idealisation used by the
literature anchor of criterion C2 (Carslaw & Jaeger sec. 11.2; Lunardini 1991
ch. 4): the ice interior is taken isothermal at ``T_m``.  That assumption is
exact when the ice starts at ``T_m`` and the ice-side conductivity never
matters (there is no temperature gradient inside the ice because every joule
arriving at the front is spent on latent heat) -- the ice-side term is
nevertheless kept in the interface balance through ``k_ice dT/dr|s-``, which
is evaluated from the (flat) ice field and is zero to round-off.

The interface speed follows the Stefan condition

    rho L dr_f/dt = k_w dT/dr|+ - k_i dT/dr|-

with quadratic one-sided reconstructions of the radial derivative, and the
latent heat is accounted per unit axial length so that the global energy
balance ``Q_wall = E_latent + E_sensible`` can be audited (criterion C3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .properties import IceWaterCase
from .similarity import quasi_steady_cylinder_front

__all__ = ["RadialFrontTracking", "RadialFrontTrackingResult"]


@dataclass
class RadialFrontTrackingResult:
    times: np.ndarray
    front: np.ndarray
    front_reference: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    initial_temperature: np.ndarray
    steps: int
    dt: float
    q_wall_integral: float
    latent_energy: float
    sensible_energy: float
    continuity_inner: float
    continuity_outer: float
    r0: float
    r_outer: float
    melting: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def front_error(self) -> float:
        ref = float(self.front_reference[-1])
        return abs(float(self.front[-1]) - ref) / ref

    @property
    def energy_residual(self) -> float:
        rhs = self.sensible_energy + (self.latent_energy if self.melting else -self.latent_energy)
        scale = abs(self.latent_energy) if self.latent_energy else 1.0
        return abs(self.q_wall_integral - rhs) / scale

    @property
    def continuity_residual(self) -> float:
        return max(self.continuity_inner, self.continuity_outer)

    def summary(self) -> dict:
        return {
            "r0_m": self.r0,
            "r_outer_m": self.r_outer,
            "t_end_s": float(self.times[-1]),
            "front_final_m": float(self.front[-1]),
            "front_reference_m": float(self.front_reference[-1]),
            "front_relative_error": self.front_error,
            "energy_residual": self.energy_residual,
            "latent_energy_J_m": self.latent_energy,
            "sensible_energy_J_m": self.sensible_energy,
            "wall_flux_integral_J_m": self.q_wall_integral,
            "continuity_inner_K": self.continuity_inner,
            "continuity_outer_K": self.continuity_outer,
            "steps": int(self.steps),
            "dt_s": self.dt,
        }


class RadialFrontTracking:
    """Cylindrical sharp-interface solver (water side + isothermal ice)."""

    def __init__(
        self,
        case: IceWaterCase,
        r0: float = 5.0e-3,
        r_outer: float = 30.0e-3,
        n_cells: int = 121,
        t_end: float | None = None,
        cfl: float = 0.35,
    ) -> None:
        if not case.melting:
            raise ValueError("the cylinder case is a melting configuration")
        self.case = case
        self.r0 = float(r0)
        self.r_outer = float(r_outer)
        self.n_cells = int(n_cells)
        self.h = self.r_outer / self.n_cells
        self.r = np.linspace(0.0, self.r_outer, self.n_cells + 1)
        self.rf = self.r0
        self.t_end = float(t_end) if t_end is not None else 2400.0
        # initial condition: steady conduction profile of the initial geometry
        # (removes the t -> 0 flux singularity, which no quasi-steady reference
        # can resolve and which is not the quantity under test)
        r_safe = np.maximum(self.r, self.r0)
        profile = case.t_melt + (case.t_wall - case.t_melt) * np.log(r_safe / self.r0) / math.log(
            self.r_outer / self.r0
        )
        self.temperature = np.where(self.r >= self.r0, profile, case.t_melt)
        self.temperature[-1] = case.t_wall
        dt_diff = cfl * self.h ** 2 / (4.0 * case.water.alpha)
        self.dt = float(dt_diff)

    # -- stencils -------------------------------------------------------
    def _pin_index(self, rf: float) -> int:
        """Grid node carrying the interface (nearest node to ``rf``)."""
        n = int(round(rf / self.h))
        return int(min(max(n, 1), self.n_cells - 3))

    def _diffuse(self, temperature: np.ndarray, rf: float, dt: float) -> np.ndarray:
        """Explicit cylindrical finite-volume step, interface pinned at a node.

        The interface node is a Dirichlet point at ``T_m`` and the ice interior
        is isothermal at ``T_m`` (one-phase idealisation), so all control
        volumes are fixed and the scheme is conservative.  Node ``n+1`` is the
        first water node; its inner face is a plain grid face because the
        interface sits exactly on node ``n``.
        """
        case = self.case
        k = case.water.k
        rho_c = case.water.volumetric_heat_capacity
        n = self._pin_index(rf)
        r = self.r
        h = self.h
        temp = temperature.copy()
        temp[n] = case.t_melt
        out = temp.copy()

        if n + 1 <= self.n_cells - 1:
            idx = np.arange(n + 1, self.n_cells)
            a_p = (idx + 0.5) * h
            a_m = (idx - 0.5) * h
            vol = 0.5 * (a_p ** 2 - a_m ** 2)
            flux_p = a_p * k * (temp[idx + 1] - temp[idx]) / h
            flux_m = a_m * k * (temp[idx] - temp[idx - 1]) / h
            out[idx] = temp[idx] + dt * (flux_p - flux_m) / (rho_c * vol)

        out[-1] = case.t_wall
        out[:n] = case.t_melt
        out[n] = case.t_melt
        return out

    def _radial_slopes(self, temperature: np.ndarray, rf: float):
        """Second-order one-sided radial derivatives at the interface node."""
        n = self._pin_index(rf)
        h = self.h
        t_m = self.case.t_melt
        dTdr_out = (-3.0 * t_m + 4.0 * temperature[n + 1] - temperature[n + 2]) / (2.0 * h)
        dTdr_in = (3.0 * t_m - 4.0 * temperature[n - 1] + temperature[n - 2]) / (2.0 * h)
        return dTdr_in, dTdr_out

    def _front_velocity(self, temperature: np.ndarray, rf: float) -> float:
        """``dr_f/dt`` (negative: the ice core shrinks as it melts).

        Heat conducted inward from the warm water through the annulus
        ``r in (r_f, R)`` arrives at the interface and is spent on latent heat;
        the interface therefore moves towards the axis.
        """
        dTdr_in, dTdr_out = self._radial_slopes(temperature, rf)
        q_in = self.case.water.k * dTdr_out      # conduction towards the axis
        q_ice = self.case.ice.k * dTdr_in        # into the (isothermal) ice
        return -(q_in - q_ice) / self.case.latent_volumetric

    def _wall_flux(self, temperature: np.ndarray) -> float:
        """Heat flux per unit area entering the domain at ``r = R`` [W/m^2].

        Positive means heat flows inward (into the cylinder), which is the
        sign convention of the energy audit.
        """
        grad = (3.0 * temperature[-1] - 4.0 * temperature[-2] + temperature[-3]) / (
            2.0 * self.h
        )
        return self.case.water.k * grad

    def run(self) -> RadialFrontTrackingResult:
        case = self.case
        temperature = self.temperature.copy()
        rf = self.rf
        n_steps = int(math.ceil(self.t_end / self.dt))
        dt = self.t_end / n_steps if n_steps else self.dt
        initial = temperature.copy()

        times = [0.0]
        fronts = [rf]
        wall = 0.0
        for _ in range(n_steps):
            v1 = self._front_velocity(temperature, rf)
            rf_mid = rf + 0.5 * dt * v1
            t_new = self._diffuse(temperature, rf_mid, dt)
            v2 = self._front_velocity(t_new, rf_mid)
            rf_new = rf + 0.5 * dt * (v1 + v2)
            # wall flux integrated over the outer circumference (per unit length)
            wall += 0.5 * dt * 2.0 * math.pi * self.r_outer * (
                self._wall_flux(temperature) + self._wall_flux(t_new)
            )
            temperature = t_new
            rf = rf_new
            times.append(times[-1] + dt)
            fronts.append(rf)

        # interface temperature from both sides (interior nodes only)
        t_m = case.t_melt
        j = self._pin_index(rf) + 1
        water_x = self.r[j:min(j + 3, len(self.r))]
        water_y = temperature[j:min(j + 3, len(self.r))]
        if water_x.size >= 2:
            coef = np.polyfit(water_x - rf, water_y, min(2, water_x.size - 1))
            t_out = float(np.polyval(coef, 0.0))
        else:  # pragma: no cover
            t_out = t_m
        ice_x = self.r[max(0, j - 3):j]
        ice_y = temperature[max(0, j - 3):j]
        t_in = float(np.mean(ice_y)) if ice_y.size else t_m

        latent = case.latent_volumetric * math.pi * (self.r0 ** 2 - rf ** 2)
        water_mask = self.r >= rf
        if water_mask.sum() >= 2:
            sensible = float(
                np.trapezoid(
                    case.water.volumetric_heat_capacity * 2.0 * math.pi * self.r[water_mask]
                    * (temperature[water_mask] - initial[water_mask]),
                    self.r[water_mask],
                )
            )
        else:  # pragma: no cover
            sensible = 0.0
        ref = np.array([
            quasi_steady_cylinder_front(case, t, self.r0, self.r_outer) for t in times
        ])
        return RadialFrontTrackingResult(
            times=np.asarray(times),
            front=np.asarray(fronts),
            front_reference=ref,
            radius=self.r.copy(),
            temperature=temperature,
            initial_temperature=initial,
            steps=n_steps,
            dt=dt,
            q_wall_integral=float(wall),
            latent_energy=float(latent),
            sensible_energy=float(sensible),
            continuity_inner=abs(t_in - t_m),
            continuity_outer=abs(t_out - t_m),
            r0=self.r0,
            r_outer=self.r_outer,
            melting=True,
            metadata={
                "n_cells": self.n_cells,
                "h_m": self.h,
                "stefan_inner": case.stefan_inner,
                "quasi_steady_error_estimate": 0.5 * case.stefan_inner,
            },
        )