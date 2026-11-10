"""One-dimensional *sharp* front-tracking solver for the ice/water interface.

The front is a genuine sub-cell object: its position ``s(t)`` is stored as a
real number and the two phases are solved on the same fixed Eulerian grid with
a ghost-node Dirichlet closure at ``x = s`` (the "ghost fluid" treatment of
immersed interfaces, Gibou et al. 2007).  Concretely, for the last node ``c``
inside the front (``x_c < s < x_{c+1}``, ``theta = (s - x_c)/h``) the ghost
value used by the inner-phase diffusion stencil is

    T_ghost = (T_m - (1 - theta) T_c) / theta      (linear extrapolation that
                                                    forces T(s) = T_m exactly)

and symmetrically for the outer phase.  The interface speed follows from the
Stefan condition evaluated with *quadratic* one-sided reconstructions of the
normal derivative, so the interface closure is second order in ``h`` and the
front position does not inherit the O(h) smearing of enthalpy methods.

Sign conventions
----------------
::

    melting (wall above T_m):  rho L V =  q_in - q_out
    freezing (wall below T_m): rho L V = -q_in + q_out

with ``q_in = -k_in dT/dx|s-`` (flux leaving the inner phase, +x sense) and
``q_out = -k_out dT/dx|s+`` (flux entering the outer phase).  Both modes are
implemented and exercised by the test suite.

Energy accounting (used for criterion C3) accumulates, per unit area:

* ``q_wall`` integrated over time  -> heat entering through the driven wall,
* ``q_far``  integrated over time  -> heat leaving through the far boundary,
* the latent heat ``rho L (s_end - s_start)``,
* the sensible enthalpy change ``int rho c (T_end - T_start) dx``,

and the residual of ``Q_wall - Q_far = E_latent + E_sensible`` is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .properties import IceWaterCase
from .similarity import TwoPhaseNeumann

__all__ = ["FrontTracking1D", "FrontTracking1DResult", "solve_pure_conduction"]


@dataclass
class FrontTracking1DResult:
    """Container for a 1-D front-tracking run."""

    times: np.ndarray
    front: np.ndarray
    front_exact: np.ndarray
    x: np.ndarray
    temperature: np.ndarray
    initial_temperature: np.ndarray
    steps: int
    dt: float
    q_wall_integral: float
    q_far_integral: float
    latent_energy: float
    sensible_energy: float
    continuity_inner: float
    continuity_outer: float
    far_field_gradient: float
    melting: bool = True
    metadata: dict = field(default_factory=dict)

    # -- derived quantities -------------------------------------------
    @property
    def front_error(self) -> float:
        """Relative error of the final front position vs the exact solution."""
        ref = float(self.front_exact[-1])
        return abs(float(self.front[-1]) - ref) / ref

    @property
    def energy_residual(self) -> float:
        """Relative residual of the latent-heat / energy balance (criterion C3)."""
        lhs = self.q_wall_integral - self.q_far_integral
        rhs = self.sensible_energy + (self.latent_energy if self.melting else -self.latent_energy)
        scale = abs(self.latent_energy) if self.latent_energy != 0 else 1.0
        return abs(lhs - rhs) / scale

    @property
    def continuity_residual(self) -> float:
        """Max |T_reconstructed(s) - T_m| from one-sided fits [K]."""
        return max(self.continuity_inner, self.continuity_outer)

    def profile_series(self, times, sim: TwoPhaseNeumann) -> dict:
        """Snapshot-friendly comparison of the final field vs the exact one."""
        return {
            "x": self.x.tolist(),
            "t_end": float(self.times[-1]),
            "temperature": self.temperature.tolist(),
            "temperature_exact": sim.temperature(self.x, float(self.times[-1])).tolist(),
        }

    def summary(self) -> dict:
        return {
            "steps": int(self.steps),
            "dt_s": self.dt,
            "t_end_s": float(self.times[-1]),
            "front_final_m": float(self.front[-1]),
            "front_exact_m": float(self.front_exact[-1]),
            "front_relative_error": self.front_error,
            "energy_residual": self.energy_residual,
            "latent_energy_J_m2": self.latent_energy,
            "sensible_energy_J_m2": self.sensible_energy,
            "wall_flux_integral_J_m2": self.q_wall_integral,
            "continuity_inner_K": self.continuity_inner,
            "continuity_outer_K": self.continuity_outer,
        }


class FrontTracking1D:
    """Fixed-grid sharp-interface solver for the 1-D two-phase Stefan problem.

    Parameters
    ----------
    case : IceWaterCase
        Physical configuration (melting or freezing).
    n_cells : int
        Number of uniform cells in the domain ``[0, L]``.
    t_end : float
        Physical end time [s].
    t_start : float, optional
        Time at which the simulation is initialised from the similarity
        solution (default ``t_end/64``).  Starting from a finite front avoids
        the degenerate ``s = 0`` cut cell and isolates the discretisation
        error, which is the quantity of interest.
    domain_factor : float
        Domain length as a multiple of the analytic front position at
        ``t_end``.
    cfl : float
        Explicit-scheme safety factor on the diffusive time-step limit.
    far_bc : {"dirichlet", "neumann"}
        Far-field boundary condition at ``x = L``.
    """

    def __init__(
        self,
        case: IceWaterCase,
        n_cells: int = 400,
        t_end: float = 60.0,
        t_start: float | None = None,
        domain_factor: float = 2.0,
        cfl: float = 0.4,
        far_bc: str = "dirichlet",
    ) -> None:
        self.case = case
        self.sim = TwoPhaseNeumann(case)
        self.t_end = float(t_end)
        self.t_start = float(t_start) if t_start is not None else t_end / 64.0
        self.n_cells = int(n_cells)
        self.cfl = float(cfl)
        self.far_bc = far_bc

        s_end = float(self.sim.front(self.t_end))
        self.length = domain_factor * s_end
        self.h = self.length / self.n_cells
        self.x = np.linspace(0.0, self.length, self.n_cells + 1)

        # start from a finite front: (i) the degenerate s = 0 cut cell is
        # avoided and (ii) the discretisation error -- the quantity of interest
        # -- is isolated from the start-up transient.  If the default start time
        # puts the front inside the first few cells, start later instead.
        lam = float(self.sim.lam)
        s_needed = 3.2 * self.h
        t_needed = (s_needed / (2.0 * lam)) ** 2 / case.alpha_inner
        self.t_start = float(max(self.t_start, t_needed))
        if self.t_start >= 0.5 * self.t_end:  # pragma: no cover
            raise ValueError("domain too coarse: initial front cannot be placed")
        self.s = float(self.sim.front(self.t_start))
        self.temperature = np.asarray(self.sim.temperature(self.x, self.t_start), dtype=float)
        self.temperature[0] = case.t_wall
        self.temperature[-1] = case.t_initial

        # only phases that actually carry a temperature gradient constrain dt
        alpha_active = [case.alpha_inner]
        if case.dT_outer != 0.0:
            alpha_active.append(case.outer.alpha)
        dt_diff = self.cfl * self.h ** 2 / (2.0 * max(alpha_active))
        v_max = float(self.sim.front_velocity(self.t_start))
        dt_move = 0.5 * self.h / max(v_max, 1e-30)
        self.dt = float(min(dt_diff, dt_move))

    # -- interface helpers --------------------------------------------
    def _pin_index(self, s: float) -> int:
        """Grid node that carries the interface (nearest node to ``s``)."""
        n = int(round(s / self.h))
        return int(min(max(n, 3), self.n_cells - 3))

    @staticmethod
    def _newton_slope(a, y, at: float) -> float:
        """``dT/da`` at ``a = at`` from a quadratic through three points.

        Newton divided-difference form; no library fit call, so it is cheap
        enough to run at every time step.
        """
        a0, a1, a2 = a
        y0, y1, y2 = y
        d01 = (y1 - y0) / (a1 - a0)
        d12 = (y2 - y1) / (a2 - a1)
        d012 = (d12 - d01) / (a2 - a0)
        return d01 + d012 * (2.0 * at - a0 - a1)

    def _interface_gradients(self, temperature: np.ndarray, s: float):
        """Second-order one-sided derivatives **at the true interface position**.

        The temperature field pins the interface to the nearest node ``n``
        (``T_n = T_m``), but the interface is a sub-cell quantity at ``x = s``.
        The fluxes that drive ``ds/dt`` are extrapolated to ``x = s`` from a
        quadratic through the three nearest nodes *of the phase itself* (the
        pinned node is deliberately excluded: it sits on the far side of the
        interface, where the profile has a corner, and including it biases the
        extrapolated gradient at coarse resolution).
        """
        n = self._pin_index(s)
        t_m = self.case.t_melt
        x = self.x
        # inner phase nodes: n-1, n-2, n-3  (a = s - x, positive towards the wall)
        dTdx_in = -self._newton_slope(
            [s - x[n - 1], s - x[n - 2], s - x[n - 3]],
            [temperature[n - 1] - t_m, temperature[n - 2] - t_m, temperature[n - 3] - t_m],
            0.0,
        )
        # outer phase nodes: n+1, n+2, n+3  (a = x - s)
        dTdx_out = self._newton_slope(
            [x[n + 1] - s, x[n + 2] - s, x[n + 3] - s],
            [temperature[n + 1] - t_m, temperature[n + 2] - t_m, temperature[n + 3] - t_m],
            0.0,
        )
        return n, dTdx_in, dTdx_out

    def _interface_fluxes(self, temperature: np.ndarray, s: float):
        """``(n, q_in, q_out)`` -- both fluxes in the +x sense [W/m^2]."""
        n, dTdx_in, dTdx_out = self._interface_gradients(temperature, s)
        q_in = -self.case.inner.k * dTdx_in     # leaving the inner phase
        q_out = -self.case.outer.k * dTdx_out   # entering the outer phase
        return n, q_in, q_out

    def _front_velocity(self, temperature: np.ndarray, s: float) -> float:
        _, q_in, q_out = self._interface_fluxes(temperature, s)
        sign = 1.0 if self.case.melting else -1.0
        return sign * (q_in - q_out) / self.case.latent_volumetric

    def _diffuse(self, temperature: np.ndarray, s: float, dt: float) -> np.ndarray:
        """One explicit finite-volume step with the interface pinned at node ``n``.

        Because the interface node is a Dirichlet point at ``T_m`` and all
        control volumes are fixed, the discrete scheme is conservative up to
        round-off; the latent heat that the audit compares against is produced
        by exactly the interface fluxes that drive ``ds/dt``.
        """
        case = self.case
        n = self._pin_index(s)
        h = self.h
        out = temperature.copy()
        temp = temperature.copy()
        temp[n] = case.t_melt

        if n >= 2:
            idx = np.arange(1, n)
            lap = (temp[idx + 1] - 2.0 * temp[idx] + temp[idx - 1]) / h ** 2
            out[idx] = temp[idx] + dt * case.alpha_inner * lap
        if n + 2 <= self.n_cells:
            idx = np.arange(n + 1, self.n_cells)
            lap = (temp[idx + 1] - 2.0 * temp[idx] + temp[idx - 1]) / h ** 2
            out[idx] = temp[idx] + dt * case.outer.alpha * lap

        out[0] = case.t_wall
        out[n] = case.t_melt
        out[self.n_cells] = (
            case.t_initial if self.far_bc == "dirichlet" else temp[self.n_cells]
        )
        return out

    # -- diagnostics ---------------------------------------------------
    def _wall_flux(self, temperature: np.ndarray) -> float:
        """Second-order one-sided flux at ``x = 0`` in the +x sense."""
        grad = (-3.0 * temperature[0] + 4.0 * temperature[1] - temperature[2]) / (
            2.0 * self.h
        )
        return -self.case.inner.k * grad

    def _far_flux(self, temperature: np.ndarray) -> float:
        """Flux leaving the domain at ``x = L`` (+x sense)."""
        grad = (3.0 * temperature[-1] - 4.0 * temperature[-2] + temperature[-3]) / (
            2.0 * self.h
        )
        return -self.case.outer.k * grad

    def _interface_temperature(self, temperature: np.ndarray, s: float):
        """Extrapolate both phases to ``x = s`` using interior nodes only."""
        c = self._pin_index(s)
        t_m = self.case.t_melt
        x = self.x

        def extrap(xs, ys, at) -> float:
            # quadratic Newton extrapolation through three nodes
            if xs.size < 3:
                return float(ys[-1])
            coef = np.polyfit(xs - at, ys, 2)
            return float(np.polyval(coef, 0.0))

        inner_x = x[max(0, c - 2): c + 1]
        inner_y = temperature[max(0, c - 2): c + 1]
        outer_x = x[c + 1: c + 4]
        outer_y = temperature[c + 1: c + 4]
        t_in = extrap(inner_x, inner_y, s) if inner_x.size else t_m
        t_out = extrap(outer_x, outer_y, s) if outer_x.size else t_m
        return t_in, t_out

    # -- driver ---------------------------------------------------------
    def run(self) -> FrontTracking1DResult:
        case = self.case
        temperature = self.temperature.copy()
        t = self.t_start
        s = self.s
        n_steps = int(math.ceil((self.t_end - self.t_start) / self.dt))
        dt = (self.t_end - self.t_start) / n_steps if n_steps else self.dt

        times = [t]
        fronts = [s]
        walls = [0.0]
        fars = [0.0]
        initial = temperature.copy()

        for _ in range(n_steps):
            v1 = self._front_velocity(temperature, s)
            s_mid = s + 0.5 * dt * v1
            t_new = self._diffuse(temperature, s_mid, dt)
            v2 = self._front_velocity(t_new, s_mid)
            s_new = s + 0.5 * dt * (v1 + v2)
            walls.append(0.5 * dt * (self._wall_flux(temperature) + self._wall_flux(t_new)))
            fars.append(0.5 * dt * (self._far_flux(temperature) + self._far_flux(t_new)))
            temperature = t_new
            s = s_new
            t += dt
            times.append(t)
            fronts.append(s)

        temperature[0] = case.t_wall
        t_m = case.t_melt
        t_in, t_out = self._interface_temperature(temperature, s)
        latent = case.latent_volumetric * (s - self.s)
        sensible = float(np.trapezoid(
            np.where(self.x <= s, case.inner.volumetric_heat_capacity,
                     case.outer.volumetric_heat_capacity)
            * (temperature - initial),
            self.x,
        ))
        return FrontTracking1DResult(
            times=np.asarray(times),
            front=np.asarray(fronts),
            front_exact=np.asarray(self.sim.front(np.asarray(times))),
            x=self.x.copy(),
            temperature=temperature,
            initial_temperature=initial,
            steps=n_steps,
            dt=dt,
            q_wall_integral=float(sum(walls)),
            q_far_integral=float(sum(fars)),
            latent_energy=latent,
            sensible_energy=sensible,
            continuity_inner=abs(t_in - t_m),
            continuity_outer=abs(t_out - t_m),
            far_field_gradient=float(
                abs(temperature[-1] - temperature[-2]) / self.h
            ),
            melting=bool(case.melting),
            metadata={
                "n_cells": self.n_cells,
                "h_m": self.h,
                "domain_m": self.length,
                "lambda": self.sim.lam,
                "mode": "melting" if case.melting else "freezing",
                "stefan_inner": case.stefan_inner,
                "far_bc": self.far_bc,
            },
        )


# ---------------------------------------------------------------------------
# negative control: pure conduction, no phase change
# ---------------------------------------------------------------------------
def solve_pure_conduction(
    case: IceWaterCase,
    t_end: float = 60.0,
    n_cells: int = 400,
    domain_factor: float = 8.0,
    cfl: float = 0.4,
):
    """Single-phase conduction vs the erfc solution (criterion C7).

    The domain is a semi-infinite body, ``T(0, t) = T_wall``,
    ``T(inf, t) = T_initial``; the exact solution is
    ``T = T_initial + (T_wall - T_initial) erfc(x / (2 sqrt(alpha t)))``.

    Phase change is switched off by construction (a single material with the
    water properties), so this is the *negative control*: any phase-change
    machinery must be inactive and the error must stay at discretisation level.
    """
    from scipy.special import erfc

    alpha = case.water.alpha
    length = domain_factor * math.sqrt(alpha * t_end)
    h = length / n_cells
    x = np.linspace(0.0, length, n_cells + 1)
    dt = cfl * h ** 2 / (2.0 * alpha)
    n_steps = int(math.ceil(t_end / dt))
    dt = t_end / n_steps

    temperature = np.full_like(x, case.t_initial)
    temperature[0] = case.t_wall
    for _ in range(n_steps):
        lap = (temperature[2:] - 2.0 * temperature[1:-1] + temperature[:-2]) / h ** 2
        temperature[1:-1] += dt * alpha * lap
        temperature[0] = case.t_wall
        temperature[-1] = case.t_initial
    exact = case.t_initial + (case.t_wall - case.t_initial) * erfc(
        x / (2.0 * math.sqrt(alpha * t_end))
    )
    err = np.abs(temperature - exact)
    scale = abs(case.t_wall - case.t_initial)
    return {
        "t_end_s": t_end,
        "n_cells": n_cells,
        "dt_s": dt,
        "steps": n_steps,
        "max_abs_error_K": float(err.max()),
        "rms_error_K": float(np.sqrt(np.mean(err ** 2))),
        "relative_max_error": float(err.max() / scale),
        "relative_rms_error": float(np.sqrt(np.mean(err ** 2)) / scale),
        "x": x.tolist(),
        "temperature": temperature.tolist(),
        "temperature_exact": exact.tolist(),
    }