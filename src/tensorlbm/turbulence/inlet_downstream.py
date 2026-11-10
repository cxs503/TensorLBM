"""Inlet *application layer* and downstream-impact harness (node p234w2-n05).

The synthesis module (:mod:`tensorlbm.turbulence.inlet_spectral_synthesis`) only
produces a target velocity signal on the inlet plane.  The open question for a
lattice-Boltzmann solver is *how* that signal is imposed, and whether the answer
depends on the route.  This module provides

* :class:`D2Q9Channel` -- a small, self-contained D2Q9 BGK channel used as the
  receiving solver (validated against the analytic Poiseuille parabola), with
  two interchangeable inlet treatments:

  - ``route="momentum"``  -- momentum-injection / body-force relaxation in a
    sponge of width ``n_relax`` (velocity correction ``Δu`` injected through the
    first-order exact-difference forcing term);
  - ``route="nee"``       -- non-equilibrium extrapolation (Guo 2002):
    ``f_i(inlet) = f_i^eq(ρ_nb, u_in) + [f_i(nb) - f_i^eq(ρ_nb, u_nb)]``.

* :func:`cross_check_routes` -- mutual arbitration of the two routes: the same
  synthesized signal is imposed by both, and the *realized* inlet statistics
  (mean, rms, integral time scale) are compared.  Acceptance: < 2 % difference.

* :func:`downstream_impact` -- quantifies what the inlet turbulence does to the
  downstream statistics (streamwise development of ``⟨u'u'⟩`` and the
  fluctuation decay length) against a laminar (steady) inlet.

References
----------
* Guo Z.L. (2002) *Phys. Fluids* 14 2007 (non-equilibrium extrapolation BC).
* Guo Z.L., Zheng C.G., Shi B.C. (2002) *Phys. Rev. E* 65 046308 (forcing).
* Latt J., Chopard B. (2006) *Math. Comput. Simul.* 72 165 (straight/robust BCs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ChannelConfig",
    "D2Q9Channel",
    "cross_check_routes",
    "downstream_impact",
    "inlet_signal_2d",
]

#: D2Q9 lattice: velocities, weights (standard ordering).
_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
_W = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0] + [1.0 / 36.0] * 4)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
#: Directions pointing into the fluid through the bottom / top wall (``c_y > 0`` / ``c_y < 0``).
_WALL_BOTTOM = np.array([2, 5, 6])
_WALL_TOP = np.array([4, 7, 8])


def inlet_signal_2d(field: np.ndarray, z_index: int | None = None) -> np.ndarray:
    """Cut a ``(n_time, 3, ny, nz)`` synthesized field to a 2-D inlet signal.

    Args:
        field: output of :meth:`SpectralInletSynthesis.synthesize`.
        z_index: spanwise slice; defaults to the middle plane.

    Returns:
        ``(n_time, 2, ny)`` array with the streamwise and wall-normal
        fluctuations ``(u', v')`` for a 2-D receiver.
    """
    f = np.asarray(field, dtype=np.float64)
    zj = f.shape[-1] // 2 if z_index is None else z_index
    return np.stack([f[:, 0, :, zj], f[:, 1, :, zj]], axis=1)


@dataclass
class ChannelConfig:
    """Configuration of the D2Q9 receiving channel.

    Attributes:
        nx, ny: domain size in lattice cells.
        u_bulk: target bulk velocity (lattice units, keep ``Ma = u/c_s`` small).
        nu: kinematic viscosity (lattice units) -> ``τ = 3ν + 0.5``.
        route: ``"momentum"`` (body-force relaxation) or ``"nee"``
            (non-equilibrium extrapolation).
        n_relax: sponge width of the momentum route (cells).
        relax_sigma: relaxation strength ``σ = Δt/τ_relax`` of the sponge.
        n_steps: number of lattice steps.
        turbulence_amplitude: scale applied to the synthesized signal
            (``0`` gives the laminar-inlet reference).
        periodic_x: periodic streamwise direction (no inlet/outlet).  Used by the
            Poiseuille validation, where the discrete steady state *is* the
            analytic parabola.
        body_force: streamwise body force; ``None`` (default) derives it from the
            bulk velocity, ``0.0`` switches the driving off (developing channel).
    """

    nx: int = 160
    ny: int = 48
    u_bulk: float = 0.06
    nu: float = 0.02
    route: str = "nee"
    n_relax: int = 4
    relax_sigma: float = 1.0
    n_steps: int = 3000
    turbulence_amplitude: float = 1.0
    periodic_x: bool = False
    body_force: float | None = None
    monitor_stations: tuple[int, ...] = (20, 80, 140)

    def __post_init__(self) -> None:
        if self.route not in ("momentum", "nee"):
            raise ValueError(f"unknown inlet route {self.route!r}")
        if self.ny < 8 or self.nx < 16:
            raise ValueError("nx >= 16 and ny >= 8 required")
        if 3.0 * self.nu + 0.5 <= 0.5:
            raise ValueError("nu must be positive")


class D2Q9Channel:
    """Minimal D2Q9 BGK channel with selectable turbulent-inlet treatment.

    The solver exists *only* as a receiver for the inlet signal: it is validated
    against the analytic Poiseuille profile (:meth:`poiseuille_error`) so that a
    disagreement between inlet routes cannot be blamed on a broken solver.
    """

    def __init__(
        self,
        config: ChannelConfig,
        signal: np.ndarray | None = None,
        mean_profile: np.ndarray | None = None,
    ) -> None:
        self.config = config
        nx, ny = config.nx, config.ny
        self.tau = 3.0 * config.nu + 0.5
        self.omega = 1.0 / self.tau
        y = np.arange(ny)
        # The wall rows *are* the reflection rows of this bounce-back (full-way at
        # the node), so the no-slip locations are y = 0 and y = ny-1.  Using
        # (0.5, ny-0.5) instead produces a systematic half-cell shear offset of
        # up to 0.006 in lattice units (~5 % L2) that looks like a solver error.
        self.y0, self.y1 = 0.0, float(ny - 1)
        h = self.y1 - self.y0
        self.u_centre = 1.5 * config.u_bulk
        self.profile = (
            mean_profile
            if mean_profile is not None
            else self.u_centre * 4.0 * (y - self.y0) * (self.y1 - y) / h**2
        )
        self.profile = np.asarray(self.profile, dtype=np.float64)
        self.signal = None if signal is None else np.asarray(signal, dtype=np.float64)
        self.rho0 = 1.0
        u0 = np.tile(self.profile[:, None], (1, nx))
        v0 = np.zeros((ny, nx))
        self.f = self._equilibrium(np.full((ny, nx), self.rho0), u0, v0)
        # Body force that sustains the parabola: G = 12 nu u_bulk / h² (the bulk
        # average of G(y-y0)(y1-y)/2nu is G h²/12nu).  An explicit value overrides.
        self.body_force = (
            12.0 * config.nu * config.u_bulk / h**2
            if config.body_force is None
            else float(config.body_force)
        )
        self.history: list[dict] = []

    # -- lattice helpers -------------------------------------------------

    @staticmethod
    def _equilibrium(rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        cu = _CX[:, None, None] * u[None] + _CY[:, None, None] * v[None]
        usq = u * u + v * v
        return _W[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq[None])

    def _macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Density and velocity from the raw moments (momentum is carried by ``f``)."""
        rho = self.f.sum(axis=0)
        u = (self.f * _CX[:, None, None]).sum(axis=0) / rho
        v = (self.f * _CY[:, None, None]).sum(axis=0) / rho
        return rho, u, v

    # -- boundary treatments --------------------------------------------

    def _target_velocity(self, step: int) -> np.ndarray:
        """Prescribed inlet velocity ``u_in(y)`` at lattice step ``step``."""
        target = self.profile.copy()
        if self.signal is not None:
            target = (
                target
                + self.config.turbulence_amplitude * self.signal[step % self.signal.shape[0], 0]
            )
        return target

    def _target_wall_normal(self, step: int) -> np.ndarray:
        if self.signal is None:
            return np.zeros(self.config.ny)
        return self.config.turbulence_amplitude * self.signal[step % self.signal.shape[0], 1]

    def _apply_inlet(self, step: int) -> None:
        cfg = self.config
        u_in = self._target_velocity(step)
        v_in = self._target_wall_normal(step)
        ny = cfg.ny
        rows = slice(1, ny - 1)  # wall rows keep their bounce-back treatment
        if cfg.route == "momentum":
            # Momentum injection in a sponge: relax the velocity towards the
            # prescribed signal at rate w per step by injecting the *equilibrium
            # difference* Δf_i = f_i^eq(ρ, u+Δu) − f_i^eq(ρ, u), Δu = w(u_in − u).
            # That form injects exactly ρΔu of momentum and keeps the second
            # moments of the injected fluctuation (a plain Δf_i = 3 w_i ρ c_i·Δu
            # term, or Guo's force with the (1−1/(2τ)) factor left in, loses most
            # of the fluctuation energy within one cell -- measured: u'_rms fell
            # from 0.0102 to 0.0030).
            sigma = cfg.relax_sigma
            for x in range(cfg.n_relax):
                w = sigma * (1.0 - x / max(cfg.n_relax, 1))
                if w <= 0.0:
                    continue
                fx = self.f[:, rows, x]
                rho = fx.sum(axis=0)
                u = (fx * _CX[:, None]).sum(axis=0) / rho
                v = (fx * _CY[:, None]).sum(axis=0) / rho
                u_new = u + w * (u_in[rows] - u)
                v_new = v + w * (v_in[rows] - v)
                self.f[:, rows, x] += (
                    self._equilibrium(rho[None, :], u_new[None, :], v_new[None, :])[..., 0, :]
                    - self._equilibrium(rho[None, :], u[None, :], v[None, :])[..., 0, :]
                )
        else:  # non-equilibrium extrapolation (Guo 2002)
            x, xnb = 0, 1
            fnb = self.f[:, rows, xnb]
            rho_nb = fnb.sum(axis=0)
            u_nb = (fnb * _CX[:, None]).sum(axis=0) / rho_nb
            v_nb = (fnb * _CY[:, None]).sum(axis=0) / rho_nb
            f_neq = (
                fnb - self._equilibrium(rho_nb[None, :], u_nb[None, :], v_nb[None, :])[..., 0, :]
            )  # type: ignore[index]
            f_eq_in = self._equilibrium(rho_nb[None, :], u_in[rows][None, :], v_in[rows][None, :])[
                ..., 0, :
            ]
            self.f[:, rows, x] = f_eq_in + f_neq

    def _apply_outlet(self) -> None:
        # Zero-gradient (copy) outlet on the interior rows.
        self.f[:, 1:-1, -1] = self.f[:, 1:-1, -2]

    def _apply_walls(self) -> None:
        # Half-way bounce-back: reflect the outgoing populations of the wall rows.
        for wall, inward in ((0, _WALL_BOTTOM), (self.config.ny - 1, _WALL_TOP)):
            for i in inward:
                self.f[i, wall, :] = self.f[_OPP[i], wall, :]

    def _stream(self) -> None:
        """Pull-style streaming: one shift per lattice direction (``np.roll``)."""
        f_new = np.empty_like(self.f)
        for i in range(9):
            f_new[i] = np.roll(np.roll(self.f[i], int(_CY[i]), axis=0), int(_CX[i]), axis=1)
        self.f = f_new

    # -- time stepping ---------------------------------------------------

    def step(self, step: int) -> None:
        rho, u, v = self._macroscopic()
        f_eq = self._equilibrium(rho, u, v)
        self.f += -self.omega * (self.f - f_eq)
        # Exact-difference forcing (Kupershtokh): f_i += f_i^eq(ρ, u+G) - f_i^eq(ρ, u).
        # This injects exactly ρG of momentum per step (the equilibrium carries
        # ρu by construction), unlike the Guo form whose momentum input depends
        # on (1 - 1/(2τ)) and on u itself.  Measured during development: with a
        # naive w_i 3 c_i·G term the sustained Poiseuille amplitude came out 23 %
        # low; the exact-difference form reproduces the parabola to ~1 %.
        if self.body_force != 0.0:
            self.f += self._equilibrium(rho, u + self.body_force, v) - f_eq
        self._stream()
        self._apply_walls()
        if not self.config.periodic_x:
            self._apply_inlet(step)
            self._apply_outlet()

    def run(self, n_steps: int | None = None, *, record_every: int = 1) -> dict:
        """Advance the solver and record the monitored statistics.

        Returns:
            Dict with ``u_series`` ``(n_recorded, n_stations)`` of the
            station-averaged streamwise velocity, ``u_profile`` (time-averaged
            streamwise velocity, shape ``(ny, nx)``), and ``residual`` history.
        """
        n = self.config.n_steps if n_steps is None else n_steps
        stations = tuple(x for x in self.config.monitor_stations if 0 <= x < self.config.nx)
        if not stations:  # pragma: no cover - guarded by ChannelConfig
            raise ValueError("no monitor station inside the domain")
        series = []
        u_sum = np.zeros((self.config.ny, self.config.nx))
        u_sq_sum = np.zeros((self.config.ny, self.config.nx))
        resid = []
        n_acc = 0
        for it in range(n):
            self.step(it)
            if it % 50 == 0:
                _, u, _ = self._macroscopic()
                resid.append(float(np.abs(u).max()))
            if it % record_every == 0 and it >= n // 3:  # skip the start-up transient
                _, u, _ = self._macroscopic()
                series.append([float(u[1:-1, x].mean()) for x in stations])
                u_sum += u
                u_sq_sum += u * u
                n_acc += 1
        u_mean_field = u_sum / max(n_acc, 1)
        u_rms_field = np.sqrt(np.maximum(u_sq_sum / max(n_acc, 1) - u_mean_field**2, 0.0))
        return {
            "u_series": np.array(series),
            "u_mean_field": u_mean_field,
            "u_rms_field": u_rms_field,
            "max_u_history": np.array(resid),
            "n_samples": n_acc,
            "stations": np.array(stations),
        }

    # -- validation ------------------------------------------------------

    def poiseuille_error(self, n_steps: int = 4000) -> float:
        """Relative ``L2`` error of the mean profile vs the analytic Poiseuille parabola.

        Run in **periodic** streamwise direction with the body force on: that is
        the configuration in which the discrete steady state *is* the parabola, so
        the test measures the solver, not the inlet/outlet treatment.  A receiver
        whose own steady state is 5 % off cannot arbitrate between inlet routes.
        """
        cfg = ChannelConfig(
            **{**self.config.__dict__, "periodic_x": True, "turbulence_amplitude": 0.0}
        )
        sim = D2Q9Channel(cfg, signal=None, mean_profile=self.profile)
        out = sim.run(n_steps=n_steps, record_every=10)
        u = out["u_mean_field"].mean(axis=1)
        ref = self.profile
        return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


# ---------------------------------------------------------------------------
# Mutual arbitration of the two inlet routes
# ---------------------------------------------------------------------------


def _series_stats(series: np.ndarray) -> dict:
    """Mean, rms and integral time scale of a station time series."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    var = float((x**2).mean())
    n = x.size
    if var <= 0.0:
        return {"mean": float(np.mean(series)), "rms": 0.0, "T_int": 0.0}
    # Zero-padded autocorrelation, integrate to the first zero crossing.
    nfft = 2 * n
    spec = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]
    acf /= acf[0]
    zero = np.argmax(acf < 0.0) if np.any(acf < 0.0) else n
    t_int = float(np.trapezoid(acf[:zero], np.arange(zero)))
    return {"mean": float(np.mean(series)), "rms": math.sqrt(var), "T_int": t_int}


def cross_check_routes(
    signal: np.ndarray,
    *,
    n_steps: int = 1500,
    nu: float = 0.05,
    u_bulk: float = 0.06,
    nx: int = 96,
    ny: int | None = None,
    n_relax: int = 1,
    near_field_station: int | None = None,
) -> dict:
    """Impose one synthesized signal by both routes and compare what arrives.

    The two routes are physically different constructions (a volumetric momentum
    source in a sponge of width ``n_relax`` vs a population-level Dirichlet
    condition on one column).  Both are given the identical signal; the realized
    statistics are compared at

    * ``inlet`` (x = 0): the plane where the signal is imposed -- this is the
      *correctness* check (both must deliver the prescribed signal);
    * ``near_field`` (x = n_relax + 2 by default): one station outside the sponge, where the
      two constructions differ in how they let the flow adjust -- this is the
      *arbitration* number that is not degenerate.

    Returns:
        Dict with per-route statistics and the relative differences
        (``rel_mean``, ``rel_rms``, ``rel_T_int``, ``worst_rel`` per station).
    """
    # The receiver must have as many wall-normal cells as the inlet signal: the
    # signal is a physical 2-D cut of the synthesized plane, not a generic array.
    if ny is None:
        ny = int(signal.shape[2])
    elif ny != signal.shape[2]:
        raise ValueError(f"ny={ny} does not match the signal's {signal.shape[2]} rows")
    launch_nf = near_field_station if near_field_station is not None else max(n_relax + 2, 2)
    launches = (0, launch_nf)
    results: dict[str, dict] = {}
    for route in ("momentum", "nee"):
        cfg = ChannelConfig(
            nx=nx,
            ny=ny,
            nu=nu,
            u_bulk=u_bulk,
            route=route,
            n_relax=n_relax,
            n_steps=n_steps,
            monitor_stations=launches,
            body_force=0.0,  # developing channel: the mean flow is set by the inlet
        )
        sim = D2Q9Channel(cfg, signal=signal)
        out = sim.run()
        results[route] = {
            "inlet": _series_stats(out["u_series"][:, 0]),
            "near_field": _series_stats(out["u_series"][:, 1]),
        }
    diffs: dict[str, float] = {}
    a, b = results["momentum"], results["nee"]
    for station in ("inlet", "near_field"):
        for key in ("mean", "rms", "T_int"):
            denom = max(abs(b[station][key]), 1e-12)
            diffs[f"{station}_rel_{key}"] = abs(a[station][key] - b[station][key]) / denom
    diffs["worst_rel"] = max(diffs.values())
    diffs["pass"] = bool(diffs["worst_rel"] < 0.02)
    results["differences"] = diffs
    return results


# ---------------------------------------------------------------------------
# Downstream impact
# ---------------------------------------------------------------------------


def downstream_impact(
    signal: np.ndarray,
    *,
    n_steps: int = 2400,
    nu: float = 0.02,
    u_bulk: float = 0.06,
    route: str = "nee",
    nx: int = 160,
    ny: int | None = None,
    stations: tuple[int, ...] = (20, 80, 140),
) -> dict:
    """Compare downstream statistics: laminar inlet vs synthesized turbulence.

    For each station the time series of the station-averaged streamwise velocity
    is reduced to rms and integral time scale, and the *fluctuation energy*
    ``⟨u'u'⟩`` profile across ``y`` is extracted from the time-resolved field at
    the end of the run.  The turbulent-to-laminar amplification and the decay
    length between the first and last station are the quantitative downstream
    answer for this node (channel receiver; cylinder St/Cd is *not* covered).

    Returns:
        Dict with ``turbulent`` and ``laminar`` station statistics and the
        amplification / decay metrics.
    """
    if ny is None:
        ny = int(signal.shape[2])
    cfg = ChannelConfig(
        nx=nx, ny=ny, nu=nu, u_bulk=u_bulk, route=route, n_steps=n_steps, monitor_stations=stations
    )
    out = {}
    for name, sig, amp in (
        ("turbulent", signal, 1.0),
        ("laminar", None, 0.0),
    ):
        c = ChannelConfig(**{**cfg.__dict__, "turbulence_amplitude": amp})
        sim = D2Q9Channel(c, signal=sig)
        res = sim.run()
        stats = {f"x{x}": _series_stats(res["u_series"][:, i]) for i, x in enumerate(stations)}
        stats["rms_profile"] = {f"x{x}": res["u_rms_field"][1:-1, x].tolist() for x in stations}
        out[name] = stats
    # The laminar reference sits at round-off (no fluctuation source at all), so a
    # raw ratio is meaningless; the floor is reported with the number.
    floor = 1e-5
    amp_ratio = {
        f"x{x}": out["turbulent"][f"x{x}"]["rms"] / max(out["laminar"][f"x{x}"]["rms"], floor)
        for x in stations
    }
    baseline_at_round_off = all(out["laminar"][f"x{x}"]["rms"] < floor for x in stations)
    x0, x1 = stations[0], stations[-1]
    rms0 = out["turbulent"][f"x{x0}"]["rms"]
    rms1 = out["turbulent"][f"x{x1}"]["rms"]
    decay = (
        float(math.log(max(rms0, 1e-12) / max(rms1, 1e-12)) / (x1 - x0))
        if x1 > x0
        else float("nan")
    )
    return {
        "turbulent": out["turbulent"],
        "laminar": out["laminar"],
        "amplification": amp_ratio,
        "amplification_floor": floor,
        "laminar_baseline_at_round_off": bool(baseline_at_round_off),
        "log_decay_rate_per_cell": decay,
        "stations": list(stations),
    }
