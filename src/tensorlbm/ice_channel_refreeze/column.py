"""One-dimensional enthalpy lattice-Boltzmann model of channel refreezing.

Numerics
--------
Conserved variable is the volumetric enthalpy ``H`` (J/m^3) referred to the
freezing point::

    H = rho_i cp_i (T - T_f)                     T <= T_f   (ice)
    H = rho_L f_l,  T = T_f,   0 <= f_l <= 1                (mushy)
    H = rho_L + rho_w cp_w (T - T_f)              f_l = 1   (water)

The sensible part is advanced by a one-dimensional thermal LBE (D1Q3,
``c_s^2 = 1/3``, diffusivity ``alpha = (tau - 1/2)/3``) with a *per-node*
relaxation time so that ice and water keep their own molecular diffusivity;
latent heat is released/absorbed by the equilibrium enthalpy inversion after
each LBE step (operator splitting, identical in structure to
``tensorlbm.phase_change.stefan_verification.EnthalpyThermalLBM``; see
``docs/P155W1_N01_STEFAN_PHASE_CHANGE.md``).

Boundary conditions
-------------------
* surface (node 0): either Dirichlet (used for the Neumann verification) or a
  Robin surface energy balance ``q_s = h_tot (T_0 - T_eff)`` applied as a
  volumetric sink in the surface cell;
* ocean side: the heat flux ``q_o`` is injected at the *ice-water interface*
  (last cell containing ice), which is the correct place for a basal flux: it
  melts ice from below / slows growth.  ``q_o`` itself comes from either a
  constant reservoir or a depleting mixed layer (``MixedLayerOcean``).

Units are SI; the grid is cell-centred with spacing ``dx`` and time step ``dt``
chosen from the diffusive stability limit ``dt = safety * dx^2 / alpha_max``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_PROPERTIES, IceWaterProperties
from .surface import AtmosphericForcing, effective_air_temperature

_W = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
_C = np.array([0, 1, -1])


@dataclass
class MixedLayerOcean:
    """Depleting mixed-layer heat reservoir driving the basal flux."""

    depth: float = 50.0            # m
    initial_superheat: float = 0.05  # K
    u_surface: float = 0.0         # m/s (set by the wind-driven flow model)
    props: IceWaterProperties = DEFAULT_PROPERTIES
    superheat: float = field(init=False)

    def __post_init__(self) -> None:
        self.superheat = float(self.initial_superheat)

    def flux(self) -> float:
        return (
            self.props.rho_w
            * self.props.cp_w
            * self.props.stanton_water
            * max(self.u_surface, 0.0)
            * max(self.superheat, 0.0)
        )

    def consume(self, heat_per_area: float) -> None:
        """Remove ``heat_per_area`` (J/m^2) from the mixed layer."""
        cap = self.props.rho_w * self.props.cp_w * max(self.depth, 1e-6)
        self.superheat = max(self.superheat - float(heat_per_area) / cap, 0.0)


@dataclass
class RefreezingColumn:
    """Enthalpy-LBM refreezing column (ice grows downwards from the surface)."""

    depth: float = 1.5                     # m, total column height
    n: int = 240                           # number of cells
    props: IceWaterProperties = DEFAULT_PROPERTIES
    safety: float = 0.30                   # dt = safety*dx^2/alpha_max
    forcing: AtmosphericForcing | None = None
    q_ocean: float = 0.0                   # W/m^2 (constant reservoir)
    ocean: MixedLayerOcean | None = None
    surface_dirichlet: float | None = None  # degC (verification mode)
    # --- state ---------------------------------------------------------
    dx: float = field(init=False)
    dt: float = field(init=False)
    H: np.ndarray = field(init=False)
    T: np.ndarray = field(init=False)
    f_l: np.ndarray = field(init=False)
    g: np.ndarray = field(init=False)
    tau: np.ndarray = field(init=False)
    steps: int = field(default=0, init=False)
    time: float = field(default=0.0, init=False)
    ocean_heat_removed: float = field(default=0.0, init=False)  # J/m^2

    def __post_init__(self) -> None:
        self.dx = self.depth / self.n
        alpha_max = max(self.props.alpha_i, self.props.alpha_w)
        self.dt = self.safety * self.dx ** 2 / alpha_max
        self.H = np.full(self.n, self.props.rho_latent, dtype=float)
        self.T = np.full(self.n, self.props.t_freeze, dtype=float)
        self.f_l = np.ones(self.n, dtype=float)
        self.g = np.tile(_W[:, None], (1, self.n)) * self.T[None, :]
        self._update_tau()

    # ------------------------------------------------------------------ setup
    def _update_tau(self) -> None:
        alpha = np.where(self.f_l < 1.0, self.props.alpha_i, self.props.alpha_w)
        alpha_lat = alpha * self.dt / self.dx ** 2
        self.tau = 0.5 + 3.0 * alpha_lat

    def invert(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Equilibrium enthalpy inversion ``H -> (T, f_l)``."""
        cap_i = self.props.volumetric_cp_i
        cap_w = self.props.volumetric_cp_w
        latent = self.props.rho_latent
        tf = self.props.t_freeze
        T = np.where(
            H < 0.0,
            tf + H / cap_i,
            np.where(H < latent, tf, tf + (H - latent) / cap_w),
        )
        f_l = np.clip(np.where(H < 0.0, 0.0, H / latent), 0.0, 1.0)
        return T, f_l

    def set_open_water(self, superheat: float = 0.0) -> None:
        """Initial state: ice-free water at the freezing point."""
        self.H = np.full(self.n, self.props.rho_latent + self.props.volumetric_cp_w * superheat)
        self.T, self.f_l = self.invert(self.H)
        self.g = np.tile(_W[:, None], (1, self.n)) * self.T[None, :]
        self._update_tau()
        self.steps = 0
        self.time = 0.0
        self.ocean_heat_removed = 0.0

    def set_state(self, T: np.ndarray, f_l: np.ndarray | None = None) -> None:
        """Impose a temperature profile (and phase state) as initial condition."""
        T = np.asarray(T, dtype=float).reshape(self.n)
        if f_l is None:
            f_l = np.where(T > self.props.t_freeze, 1.0, 0.0)
        f_l = np.clip(np.asarray(f_l, dtype=float).reshape(self.n), 0.0, 1.0)
        latent = self.props.rho_latent
        H = np.where(
            f_l <= 0.0,
            self.props.volumetric_cp_i * (T - self.props.t_freeze),
            np.where(
                f_l >= 1.0,
                latent + self.props.volumetric_cp_w * (T - self.props.t_freeze),
                latent * f_l,
            ),
        )
        self.T = T.copy()
        self.f_l = f_l.copy()
        self.H = H
        self.g = np.tile(_W[:, None], (1, self.n)) * self.T[None, :]
        self._update_tau()
        self.steps = 0
        self.time = 0.0
        self.ocean_heat_removed = 0.0

    # ------------------------------------------------------------------- step
    def _surface_flux(self) -> float:
        if self.surface_dirichlet is not None:
            return 0.0
        if self.forcing is None:
            return 0.0
        t_eff, h_tot = effective_air_temperature(self.forcing, self.props)
        return h_tot * (self.T[0] - t_eff)

    def _basal_flux(self) -> float:
        if self.ocean is not None:
            return self.ocean.flux()
        return float(self.q_ocean)

    def step(self, n_steps: int = 1) -> None:
        cap_i = self.props.volumetric_cp_i
        latent = self.props.rho_latent
        for _ in range(int(n_steps)):
            T_old = self.T
            # 1) collide (per-node relaxation) and stream (D1Q3)
            geq = _W[:, None] * T_old[None, :]
            self.g -= (self.g - geq) / self.tau[None, :]
            self.g[1] = np.roll(self.g[1], 1)
            self.g[2] = np.roll(self.g[2], -1)
            # zero-flux closure of the two ends (np.roll would wrap around)
            self.g[1, 0] = self.g[1, 1]
            self.g[2, -1] = self.g[2, -2]
            T_lbm = self.g.sum(axis=0)
            # 2) sensible enthalpy update
            cap_local = np.where(self.f_l < 1.0, cap_i, self.props.volumetric_cp_w)
            self.H += cap_local * (T_lbm - T_old)
            # 3) external fluxes
            if self.surface_dirichlet is None:
                q_s = self._surface_flux()
                if q_s:
                    self.H[0] -= self.dt * q_s / self.dx
            q_o = self._basal_flux()
            if q_o:
                ice = np.nonzero(self.f_l < 1.0)[0]
                idx = int(ice[-1]) if ice.size else 0
                self.H[idx] += self.dt * q_o / self.dx
                self.ocean_heat_removed += self.dt * q_o
            # 4) equilibrium inversion
            self.T, self.f_l = self.invert(self.H)
            # 5) write non-equilibrium part back to the distributions
            self.g += _W[:, None] * (self.T - T_lbm)[None, :]
            # 6) Dirichlet surface (verification mode only)
            if self.surface_dirichlet is not None:
                self.T[0] = float(self.surface_dirichlet)
                self.f_l[0] = 0.0
                self.H[0] = cap_i * (self.T[0] - self.props.t_freeze)
                self.g[:, 0] = _W * self.T[0]
            self.time += self.dt
            self.steps += 1
            self._update_tau()
        if self.ocean is not None:
            self.ocean.consume(self.ocean_heat_removed)
            self.ocean_heat_removed = 0.0

    # -------------------------------------------------------------- diagnostics
    def thickness(self) -> float:
        """Equivalent ice thickness ``h = dx * sum(1 - f_l)``, m."""
        return float(self.dx * np.sum(1.0 - self.f_l))

    def interface_index(self) -> int:
        ice = np.nonzero(self.f_l < 1.0)[0]
        return int(ice[-1]) if ice.size else 0

    def liquid_fraction(self) -> np.ndarray:
        return self.f_l.copy()

    def temperature_profile(self) -> np.ndarray:
        return self.T.copy()

    def run_to(self, t_end: float, sample_every: int = 0) -> dict:
        """Integrate to ``t_end`` seconds; optionally sample ``h(t)``."""
        n_target = int(round((t_end - self.time) / self.dt))
        if n_target <= 0:
            return {"t": np.array([self.time]), "h": np.array([self.thickness()])}
        if not sample_every:
            self.step(n_target)
            return {"t": np.array([self.time]), "h": np.array([self.thickness()])}
        ts, hs = [], []
        done = 0
        while done < n_target:
            chunk = min(sample_every, n_target - done)
            self.step(chunk)
            done += chunk
            ts.append(self.time)
            hs.append(self.thickness())
        return {"t": np.asarray(ts), "h": np.asarray(hs)}


__all__ = ["MixedLayerOcean", "RefreezingColumn"]