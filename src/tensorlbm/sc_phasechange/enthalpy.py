"""Enthalpy-method ice/water phase change with an explicit Stefan-condition audit.

Physical model
--------------
One-dimensional melting of ice by a wall held above the melting point.  The
enthalpy per unit mass relative to the melting temperature ``T_m`` is

.. math::

    H = \\int_{T_m}^{T} c_p\\,dT + L\\,\\phi_l ,

with ``phi_l`` the liquid fraction: ``H < 0`` -> solid (``T < T_m``),
``0 <= H <= L`` -> isothermal mushy state (``T = T_m``), ``H > L`` -> liquid.
The temperature is the inverse of that relation, so the moving interface is
*implicit* -- no front tracking -- and the Stefan condition

.. math::

    \\rho L v_n = \\left[k\\,\\frac{\\partial T}{\\partial n}\\right]_{s}^{l}

is a property of the discrete enthalpy balance rather than an extra equation.
It is nevertheless audited here *independently*: the left-hand side from the
phase-field evolution, the right-hand side from one-sided temperature
gradients two cells away from the interface, both accumulated over a window of
steps (a single step's latent uptake is quantised by the cell size, which
produces spurious +/-17 % single-step residuals -- see the report).

Numerics
--------
Finite volume, explicit in time, harmonic-mean face conductivity (exact for a
face sitting on a conductivity jump), Dirichlet wall at ``x = 0`` through a
ghost cell at half a cell distance, Dirichlet ``T = T_i`` at the far end.  That
far condition is only legitimate while the subcooled boundary layer stays in
the domain, so the domain length is sized from
:meth:`NeumannMelting.solid_penetration` (the runner does this automatically).

Units are SI (``K``, ``J``, ``kg``, ``m``, ``s``), unlike the lattice-unit
Stefan work in :mod:`tensorlbm.phase_change`: with the real ice/water property
set, a temperature offset of ``5 K`` or a Stefan number built on the true
latent heat means the physical thing rather than a normalisation.

The *constant-density* idealisation of the classical Stefan problem is used
(same ``rho`` in both phases): the real 917 -> 1000 kg/m^3 density jump drives
a volume-change flow that a 1-D conduction model cannot represent.  The real
densities are kept on the property object for disclosure.

References
----------
Carslaw & Jaeger (1959) *Conduction of Heat in Solids*, 2nd ed., §11.4
(two-phase Neumann similarity solution); Voller & Cross (1981) Int. J. Heat
Mass Transfer 24 545 (enthalpy method); Alexiades & Solomon (1993)
*Mathematical Modeling of Melting and Freezing Processes*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc

__all__ = [
    "ICE_WATER",
    "PhaseChangeProperties",
    "NeumannMelting",
    "EnthalpyPhaseChange1D",
    "PhaseChangeStep",
]


@dataclass(frozen=True)
class PhaseChangeProperties:
    """Thermophysical properties of the solid/liquid pair (SI units).

    Defaults: ice/water at 273.15 K.  ``rho`` is shared by both phases (the
    constant-density Stefan idealisation); ``rho_solid_real``/``rho_liquid_real``
    keep the measured densities for disclosure only.
    """

    t_melt: float = 273.15  # K
    latent: float = 3.34e5  # J/kg
    cp_solid: float = 2100.0  # J/(kg K)
    cp_liquid: float = 4180.0  # J/(kg K)
    k_solid: float = 2.22  # W/(m K)
    k_liquid: float = 0.56  # W/(m K)
    rho: float = 1000.0  # kg/m^3 (both phases, idealisation)
    rho_solid_real: float = 917.0  # kg/m^3 (disclosure only)
    rho_liquid_real: float = 1000.0  # kg/m^3 (disclosure only)

    @property
    def alpha_solid(self) -> float:
        return self.k_solid / (self.rho * self.cp_solid)

    @property
    def alpha_liquid(self) -> float:
        return self.k_liquid / (self.rho * self.cp_liquid)

    def stefan_number(self, delta_t: float) -> float:
        """``Ste = c_p dT / L`` on the liquid side (superheat above ``T_m``)."""
        return self.cp_liquid * float(delta_t) / self.latent

    def superheat_for_stefan(self, stefan: float) -> float:
        """Superheat ``dT = Ste L / c_p`` (K) realising a Stefan number."""
        return float(stefan) * self.latent / self.cp_liquid


ICE_WATER = PhaseChangeProperties()


# ---------------------------------------------------------------------------
# Neumann (similarity) solution for two-phase melting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NeumannMelting:
    """Exact 1-D melting of a semi-infinite solid with optional subcooling.

    Wall ``x = 0`` is held at ``T_s = T_m + superheat`` for ``t > 0``; the solid
    is initially at ``T_i = T_m - subcooling`` (``<= T_m``).  With
    ``eta = x / (2 sqrt(alpha_l t))`` and ``lam = s(t) / (2 sqrt(alpha_l t))``

    * liquid, ``x < s``:  ``T = T_s - (T_s - T_m) erf(eta)/erf(lam)``
    * solid,  ``x > s``:  ``T = T_i + (T_m - T_i) erfc(eta_s)/erfc(gam)``,
      ``eta_s = (x - s)/(2 sqrt(alpha_s t))``, ``gam = lam sqrt(alpha_l/alpha_s)``

    and ``lam`` solves the transcendental Stefan balance (heat conducted in
    through the liquid minus heat conducted away into the subcooled solid =
    latent heat of the melted layer):

    .. math::

        \\lambda = \\frac{c_{p,l}(T_s-T_m)}{L\\sqrt{\\pi}}\\,
                   \\frac{e^{-\\lambda^2}}{\\mathrm{erf}\\,\\lambda}
                 - \\frac{c_{p,s}(T_m-T_i)}{L\\sqrt{\\pi}}\\,
                   \\sqrt{\\frac{\\alpha_s}{\\alpha_l}}\\,
                   \\frac{e^{-\\lambda^2\\alpha_l/\\alpha_s}}
                        {\\mathrm{erfc}(\\lambda\\sqrt{\\alpha_l/\\alpha_s})}

    For ``subcooling = 0`` this reduces to the classical one-phase relation
    ``lam exp(lam^2) erf(lam) = Ste/sqrt(pi)``, cross-checked in the tests
    against :func:`tensorlbm.phase_change.stefan_exact.planar_lambda`.
    """

    superheat: float
    subcooling: float = 0.0
    props: PhaseChangeProperties = ICE_WATER

    def __post_init__(self) -> None:
        if self.superheat <= 0.0:
            raise ValueError("superheat must be positive")
        if self.subcooling < 0.0:
            raise ValueError("subcooling must be non-negative")

    @property
    def alpha_l(self) -> float:
        return self.props.alpha_liquid

    @property
    def alpha_s(self) -> float:
        return self.props.alpha_solid

    @property
    def t_surface(self) -> float:
        return self.props.t_melt + self.superheat

    @property
    def t_initial(self) -> float:
        return self.props.t_melt - self.subcooling

    def _balance(self, lam: float) -> float:
        p = self.props
        liquid = (
            p.cp_liquid
            * self.superheat
            / (p.latent * math.sqrt(math.pi))
            * math.exp(-lam * lam)
            / erf(lam)
        )
        solid = 0.0
        if self.subcooling > 0.0:
            ratio = self.alpha_l / self.alpha_s
            gam = lam * math.sqrt(ratio)
            solid = (
                p.cp_solid
                * self.subcooling
                / (p.latent * math.sqrt(math.pi))
                * math.sqrt(self.alpha_s / self.alpha_l)
                * math.exp(-lam * lam * ratio)
                / erfc(gam)
            )
        return lam - liquid + solid

    @property
    def lam(self) -> float:
        # balance(lam) -> -inf as lam -> 0+ (the liquid term diverges as
        # 1/erf(lam)) and grows without bound for large lam: one sign change.
        lo = 1e-9
        hi = 1.0
        while self._balance(hi) < 0.0:
            hi *= 2.0
            if hi > 64.0:  # pragma: no cover - guarded by the property ranges
                raise ValueError("no Neumann similarity solution found")
        return float(brentq(self._balance, lo, hi, xtol=1e-15, rtol=1e-15))

    def front(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return 2.0 * self.lam * math.sqrt(self.alpha_l * t)

    def time_for_front(self, s: float) -> float:
        return (s / (2.0 * self.lam)) ** 2 / self.alpha_l

    def front_velocity(self, t: float) -> float:
        if t <= 0.0:
            return float("inf")
        return self.lam * math.sqrt(self.alpha_l / t)

    def solid_penetration(self, t: float) -> float:
        """``3 sqrt(alpha_s t)``: where the subcooled layer is ~0.4 % depleted."""
        return 3.0 * math.sqrt(self.alpha_s * t)

    def temperature(self, x, t: float) -> np.ndarray:
        p = self.props
        x = np.asarray(x, dtype=float)
        if t <= 0.0:
            return np.full_like(x, self.t_initial)
        s = self.front(t)
        lam = self.lam
        eta = x / (2.0 * math.sqrt(self.alpha_l * t))
        liquid = self.t_surface - self.superheat * erf(eta) / erf(lam)
        if self.subcooling > 0.0:
            ratio = self.alpha_l / self.alpha_s
            eta_s = (x - s) / (2.0 * math.sqrt(self.alpha_s * t))
            solid = self.t_initial + self.subcooling * erfc(eta_s) / erfc(lam * math.sqrt(ratio))
        else:
            solid = np.full_like(x, p.t_melt)
        return np.where(x <= s, liquid, solid)

    def wall_heat_flux(self, t: float) -> float:
        """``-k_l dT/dx|_0`` (W/m^2, positive into the domain)."""
        if t <= 0.0:
            return float("inf")
        p = self.props
        return p.k_liquid * self.superheat / (math.sqrt(math.pi * self.alpha_l * t) * erf(self.lam))


# ---------------------------------------------------------------------------
# Enthalpy solver
# ---------------------------------------------------------------------------
@dataclass
class PhaseChangeStep:
    """Diagnostics at one recorded time level."""

    time: float
    front: float
    front_neumann: float
    front_error_pct: float
    front_crossing: float
    liquid_fraction_mean: float
    stefan_latent_rate: float
    stefan_flux_jump: float
    stefan_residual_pct: float
    energy_total: float
    energy_imbalance_pct: float
    energy_cumulative_imbalance_pct: float


@dataclass
class EnthalpyPhaseChange1D:
    """Explicit finite-volume enthalpy solver for 1-D melting.

    ``n_cells`` cells over ``length`` (SI); wall at ``x = 0``, far field held at
    the initial temperature.  ``cfl`` is ``dt / (dx^2 / alpha_max)`` and must
    stay below 0.5 for the explicit scheme.
    """

    length: float = 0.8
    n_cells: int = 800
    props: PhaseChangeProperties = ICE_WATER
    superheat: float = 20.0
    subcooling: float = 0.0
    cfl: float = 0.45
    x: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)
    dt: float = field(init=False)
    enthalpy: np.ndarray = field(init=False, repr=False)
    time: float = field(init=False, default=0.0)
    steps: int = field(init=False, default=0)
    _enthalpy_prev: np.ndarray | None = field(init=False, repr=False, default=None)
    _energy_initial: float = field(init=False, default=0.0)
    _energy_window_start: float = field(init=False, default=0.0)
    _wall_heat: float = field(init=False, default=0.0)
    _wall_heat_total: float = field(init=False, default=0.0)
    _acc_latent: float = field(init=False, default=0.0)
    _acc_flux: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.n_cells < 8:
            raise ValueError("need length > 0 and n_cells >= 8")
        if not 0.0 < self.cfl < 0.5:
            raise ValueError("cfl must be in (0, 0.5) for the explicit scheme")
        self.dx = self.length / self.n_cells
        self.x = (np.arange(self.n_cells) + 0.5) * self.dx
        alpha_max = max(self.props.alpha_liquid, self.props.alpha_solid)
        self.dt = self.cfl * self.dx * self.dx / alpha_max
        self.enthalpy = np.full(self.n_cells, -self.props.cp_solid * self.subcooling)
        self._enthalpy_prev = None
        self.time = 0.0
        self.steps = 0
        self._energy_initial = float(self.props.rho * np.sum(self.enthalpy) * self.dx)
        self._energy_window_start = self._energy_initial
        self._wall_heat = 0.0
        self._wall_heat_total = 0.0
        self._acc_latent = 0.0
        self._acc_flux = 0.0

    # -- enthalpy <-> temperature -------------------------------------------
    def temperature_of(self, h: np.ndarray | None = None) -> np.ndarray:
        p = self.props
        h = self.enthalpy if h is None else h
        t = np.empty_like(h)
        solid = h <= 0.0
        mushy = (h > 0.0) & (h < p.latent)
        liquid = h >= p.latent
        t[solid] = p.t_melt + h[solid] / p.cp_solid
        t[mushy] = p.t_melt
        t[liquid] = p.t_melt + (h[liquid] - p.latent) / p.cp_liquid
        return t

    def liquid_fraction_of(self, h: np.ndarray | None = None) -> np.ndarray:
        p = self.props
        h = self.enthalpy if h is None else h
        phi = np.zeros_like(h)
        mushy = (h > 0.0) & (h < p.latent)
        phi[mushy] = h[mushy] / p.latent
        phi[h >= p.latent] = 1.0
        return phi

    @property
    def liquid_fraction(self) -> np.ndarray:
        return self.liquid_fraction_of()

    @property
    def temperature(self) -> np.ndarray:
        return self.temperature_of()

    def conductivity_of(self, temp: np.ndarray) -> np.ndarray:
        p = self.props
        return np.where(temp >= p.t_melt, p.k_liquid, p.k_solid)

    # -- front / integral diagnostics ---------------------------------------
    @property
    def front(self) -> float:
        """Phase-integral front ``int_0^L phi_l dx`` (conserved, noise-free)."""
        return float(self.liquid_fraction.sum() * self.dx)

    def front_crossing(self) -> float:
        """``phi_l = 0.5`` crossing (independent, sharper near ``t -> 0``)."""
        phi = self.liquid_fraction
        idx = np.nonzero(phi < 0.5)[0]
        if idx.size == 0:
            return self.length
        j = int(idx[0])
        if j == 0:
            return 0.0
        p0, p1 = phi[j - 1], phi[j]
        frac = (p0 - 0.5) / (p0 - p1) if p0 != p1 else 0.5
        return float((j - 1 + frac + 0.5) * self.dx)

    @property
    def total_energy(self) -> float:
        """Sensible + latent energy per unit area (J/m^2), zero for solid at Tm."""
        return float(self.props.rho * np.sum(self.enthalpy) * self.dx)

    @property
    def t_surface(self) -> float:
        return self.props.t_melt + self.superheat

    @property
    def t_initial(self) -> float:
        return self.props.t_melt - self.subcooling

    def wall_flux_numeric(self) -> float:
        """Heat flux entering the domain through the wall (W/m^2)."""
        temp = self.temperature
        p = self.props
        k0 = p.k_liquid if temp[0] >= p.t_melt else p.k_solid
        return float(k0 * (self.t_surface - temp[0]) / (0.5 * self.dx))

    def wall_flux_audit(self, temp: np.ndarray | None = None) -> float:
        """Wall heat flux from an *independent* second-order extrapolation.

        The update uses the ghost-cell formula (half-cell distance); the audit
        uses the parabola through cells 0,1,2 evaluated at ``x = 0``.  Two
        different discretisations of the same boundary condition make the
        energy-budget check non-circular.
        """
        temp = self.temperature if temp is None else temp
        k0 = self.props.k_liquid if temp[0] >= self.props.t_melt else self.props.k_solid
        slope = self._quadratic_slope(self.x[:3], temp[:3], 0.0)
        return float(-k0 * slope)

    def far_flux_audit(self, temp: np.ndarray | None = None) -> float:
        """Far-boundary heat flux (``+x`` positive), second-order extrapolated."""
        temp = self.temperature if temp is None else temp
        k1 = self.props.k_liquid if temp[-1] >= self.props.t_melt else self.props.k_solid
        xs = self.x[-3:]
        slope = self._quadratic_slope(xs, temp[-3:], self.length)
        return float(-k1 * slope)

    def _flux_jump(self, temp: np.ndarray | None = None) -> float:
        """``q_l - q_s`` across the interface (W/m^2), interface-extrapolated.

        The interface position ``x_f`` comes from the phase field
        (``x_f = int phi_l dx``); the temperature gradient is taken from a
        local parabola through the three bulk liquid cells and the three bulk
        solid cells adjacent to the interface and evaluated *at* ``x_f``.  The
        partial mushy cell (where ``dT/dx`` is an O(1/dx) artefact of the
        isothermal-mushy idealisation) is excluded.  A one-sided first-order
        gradient two cells behind the front is ~4 % low at ``Ste = 0.5`` and
        grows with time, which is exactly the bias this extrapolation removes.
        """
        temp = self.temperature if temp is None else temp
        x_f = self.front
        if x_f <= 2.5 * self.dx or x_f >= self.length - 3.5 * self.dx:
            return 0.0
        k = self.conductivity_of(temp)
        j = int(np.searchsorted(self.x, x_f))  # first cell right of x_f
        liq = (j - 1, j - 2, j - 3)  # bulk liquid cells
        sol = (j, j + 1, j + 2)  # bulk solid cells
        if min(liq) < 0 or max(sol) > self.n_cells - 1:
            return 0.0
        q_l = -k[liq[0]] * self._quadratic_slope(self.x[list(liq)], temp[list(liq)], x_f)
        q_s = -k[sol[0]] * self._quadratic_slope(self.x[list(sol)], temp[list(sol)], x_f)
        return float(q_l - q_s)

    @staticmethod
    def _quadratic_slope(xs: np.ndarray, ts: np.ndarray, x0: float) -> float:
        """Slope at ``x0`` of the parabola through ``(xs, ts)`` (3 points)."""
        x1, x2, x3 = xs
        y1, y2, y3 = ts
        d1 = (y2 - y1) / (x2 - x1)
        d2 = (y3 - y2) / (x3 - x2)
        c2 = (d2 - d1) / (x3 - x1)  # second derivative coefficient
        return d1 + c2 * ((x0 - x2) + (x0 - (x1 + x2) / 2.0))

    # -- one explicit step ---------------------------------------------------
    def step(self) -> None:
        p = self.props
        temp = self.temperature
        k = self.conductivity_of(temp)
        k_face = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])  # harmonic mean
        q_int = -k_face * np.diff(temp) / self.dx
        q_wall = self.wall_flux_numeric()
        q_far = -k[-1] * (self.t_initial - temp[-1]) / (0.5 * self.dx)
        q = np.empty(self.n_cells + 1)
        q[0] = q_wall
        q[1:-1] = q_int
        q[-1] = q_far

        prev_phi = self.liquid_fraction_of()
        self._enthalpy_prev = self.enthalpy.copy()
        self.enthalpy = self.enthalpy + self.dt * (q[:-1] - q[1:]) / (p.rho * self.dx)
        self.time += self.dt
        self.steps += 1

        # audit accumulators (windowed: a single step's latent uptake is
        # quantised by the cell size)
        d_phi = float((self.liquid_fraction - prev_phi).sum()) * self.dx
        self._acc_latent += p.rho * p.latent * d_phi
        self._acc_flux += self._flux_jump() * self.dt
        wall_now = self.wall_flux_audit() - self.far_flux_audit()
        self._wall_heat += wall_now * self.dt
        self._wall_heat_total += wall_now * self.dt

    # -- drivers -------------------------------------------------------------
    def run(
        self,
        t_end: float,
        record_times: list[float] | None = None,
        reference: NeumannMelting | None = None,
    ) -> list[PhaseChangeStep]:
        """Integrate to ``t_end``; diagnostics at each entry of ``record_times``."""
        if reference is None:
            reference = NeumannMelting(self.superheat, self.subcooling, self.props)
        record = sorted(float(t) for t in (record_times or []))
        out: list[PhaseChangeStep] = []
        for target in record:
            target = min(target, t_end)
            n = int(math.ceil((target - self.time) / self.dt))
            for _ in range(max(n, 1)):
                self.step()
            out.append(self.diagnostic(reference))
        return out

    def diagnostic(self, reference: NeumannMelting | None = None) -> PhaseChangeStep:
        reference = reference or NeumannMelting(self.superheat, self.subcooling, self.props)
        front = self.front
        front_ref = reference.front(self.time)
        resid = (
            100.0 * (self._acc_latent - self._acc_flux) / abs(self._acc_flux)
            if abs(self._acc_flux) > 0.0
            else float("nan")
        )
        energy_now = self.total_energy
        # Windowed energy budget: the t -> 0 singularity of the imposed wall
        # temperature is not resolved by any grid, so the audit is applied to
        # each record window separately (and the cumulative value reported).
        imbalance = (
            100.0
            * (energy_now - self._energy_window_start - self._wall_heat)
            / abs(self._wall_heat)
            if abs(self._wall_heat) > 0.0
            else float("nan")
        )
        cumulative = (
            100.0
            * (energy_now - self._energy_initial - self._wall_heat_total)
            / abs(self._wall_heat_total)
            if abs(self._wall_heat_total) > 0.0
            else float("nan")
        )
        out = PhaseChangeStep(
            time=self.time,
            front=front,
            front_neumann=front_ref,
            front_error_pct=(
                100.0 * (front - front_ref) / front_ref if front_ref > 0 else float("nan")
            ),
            front_crossing=self.front_crossing(),
            liquid_fraction_mean=float(self.liquid_fraction.mean()),
            stefan_latent_rate=self._acc_latent,  # window-integrated latent heat (J/m^2)
            stefan_flux_jump=self._acc_flux,  # window-integrated flux jump (J/m^2)
            stefan_residual_pct=resid,
            energy_total=energy_now,
            energy_imbalance_pct=imbalance,
            energy_cumulative_imbalance_pct=cumulative,
        )
        self._acc_latent = 0.0
        self._acc_flux = 0.0
        self._wall_heat = 0.0
        self._energy_window_start = energy_now
        return out
