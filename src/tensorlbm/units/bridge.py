"""Physical <-> lattice unit bridge with explicit, individually testable sites.

Why this module exists
----------------------
LBM is solved in lattice units (``dx = dt = rho = 1``).  Every result that a
user sees is produced by a *dimensionalisation chain*: lattice spacing ``dx``
[m], time step ``dt`` [s] and reference density ``rho0`` [kg/m^3] map lattice
numbers onto physical ones.  A factor error anywhere in that chain silently
rescales the answer while leaving the simulation looking perfectly healthy
(the historical ``F/2`` drag defect: the reported force was half of the true
value because the force conversion site was wrong, not because the solver
was wrong).

This module makes every conversion site explicit and separately addressable
so that :mod:`tensorlbm.units.gate` can (a) verify round-trip closure to
machine precision and (b) inject single-site factor errors ("mutation
battery") and require the anchor suite to expose them.

Conversion sites (SI unless noted)
----------------------------------
=================  =========================  =============================
site               lattice -> physical        formula
=================  =========================  =============================
length             ``L``                      ``L * dx``
time               ``t``                      ``t * dt``
velocity           ``u``                      ``u * dx / dt``
viscosity          ``nu``                     ``nu * dx^2 / dt``
density            ``rho``                    ``rho * rho0``
stress             ``tau``                    ``tau * rho0 * dx^2 / dt^2``
force              ``F``                      ``F * rho0 * dx^4 / dt^2``
=================  =========================  =============================

Mass conservation of the mapping: ``force = stress * area`` is preserved by
the two formulas above because a lattice area is ``dx^2``.

Mutation semantics (deliberate, documented restriction)
-------------------------------------------------------
The four site factors (``length_factor``, ``time_factor``,
``density_factor``, ``viscosity_factor``) are applied **only on the
lattice -> physical (output) side**; the physical -> lattice (set-up) side
always uses the true ``dx``/``dt``/``rho0``.  This models one bug class
exactly: a driver that dimensionalises its *results* with a wrong
coefficient.

A *consistent* factor error applied on both the set-up and the output side is
**not** detectable by any physical test, because the lattice algorithm is
similarity invariant (that is the whole point of LBM non-dimensionalisation):
``Re_lb`` and every non-dimensional number are unchanged when ``dx`` or
``dt`` is rescaled.  Claiming otherwise would be dishonest, so the gate
tests the output site, which is where real defects (e.g. a drag force off by
two) are observable, and the round-trip identity tests algebraic closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = [
    "LatticeBridge",
    "MUTATIONS",
    "CS_LATTICE",
    "MAX_MA_MUTATION_SAFE",
    "bootstrap_bridge",
    "mutate",
]

#: lattice speed of sound ``cs = 1/sqrt(3)``
CS_LATTICE: float = 1.0 / math.sqrt(3.0)

#: upper Ma accepted by :func:`bootstrap_bridge` (compressibility control)
MAX_MA_MUTATION_SAFE: float = 0.1


@dataclass(frozen=True)
class LatticeBridge:
    """Dimensional bridge between lattice units and SI units.

    Parameters
    ----------
    dx:
        Physical size of one lattice spacing [m] (true value).
    dt:
        Physical duration of one lattice step [s] (true value).
    rho0:
        Physical density [kg m^-3] that is mapped to lattice ``rho = 1``.
    length_factor, time_factor, density_factor, viscosity_factor:
        Output-site site factors, ``1.0`` for a correct bridge.  Values other
        than ``1.0`` represent a single-site dimensionalisation defect and are
        only used by the mutation battery.
    """

    dx: float
    dt: float
    rho0: float
    length_factor: float = 1.0
    time_factor: float = 1.0
    density_factor: float = 1.0
    viscosity_factor: float = 1.0

    # ------------------------------------------------------------------ time
    def time_to_phys(self, t_lb: float) -> float:
        return t_lb * self.dt * self.time_factor

    def time_to_lattice(self, t: float) -> float:
        return t / self.dt

    def rate_to_phys(self, rate_lb: float) -> float:
        """Per-step rate -> per-second rate [1/s]."""
        return rate_lb / (self.dt * self.time_factor)

    # ---------------------------------------------------------------- length
    def length_to_phys(self, x_lb: float) -> float:
        return x_lb * self.dx * self.length_factor

    def length_to_lattice(self, x: float) -> float:
        return x / self.dx

    def wave_number_to_phys(self, k_lb: float) -> float:
        """Lattice wave number [1/lattice] -> SI wave number [1/m]."""
        return k_lb / (self.dx * self.length_factor)

    # -------------------------------------------------------------- velocity
    def velocity_to_phys(self, u_lb: float) -> float:
        return u_lb * self.dx * self.length_factor / (self.dt * self.time_factor)

    def velocity_to_lattice(self, u: float) -> float:
        return u * self.dt / self.dx

    # ------------------------------------------------------------- viscosity
    def viscosity_to_phys(self, nu_lb: float) -> float:
        return (
            nu_lb
            * (self.dx * self.length_factor) ** 2
            / (self.dt * self.time_factor)
            * self.viscosity_factor
        )

    def viscosity_to_lattice(self, nu: float) -> float:
        return nu * self.dt / self.dx**2

    def tau_from_viscosity(self, nu_lb: float) -> float:
        """BGK relaxation time for a lattice viscosity (``tau = 1/2 + 3 nu``)."""
        return 0.5 + 3.0 * nu_lb

    # --------------------------------------------------------------- density
    def density_to_phys(self, rho_lb: float) -> float:
        return rho_lb * self.rho0 * self.density_factor

    def density_to_lattice(self, rho: float) -> float:
        return rho / self.rho0

    # ------------------------------------------------------ stress and force
    def stress_to_phys(self, tau_lb: float) -> float:
        """Lattice stress (unit density) -> SI stress [Pa]."""
        return (
            tau_lb
            * self.rho0
            * self.density_factor
            * (self.dx * self.length_factor) ** 2
            / (self.dt * self.time_factor) ** 2
        )

    def stress_to_lattice(self, stress: float) -> float:
        return stress / (self.rho0 * self.dx**2 / self.dt**2)

    def force_to_phys(self, force_lb: float) -> float:
        """Lattice force (unit density) -> SI force [N].

        ``F = m a`` with mass scale ``rho0 dx^3`` and acceleration scale
        ``dx/dt^2``, i.e. ``rho0 dx^4/dt^2``.
        """
        return (
            force_lb
            * self.rho0
            * self.density_factor
            * (self.dx * self.length_factor) ** 4
            / (self.dt * self.time_factor) ** 2
        )

    def force_to_lattice(self, force: float) -> float:
        return force / (self.rho0 * self.dx**4 / self.dt**2)

    # ------------------------------------------------------------ diagnostics
    def reynolds_lattice(self, u_lb: float, length_lb: float, nu_lb: float) -> float:
        return u_lb * length_lb / nu_lb

    def reynolds_physical(self, u: float, length: float, nu: float) -> float:
        return u * length / nu

    def mach(self, u_lb: float) -> float:
        return abs(u_lb) / CS_LATTICE


def bootstrap_bridge(
    *,
    length_phys: float,
    velocity_phys: float,
    nu_phys: float,
    nx: int,
    rho0: float = 1000.0,
    u_lb: float = 0.05,
) -> LatticeBridge:
    """Derive ``dx``/``dt`` from a physical operating point and a grid.

    This is the standard LBM bootstrap: the grid fixes ``dx = L/nx``; the
    compressibility constraint fixes the lattice velocity ``u_lb`` (hence
    ``Ma = u_lb*sqrt(3)``); ``dt`` then follows from
    ``u_lb = u_phys dt / dx``.

    Raises
    ------
    ValueError
        If the resulting Mach number is not below ``0.1`` or the relaxation
        time falls outside ``[0.51, 2.0]``.
    """
    if nx < 4:
        raise ValueError("nx must be >= 4 lattice cells")
    dx = length_phys / nx
    dt = u_lb * dx / velocity_phys
    bridge = LatticeBridge(dx=dx, dt=dt, rho0=rho0)
    ma = bridge.mach(u_lb)
    tau = bridge.tau_from_viscosity(bridge.viscosity_to_lattice(nu_phys))
    if ma > MAX_MA_MUTATION_SAFE:
        raise ValueError(f"lattice Mach number {ma:.3f} exceeds 0.1")
    if not (0.51 <= tau <= 2.0):
        raise ValueError(f"relaxation time {tau:.3f} outside [0.51, 2.0]")
    return bridge


#: Single-site dimensionalisation defects used by the mutation battery.
MUTATIONS: dict[str, dict[str, float]] = {
    "length_x2": {"length_factor": 2.0},
    "length_div2": {"length_factor": 0.5},
    "rho0_x2": {"density_factor": 2.0},
    "nu_x2": {"viscosity_factor": 2.0},
    "dt_x2": {"time_factor": 2.0},
}


def mutate(bridge: LatticeBridge, name: str) -> LatticeBridge:
    """Return ``bridge`` with one conversion site multiplied by a factor."""
    if name not in MUTATIONS:
        raise KeyError(f"unknown mutation {name!r}; known: {sorted(MUTATIONS)}")
    return replace(bridge, **MUTATIONS[name])