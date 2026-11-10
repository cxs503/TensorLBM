"""Four analytic anchors that make the unit bridge machine-verifiable.

Each anchor is a *physical* problem whose solution is known in closed form.
The case is set up in physical (SI) parameters, the bridge produces the
lattice parameters, a real D2Q9 simulation is run in lattice units, and the
lattice result is dimensionalised again with the bridge and compared against
the closed-form physical value.  Any factor error in a conversion site shows
up as a factor error in the reported physical number, while the simulation
itself looks perfectly healthy -- which is exactly the failure mode the gate
exists to catch.

Anchors and reference laws
--------------------------
``poiseuille``   plane channel, ``U_b^+ = Re_tau / 3`` (wall units, ``h`` =
                 channel half-height) and ``u_max = f h^2 / (8 rho nu)``
``couette``      linear profile, wall shear ``tau_w = mu U / h``
``stokes``       Stokes drag on a sphere, ``F = 6 pi mu a U``
``taylor_green`` 2D Taylor-Green decay ``exp(-2 nu k^2 t)``

Regime note (honest scoping)
----------------------------
The ``Ma < 0.1`` and ``tau in [0.51, 2]`` constraints required by the gate
force a low lattice Reynolds number for the channel anchors, i.e. the anchors
live in the creeping-flow regime (``Re_h`` of order one, micro-fluidic
scaling).  That is deliberate: these anchors verify the *unit conversion
chain*, not turbulence modelling, and a creeping-flow solution is exactly the
closed form quoted above, so the residual error is attributable to the
conversion chain rather than to discretisation.

The Stokes anchor is a *conversion-chain* anchor: the lattice drag is
evaluated from the lattice similarity relations rather than from a resolved
3D simulation (a 2% accurate resolved sphere would need a cluster run and
would inject solver discretisation error into a conversion test).  Its role in
the mutation battery is to exercise the mass/length/time site families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bridge import LatticeBridge
from .lattice import channel_couette, channel_poiseuille, taylor_green

__all__ = [
    "ANCHORS",
    "TOLERANCES",
    "AnchorCase",
    "build_case",
    "evaluate",
]

#: Pre-declared acceptance tolerances (relative error on the anchor number).
TOLERANCES: dict[str, float] = {
    "poiseuille": 4.0e-3,
    "couette": 5.0e-3,
    "stokes": 2.0e-2,
    "taylor_green": 2.0e-2,
}

ANCHORS: tuple[str, ...] = ("poiseuille", "couette", "stokes", "taylor_green")

#: lattice velocity used by every anchor (Ma = 0.02*sqrt(3) ~ 0.035 << 0.1)
U_LB: float = 0.02

#: default grid families (three independent (dx, dt) groups per anchor)
GRIDS: dict[str, tuple[int, ...]] = {
    "poiseuille": (48, 64, 80),  # channel height in lattice cells
    "couette": (8, 12, 16),
    "stokes": (24, 32, 48),      # cells per reference diameter
    "taylor_green": (32, 48, 64),
}


@dataclass
class AnchorCase:
    """A physical anchor: SI operating point + lattice set-up + measurement."""

    anchor: str
    grid: int
    physical: dict[str, float] = field(default_factory=dict)
    lattice: dict[str, float] = field(default_factory=dict)
    bridge: LatticeBridge | None = None

    @property
    def name(self) -> str:
        return f"{self.anchor}[{self.grid}]"


def _draw(seed: int, anchor: str) -> dict[str, float]:
    """Deterministic physical operating point for a seed.

    Values are drawn in ``+-10%`` around a micro-fluidic reference point so
    that the three seeds are genuinely different physical problems and the
    ``tau`` window required by the gate stays satisfied.
    """
    rng = np.random.RandomState(1000 + seed)
    jitter = lambda: float(rng.uniform(0.9, 1.1))  # noqa: E731
    base = {
        # micro-fluidic reference points; Re_h ~ 5 for the channel anchors so
        # that tau stays inside [0.51, 2] on every grid of the ladder.
        "poiseuille": dict(h=1.0e-3, nu=1.0e-6, rho0=998.0, u_ref=2.5e-3),
        "couette": dict(h=1.0e-3, nu=1.0e-6, rho0=998.0, u_ref=2.0e-3),
        "stokes": dict(a=5.0e-5, nu=1.0e-6, rho0=998.0, u_ref=1.0e-3),
        "taylor_green": dict(h=1.0e-3, nu=1.0e-6, rho0=998.0, u_ref=1.0e-2),
    }[anchor]
    return {k: v * jitter() for k, v in base.items()}


def build_case(anchor: str, seed: int, grid: int) -> AnchorCase:
    """Set up the physical case, derive lattice parameters, run the solver."""
    if anchor not in ANCHORS:
        raise KeyError(f"unknown anchor {anchor!r}")
    phys = _draw(seed, anchor)
    case = AnchorCase(anchor=anchor, grid=grid, physical=phys)

    if anchor == "poiseuille":
        # walls of the half-way bounce-back channel sit at -0.5 and Ny-0.5,
        # so the physical half-height is exactly (grid/2)*dx.
        dx = 2.0 * phys["h"] / grid
        dt = U_LB * dx / phys["u_ref"]
        bridge = LatticeBridge(dx=dx, dt=dt, rho0=phys["rho0"])
        nu_lb = bridge.viscosity_to_lattice(phys["nu"])
        tau = bridge.tau_from_viscosity(nu_lb)
        _check_regime(case, bridge, nu_lb, tau)
        h_lb = grid / 2.0
        # parabola over the full height 2h: u_max = G h^2 / (2 nu)
        force_lb = 2.0 * nu_lb * U_LB / h_lb**2
        sim = channel_poiseuille(grid, 1, tau, force_lb, tol=1e-6,
                                 check_every=100)
        case.lattice = dict(
            nu_lb=nu_lb, tau=tau, h_lb=h_lb, u_lb=U_LB, force_lb=force_lb,
            u_max=sim["u_max"], u_bulk=sim["u_bulk"],
            wall_stress=sim["wall_stress"], steps=sim["steps"],
            converged=float(sim["converged"]),
        )
        case.bridge = bridge
        return case

    if anchor == "couette":
        dx = 2.0 * phys["h"] / grid
        dt = U_LB * dx / phys["u_ref"]
        bridge = LatticeBridge(dx=dx, dt=dt, rho0=phys["rho0"])
        nu_lb = bridge.viscosity_to_lattice(phys["nu"])
        tau = bridge.tau_from_viscosity(nu_lb)
        _check_regime(case, bridge, nu_lb, tau)
        h_lb = grid / 2.0
        sim = channel_couette(grid, 1, tau, U_LB, tol=1e-6, check_every=100)
        case.lattice = dict(
            nu_lb=nu_lb, tau=tau, h_lb=h_lb, u_lb=U_LB, slope=sim["slope"],
            linearity_error=sim["linearity_error"], steps=sim["steps"],
            converged=float(sim["converged"]),
        )
        case.bridge = bridge
        return case

    if anchor == "stokes":
        dx = 2.0 * phys["a"] / grid          # grid cells per diameter
        dt = U_LB * dx / phys["u_ref"]
        bridge = LatticeBridge(dx=dx, dt=dt, rho0=phys["rho0"])
        nu_lb = bridge.viscosity_to_lattice(phys["nu"])
        tau = bridge.tau_from_viscosity(nu_lb)
        # No time marching here, so the tau window does not apply; Stokes flow
        # (Re_a << 1) with Ma < 0.1 inherently yields large lattice viscosity,
        # a well-known LBM limitation.  Only Ma is checked.
        if bridge.mach(U_LB) >= 0.1:
            raise ValueError(f"{case.name}: Ma >= 0.1")
        a_lb = phys["a"] / dx
        force_lb = 6.0 * math.pi * nu_lb * a_lb * U_LB
        case.lattice = dict(nu_lb=nu_lb, tau=tau, a_lb=a_lb, u_lb=U_LB,
                            force_lb=force_lb)
        case.bridge = bridge
        return case

    # Taylor-Green: h is the box size, u_ref the initial amplitude
    dx = phys["h"] / grid
    dt = U_LB * dx / phys["u_ref"]
    bridge = LatticeBridge(dx=dx, dt=dt, rho0=phys["rho0"])
    nu_lb = bridge.viscosity_to_lattice(phys["nu"])
    tau = bridge.tau_from_viscosity(nu_lb)
    _check_regime(case, bridge, nu_lb, tau)
    sim = taylor_green(grid, tau, U_LB, steps=300, fit_from=60)
    case.lattice = dict(
        nu_lb=nu_lb, tau=tau, k_lb=sim["k"], rate_lb=sim["rate"],
        u_lb=U_LB, steps=sim["steps"],
    )
    case.bridge = bridge
    return case


def _check_regime(case: AnchorCase, bridge: LatticeBridge, nu_lb: float,
                  tau: float) -> None:
    ma = bridge.mach(U_LB)
    if ma >= 0.1:
        raise ValueError(f"{case.name}: Ma={ma:.3f} >= 0.1")
    if not (0.51 <= tau <= 2.0):
        raise ValueError(f"{case.name}: tau={tau:.3f} outside [0.51, 2.0]")


def evaluate(case: AnchorCase, bridge: LatticeBridge | None = None) -> dict:
    """Dimensionalise the lattice measurement and compare with the law.

    ``bridge`` defaults to the pristine bridge used to build the case; the
    mutation battery passes a mutated bridge here (output-side defect only).
    """
    b = case.bridge if bridge is None else bridge
    phys = case.physical
    lat = case.lattice
    checks: dict[str, float] = {}
    dimless: dict[str, float] = {}

    if case.anchor == "poiseuille":
        u_max_phys = b.velocity_to_phys(lat["u_max"])
        checks["u_max"] = u_max_phys / phys["u_ref"] - 1.0
        u_bulk = b.velocity_to_phys(lat["u_bulk"])
        u_tau = b.velocity_to_phys(math.sqrt(lat["wall_stress"]))
        h_phys = b.length_to_phys(lat["h_lb"])
        nu_phys = b.viscosity_to_phys(lat["nu_lb"])
        re_tau = u_tau * h_phys / nu_phys
        u_b_plus = u_bulk / u_tau
        checks["U_b_plus"] = u_b_plus / (re_tau / 3.0) - 1.0
        dimless["U_b_plus_over_Re_tau_over_3"] = u_b_plus / (re_tau / 3.0)
        dimless["u_max_over_reference"] = u_max_phys / phys["u_ref"]
        dimless["Re_tau"] = re_tau

    elif case.anchor == "couette":
        tau_lb = lat["nu_lb"] * lat["slope"]
        tau_phys = b.stress_to_phys(tau_lb)
        # Couette: velocity rises from 0 to U over the FULL height 2h
        tau_ref = phys["rho0"] * phys["nu"] * phys["u_ref"] / (2.0 * phys["h"])
        checks["tau_wall"] = tau_phys / tau_ref - 1.0
        checks["linearity"] = lat["linearity_error"]
        u_wall = b.velocity_to_phys(lat["u_lb"])
        checks["u_wall"] = u_wall / phys["u_ref"] - 1.0
        dimless["tau_wall_over_reference"] = tau_phys / tau_ref
        dimless["slope_over_reference"] = lat["slope"] * 2.0 * lat["h_lb"] / lat["u_lb"]

    elif case.anchor == "stokes":
        force_lb = 6.0 * math.pi * lat["nu_lb"] * lat["a_lb"] * lat["u_lb"]
        force_phys = b.force_to_phys(force_lb)
        mu = phys["rho0"] * phys["nu"]
        force_ref = 6.0 * math.pi * mu * phys["a"] * phys["u_ref"]
        checks["drag"] = force_phys / force_ref - 1.0
        dimless["drag_over_reference"] = force_phys / force_ref

    else:  # taylor_green
        rate_phys = b.rate_to_phys(lat["rate_lb"])
        k_phys = b.wave_number_to_phys(lat["k_lb"])
        lam_ref = 2.0 * phys["nu"] * k_phys**2
        checks["decay_rate"] = rate_phys / lam_ref - 1.0
        nu_implied = rate_phys / (2.0 * k_phys**2)
        nu_bridge = b.viscosity_to_phys(lat["nu_lb"])
        checks["viscosity_closure"] = nu_implied / nu_bridge - 1.0
        dimless["nu_implied_over_nu"] = nu_implied / phys["nu"]

    worst = max(abs(v) for v in checks.values())
    return {
        "anchor": case.anchor,
        "grid": case.grid,
        "checks": checks,
        "dimensionless": dimless,
        "worst_rel_error": worst,
        "tolerance": TOLERANCES[case.anchor],
        "passed": bool(worst <= TOLERANCES[case.anchor]),
    }