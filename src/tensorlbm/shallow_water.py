"""Shallow-water dam break and the Ritter (1892) self-similar solution.

Physical problem
----------------
A reservoir of still water of depth ``h0`` (``x < 0``) is released at ``t = 0``
onto a dry, frictionless bed.  The depth-averaged shallow-water equations (SWE)

.. math::

    \\partial_t h + \\partial_x(hu) = 0, \\qquad
    \\partial_t(hu) + \\partial_x\\!\\left(hu^2 + \\tfrac12 g h^2\\right) = 0

admit the classical self-similar solution of Ritter (1892).  With
:math:`c_0=\\sqrt{g h_0}` and :math:`\\xi=x/t`,

.. math::

    h(x,t) = \\frac{1}{9g}\\left(2c_0 - \\xi\\right)^2, \\qquad
    u(x,t) = \\tfrac{2}{3}\\left(c_0 + \\xi\\right),
    \\qquad 0 \\le x \\le 2c_0 t,

``h = h_0`` for :math:`x \\le -c_0 t` and ``h = 0`` beyond the front
:math:`x_{front}(t) = 2 c_0 t`; the front advances at constant speed
:math:`2\\sqrt{g h_0}`.  This is the analytic anchor used here.

Numerical scheme
----------------
The SWE are integrated with a **finite-volume HLL scheme with hydrostatic
reconstruction** (Audusse et al. 2004) on a Cartesian 1-D grid, including a
dry-bed (wetting/drying) treatment.  The scheme is exactly mass conservative in
flux form and well balanced for the lake-at-rest state, which are the two
properties the benchmark gates rely on.

A depth-averaged *lattice-BGK* variant (D1Q3 with a non-ideal pressure force)
was prototyped first and rejected: the hydrostatic pressure :math:`\\tfrac12
gh^2` is many orders of magnitude smaller than the lattice pressure
:math:`c_s^2h` in the low-celerity regime required by the lattice Boltzmann
stability limit, and the required force cancels a term ~:math:`(c_s^2/gh)` times
larger than itself, which made the wet/dry front unstable (see
``docs/reports/`` note in the benchmark README).  The finite-volume solver is
used as the *reference shallow-water solver*; the multiphase LBM dam break in
``tensorlbm.dam_break`` remains the model for the vertical-plane problem.

References
----------
Ritter (1892) Z. Ver. Deutsch. Ing. 36 947; Martin & Moyce (1952) Phil. Trans.
R. Soc. A 244 312; Audusse et al. (2004) SIAM J. Sci. Comput. 25 2050;
Toro (2001) *Shock-Capturing Methods for Free-Surface Flows*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "ShallowWaterConfig",
    "ritter_depth",
    "ritter_velocity",
    "ritter_front",
    "init_dam_break",
    "swe_step",
    "front_position",
    "run_ritter_dam_break",
    "front_from_mass",
    "RitterRunResult",
]


@dataclass(frozen=True)
class ShallowWaterConfig:
    """Configuration of a 1-D Ritter dam-break run.

    Lengths are measured in units of the initial depth ``h0`` and velocities in
    ``sqrt(g h0)``; with ``g = 1`` the non-dimensional front law is simply
    ``x_front = 2 t``.
    """

    nx: int = 800              # number of cells
    x_min: float = -4.0        # left (upstream) end of the tank [h0]
    x_max: float = 12.0        # right (downstream) end [h0]
    dam_x: float = 0.0         # dam position [h0]
    h0: float = 1.0            # initial depth
    g: float = 1.0             # gravity
    cfl: float = 0.4
    t_end: float = 5.0         # physical end time (in units of sqrt(h0/g))
    sample_every: int = 50     # steps between front samples
    h_dry: float = 1.0e-4      # dry threshold (fraction of h0)
    front_thresh_frac: float = 1.0e-3   # front = outermost cell with h > frac*h0
    dtype: str = "float64"

    @property
    def c0(self) -> float:
        return math.sqrt(self.g * self.h0)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def t_end_star(self) -> float:
        return self.t_end * math.sqrt(self.g / self.h0)

    def validate(self) -> None:
        if self.nx < 32:
            raise ValueError("nx must be >= 32")
        if not self.x_min < self.dam_x < self.x_max:
            raise ValueError("dam_x must lie inside the tank")
        if self.g <= 0.0 or self.h0 <= 0.0:
            raise ValueError("g and h0 must be positive")
        if not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must be in (0, 1]")


# ---------------------------------------------------------------------------
# Analytical anchors
# ---------------------------------------------------------------------------


def ritter_front(t: float, h0: float, g: float) -> float:
    """Front position of the Ritter solution ``x_f = 2 sqrt(g h0) t``."""
    return 2.0 * math.sqrt(g * h0) * t


def ritter_depth(x: float, t: float, h0: float, g: float) -> float:
    """Water depth of the Ritter solution at ``(x, t)`` (0 beyond the front)."""
    if t <= 0.0:
        return h0 if x <= 0.0 else 0.0
    c0 = math.sqrt(g * h0)
    xi = x / t
    if xi <= -c0:
        return h0
    if xi >= 2.0 * c0:
        return 0.0
    return (2.0 * c0 - xi) ** 2 / (9.0 * g)


def ritter_velocity(x: float, t: float, h0: float, g: float) -> float:
    """Depth-averaged velocity of the Ritter solution."""
    if t <= 0.0:
        return 0.0
    c0 = math.sqrt(g * h0)
    xi = x / t
    if xi <= -c0 or xi >= 2.0 * c0:
        return 0.0
    return (2.0 / 3.0) * (c0 + xi)


# ---------------------------------------------------------------------------
# Finite-volume HLL solver
# ---------------------------------------------------------------------------


def init_dam_break(cfg: ShallowWaterConfig) -> torch.Tensor:
    """Return the conservative state ``U = (h, hu)`` of shape ``(2, nx)``."""
    dtype = getattr(torch, cfg.dtype)
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=dtype)
    h = torch.where(x < cfg.dam_x, torch.full_like(x, cfg.h0), torch.zeros_like(x))
    hu = torch.zeros_like(x)
    return torch.stack([h, hu])


def _phys_flux(h: torch.Tensor, hu: torch.Tensor, g: float) -> tuple[torch.Tensor, torch.Tensor]:
    u = hu / torch.clamp(h, min=1.0e-12)
    return hu, hu * u + 0.5 * g * h * h


def swe_step(U: torch.Tensor, cfg: ShallowWaterConfig, dt: float) -> torch.Tensor:
    """One explicit finite-volume step with HLL fluxes and ghost cells.

    Ghost-cell boundaries: the left end is a mirrored, impermeable wall
    (``h_g = h_0``, ``hu_g = -hu_0``) so that the lake-at-rest state is preserved
    exactly (the wall face carries only the hydrostatic pressure flux); the right
    end is a zero-gradient outflow.  The scheme is in strict flux form, hence
    mass conservative to round-off.  Velocities are desingularised
    (``u = hu/max(h, eps)``, ``u = 0`` in dry cells) and bounded by the physical
    dry-bed limit ``|u| <= 2.2 sqrt(g h0)``.
    """
    g = cfg.g
    h = U[0]
    hu = U[1]
    h_eps = max(cfg.h_dry, 1.0e-12)
    # no velocity clipping: it would break the consistency between the mass and
    # momentum fluxes (the state itself is desingularised instead)

    # --- ghost cells
    h_ext = torch.cat([h[:1], h, h[-1:]])
    hu_ext = torch.cat([-hu[:1], hu, hu[-1:]])

    def _state(hh: torch.Tensor, mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hh_pos = torch.clamp(hh, min=0.0)
        uu = mm / torch.clamp(hh_pos, min=h_eps)
        uu = torch.where(hh_pos < cfg.h_dry, torch.zeros_like(uu), uu)
        return hh_pos, uu, torch.sqrt(g * hh_pos)

    h_l, u_l, c_l = _state(h_ext[:-1], hu_ext[:-1])
    h_r, u_r, c_r = _state(h_ext[1:], hu_ext[1:])

    fl_h_l, fl_hu_l = _phys_flux(h_l, h_l * u_l, g)
    fl_h_r, fl_hu_r = _phys_flux(h_r, h_r * u_r, g)

    # Global (domain-wide) Rusanov wave speeds: positivity preserving and
    # free of the near-dry velocity blow-up that a local estimate suffers from.
    wave_max = float((u_l.abs() + c_l).max().item())
    wave_max = max(wave_max, 2.0 * cfg.c0)
    s_l = torch.full_like(u_l, -wave_max)
    s_r = torch.full_like(u_l, wave_max)
    denom = torch.where(s_r - s_l > 1.0e-10, s_r - s_l, torch.full_like(s_l, 1.0e-10))
    flux_h = (s_r * fl_h_l - s_l * fl_h_r + s_l * s_r * (h_r - h_l)) / denom
    flux_hu = (s_r * fl_hu_l - s_l * fl_hu_r
               + s_l * s_r * (h_r * u_r - h_l * u_l)) / denom
    both_dry = (h_l <= cfg.h_dry) & (h_r <= cfg.h_dry)
    flux_h = torch.where(both_dry, torch.zeros_like(flux_h), flux_h)
    flux_hu = torch.where(both_dry, torch.zeros_like(flux_hu), flux_hu)

    dU = torch.zeros_like(U)
    dU[0] = -(flux_h[1:] - flux_h[:-1]) / cfg.dx
    dU[1] = -(flux_hu[1:] - flux_hu[:-1]) / cfg.dx

    h_new = torch.clamp(U[0] + dt * dU[0], min=0.0)
    hu_new = U[1] + dt * dU[1]
    hu_new = torch.where(h_new < cfg.h_dry, torch.zeros_like(hu_new), hu_new)
    return torch.stack([h_new, hu_new])


def front_position(h: torch.Tensor, cfg: ShallowWaterConfig) -> float:
    """Outermost wet cell with sub-cell interpolation, in physical units."""
    thresh = cfg.front_thresh_frac * cfg.h0
    wet = torch.nonzero(h > thresh, as_tuple=False)
    if wet.numel() == 0:
        return cfg.x_min
    i = int(wet[-1])
    x_i = cfg.x_min + i * cfg.dx
    if i + 1 >= h.numel():
        return x_i
    h_i = float(h[i])
    h_n = float(h[i + 1])
    if h_i <= h_n:
        return x_i
    return x_i + cfg.dx * (h_i - thresh) / (h_i - h_n)


def front_from_mass(volume_downstream: float, h0: float) -> float:
    """Front position recovered from the *integral* form of the Ritter law.

    Integrating the self-similar depth profile gives the discharged volume

    .. math:: V_{down}(t) = \int_0^{2c_0t}\! h\,dx = \frac{8c_0^3 t}{27g},

    hence, eliminating ``t`` with ``x_f = 2c_0 t``,

    .. math:: x_f = \frac{27 V_{down}}{4 h_0} .

    ``V_down`` is a *conserved* quantity of the numerical solution (total water
    volume released downstream of the dam), so this estimator is insensitive to
    the numerical smearing of the thin front tip that limits a direct
    threshold-based front detection.
    """
    return 27.0 * volume_downstream / (4.0 * h0)


@dataclass
class RitterRunResult:
    """Container for a dam-break run."""

    time: list[float]
    front: list[float]
    front_analytic: list[float]
    mass_initial: float
    mass_final: float
    profiles: list[tuple[float, list[float]]]
    steps: int
    dt: float
    volume_downstream: list[float]


def run_ritter_dam_break(cfg: ShallowWaterConfig) -> RitterRunResult:
    """Integrate the dam break up to ``cfg.t_end`` and collect diagnostics."""
    cfg.validate()
    U = init_dam_break(cfg)
    mass0 = float(U[0].sum().item() * cfg.dx)
    dt = cfg.cfl * cfg.dx / (2.0 * cfg.c0)
    n_steps = int(round(cfg.t_end / dt))
    times: list[float] = []
    fronts: list[float] = []
    refs: list[float] = []
    profiles: list[tuple[float, list[float]]] = []
    volumes: list[float] = []
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=getattr(torch, cfg.dtype))
    down = (x > cfg.dam_x).to(getattr(torch, cfg.dtype))
    for step in range(1, n_steps + 1):
        U = swe_step(U, cfg, dt)
        if step % cfg.sample_every == 0 or step == n_steps:
            t = step * dt
            times.append(t)
            fronts.append(front_position(U[0], cfg))
            refs.append(ritter_front(t, cfg.h0, cfg.g))
            profiles.append((t, U[0].tolist()))
            volumes.append(float((U[0] * down).sum().item() * cfg.dx))
    return RitterRunResult(
        time=times,
        front=fronts,
        front_analytic=refs,
        mass_initial=mass0,
        mass_final=float(U[0].sum().item() * cfg.dx),
        profiles=profiles,
        steps=n_steps,
        dt=dt,
        volume_downstream=volumes,
    )