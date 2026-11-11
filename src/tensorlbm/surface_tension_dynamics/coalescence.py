"""Coalescence of two equal 2D drops: neck growth law and capillary time scales.

Physics
-------
Two equal drops (radius ``R``, viscosity ``mu``, density ``rho``) are brought into
contact.  Surface tension then drives a liquid bridge whose neck radius ``r_b(t)``
grows.  Dimensional analysis of the *inertially limited* problem (no length scale
other than ``r_b`` and ``R``) leaves one dimensionless group, the Ohnesorge
number ``Oh = mu / sqrt(rho sigma R)``:

    r_b(t) = (sigma/rho)^{1/3} t^{2/3} f(...)        inertial regime      (1a)
    r_b(t) ~ (sigma/mu) t                            viscous regime       (1b)

(Eggers, Lister & Stone, *J. Fluid Mech.* **401** (1999) 293 for the inertial
self-similar neck; the linear viscous growth goes back to Hopper 1990 and was
measured for drops by Aarts et al., *Phys. Rev. Lett.* **95** (2005) 164503).
The capillary-inertial time of the *merged* drop is

    tau_ci = sqrt(rho R^3 / sigma)                                        (2)

and the experiment-level statement we test is the dimensionless coalescence time
``tau* = t(r_b = R/2) / tau_ci``: a single O(1) number that the scaling laws above
fix up to an O(1) prefactor.

Setup and honest limitations
----------------------------
The verification substrate (:class:`tensorlbm.free_surface_tension.solver.
CapillarySolver`) carries a **single density**: both "phases" are the same fluid,
so the mass that must be accelerated is the liquid *and* the surrounding medium.
This is stated in the report: the measured ``tau*`` therefore corresponds to a
density ratio of 1, not to a drop in air, and the literature coefficient is only
comparable through the equal-density modification of (1) -- which is why the
report treats (1a)'s *exponent* as the primary falsifiable prediction and the
prefactor as a lower-confidence comparison.

Both drops are placed on the centre line, just touching, in a periodic box
``8R x 4R``; the neck is measured on the symmetry plane as the half-height of the
``phi = 0.5`` crossing interval by linear interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..free_surface_tension.solver import CapillarySolver, SolverConfig

__all__ = [
    "CoalescenceConfig",
    "neck_radius",
    "run_coalescence",
    "two_drop_phi",
]


@dataclass
class CoalescenceConfig:
    """Geometry, fluid properties and measurement settings of one coalescence run."""

    radius: float = 10.0
    nx: int | None = None
    ny: int | None = None
    sigma: float = 0.05
    tau_liquid: float = 0.6
    tau_gas: float = 0.6
    width: float = 2.0
    steps: int = 6000
    sample: int = 10
    advection: str = "sl"
    gap: float = 0.0
    label: str = ""
    stop_at_neck_frac: float = 0.95
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nx is None:
            self.nx = int(8 * self.radius)
        if self.ny is None:
            self.ny = int(4 * self.radius)

    @property
    def mu(self) -> float:
        return (self.tau_liquid - 0.5) / 3.0

    @property
    def tau_ci(self) -> float:
        """Capillary-inertial time ``sqrt(rho R^3 / sigma)`` (rho = 1 in lattice units)."""
        return math.sqrt(self.radius**3 / self.sigma)

    @property
    def Oh(self) -> float:
        """Ohnesorge number ``mu / sqrt(rho sigma R)``."""
        return self.mu / math.sqrt(self.sigma * self.radius)

    def validate(self) -> None:
        if self.radius < 4:
            raise ValueError("radius must be at least 4 lattice cells")
        if self.nx is None or self.ny is None:  # pragma: no cover - post_init fills
            raise ValueError("nx/ny not initialised")
        if self.nx < 4 * self.radius or self.ny < 4 * self.radius:
            raise ValueError("box too small: need nx >= 4R and ny >= 4R")
        if self.advection not in ("sl", "upwind"):
            raise ValueError(f"unknown advection {self.advection!r}")


def two_drop_phi(config: CoalescenceConfig, *, device: str = "cpu") -> torch.Tensor:
    """Union of two ``tanh`` disks touching at the box centre (``x = nx/2``)."""
    config.validate()
    assert config.nx is not None and config.ny is not None
    x = torch.arange(config.nx, dtype=torch.float64, device=device).view(1, -1) + 0.5
    y = torch.arange(config.ny, dtype=torch.float64, device=device).view(-1, 1) + 0.5
    y0 = 0.5 * config.ny
    half = config.radius + 0.5 * config.gap
    c1 = (0.5 * config.nx - half, y0)
    c2 = (0.5 * config.nx + half, y0)
    d1 = torch.sqrt((x - c1[0]) ** 2 + (y - c1[1]) ** 2)
    d2 = torch.sqrt((x - c2[0]) ** 2 + (y - c2[1]) ** 2)
    d = torch.minimum(d1, d2)
    return 0.5 + 0.5 * torch.tanh(2.0 * (config.radius - d) / config.width)


def neck_radius(phi: torch.Tensor, *, level: float = 0.5) -> tuple[float, int]:
    """Half-height of the ``phi = level`` interval on the symmetry plane.

    The symmetry plane ``x = nx/2`` sits between cells ``nx/2 - 1`` and ``nx/2``
    for an even ``nx``; the indicator is averaged over those two columns, then
    the contiguous run around the centre row where ``phi > level`` is located and
    its two boundaries are interpolated linearly.  Returns ``(r_b, n_cells)``;
    ``r_b = 0`` when the drops are not yet connected.
    """
    if phi.dim() != 2:
        raise ValueError(f"phi must be 2-D, got {tuple(phi.shape)}")
    ny, nx = phi.shape
    col = 0.5 * (phi[:, nx // 2 - 1] + phi[:, nx // 2])
    above = col > level
    j0 = ny // 2
    if not bool(above[j0]):
        return 0.0, 0
    j_lo = j0
    while j_lo - 1 >= 0 and bool(above[j_lo - 1]):
        j_lo -= 1
    j_hi = j0
    while j_hi + 1 < ny and bool(above[j_hi + 1]):
        j_hi += 1
    lo_val = float(col[j_lo - 1]) if j_lo - 1 >= 0 else 0.0
    hi_val = float(col[j_hi + 1]) if j_hi + 1 < ny else 0.0
    frac_lo = (level - lo_val) / (float(col[j_lo]) - lo_val) if float(col[j_lo]) != lo_val else 0.0
    frac_hi = (level - hi_val) / (float(col[j_hi]) - hi_val) if float(col[j_hi]) != hi_val else 0.0
    y_lo = j_lo - frac_lo
    y_hi = j_hi + frac_hi
    return 0.5 * (y_hi - y_lo), j_hi - j_lo + 1


def _power_law(t: list[float], r: list[float], lo: float, hi: float) -> dict[str, float]:
    """Log-log least-squares fit ``r = C t^n`` on the window ``lo <= r <= hi``."""
    pts = [(math.log(tt), math.log(rr)) for tt, rr in zip(t, r) if lo <= rr <= hi and rr > 0.0]
    if len(pts) < 3:
        return {"exponent": float("nan"), "prefactor": float("nan"), "n_points": len(pts)}
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den if den else float("nan")
    intercept = (sy - slope * sx) / n
    # R^2 of the log-log fit
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in pts)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts)
    return {
        "exponent": slope,
        "prefactor": math.exp(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "n_points": n,
    }


def run_coalescence(config: CoalescenceConfig, *, device: str = "cpu") -> dict[str, Any]:
    """Run one coalescence case and return the neck-growth law and time scales."""
    config.validate()
    assert config.nx is not None and config.ny is not None
    solver = CapillarySolver(
        (config.ny, config.nx),
        SolverConfig(
            sigma=config.sigma,
            tau_liquid=config.tau_liquid,
            tau_gas=config.tau_gas,
            width=config.width,
            init_laplace=False,
            advection=config.advection,
        ),
    )
    phi0 = two_drop_phi(config, device=device)
    solver.set_phi(phi0, init_laplace=False)

    ts: list[int] = []
    rb: list[float] = []
    umax: list[float] = []
    done = 0
    while done < config.steps:
        n = min(config.sample, config.steps - done)
        solver.step(n, advect=True)
        done += n
        r, _ = neck_radius(solver.phi)
        ts.append(done)
        rb.append(r)
        umax.append(solver.max_speed())
        if r >= config.stop_at_neck_frac * config.radius:
            break

    R = config.radius
    tau_ci = config.tau_ci
    # time to reach r_b = R/2 by linear interpolation of the sampled series
    def time_at(target: float) -> float:
        for i in range(1, len(rb)):
            if rb[i - 1] < target <= rb[i]:
                f = (target - rb[i - 1]) / (rb[i] - rb[i - 1])
                return ts[i - 1] + f * (ts[i] - ts[i - 1])
        return float("nan")

    t_half = time_at(0.5 * R)
    fit = _power_law(ts, rb, lo=0.05 * R, hi=0.6 * R)
    return {
        "case": "drop_coalescence",
        "label": config.label,
        "params": {
            "radius": R,
            "nx": config.nx,
            "ny": config.ny,
            "sigma": config.sigma,
            "mu": config.mu,
            "tau_liquid": config.tau_liquid,
            "tau_gas": config.tau_gas,
            "width": config.width,
            "steps_run": done,
            "sample": config.sample,
            "Oh": config.Oh,
            "tau_ci": tau_ci,
            "density_ratio": 1.0,
        },
        "metrics": {
            "neck_max": max(rb),
            "neck_final_over_R": rb[-1] / R,
            "t_half_R": t_half,
            "tau_star": t_half / tau_ci if tau_ci else float("nan"),
            "growth_exponent": fit["exponent"],
            "growth_prefactor": fit["prefactor"],
            "growth_r2": fit.get("r2", float("nan")),
            "umax": max(umax),
        },
        "series": {"t": ts, "r_b": rb, "r_b_over_R": [v / R for v in rb], "umax": umax},
    }