"""Marangoni (interfacial-shear) driven channel flow against the exact Stokes solution.

Physics
-------
A flat interface carries a constant surface-tension gradient ``dsigma/dx``
(e.g. a linear surfactant or temperature gradient along a tank).  The tangential
stress balance at the interface is ``mu du/dn = dsigma/dx``; with no-slip walls at
``y = y_wall,low/high`` and negligible inertia (Stokes) the steady state is a
piecewise-linear profile whose interface value is

    u_i = (dsigma/dx) h1 h2 / ((h1 + h2) mu)                             (1)

with ``h1 = y0 - y_wall,low`` and ``h2 = y_wall,high - y0`` (Landau & Lifshitz
*Fluid Mechanics* §61-62; the two-layer Couette problem with an interfacial
shear).  The whole velocity field is

    u_x(y) = u_i (y - y_wall,low) / h1     (lower layer)
    u_x(y) = u_i (y_wall,high - y) / h2    (upper layer)                 (2)

so *every* row, not only the interface, is a prediction.  The verification
metric is the relative L2 error of the measured profile against (2), which is a
much stricter statement than the interface-velocity point value used by the
earlier static CSF report, and it cannot be satisfied by a force of the wrong
shape (a force spread over the wrong band would bend the profile).

The Marangoni force of the solver is ``f = (I - n n^T) grad(sigma) |grad phi|``
whose normal integral is exactly ``dsigma/dx`` because
``int |grad phi| dn = 1`` for a ``tanh`` indicator -- that identity is what the
profile test checks in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..free_surface_tension.csf import gradient
from ..free_surface_tension.solver import CapillarySolver, SolverConfig

__all__ = [
    "MarangoniConfig",
    "analytic_couette_profile",
    "run_marangoni_channel",
]


@dataclass
class MarangoniConfig:
    """One Marangoni channel case (constant ``dsigma/dx`` along a flat interface)."""

    dsdx: float = 1.0e-5
    Ly: int = 64
    Lx: int = 4
    y0: float | None = None
    tau: float = 0.9
    width: float = 3.0
    steps: int = 6000
    sample: int = 100
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def mu(self) -> float:
        """Dynamic viscosity of the LBGK collision, ``mu = (tau - 1/2)/3``."""
        return (self.tau - 0.5) / 3.0

    def heights(self) -> tuple[float, float]:
        y0 = 0.5 * self.Ly if self.y0 is None else self.y0
        h1 = y0 - 0.5
        h2 = (self.Ly - 0.5) - y0
        return h1, h2

    def u_interface(self) -> float:
        h1, h2 = self.heights()
        return self.dsdx * h1 * h2 / ((h1 + h2) * self.mu)

    def validate(self) -> None:
        if self.Lx < 2:
            raise ValueError("Lx must be at least 2 (periodic in x)")
        h1, h2 = self.heights()
        if h1 < 3 or h2 < 3:
            raise ValueError(f"layer heights too small: h1={h1}, h2={h2}")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")


def analytic_couette_profile(config: MarangoniConfig) -> list[float]:
    """``u_x(y)`` of (2) at the cell centres ``y = j + 1/2`` (row 0 at the wall)."""
    y0 = 0.5 * config.Ly if config.y0 is None else config.y0
    h1, h2 = config.heights()
    ui = config.u_interface()
    prof: list[float] = []
    for j in range(config.Ly):
        y = j + 0.5
        if j in (0, config.Ly - 1):
            prof.append(0.0)  # bounce-back wall rows: solid, u = 0
        elif y <= y0:
            prof.append(ui * (y - 0.5) / h1)
        else:
            prof.append(ui * ((config.Ly - 0.5) - y) / h2)
    return prof


def _rel_l2(num: list[float], ref: list[float], scale: float) -> float:
    s = math.sqrt(sum((a - b) ** 2 for a, b in zip(num, ref)) / len(ref))
    return s / scale if scale else float("nan")


def run_marangoni_channel(config: MarangoniConfig, *, device: str = "cpu") -> dict[str, Any]:
    """Run one Marangoni case; return the field comparison and the drift history."""
    config.validate()
    y0 = 0.5 * config.Ly if config.y0 is None else config.y0
    solver = CapillarySolver(
        (config.Ly, config.Lx),
        SolverConfig(
            sigma=0.0,  # flat interface: no capillary force, only Marangoni
            tau_liquid=config.tau,
            tau_gas=config.tau,
            width=config.width,
            init_laplace=False,
        ),
    )
    x = torch.arange(config.Lx, dtype=torch.float64).view(1, -1)
    y = torch.arange(config.Ly, dtype=torch.float64).view(-1, 1)
    phi = 0.5 + 0.5 * torch.tanh(2.0 * (y0 - y) / config.width)
    solver.set_phi(phi, init_laplace=False)
    solver.add_wall(axis=1, side="both")

    grad_sigma = (config.dsdx, 0.0)
    hist: list[tuple[int, float, float]] = []
    done = 0
    while done < config.steps:
        n = min(config.sample, config.steps - done)
        solver.step(n, advect=False, grad_sigma=grad_sigma)
        done += n
        ux = solver.u[0].mean(dim=1)
        hist.append((done, float(ux[config.Ly // 2]), solver.max_speed()))

    prof = [float(v) for v in solver.u[0].mean(dim=1)]
    ref = analytic_couette_profile(config)
    ui = config.u_interface()

    # mask: drop the two bounce-back rows next to each wall (half-way wall
    # closure is first order there, and the analytic no-slip value is exact)
    inner = list(range(2, config.Ly - 2))
    band = [j for j in range(config.Ly) if abs((j + 0.5) - y0) <= 2.5]
    rows = {
        "all_inner": inner,
        "excl_interface_band": [j for j in inner if j not in band],
        "interface_band": [j for j in inner if j in band],
    }
    errs = {
        name: _rel_l2([prof[j] for j in idx], [ref[j] for j in idx], abs(ui))
        for name, idx in rows.items()
    }
    # interface velocity by parabolic interpolation across the band (the raw
    # cell value sits half a cell off the interface and is not the peak)
    j0 = int(math.floor(y0 - 0.5))
    trip = prof[max(j0 - 1, 0) : j0 + 2]
    if len(trip) == 3:
        denom = trip[0] - 2 * trip[1] + trip[2]
        offset = 0.5 * (trip[0] - trip[2]) / denom if abs(denom) > 1e-30 else 0.0
        ui_meas = trip[1] - 0.25 * (trip[0] - trip[2]) * offset
    else:  # pragma: no cover - defensive
        ui_meas = prof[config.Ly // 2]

    # interface velocity from a linear fit of the *interior* of the lower layer:
    # the V vertex itself is rounded off by the diffuse interface band, so the
    # point value (parabolic interpolation above) is biased low; the interior
    # rows carry the exact Stokes slope.
    fit_lo = max(int(0.15 * y0), 2)
    fit_hi = int(y0) - 3
    u_fit = float("nan")
    slope_fit = float("nan")
    if fit_hi - fit_lo >= 3:
        ys = torch.arange(fit_lo, fit_hi + 1, dtype=torch.float64) + 0.5
        us = torch.tensor(prof[fit_lo : fit_hi + 1], dtype=torch.float64)
        a_mat = torch.stack([ys, torch.ones_like(ys)], dim=1)
        sol = torch.linalg.lstsq(a_mat, us.unsqueeze(1)).solution
        slope_fit = float(sol[0])
        u_fit = float(sol[0] * y0 + sol[1])

    sharp = float(torch.sqrt(sum(g * g for g in gradient(solver.phi)) + 1e-30).max())
    return {
        "case": "marangoni_channel",
        "label": config.label,
        "params": {
            "dsdx": config.dsdx,
            "Ly": config.Ly,
            "Lx": config.Lx,
            "y0": y0,
            "tau": config.tau,
            "mu": config.mu,
            "width": config.width,
            "steps": config.steps,
            "h1": config.heights()[0],
            "h2": config.heights()[1],
            "u_interface_theory": ui,
            "Re_marangoni": abs(ui) * config.heights()[0] / (config.mu / 1.0),
        },
        "metrics": {
            "u_interface": ui_meas,
            "u_interface_rel_err": ui_meas / ui - 1.0 if ui else float("nan"),
            "u_interface_fit": u_fit,
            "u_interface_fit_rel_err": u_fit / ui - 1.0 if ui else float("nan"),
            "slope_lower": slope_fit,
            "slope_lower_theory": ui / config.heights()[0] if ui else float("nan"),
            "profile_rel_l2": errs["all_inner"],
            "profile_rel_l2_excl_band": errs["excl_interface_band"],
            "profile_rel_l2_band": errs["interface_band"],
            "umax": solver.max_speed(),
            "u_drift_rel": abs(hist[-1][1] - hist[-2][1]) / max(abs(hist[-1][1]), 1e-30)
            if len(hist) > 1
            else float("nan"),
            "sharpness": sharp,
        },
        "series": {
            "y": [j + 0.5 for j in range(config.Ly)],
            "ux": prof,
            "ux_theory": ref,
            "t": [h[0] for h in hist],
            "u_interface": [h[1] for h in hist],
            "umax": [h[2] for h in hist],
        },
    }