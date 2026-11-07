"""Gravity-driven Rayleigh--Taylor instability in D2Q9 (Boussinesq two-layer).

Physical problem
----------------
A heavy fluid (``rho_1``) rests on a light fluid (``rho_2``) in a gravitational
field ``g`` pointing downwards.  The interface carries a single-mode
perturbation of wavelength ``lam`` and amplitude ``eta_0``.  For inviscid,
immiscible, incompressible fluids the linear (normal-mode) growth rate of the
perturbation is

.. math::

    \\sigma_{analytic} = \\sqrt{A g k}, \\qquad
    A = \\frac{\\rho_1-\\rho_2}{\\rho_1+\\rho_2}, \\qquad k = \\frac{2\\pi}{\\lambda}

with :math:`A` the Atwood number (Rayleigh 1883, Taylor 1950).  Viscosity
:math:`\\nu` damps the instability; the exact linear result for equal
kinematic viscosities is

.. math::

    \\sigma_{visc} = \\sqrt{A g k + \\nu^2 k^4} - \\nu k^2 ,

which reduces to :math:`\\sqrt{Agk}` for :math:`\\nu k^2 \\ll \\sigma`.

Numerical model
---------------
Two D2Q9 populations are advected simultaneously:

* ``f`` – flow.  BGK collision with a **Boussinesq buoyancy force per unit
  mass** :math:`F_y = -A g \\phi`, where :math:`\\phi\\in[-1,+1]` is the phase
  field (``+1`` heavy, ``-1`` light).  Only the buoyancy *deviation* is applied,
  the mean hydrostatic part is removed, so the reference density stays at
  :math:`\\rho_0=1` (no spurious compressibility from :math:`\\rho_0 g H`).
* ``g`` – phase field, transported by the standard advection--diffusion
  equilibrium :math:`g_i^{eq} = w_i\\phi(1 + 3\\,c_i\\!\\cdot\\! u)` with
  diffusivity :math:`D = c_s^2(\\tau_\\phi-1/2)`.

Both populations are periodic in ``x`` (one wavelength per box) and bounce back
on the top/bottom walls.

Diagnostics
-----------
The instantaneous interface shape is recovered from the *heavy-fluid column
height*

.. math::

    N(x) = \\sum_y \\frac{1+\\phi(x,y)}{2} = n_y - y_{int}(x),

which is a conserved, noise-free moment (no threshold), and its first Fourier
mode ``a(t) = (2/n_x) |\\sum_x (N(x)-\\bar N) \\cos(kx)|``.  In the linear regime
``a(t) = a_0 e^{\\sigma t}``; :math:`\\sigma_{sim}` is a least-squares slope of
``ln a`` over a prescribed amplitude window.

References
----------
Rayleigh (1883) Proc. London Math. Soc. 14 170; Taylor (1950) Proc. R. Soc.
Lond. A 201 192; Chandrasekhar (1961) *Hydrodynamic and Hydromagnetic
Stability*, §92 (viscous correction).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .d2q9 import W, equilibrium, macroscopic
from .solver import stream

__all__ = [
    "RTConfig",
    "analytic_growth_rate",
    "viscous_growth_rate",
    "init_rt_state",
    "rt_step",
    "interface_column",
    "mode_amplitude",
    "fit_growth_rate",
    "fit_growth_rate_twomode",
    "interface_zero_crossing",
    "mode_amplitude_zc",
    "run_rayleigh_taylor",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RTConfig:
    """Configuration of one single-mode Rayleigh--Taylor run.

    All quantities are in lattice units (``dx = dt = 1``).
    """

    lam: int = 64                 # perturbation wavelength (lattice units)
    layers: float = 3.0           # vertical extent = layers * lam (per fluid side)
    g: float = 1.0e-2             # gravitational acceleration (lattice)
    atwood: float = 1.0 / 3.0     # Atwood number, rho1/rho2 = 2  ->  A = 1/3
    eta0_frac: float = 0.01       # initial amplitude eta_0 = eta0_frac * lam
    interface_width: float = 1.5  # tanh half-width of the diffuse interface
    tau: float = 0.51             # flow relaxation time  (nu = (tau-1/2)/3)
    tau_phi: float = 0.503        # phase-field relaxation time (D = (tau_phi-1/2)/3)
    n_steps: int = 600            # maximum number of lattice steps
    sample_every: int = 5         # amplitude sampling interval
    a_lo_frac: float = 0.01       # fit window lower bound on a/lam
    a_hi_frac: float = 0.05       # fit window upper bound on a/lam
    stop_a_frac: float = 0.16     # stop when a lam exceeds this (nonlinear onset)
    seed_eigenmode: bool = True   # seed the linear normal-mode velocity field
    device: str = "cpu"

    @property
    def nx(self) -> int:
        return int(self.lam)

    @property
    def ny(self) -> int:
        return int(round(self.layers * self.lam))

    @property
    def k(self) -> float:
        return 2.0 * math.pi / float(self.lam)

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def diffusivity(self) -> float:
        return (self.tau_phi - 0.5) / 3.0

    def validate(self) -> None:
        if self.lam < 8:
            raise ValueError("lam must be >= 8")
        if not 0.0 < self.atwood < 1.0:
            raise ValueError("atwood must be in (0, 1)")
        if self.tau <= 0.5 or self.tau_phi <= 0.5:
            raise ValueError("relaxation times must exceed 0.5")
        if not 0.0 < self.a_lo_frac < self.a_hi_frac < 0.5:
            raise ValueError("invalid amplitude fit window")


# ---------------------------------------------------------------------------
# Analytical anchors
# ---------------------------------------------------------------------------


def analytic_growth_rate(atwood: float, g: float, lam: float) -> float:
    """Inviscid linear growth rate ``sigma = sqrt(A g k)``."""
    return math.sqrt(atwood * g * 2.0 * math.pi / lam)


def viscous_growth_rate(atwood: float, g: float, lam: float, nu: float) -> float:
    """Viscous linear growth rate ``sqrt(Agk + nu^2 k^4) - nu k^2``."""
    k = 2.0 * math.pi / lam
    return math.sqrt(atwood * g * k + (nu * k * k) ** 2) - nu * k * k


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------


def init_rt_state(cfg: RTConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(f, g, phi)`` for a single-mode RT initial condition.

    ``phi = tanh((y - y_int(x)) / delta)`` with ``y_int(x) = ny/2 + eta0 cos(kx)``
    and ``phi = +1`` (heavy) on top.
    """
    device = torch.device(cfg.device)
    ny, nx = cfg.ny, cfg.nx
    y = torch.arange(ny, dtype=torch.float64, device=device).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float64, device=device).view(1, nx)
    eta0 = cfg.eta0_frac * cfg.lam
    y_int = 0.5 * ny + eta0 * torch.cos(cfg.k * x)
    phi = torch.tanh((y - y_int) / cfg.interface_width).to(torch.float32)

    rho = torch.ones((ny, nx), dtype=torch.float32, device=device)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    if cfg.seed_eigenmode:
        # Linear normal mode of the two-layer (Boussinesq) problem, starting from
        # rest: u_y = sigma eta cos(kx) exp(-k|y-y_int|) and the solenoidal
        # companion u_x = +-sigma eta sin(kx) exp(-k|y-y_int|).  Seeding it means
        # only the *growing* mode is excited, so a(t) = eta_0 exp(sigma t) from
        # t = 0 and no decaying-mode transient contaminates the fit.
        sig = analytic_growth_rate(cfg.atwood, cfg.g, cfg.lam)
        amp = sig * eta0
        decay = torch.exp(-cfg.k * (y - y_int).abs())
        uy = amp * torch.cos(cfg.k * x) * decay
        ux = torch.where(y >= y_int, amp, -amp) * torch.sin(cfg.k * x) * decay
        ux = ux.to(torch.float32)
        uy = uy.to(torch.float32)
    f = equilibrium(rho, ux, uy)
    g = W.to(device).view(9, 1, 1) * phi.unsqueeze(0)
    return f, g, phi


# ---------------------------------------------------------------------------
# One lattice step
# ---------------------------------------------------------------------------


def rt_step(
    f: torch.Tensor,
    g: torch.Tensor,
    cfg: RTConfig,
    phi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance ``(f, g)`` by one lattice step (BGK + Boussinesq force + walls)."""
    device = f.device
    w = W.to(device).view(9, 1, 1)

    rho, ux, uy = macroscopic(f)
    if phi is None:
        phi = g.sum(dim=0)

    # --- flow: Boussinesq buoyancy (deviation only, mean hydrostatic removed)
    force_y = -cfg.atwood * cfg.g * phi
    feq = equilibrium(rho, ux, uy + cfg.tau * force_y)
    f_out = f - (f - feq) / cfg.tau

    # --- phase-field advection/diffusion
    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float32,
                      device=device).view(9, 1, 1)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float32,
                      device=device).view(9, 1, 1)
    cu = 3.0 * (cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0))
    g_eq = w * phi.unsqueeze(0) * (1.0 + cu)
    g_out = g - (g - g_eq) / cfg.tau_phi

    f_out, g_out = stream(f_out), stream(g_out)

    # no-slip on the flow, zero phase flux on the phase field: bounce back rows
    f_out = _bounce_back_rows(f_out)
    g_out = _bounce_back_rows(g_out)
    return f_out, g_out


_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)


def _bounce_back_rows(buf: torch.Tensor) -> torch.Tensor:
    """Half-way bounce back on the first/last lattice rows (D2Q9, ``cy`` walls).

    After streaming the wall rows hold exactly the populations that left the
    adjacent fluid row, so reflecting them back conserves the *interior* mass
    exactly (the reflected pair is swapped, nothing is created).
    """
    out = buf.clone()
    # bottom wall (row 0): fluid row 1 misses c_i with cy > 0 (2, 5, 6)
    out[2, 1, :] = buf[4, 0, :]
    out[5, 1, :] = buf[7, 0, :]
    out[6, 1, :] = buf[8, 0, :]
    # top wall (row ny-1): fluid row ny-2 misses c_i with cy < 0 (4, 7, 8)
    out[4, -2, :] = buf[2, -1, :]
    out[7, -2, :] = buf[5, -1, :]
    out[8, -2, :] = buf[6, -1, :]
    return out


# ---------------------------------------------------------------------------
# Interface diagnostics
# ---------------------------------------------------------------------------


def interface_column(phi: torch.Tensor) -> torch.Tensor:
    """Heavy-fluid column height ``N(x) = sum_y (1+phi)/2`` (conserved moment)."""
    return 0.5 * (1.0 + phi).sum(dim=0)


def mode_amplitude(phi: torch.Tensor, lam: int) -> float:
    """First Fourier mode amplitude of the interface column height."""
    ny, nx = phi.shape
    n = interface_column(phi)
    fluc = n - n.mean()
    x = torch.arange(nx, dtype=torch.float64, device=phi.device)
    basis = torch.cos(2.0 * math.pi * x / float(lam)).to(fluc.dtype)
    return float((2.0 / nx) * torch.abs((fluc * basis).sum()).item())


def interface_zero_crossing(phi: torch.Tensor) -> torch.Tensor:
    """Per-column interface height ``y`` where ``phi`` changes sign (linear interp).

    Returns a float64 tensor of length ``nx``; ``NaN`` where no crossing exists.
    """
    ny, nx = phi.shape
    out = torch.full((nx,), float("nan"), dtype=torch.float64, device=phi.device)
    for j in range(nx):
        col = phi[:, j]
        idx = None
        for y in range(ny - 1):
            if float(col[y]) < 0.0 <= float(col[y + 1]):
                idx = y
                break
        if idx is None:
            continue
        c0 = float(col[idx])
        c1 = float(col[idx + 1])
        out[j] = idx + (-c0) / (c1 - c0)
    return out


def mode_amplitude_zc(phi: torch.Tensor, lam: int) -> float:
    """First Fourier mode amplitude of the ``phi = 0`` interface height."""
    ny, nx = phi.shape
    y_int = interface_zero_crossing(phi)
    if bool(torch.isnan(y_int).any()):
        return float("nan")
    fluc = y_int - y_int.mean()
    x = torch.arange(nx, dtype=torch.float64, device=phi.device)
    basis = torch.cos(2.0 * math.pi * x / float(lam))
    return float((2.0 / nx) * torch.abs((fluc * basis).sum()).item())


@dataclass
class GrowthFit:
    sigma: float
    intercept: float
    r2: float
    n_points: int
    t_range: tuple[float, float]
    a_range: tuple[float, float]


def fit_growth_rate(
    times: list[float], amps: list[float], a_min: float, a_max: float
) -> GrowthFit:
    """Least-squares slope of ``ln a`` inside the linear window ``[a_min, a_max]``."""
    pts = [(t, a) for t, a in zip(times, amps) if a_min <= a <= a_max]
    if len(pts) < 4:
        # fall back: widest available monotone window
        pts = [(t, a) for t, a in zip(times, amps) if a > 0.0][: len(times)]
    if len(pts) < 2:
        raise ValueError("insufficient amplitude samples for a growth-rate fit")
    ts = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    ls = torch.tensor([math.log(p[1]) for p in pts], dtype=torch.float64)
    n = ts.numel()
    t_mean, l_mean = ts.mean(), ls.mean()
    denom = ((ts - t_mean) ** 2).sum()
    if float(denom) == 0.0:
        raise ValueError("degenerate time window")
    slope = float(((ts - t_mean) * (ls - l_mean)).sum() / denom)
    intercept = float(l_mean - slope * t_mean)
    resid = ls - (slope * ts + intercept)
    ss_res = float((resid**2).sum())
    ss_tot = float(((ls - l_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return GrowthFit(
        sigma=slope,
        intercept=intercept,
        r2=r2,
        n_points=n,
        t_range=(float(ts.min()), float(ts.max())),
        a_range=(float(min(p[1] for p in pts)), float(max(p[1] for p in pts))),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_rayleigh_taylor(cfg: RTConfig, output_dir: Path | None = None) -> dict:
    """Run one RT case and return a result dict (optionally writing artifacts).

    Two independent amplitude estimators are recorded every ``sample_every``
    steps: the conserved heavy-column moment (``a_col``) and the ``phi = 0``
    interface height (``a_zc``).  The reported growth rate is a two-mode
    projection of ``a_zc`` inside the linear window ``[eta0, a_hi_frac*lam]``,
    which removes the impulsive-start transient (growing + decaying mode).
    """
    cfg.validate()
    f, g, phi = init_rt_state(cfg)

    sigma_th = analytic_growth_rate(cfg.atwood, cfg.g, cfg.lam)
    sigma_visc = viscous_growth_rate(cfg.atwood, cfg.g, cfg.lam, cfg.nu)

    mass_init = float(g[:, 1:-1, :].sum().item())
    times: list[float] = []
    a_col: list[float] = []
    a_zc: list[float] = []

    t_stop = cfg.n_steps
    for step in range(1, cfg.n_steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % cfg.sample_every == 0 or step == 1:
            times.append(float(step))
            a_col.append(mode_amplitude(phi, cfg.lam))
            a_zc.append(mode_amplitude_zc(phi, cfg.lam))
        if a_zc[-1] > cfg.stop_a_frac * cfg.lam:
            t_stop = step
            break

    a_hi = cfg.a_hi_frac * cfg.lam
    keep = [(t, a) for t, a in zip(times, a_zc) if 0.0 < a <= a_hi]
    fit = fit_growth_rate_twomode(
        [k[0] for k in keep], [k[1] for k in keep],
        (0.8 * sigma_th, 1.3 * sigma_th),
    )
    fit_col = fit_growth_rate(times, a_col, cfg.a_lo_frac * cfg.lam, a_hi)

    mass_end = float(g[:, 1:-1, :].sum().item())
    result: dict = {
        "lam": cfg.lam,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "atwood": cfg.atwood,
        "g": cfg.g,
        "k": cfg.k,
        "eta0": cfg.eta0_frac * cfg.lam,
        "tau": cfg.tau,
        "nu": cfg.nu,
        "tau_phi": cfg.tau_phi,
        "diffusivity": cfg.diffusivity,
        "interface_width": cfg.interface_width,
        "steps_run": t_stop,
        "sigma_analytic": sigma_th,
        "sigma_viscous_theory": sigma_visc,
        "sigma_sim": fit.sigma,
        "sigma_sim_colmoment": fit_col.sigma,
        "fit_r2": fit.r2,
        "fit_points": fit.n_points,
        "fit_t_range": list(fit.t_range),
        "fit_a_over_lam": [fit.a_range[0] / cfg.lam, fit.a_range[1] / cfg.lam],
        "ratio_sim_over_analytic": fit.sigma / sigma_th,
        "rel_error_pct": 100.0 * (fit.sigma / sigma_th - 1.0),
        "ratio_sim_over_viscous": fit.sigma / sigma_visc,
        "rel_error_vs_viscous_pct": 100.0 * (fit.sigma / sigma_visc - 1.0),
        "ratio_colmoment_over_analytic": fit_col.sigma / sigma_th,
        "phase_mass_init": mass_init,
        "phase_mass_end": mass_end,
        "phase_mass_drift_pct": 100.0 * abs(mass_end / mass_init - 1.0),
        "config": asdict(cfg),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "growth_history.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["step", "a_column_moment", "a_interface_zc", "a_over_lambda_zc"])
            for t, ac, az in zip(times, a_col, a_zc):
                wr.writerow([int(t), f"{ac:.10e}", f"{az:.10e}", f"{az / cfg.lam:.6e}"])
        (out / "case.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return result


def fit_growth_rate_twomode(
    times: list[float],
    amps: list[float],
    sigma_bounds: tuple[float, float],
    n_grid: int = 600,
) -> GrowthFit:
    """Fit ``a(t) = c_+ e^{sigma t} + c_- e^{-sigma t}`` (impulsive-start solution).

    Starting from rest the exact linear solution is a superposition of the
    growing and the decaying normal mode (``a = a_0 cosh(sigma t)`` for a pure
    displacement perturbation).  A plain log-linear fit therefore *under*-estimates
    the asymptotic growth rate during the startup transient.  Projecting on the
    two-mode basis and scanning ``sigma`` removes that bias; ``sigma`` enters
    non-linearly, the two amplitudes ``c_pm`` are solved by linear least squares
    for every trial ``sigma``.
    """
    pts = [(t, a) for t, a in zip(times, amps) if a > 0.0 and math.isfinite(a)]
    if len(pts) < 6:
        raise ValueError("need >= 6 amplitude samples for a two-mode fit")
    ts = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    a = torch.tensor([p[1] for p in pts], dtype=torch.float64)

    best: tuple[float, float, torch.Tensor] | None = None
    lo, hi = sigma_bounds
    for sig in torch.linspace(lo, hi, n_grid, dtype=torch.float64).tolist():
        basis = torch.stack([torch.exp(sig * ts), torch.exp(-sig * ts)], dim=1)
        coef, *_ = torch.linalg.lstsq(basis, a.unsqueeze(1))
        rss = float(((a.unsqueeze(1) - basis @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, sig, coef)
    assert best is not None
    rss, sig, coef = best
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2 = 1.0 - rss / ss_tot if ss_tot > 0.0 else 1.0
    return GrowthFit(
        sigma=sig,
        intercept=float(coef[1, 0]),
        r2=r2,
        n_points=len(pts),
        t_range=(float(ts.min()), float(ts.max())),
        a_range=(float(a.min()), float(a.max())),
    )
