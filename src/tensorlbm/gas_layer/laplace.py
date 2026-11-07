"""Static-bubble Young-Laplace gate for the Shan-Chen two-phase kernel.

Purpose
-------
Before a gas layer is sheared under a hull, the interface model must be shown to
carry the right capillary pressure.  This module runs a quiescent vapour bubble
in a liquid bath (one Shan-Chen single-component fluid with a Carnahan-Starling
equation of state), measures the mechanical-pressure jump across the interface,
and regresses ``Delta p`` against ``1/R``:

    ``Delta p(R) = sigma (1/R1 + 1/R2) = sigma / R``   (2-D circular interface)

The gate is the *self-consistency of that law across radii* -- a wrong surface
tension shows up as a radius-dependent ``sigma``.  The measured mechanical
pressure is the Shan-Chen pressure ``p = rho cs^2 - (G/2) cs^2 psi^2``, not a
density proxy: this is what makes the check a pressure check.

Reported alongside: the coexisting density ratio ``rho_liquid/rho_gas`` (the
density-contrast requirement) and the 10-90 % interface width in lattice units
(the interface-resolution requirement), plus the maximum spurious current.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..multiphase import make_psi_eos
from . import kernels


@dataclass
class LaplaceConfig:
    """Configuration of the static-bubble Laplace gate."""

    nx: int = 88
    ny: int = 88
    radii: tuple[float, ...] = (12.0, 16.0, 20.0)
    eos_temperature: float = 0.04010547044515562
    G: float = 1.0
    tau: float = 1.0
    n_steps: int = 3000
    rho_liquid_init: float = 7.997835999814006
    rho_gas_init: float = 0.0649052793822428
    init_width: float = 4.0
    avg_every: int = 25
    eos: str = "pr"
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def psi(self):
        """Pseudopotential of the configured equation of state."""
        return make_psi_eos(self.eos, self.eos_temperature)


@dataclass
class LaplaceResult:
    """Outcome of the Laplace gate for one bubble radius."""

    radius: float
    rho_liquid: float
    rho_gas: float
    density_ratio: float
    delta_p: float
    sigma_eff: float
    interface_width: float
    max_spurious_velocity: float


def _radial_grid(nx: int, ny: int, dtype: torch.dtype, device: torch.device):
    yy = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
    xx = torch.arange(nx, dtype=dtype, device=device).view(1, nx)
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    return torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def _interface_width(r: torch.Tensor, rho: torch.Tensor, rho_l: float, rho_g: float,
                     radius: float) -> float:
    """10-90 % radial width of the density transition, in lattice units."""
    nbins = 160
    edges = torch.linspace(0.0, float(r.max()), nbins + 1, dtype=rho.dtype, device=rho.device)
    prof = torch.full((nbins,), float("nan"), dtype=rho.dtype, device=rho.device)
    for k in range(nbins - 1):
        mask = (r >= edges[k]) & (r < edges[k + 1])
        if bool(mask.any()):
            prof[k] = rho[mask].mean()
    ok = ~torch.isnan(prof)
    xs = 0.5 * (edges[:-1] + edges[1:])[ok]
    ys = prof[ok]
    level_hi = rho_g + 0.9 * (rho_l - rho_g)
    level_lo = rho_g + 0.1 * (rho_l - rho_g)

    def crossing(level: float) -> float:
        sign = ys - level
        for k in range(len(ys) - 1):
            if float(sign[k]) * float(sign[k + 1]) <= 0.0:
                denom = float(ys[k] - ys[k + 1])
                if abs(denom) < 1e-30:
                    return float(xs[k])
                t = float(ys[k] - level) / denom
                return float(xs[k] + t * (xs[k + 1] - xs[k]))
        return float("nan")

    r_hi = crossing(level_hi)
    r_lo = crossing(level_lo)
    if math.isnan(r_hi) or math.isnan(r_lo):
        return float("nan")
    return abs(r_hi - r_lo)


def _run_one(
    config: LaplaceConfig, radius: float, device: torch.device
) -> tuple[LaplaceResult, dict[str, object]]:
    nx, ny = config.nx, config.ny
    r = _radial_grid(nx, ny, config.dtype, device)
    if config.init_width > 0.0:
        rho = config.rho_gas_init + 0.5 * (config.rho_liquid_init - config.rho_gas_init) * (
            1.0 + torch.tanh((radius - r) / config.init_width)
        )
    else:
        rho = torch.where(
            r < radius,
            torch.full_like(r, config.rho_liquid_init),
            torch.full_like(r, config.rho_gas_init),
        )
    zero = torch.zeros_like(rho)
    f = kernels.equilibrium(rho, zero, zero)
    psi = config.psi()

    inner = r < radius - 6.0
    outer = r > radius + max(10.0, 0.5 * radius)
    if not bool(outer.any()):
        outer = r > radius + 6.0
    # Time-average the pressure over the last third of the run: the Shan-Chen
    # spurious currents oscillate, and a single snapshot biases Delta p.
    n_avg = max(1, config.n_steps // 3 // config.avg_every)
    p_in_sum = 0.0
    p_out_sum = 0.0
    for step in range(config.n_steps):
        f_post, _, _ = kernels.collide_scp(
            f, config.tau, psi, G=config.G, periodic_x=True
        )
        f = kernels.stream(f_post, solid=None, periodic_x=True, periodic_y=True)["f"]
        if step >= config.n_steps - n_avg * config.avg_every and step % config.avg_every == 0:
            rho_s = kernels.macroscopic(f)[0]
            p_s = kernels.mechanical_pressure(rho_s, psi, G=config.G)
            p_in_sum += float(p_s[inner].mean())
            p_out_sum += float(p_s[outer].mean())

    rho_f, ux, uy = kernels.macroscopic(f)
    rho_l = float(rho_f[inner].mean())
    rho_g = float(rho_f[outer].mean())
    dp = (p_in_sum - p_out_sum) / max(n_avg, 1)
    width = _interface_width(r, rho_f, rho_l, rho_g, radius)
    umax = float(torch.sqrt(ux * ux + uy * uy).max())
    res = LaplaceResult(
        radius=radius,
        rho_liquid=rho_l,
        rho_gas=rho_g,
        density_ratio=rho_l / max(rho_g, 1e-12),
        delta_p=dp,
        sigma_eff=dp * radius,
        interface_width=width,
        max_spurious_velocity=umax,
    )
    return res, {"rho_final": rho_f}


def run_laplace_gate(config: LaplaceConfig | None = None) -> dict[str, object]:
    """Run the static-bubble Laplace gate over all configured radii.

    Returns a JSON-serialisable report containing the per-radius measurements,
    the least-squares surface tension ``sigma`` (through the origin), the
    per-radius relative error of ``Delta p`` against ``sigma/R``, the density
    ratio, the interface width, and the boolean gate verdicts.
    """
    cfg = config or LaplaceConfig()
    device = torch.device(cfg.device)
    results: list[LaplaceResult] = []
    for radius in cfg.radii:
        res, _ = _run_one(cfg, float(radius), device)
        results.append(res)

    inv_r = [1.0 / r.radius for r in results]
    dp = [r.delta_p for r in results]
    num = sum(x * y for x, y in zip(inv_r, dp, strict=True))
    den = sum(x * x for x in inv_r)
    sigma = num / den if den > 0 else float("nan")
    rel = [abs(d - sigma * ir) / abs(sigma * ir) for d, ir in zip(dp, inv_r, strict=True)]
    max_rel = max(rel)
    ratio = min(r.density_ratio for r in results)
    width = min(r.interface_width for r in results)
    return {
        "case": "static_bubble_laplace_gate",
        "eos": cfg.eos,
        "eos_temperature": cfg.eos_temperature,
        "G": cfg.G,
        "tau": cfg.tau,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "n_steps": cfg.n_steps,
        "sigma_eff": sigma,
        "per_radius": [
            {
                "radius": r.radius,
                "rho_liquid": r.rho_liquid,
                "rho_gas": r.rho_gas,
                "density_ratio": r.density_ratio,
                "delta_p": r.delta_p,
                "sigma_eff": r.sigma_eff,
                "delta_p_laplace_fit": sigma / (1.0 / r.radius),
                "rel_error": e,
                "interface_width": r.interface_width,
                "max_spurious_velocity": r.max_spurious_velocity,
            }
            for r, e in zip(results, rel, strict=True)
        ],
        "max_rel_error": max_rel,
        "min_density_ratio": ratio,
        "min_interface_width": width,
        "gate": {
            "laplace_rel_error_le_5pct": bool(max_rel <= 0.05),
            "density_ratio_ge_100": bool(ratio >= 100.0),
            "interface_width_ge_3": bool(width >= 3.0),
        },
        "passed": bool(max_rel <= 0.05 and ratio >= 100.0 and width >= 3.0),
    }