"""Cavitation model for LBM multiphase flows (Schnerr–Sauer).

Implements the Schnerr–Sauer (2001) mass-transfer model for cavitation,
coupled to the Shan–Chen single-component (SCMP) multiphase LBM.  This
provides vapour-bubble nucleation, growth and collapse dynamics analogous
to the cavitation module in XFlow and PowerFlow.

Physics
-------
The Schnerr–Sauer model expresses the phase-change mass-transfer rate as:

    ṁ⁺ = C_vap * (ρ_l ρ_v / ρ) * (3 R_b / (ρ_l − ρ_v)) * ṁ_bulk  (evaporation)
    ṁ⁻ = C_cond * similar term                                       (condensation)

where R_b is the bubble radius derived from the vapour volume fraction α_v:

    R_b = (α_v / (1 − α_v) * 3/(4π n_nuc))^(1/3)

In LBM, this is incorporated via a modified interaction potential that
applies a vapour-pressure correction to the standard Shan–Chen force,
effectively adding a mass-transfer source term to the collision operator.

The implementation uses the same D2Q9 SCMP infrastructure as
:mod:`tensorlbm.multiphase` and can be used as a drop-in enhancement.

References
----------
Schnerr, G. H., & Sauer, J. (2001). Physical and numerical modeling of
    unsteady cavitation dynamics. *4th Int. Conf. Multiphase Flow*, New
    Orleans, USA.
Shan, X., & Chen, H. (1993). Lattice Boltzmann model for simulating flows
    with multiple phases and components. *Phys. Rev. E* 47, 1815.
Sukop, M. C., & Thorne, D. T. (2006). *Lattice Boltzmann Modeling*. Springer.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

__all__ = [
    "CavitationConfig",
    "psi_cavitation",
    "schnerr_sauer_source",
    "apply_cavitation_force",
    "run_cavitation_flow",
]

# Physical reference values (lattice units)
_RHO_L = 2.65    # liquid reference density (SC at G≈−5.5)
_RHO_V = 0.038   # vapour reference density (SC at G≈−5.5)


# ---------------------------------------------------------------------------
# Pseudopotential and source term
# ---------------------------------------------------------------------------

def psi_cavitation(
    rho: torch.Tensor,
    rho_l: float = _RHO_L,
    rho_v: float = _RHO_V,
    psi0: float = 4.0,
) -> torch.Tensor:
    """Enhanced pseudopotential for cavitation (Shan–Chen + vapour-pressure correction).

    Uses the standard exponential ψ(ρ) = ψ₀ exp(−ρ₀/ρ) but with modified
    parameters tuned for large density-ratio cavitation.

    Args:
        rho:   Density field, shape ``(...)``.
        rho_l: Liquid reference density.
        rho_v: Vapour reference density.
        psi0:  Pseudopotential strength parameter.

    Returns:
        Pseudopotential ψ, same shape as *rho*.
    """
    rho_ref = (rho_l + rho_v) / 2.0
    return psi0 * torch.exp(-rho_ref / rho.clamp(min=1e-4))


def schnerr_sauer_source(
    rho: torch.Tensor,
    p: torch.Tensor,
    p_sat: float,
    rho_l: float = _RHO_L,
    rho_v: float = _RHO_V,
    n_nuc: float = 1.6e13,
    c_vap: float = 1.0,
    c_cond: float = 0.2,
) -> torch.Tensor:
    """Compute Schnerr–Sauer mass-transfer source term.

    Returns the net mass-transfer rate ṁ (> 0 = evaporation, < 0 =
    condensation) at each cell, to be added as a density source in the
    BGK collision operator.

    Args:
        rho:    Density field, shape ``(ny, nx)``.
        p:      Pressure field (ρ c_s² in LBM), shape ``(ny, nx)``.
        p_sat:  Saturation pressure in lattice units.
        rho_l:  Liquid reference density.
        rho_v:  Vapour reference density.
        n_nuc:  Nucleation site density (sites / m³ → rescaled for l.u.).
        c_vap:  Evaporation coefficient (default 1.0).
        c_cond: Condensation coefficient (default 0.2, more conservative).

    Returns:
        Mass-transfer rate ṁ, shape ``(ny, nx)``.
    """
    # Vapour volume fraction α_v estimated from local density
    alpha_v = torch.clamp((rho_l - rho) / (rho_l - rho_v + 1e-12), 0.0, 1.0)
    alpha_l = 1.0 - alpha_v

    # Bubble radius from vapour fraction
    n_nuc_lu = max(n_nuc * 1e-30, 1e-20)  # rescale to lattice units (very small)
    r_b = (alpha_v / (alpha_l.clamp(min=1e-6)) * (3.0 / (4.0 * math.pi * n_nuc_lu + 1e-40))
           ) ** (1.0 / 3.0)
    r_b = r_b.clamp(min=1e-6, max=0.5)

    # Driving pressure p − p_sat
    dp = p - p_sat
    dp_mag = dp.abs().clamp(min=1e-12)

    # Mass-transfer term (Rayleigh–Plesset simplified)
    rho_l_val = rho_l.item() if hasattr(rho_l, "item") else float(rho_l)
    inner = 2.0 / 3.0 * dp_mag / rho_l_val
    base_rate = (rho_l * rho_v / rho.clamp(min=1e-4)
                 * 3.0 / r_b.clamp(min=1e-6)
                 * torch.sqrt(inner))

    # Sign: evaporation (p < p_sat) or condensation (p > p_sat)
    m_dot = torch.where(dp < 0, c_vap * alpha_l * base_rate, -c_cond * alpha_v * base_rate)
    return m_dot


def apply_cavitation_force(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau: float,
    G: float = -5.5,
    p_sat: float | None = None,
    rho_l: float = _RHO_L,
    rho_v: float = _RHO_V,
) -> torch.Tensor:
    """Apply Shan–Chen + cavitation mass-transfer force to 2-D flow.

    Combines the standard Shan–Chen pseudopotential interaction with the
    Schnerr–Sauer density source to simulate cavitation.

    Args:
        f:     Post-collision DF, shape ``(9, ny, nx)``.
        rho:   Density, shape ``(ny, nx)``.
        ux, uy: Velocity, shape ``(ny, nx)``.
        tau:   BGK relaxation time.
        G:     Shan–Chen coupling constant (negative for phase separation).
        p_sat: Saturation pressure; defaults to critical SC pressure.
        rho_l, rho_v: Phase densities.

    Returns:
        Modified DF with cavitation force applied, shape ``(9, ny, nx)``.
    """
    from .d2q9 import C as _C  # noqa: PLC0415,F401
    from .multiphase import sc_single_component_force  # noqa: PLC0415

    # Default saturation pressure ≈ spinodal region midpoint
    if p_sat is None:
        p_sat = (rho_l + rho_v) / 2.0 / 3.0  # c_s² = 1/3

    psi = psi_cavitation(rho, rho_l, rho_v)
    F_sc = sc_single_component_force(psi, G)  # returns (Fx, Fy)

    # Pressure (EOS: p = rho c_s²)
    p = rho / 3.0

    # Schnerr–Sauer mass-transfer source
    m_dot = schnerr_sauer_source(rho, p, p_sat, rho_l, rho_v)

    # Add mass source to rest-distribution
    Fx, Fy = F_sc

    # Velocity shift due to SC force (Guo forcing)
    u_shift = 0.5 * tau  # single step
    ux_eff = ux + u_shift * Fx / rho.clamp(min=1e-4)
    uy_eff = uy + u_shift * Fy / rho.clamp(min=1e-4)

    from .d2q9 import equilibrium as feq  # noqa: PLC0415
    f_eq_forced = feq(rho + m_dot * tau, ux_eff, uy_eff)

    # Blend: full BGK with modified eq
    return f - (f - f_eq_forced) / tau


# ---------------------------------------------------------------------------
# Configuration and runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CavitationConfig:
    """Configuration for a 2-D cavitation flow simulation."""

    nx: int = 256
    ny: int = 128
    # Shan–Chen coupling
    G: float = -5.5            # SC interaction strength (induces phase separation)
    # Initial conditions
    rho_init: float = 1.5      # mean initial density (between rho_l and rho_v)
    # Nozzle / constriction geometry: narrow throat at x=x_throat with width h_throat
    throat_x: float = 0.35     # throat x position (fraction of nx)
    throat_height: float = 0.3 # throat height (fraction of ny)
    # Inlet conditions
    u_in: float = 0.04         # inlet velocity (l.u.)
    re: float = 500.0          # Reynolds number
    # Simulation
    n_steps: int = 5000
    output_interval: int = 500
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def nu(self) -> float:
        L = self.throat_height * self.ny
        return self.u_in * L / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5


def run_cavitation_flow(
    cfg: CavitationConfig | None = None,
    **kwargs: object,
) -> Path:
    """Run a 2-D nozzle cavitation simulation (Schnerr–Sauer + Shan–Chen).

    Simulates vapour-bubble formation in the low-pressure region downstream
    of a converging–diverging nozzle throat.

    Args:
        cfg:     Configuration object.
        **kwargs: Override any :class:`CavitationConfig` field.

    Returns:
        Path to the run output directory.
    """
    from .boundaries import bounce_back_cells  # noqa: PLC0415
    from .config_io import save_config_json  # noqa: PLC0415
    from .d2q9 import equilibrium, macroscopic  # noqa: PLC0415
    from .logging_config import configure_logging  # noqa: PLC0415
    from .logging_config import logger as _logger
    from .solver import stream  # noqa: PLC0415
    from .utils import (  # noqa: PLC0415
        get_reproducibility_metadata,
        prepare_run_dir,
        resolve_device,
    )

    if cfg is None:
        valid = set(CavitationConfig.__dataclass_fields__)
        cfg = CavitationConfig(**{k: v for k, v in kwargs.items() if k in valid})

    device = resolve_device(cfg.device)
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name or "cavitation_flow", cfg.overwrite)
    configure_logging(run_dir)
    save_config_json(asdict(cfg), run_dir / "config.json")

    _logger.info("Cavitation flow: nx=%d ny=%d G=%.2f Re=%.1f device=%s",
                 cfg.nx, cfg.ny, cfg.G, cfg.re, cfg.device)

    nx, ny = cfg.nx, cfg.ny

    # Build nozzle walls: channel with top/bottom walls and a symmetric constriction
    solid = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    solid[0, :] = True   # bottom wall
    solid[-1, :] = True  # top wall

    # Converging-diverging nozzle throat block
    x_t = int(cfg.throat_x * nx)
    h_half = int(0.5 * (1 - cfg.throat_height) * ny / 2)
    if h_half > 0:
        solid[:h_half, max(0, x_t - 5):x_t + 5] = True   # bottom block
        solid[ny - h_half:, max(0, x_t - 5):x_t + 5] = True  # top block

    # Init: uniform density + inlet velocity
    rho = torch.full((ny, nx), cfg.rho_init, device=device)
    ux0 = torch.full((ny, nx), cfg.u_in, device=device)
    uy0 = torch.zeros(ny, nx, device=device)
    # Perturb density slightly for phase separation seeding
    rho += 0.02 * (torch.rand(ny, nx, device=device) - 0.5)
    f = equilibrium(rho, ux0, uy0)

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    steps_out: list[int] = []
    rho_min_out: list[float] = []

    for step in range(cfg.n_steps):
        rho, ux, uy = macroscopic(f)

        # SC collision with cavitation
        f = apply_cavitation_force(f, rho, ux, uy, cfg.tau, G=cfg.G)
        f = bounce_back_cells(f, solid)

        # Inlet/outlet BCs
        f[:, :, 0] = equilibrium(rho[:, 0:1].expand(-1, 1), ux0[:, 0:1], uy0[:, 0:1])
        f = stream(f)

        if (step + 1) % cfg.output_interval == 0 or step == cfg.n_steps - 1:
            rho_pp, ux_pp, uy_pp = macroscopic(f)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(rho_pp.cpu().numpy(), origin="lower", cmap="Blues")
            axes[0].set_title(f"Density – step {step + 1}")
            plt.colorbar(im0, ax=axes[0])
            speed = torch.sqrt(ux_pp ** 2 + uy_pp ** 2)
            im1 = axes[1].imshow(speed.cpu().numpy(), origin="lower", cmap="hot")
            axes[1].set_title("Velocity |u|")
            plt.colorbar(im1, ax=axes[1])
            plt.tight_layout()
            fig.savefig(run_dir / f"step_{step + 1:06d}.png", dpi=100)
            plt.close(fig)

            rho_min = float(rho_pp.min().item())
            steps_out.append(step + 1)
            rho_min_out.append(rho_min)
            _logger.info("step=%d  rho_min=%.4f", step + 1, rho_min)

    with (run_dir / "cavitation_stats.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "rho_min"])
        w.writerows(zip(steps_out, rho_min_out, strict=True))

    meta = {
        **get_reproducibility_metadata(),
        "config": asdict(cfg),
        "steps": steps_out,
        "rho_min": rho_min_out,
    }
    with (run_dir / "run_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    _logger.info("Cavitation flow complete → %s", run_dir)
    return run_dir
