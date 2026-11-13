"""Lattice-Boltzmann solver for the 1-D shallow-water equations.

This module implements a D1Q3 lattice-BGK scheme for the depth-averaged
shallow-water equations (SWE), following the approach used in pylbm's
D1Q2Q2 shallow-water scheme.

Equations
---------
The 1-D SWE on a flat, frictionless bed:

    ∂_t h   + ∂_x(hu) = 0
    ∂_t(hu) + ∂_x(hu²/h + ½gh²) = 0

LBM formulation (following pylbm D1Q2Q2 pattern)
-------------------------------------------------
Two distribution functions with D1Q2 velocities (±λ):

    f_i  → conserved moment h  (mass),     i ∈ {1, 2}
    g_i  → conserved moment hu (momentum), i ∈ {1, 2}

Velocities: c_1 = +λ, c_2 = -λ

Equilibrium (from Chapman-Enskog analysis):
    f_1^eq = ½h + ½(hu)/λ
    f_2^eq = ½h - ½(hu)/λ

    g_1^eq = ½(hu) + ½(hu²/h + ½gh²)/λ
    g_2^eq = ½(hu) - ½(hu²/h + ½gh²)/λ

Relaxation: s = 1/(0.5 + σ), where σ controls numerical viscosity.

Boundary conditions
-------------------
- Left (x_min): reflective wall
- Right (x_max): zero-gradient outflow

References
----------
- Gouarin & Graille, pylbm demo: D1Q2Q2 shallow water
- Toro (2001) Shock-Capturing Methods for Free-Surface Flows
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "SWE_LBM_Config",
    "init_dam_break_lbm",
    "lbm_step",
    "run_swe_lbm",
    "SWE_LBM_Result",
]


@dataclass(frozen=True)
class SWE_LBM_Config:
    """Configuration for the 1-D SWE LBM solver."""

    nx: int = 400
    x_min: float = -4.0
    x_max: float = 12.0
    dam_x: float = 0.0
    h0: float = 1.0
    g: float = 1.0
    lam: float = 4.0          # lattice velocity (scheme celerity)
    sigma_h: float = 1.0e-3   # relaxation parameter for mass
    sigma_q: float = 1.0e-1   # relaxation parameter for momentum
    cfl: float = 0.4
    t_end: float = 5.0
    h_dry: float = 1.0e-4
    dtype: str = "float64"

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dt(self) -> float:
        return self.cfl * self.dx / self.lam

    @property
    def s_h(self) -> float:
        return 1.0 / (0.5 + self.sigma_h)

    @property
    def s_q(self) -> float:
        return 1.0 / (0.5 + self.sigma_q)

    @property
    def n_steps(self) -> int:
        return int(self.t_end / self.dt) + 1

    def validate(self) -> None:
        if self.nx < 32:
            raise ValueError("nx must be >= 32")
        if self.lam <= 0:
            raise ValueError("lam must be positive")


def init_dam_break_lbm(cfg: SWE_LBM_Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise f and g for a dam-break problem.

    Returns (f, g) each of shape (2, nx) where:
        f[0] = f_1 (right-going), f[1] = f_2 (left-going)
        g[0] = g_1 (right-going), g[1] = g_2 (left-going)
    """
    device = torch.device("cpu")
    dtype = getattr(torch, cfg.dtype)
    lam = cfg.lam

    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=dtype)
    h = torch.where(x < cfg.dam_x, torch.full_like(x, cfg.h0), torch.zeros_like(x))
    hu = torch.zeros_like(x)

    # Equilibrium for f (mass)
    f1 = 0.5 * h + 0.5 * hu / lam
    f2 = 0.5 * h - 0.5 * hu / lam

    # Equilibrium for g (momentum)
    h_safe = torch.clamp(h, min=1e-12)
    u = hu / h_safe
    flux_mom = hu * u + 0.5 * cfg.g * h * h
    g1 = 0.5 * hu + 0.5 * flux_mom / lam
    g2 = 0.5 * hu - 0.5 * flux_mom / lam

    f = torch.stack([f1, f2])
    g = torch.stack([g1, g2])
    return f, g


def lbm_step(f: torch.Tensor, g: torch.Tensor, cfg: SWE_LBM_Config) -> tuple[torch.Tensor, torch.Tensor]:
    """One LBM stream-collide step for the SWE (D1Q2)."""
    lam = cfg.lam
    h_dry = cfg.h_dry
    nx = f.shape[1]

    # --- Macroscopic variables ---
    h = f[0] + f[1]  # sum of f_1 + f_2
    hu = g[0] + g[1]  # sum of g_1 + g_2

    # Desingularise
    h = torch.clamp(h, min=0.0)
    hu = torch.where(h < h_dry, torch.zeros_like(hu), hu)

    # --- Equilibrium ---
    h_safe = torch.clamp(h, min=1e-12)
    u = hu / h_safe
    flux_mom = hu * u + 0.5 * cfg.g * h * h

    feq1 = 0.5 * h + 0.5 * hu / lam
    feq2 = 0.5 * h - 0.5 * hu / lam
    geq1 = 0.5 * hu + 0.5 * flux_mom / lam
    geq2 = 0.5 * hu - 0.5 * flux_mom / lam

    # --- Collision ---
    f1_coll = f[0] - cfg.s_h * (f[0] - feq1)
    f2_coll = f[1] - cfg.s_h * (f[1] - feq2)
    g1_coll = g[0] - cfg.s_q * (g[0] - geq1)
    g2_coll = g[1] - cfg.s_q * (g[1] - geq2)

    # --- Streaming ---
    f1_new = torch.zeros_like(f1_coll)
    f2_new = torch.zeros_like(f2_coll)
    g1_new = torch.zeros_like(g1_coll)
    g2_new = torch.zeros_like(g2_coll)

    # f_1 goes right
    f1_new[1:] = f1_coll[:-1]
    g1_new[1:] = g1_coll[:-1]

    # f_2 goes left
    f2_new[:-1] = f2_coll[1:]
    g2_new[:-1] = g2_coll[1:]

    # --- Boundary conditions ---
    # Left wall (i=0): reflective - incoming f_2 from wall uses reflected state
    h_0 = f1_new[0] + f2_new[0]
    hu_0 = g1_new[0] + g2_new[0]
    # Reflect: flip velocity sign for boundary equilibrium
    f2_new[0] = 0.5 * h_0 - 0.5 * (-hu_0) / lam  # use -hu for reflection
    g2_new[0] = 0.5 * (-hu_0) - 0.5 * (hu_0 * (-hu_0 / torch.clamp(h_0, min=1e-12)) + 0.5 * cfg.g * h_0 * h_0) / lam

    # Right outflow (i=nx-1): zero gradient
    f1_new[-1] = f1_new[-2]
    f2_new[-1] = f2_new[-2]
    g1_new[-1] = g1_new[-2]
    g2_new[-1] = g2_new[-2]

    f_new = torch.stack([f1_new, f2_new])
    g_new = torch.stack([g1_new, g2_new])
    return f_new, g_new


@dataclass
class SWE_LBM_Result:
    """Result of an SWE LBM simulation."""

    h: torch.Tensor
    hu: torch.Tensor
    x: torch.Tensor
    t: float
    n_steps: int
    mass_initial: float
    mass_final: float


def run_swe_lbm(cfg: SWE_LBM_Config) -> SWE_LBM_Result:
    """Run the SWE LBM simulation."""
    cfg.validate()

    f, g = init_dam_break_lbm(cfg)
    mass_initial = float((f[0] + f[1]).sum().item() * cfg.dx)

    dtype = getattr(torch, cfg.dtype)
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=dtype)

    for _ in range(cfg.n_steps):
        f, g = lbm_step(f, g, cfg)

    h = f[0] + f[1]
    hu = g[0] + g[1]
    mass_final = float(h.sum().item() * cfg.dx)

    return SWE_LBM_Result(
        h=h, hu=hu, x=x,
        t=cfg.n_steps * cfg.dt,
        n_steps=cfg.n_steps,
        mass_initial=mass_initial,
        mass_final=mass_final,
    )
