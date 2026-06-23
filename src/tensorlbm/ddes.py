"""Delayed Detached-Eddy Simulation (DDES) and Scale-Adaptive Simulation (SAS)
hybrid turbulence models for TensorLBM.

These models bridge RANS (near the wall) and LES (in separated regions),
corresponding to the **VLES** (Very Large Eddy Simulation) formulation used in
PowerFlow and the **hybrid RANS-LES** turbulence treatment in XFlow.

Background
----------
Detached Eddy Simulation (DES) was introduced by Spalart et al. (1997) as a
hybrid model that uses RANS near walls (where LES is prohibitively expensive)
and switches to LES in detached, energetic regions.

Delayed DES (DDES, Spalart et al. 2006) adds a shielding function ``f_d`` that
prevents premature activation of the DES limiter inside attached boundary
layers ("modelled-stress depletion" fix).

Scale-Adaptive Simulation (SAS, Menter & Egorov 2010) is a related approach
that uses the von Kármán length scale to automatically adapt from RANS to
LES-like behaviour in unsteady separated regions.

Implementation in LBM
---------------------
In the lattice Boltzmann framework the turbulence model enters through the
**effective relaxation time** τ_eff (or equivalently the effective kinematic
viscosity ν_eff):

    ν_eff = ν_molecular + ν_t

The sub-grid eddy viscosity ν_t is computed by the turbulence model.  For
DDES this is:

    ν_t = (C_S Δ_DDES)² |S̃|                      (Smagorinsky-like LES branch)

where Δ_DDES is the DDES length scale:

    Δ_DDES = max(RANS_length_scale, C_DES Δ_max / f_shield)

with

    Δ_max = max(Δx, Δy, Δz)                        (filter width)
    f_shield = 1 − tanh[(8 r_d)³]                  (DDES shielding function)
    r_d = (ν + ν_t) / (|∇u| κ² d_w²)             (empirical ratio)

For SAS the additional term is derived from the modelled turbulence kinetic
energy budget and the von Kármán length scale L_vK = κ |S| / |∇²u|.

This module provides:
  1. ``ddes_length_scale``  – compute the DDES hybrid length scale
  2. ``sas_source_term``    – SAS extra source term for ω equation
  3. ``ddes_eddy_viscosity`` – effective ν_t from DDES
  4. ``apply_ddes_collision`` – LBM collision step using DDES ν_eff
  5. ``DDESConfig`` / ``DDESResult``

References
----------
Spalart, P. R. et al. (1997). Comments on the feasibility of LES for wings,
    and on a hybrid RANS/LES approach. *Advances in DNS/LES*, 137–147.
Spalart, P. R. et al. (2006). A new version of detached-eddy simulation,
    resistant to ambiguous grid densities. *Theor. Comput. Fluid Dyn.* 20, 181.
Menter, F. R. & Egorov, Y. (2010). The scale-adaptive simulation method for
    unsteady turbulent flow predictions. *Flow Turbul. Combust.* 85, 113.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "DDESConfig",
    "DDESResult",
    "ddes_shielding_function",
    "ddes_length_scale",
    "ddes_eddy_viscosity",
    "sas_source_term",
    "apply_ddes_collision",
    "run_ddes_diagnostics",
]

# Model constants
_C_DES: float = 0.65      # DES calibration constant
_C_S: float = 0.1         # Smagorinsky constant
_KAPPA: float = 0.41      # von Kármán constant
_C_SAS: float = 2.0       # SAS model constant (Menter & Egorov 2010)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DDESConfig:
    """DDES / SAS model configuration."""
    mode: str = "ddes"           # "ddes" | "sas" | "les"
    nu_molecular: float = 1e-5   # kinematic viscosity [lattice or SI units]
    dx: float = 1.0              # grid spacing (isotropic)
    c_des: float = _C_DES        # DES calibration constant
    c_s: float = _C_S            # Smagorinsky constant
    kappa: float = _KAPPA        # von Kármán constant
    c_sas: float = _C_SAS        # SAS model constant
    nu_t_max: float = 0.3        # Clamp maximum ν_t (prevents τ<0.5)


@dataclass
class DDESResult:
    """Diagnostics from a DDES/SAS step."""
    nu_t_mean: float
    nu_t_max: float
    shield_fraction: float       # fraction of cells in RANS-shielded region
    rans_fraction: float         # fraction of cells using RANS length scale
    les_fraction: float          # fraction of cells using LES length scale


# ---------------------------------------------------------------------------
# Strain-rate tensor
# ---------------------------------------------------------------------------

def _strain_rate_magnitude(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Compute |S| = sqrt(2 S_ij S_ij) from 2-D velocity fields.

    S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
    """
    # Central differences
    dudx = (torch.roll(ux, -1, 1) - torch.roll(ux, 1, 1)) / 2.0
    dudy = (torch.roll(ux, -1, 0) - torch.roll(ux, 1, 0)) / 2.0
    dvdx = (torch.roll(uy, -1, 1) - torch.roll(uy, 1, 1)) / 2.0
    dvdy = (torch.roll(uy, -1, 0) - torch.roll(uy, 1, 0)) / 2.0

    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)

    return torch.sqrt(2.0 * (S11**2 + 2.0*S12**2 + S22**2) + 1e-20)


def _gradient_magnitude(field: torch.Tensor) -> torch.Tensor:
    """2-D gradient magnitude using central differences."""
    dfdx = (torch.roll(field, -1, 1) - torch.roll(field, 1, 1)) / 2.0
    dfdy = (torch.roll(field, -1, 0) - torch.roll(field, 1, 0)) / 2.0
    return torch.sqrt(dfdx**2 + dfdy**2 + 1e-20)


def _laplacian(field: torch.Tensor) -> torch.Tensor:
    """Discrete 2-D Laplacian using 5-point stencil."""
    return (
        torch.roll(field, 1, 0)
        + torch.roll(field, -1, 0)
        + torch.roll(field, 1, 1)
        + torch.roll(field, -1, 1)
        - 4.0 * field
    )


# ---------------------------------------------------------------------------
# DDES shielding function
# ---------------------------------------------------------------------------

def ddes_shielding_function(
    nu: float,
    nu_t: torch.Tensor,
    S_mag: torch.Tensor,
    d_wall: torch.Tensor,
    kappa: float = _KAPPA,
) -> torch.Tensor:
    """Compute the DDES shielding function f_d.

    f_d = 1 − tanh[(8 r_d)³]
    r_d = (ν + ν_t) / (|S| κ² d_w²)

    Returns f_d ∈ [0, 1]:
      f_d ≈ 0 → boundary-layer region (RANS shielded)
      f_d ≈ 1 → separated / far-field region (LES active)
    """
    denominator = S_mag * kappa**2 * d_wall**2 + 1e-20
    r_d = (nu + nu_t) / denominator
    f_d = 1.0 - torch.tanh((8.0 * r_d) ** 3)
    return f_d


# ---------------------------------------------------------------------------
# DDES length scale
# ---------------------------------------------------------------------------

def ddes_length_scale(
    d_wall: torch.Tensor,
    f_d: torch.Tensor,
    delta_max: float,
    l_rans: torch.Tensor,
    c_des: float = _C_DES,
) -> torch.Tensor:
    """Compute the DDES hybrid length scale.

    l_DDES = l_RANS − f_d max(0, l_RANS − C_DES Δ_max)

    where l_RANS is the RANS length scale (e.g. d_w for Smagorinsky near-wall).
    """
    l_les = c_des * delta_max
    # Shielded switch: in BL (f_d≈0), use RANS; in separated (f_d≈1), use LES
    l_ddes = l_rans - f_d * torch.clamp(l_rans - l_les, min=0.0)
    return l_ddes


# ---------------------------------------------------------------------------
# DDES eddy viscosity
# ---------------------------------------------------------------------------

def ddes_eddy_viscosity(
    ux: torch.Tensor,
    uy: torch.Tensor,
    d_wall: torch.Tensor,
    cfg: DDESConfig,
    nu_t_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Compute DDES sub-grid eddy viscosity ν_t.

    Returns (nu_t, f_d, nu_t_mean, nu_t_max, shield_frac).
    """
    nu = cfg.nu_molecular
    delta = cfg.dx  # isotropic grid

    S_mag = _strain_rate_magnitude(ux, uy)

    if nu_t_prev is None:
        nu_t_prev = torch.zeros_like(ux)

    # --- Mode switch ---
    if cfg.mode == "les":
        l_sgs = cfg.c_s * delta
        nu_t = l_sgs**2 * S_mag
        nu_t = torch.clamp(nu_t, max=cfg.nu_t_max)
        ones = torch.ones_like(nu_t)
        return nu_t, ones, float(nu_t.mean()), float(nu_t.max()), 0.0

    # RANS length scale: Smagorinsky with van Driest damping at wall
    l_rans = cfg.c_s * delta * d_wall / (d_wall + 1.0)   # simplified near-wall

    f_d = ddes_shielding_function(nu, nu_t_prev, S_mag, d_wall, cfg.kappa)
    l_ddes = ddes_length_scale(d_wall, f_d, delta, l_rans, cfg.c_des)

    nu_t = l_ddes**2 * S_mag
    nu_t = torch.clamp(nu_t, min=0.0, max=cfg.nu_t_max)

    shield_frac = float((f_d < 0.5).float().mean())
    rans_frac = float((l_ddes == l_rans).float().mean())
    les_frac = 1.0 - shield_frac

    return nu_t, f_d, float(nu_t.mean()), float(nu_t.max()), shield_frac


# ---------------------------------------------------------------------------
# SAS source term
# ---------------------------------------------------------------------------

def sas_source_term(
    ux: torch.Tensor,
    uy: torch.Tensor,
    k: torch.Tensor,
    omega: torch.Tensor,
    cfg: DDESConfig,
) -> torch.Tensor:
    """Compute the additional SAS production term Q_SAS.

    Q_SAS = max(C_SAS ρ κ S² [L/L_vK]² − C_SAS_2 ρ 2k max(|∇k|²/k², |∇ω|²/ω²), 0)

    where L_vK = κ |S| / |∇²u| is the von Kármán length scale.

    For simplicity this returns a 2-D field of the SAS production term that can
    be added as a source to the ω equation.
    """
    S_mag = _strain_rate_magnitude(ux, uy)

    # von Kármán length scale: L_vK = κ |S| / |∇²u|
    lap_ux = _laplacian(ux)
    lap_uy = _laplacian(uy)
    lap_mag = torch.sqrt(lap_ux**2 + lap_uy**2 + 1e-20)
    L_vK = cfg.kappa * S_mag / (lap_mag + 1e-12)

    # Turbulence length scale: L = k^0.5 / (C_μ^0.25 ω)
    C_mu = 0.09
    L_turb = torch.sqrt(k + 1e-20) / (C_mu**0.25 * (omega + 1e-12))

    # Ratio
    ratio_sq = (L_turb / (L_vK + 1e-12)) ** 2

    # Gradient terms (regularised)
    grad_k_mag = _gradient_magnitude(k)
    grad_o_mag = _gradient_magnitude(omega)
    psi = torch.maximum(
        grad_k_mag**2 / (k**2 + 1e-20),
        grad_o_mag**2 / (omega**2 + 1e-20),
    )

    Q = torch.clamp(
        cfg.c_sas * S_mag**2 * ratio_sq - 2.0 * k * psi,
        min=0.0,
    )
    return Q


# ---------------------------------------------------------------------------
# LBM collision with DDES
# ---------------------------------------------------------------------------

def apply_ddes_collision(
    f: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    rho: torch.Tensor,
    d_wall: torch.Tensor,
    cfg: DDESConfig,
    nu_t_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the BGK collision step with DDES effective viscosity.

    τ_eff = 0.5 + 3 (ν_mol + ν_t)

    Returns (f_post_collision, nu_t).
    """
    from tensorlbm.d2q9 import equilibrium

    nu_t, f_d, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg, nu_t_prev)

    nu_eff = cfg.nu_molecular + nu_t
    tau = 0.5 + 3.0 * nu_eff
    tau = torch.clamp(tau, min=0.5005)   # prevent τ ≤ 0.5 instability
    omega_lbm = 1.0 / tau

    f_eq = equilibrium(rho, ux, uy)
    f_out = f - omega_lbm.unsqueeze(0) * (f - f_eq)
    return f_out, nu_t


# ---------------------------------------------------------------------------
# Diagnostic runner
# ---------------------------------------------------------------------------

def run_ddes_diagnostics(
    ux: torch.Tensor,
    uy: torch.Tensor,
    d_wall: torch.Tensor,
    cfg: DDESConfig,
    nu_t_prev: torch.Tensor | None = None,
) -> DDESResult:
    """Compute DDES diagnostics for an existing velocity field."""
    nu_t, f_d, nu_t_mean, nu_t_max_val, shield_frac = ddes_eddy_viscosity(
        ux, uy, d_wall, cfg, nu_t_prev
    )

    total = nu_t.numel()
    rans_frac = float((f_d < 0.5).float().mean())

    return DDESResult(
        nu_t_mean=nu_t_mean,
        nu_t_max=nu_t_max_val,
        shield_fraction=shield_frac,
        rans_fraction=rans_frac,
        les_fraction=1.0 - rans_frac,
    )
