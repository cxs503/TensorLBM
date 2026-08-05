"""Common wave boundary-condition module — delegates to lattice-specific Zou-He inlets.

Provides a unified ``wave_bc_3d(f, face, wave_params)`` interface that:

1. Computes the Airy / JONSWAP wave velocity profile (lattice-agnostic — the
   profile is just ``(ux, uy, uz)`` fields).
2. Applies the Zou-He inlet velocity BC with that profile (lattice-specific:
   D3Q19 or D3Q27).

This places the wave BC at the **same level** as ``far_field_bc`` and channel
BCs: it is a post-streaming boundary update that is collision-agnostic and can
be freely combined with any collision operator (BGK, MRT, CG, …) or turbulence
model (Smagorinsky, WALE, RANS, …).

Supported lattices: **D3Q19** and **D3Q27**.

Hot-path invariants
-------------------
* No GPU→CPU syncs (``.item()``, ``float(tensor)``) inside the BC path.
* Direction-index lists are pre-computed Python lists (not ``.item()`` lookups).
* The velocity-profile computation is shared across lattices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .d3q19 import OPPOSITE as _OPP19
from .d3q19 import equilibrium3d
from .d3q27 import OPPOSITE as _OPP27
from .d3q27 import equilibrium27

__all__ = [
    "WaveParams",
    "wave_bc_3d",
    "zou_he_inlet_velocity_profile_19",
    "zou_he_inlet_velocity_profile_27",
]

# --------------------------------------------------------------------------- #
# Pre-computed direction-index lists (avoid .item() GPU→CPU sync in hot path)
# --------------------------------------------------------------------------- #

# D3Q19: directions with cx > 0 (unknown at x=0 inlet) and their opposites
_D3Q19_INLET_DIRS: list[int] = [1, 7, 9, 11, 13]
_D3Q19_INLET_OPP: list[int] = [int(_OPP19[k].item()) for k in _D3Q19_INLET_DIRS]

# D3Q19: cx=0 directions at x=0
_D3Q19_CX0: list[int] = [0, 3, 4, 5, 6, 15, 16, 17, 18]
# D3Q19: cx<0 directions at x=0
_D3Q19_CX_NEG: list[int] = [2, 8, 10, 12, 14]

# D3Q27: directions with cx > 0 (unknown at x=0 inlet) and their opposites
_D3Q27_INLET_DIRS: list[int] = [1, 7, 9, 11, 13, 19, 21, 23, 25]
_D3Q27_INLET_OPP: list[int] = [int(_OPP27[k].item()) for k in _D3Q27_INLET_DIRS]

# D3Q27: cx=0 directions at x=0
_D3Q27_CX0: list[int] = [0, 3, 4, 5, 6, 15, 16, 17, 18]
# D3Q27: cx<0 directions at x=0
_D3Q27_CX_NEG: list[int] = [2, 8, 10, 12, 14, 20, 22, 24, 26]


# --------------------------------------------------------------------------- #
# Wave parameter dataclass
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WaveParams:
    """Parameters for the Airy linear-wave inlet boundary condition.

    All quantities are in lattice units unless otherwise noted.

    Attributes
    ----------
    step : int
        Current simulation time step (used as *t* in LBM units).
    u_mean : float
        Mean current x-velocity (always positive).
    wave_amp : float
        Horizontal velocity amplitude at the free surface
        (i.e. *U_w = A · ω* where *A* is wave height / 2).
        Should be small (≪ *c_s* ≈ 0.577).
    wave_period : float
        Wave period in LBM time steps.
    wave_k : float
        Wave number *k = 2π / λ* (units of 1 / lattice spacing).
    water_depth : float
        Water depth *H* in lattice units.
    z_bed : float
        z-index of the sea bed (bottom of the water column).
    rho_out : float
        Prescribed density at the outlet (if outlet is also applied).
    apply_outlet : bool
        If True, also apply Zou/He pressure outlet at the opposite face.
    """

    step: int = 0
    u_mean: float = 0.0
    wave_amp: float = 0.0
    wave_period: float = 100.0
    wave_k: float = 0.01
    water_depth: float = 1.0
    z_bed: float = 0.0
    rho_out: float = 1.0
    apply_outlet: bool = True


# --------------------------------------------------------------------------- #
# Shared velocity-profile computation (lattice-agnostic)
# --------------------------------------------------------------------------- #

def _airy_wave_velocity_3d(
    nz: int,
    ny: int,
    step: int,
    u_mean: float,
    wave_amp: float,
    wave_period: float,
    wave_k: float,
    water_depth: float,
    z_bed: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Airy wave velocity components at the inlet plane for one step.

    Returns ``(ux, uy, uz)`` each of shape ``(nz, ny)``.
    """
    omega = 2.0 * math.pi / wave_period
    phase = omega * step

    z_coords = torch.arange(nz, device=device, dtype=torch.float32)
    depth_from_bed = z_coords - z_bed

    kH = wave_k * water_depth
    sinh_kH = math.sinh(kH) if kH > 1e-8 else 1e-8
    cosh_kH = math.cosh(kH)

    cosh_z = torch.cosh(wave_k * depth_from_bed.clamp(min=0.0))
    ux_profile = wave_amp * math.cos(phase) * cosh_z / cosh_kH

    sinh_z = torch.sinh(wave_k * depth_from_bed.clamp(min=0.0))
    uz_profile = -wave_amp * math.sin(phase) * sinh_z / sinh_kH

    ux = ux_profile.unsqueeze(1).expand(nz, ny) + u_mean
    uy = torch.zeros(nz, ny, device=device, dtype=torch.float32)
    uz = uz_profile.unsqueeze(1).expand(nz, ny)
    return ux, uy, uz


# --------------------------------------------------------------------------- #
# D3Q19 Zou-He inlet with velocity profile (vectorised, no .item())
# --------------------------------------------------------------------------- #

def zou_he_inlet_velocity_profile_19(
    f: torch.Tensor,
    ux_in: torch.Tensor,
    uy_in: torch.Tensor,
    uz_in: torch.Tensor,
) -> torch.Tensor:
    """Zou/He inlet velocity BC at x=0 with a non-uniform 2-D velocity field (D3Q19).

    Uses non-equilibrium bounce-back (NEBB) for every incoming direction *k*
    (cx > 0).  Vectorised — no Python loop, no ``.item()``.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        ux_in: Prescribed x-velocity field of shape ``(nz, ny)`` at the inlet.
        uy_in: Prescribed y-velocity field of shape ``(nz, ny)`` at the inlet.
        uz_in: Prescribed z-velocity field of shape ``(nz, ny)`` at the inlet.

    Returns:
        Updated distribution tensor of the same shape.
    """
    # Sum of cx=0 and cx<0 directions at x=0
    sum_cx0 = sum(f[k, :, :, 0] for k in _D3Q19_CX0)
    sum_cx_neg = sum(f[k, :, :, 0] for k in _D3Q19_CX_NEG)

    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

    # Equilibrium at the inlet — wrap to (nz, ny, 1) for equilibrium3d
    rho3 = rho.unsqueeze(-1)
    ux3 = ux_in.unsqueeze(-1)
    uy3 = uy_in.unsqueeze(-1)
    uz3 = uz_in.unsqueeze(-1)
    feq = equilibrium3d(rho3, ux3, uy3, uz3)  # (19, nz, ny, 1)

    f_new = f.clone()
    dirs = torch.tensor(_D3Q19_INLET_DIRS, device=f.device, dtype=torch.long)
    opps = torch.tensor(_D3Q19_INLET_OPP, device=f.device, dtype=torch.long)
    f_new[dirs, :, :, 0] = (
        feq[dirs, :, :, 0]
        - feq[opps, :, :, 0]
        + f[opps, :, :, 0]
    )
    return f_new


# --------------------------------------------------------------------------- #
# D3Q27 Zou-He inlet with velocity profile (vectorised, no .item())
# --------------------------------------------------------------------------- #

def zou_he_inlet_velocity_profile_27(
    f: torch.Tensor,
    ux_in: torch.Tensor,
    uy_in: torch.Tensor,
    uz_in: torch.Tensor,
) -> torch.Tensor:
    """Zou/He inlet velocity BC at x=0 with a non-uniform 2-D velocity field (D3Q27).

    Mirrors :func:`zou_he_inlet_velocity_profile_19` but for the D3Q27 lattice.
    Vectorised — no Python loop, no ``.item()``.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        ux_in: Prescribed x-velocity field of shape ``(nz, ny)`` at the inlet.
        uy_in: Prescribed y-velocity field of shape ``(nz, ny)`` at the inlet.
        uz_in: Prescribed z-velocity field of shape ``(nz, ny)`` at the inlet.

    Returns:
        Updated distribution tensor of the same shape.
    """
    sum_cx0 = sum(f[k, :, :, 0] for k in _D3Q27_CX0)
    sum_cx_neg = sum(f[k, :, :, 0] for k in _D3Q27_CX_NEG)

    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

    rho3 = rho.unsqueeze(-1)
    ux3 = ux_in.unsqueeze(-1)
    uy3 = uy_in.unsqueeze(-1)
    uz3 = uz_in.unsqueeze(-1)
    feq = equilibrium27(rho3, ux3, uy3, uz3)  # (27, nz, ny, 1)

    f_new = f.clone()
    dirs = torch.tensor(_D3Q27_INLET_DIRS, device=f.device, dtype=torch.long)
    opps = torch.tensor(_D3Q27_INLET_OPP, device=f.device, dtype=torch.long)
    f_new[dirs, :, :, 0] = (
        feq[dirs, :, :, 0]
        - feq[opps, :, :, 0]
        + f[opps, :, :, 0]
    )
    return f_new


# --------------------------------------------------------------------------- #
# Unified dispatch: wave_bc_3d
# --------------------------------------------------------------------------- #

def wave_bc_3d(
    f: torch.Tensor,
    face: str = "inlet_x",
    wave_params: WaveParams | None = None,
    *,
    lattice: str = "auto",
) -> torch.Tensor:
    """Apply the wave boundary condition at the specified face (D3Q19 / D3Q27).

    Computes the Airy linear-wave velocity profile at the inlet plane and
    applies the Zou-He inlet velocity BC.  Optionally also applies a
    pressure (Zou/He) outlet at the opposite face.

    This is a **post-streaming** boundary update that is collision-agnostic:
    it can be freely combined with any collision operator (BGK, MRT, CG, …)
    or turbulence model (Smagorinsky, WALE, RANS, …).

    Args:
        f: Distribution tensor of shape ``(Q, nz, ny, nx)`` where *Q* is 19
           (D3Q19) or 27 (D3Q27).
        face: Which face to apply the wave inlet on.  Currently only
              ``"inlet_x"`` (x=0 plane) is supported.
        wave_params: :class:`WaveParams` with wave kinematics.  If *None*,
                     a zero-amplitude (uniform current) default is used.
        lattice: Lattice name — ``"D3Q19"``, ``"D3Q27"``, or ``"auto"``
                 (inferred from ``f.shape[0]``).

    Returns:
        Updated distribution tensor of the same shape as *f*.

    Raises:
        ValueError: If the lattice or face is unsupported, or if the
                    distribution tensor has an unexpected shape.
    """
    if wave_params is None:
        wave_params = WaveParams()

    if face != "inlet_x":
        raise ValueError(
            f"Only face='inlet_x' is currently supported, got {face!r}"
        )

    # Determine lattice
    if lattice == "auto":
        q = f.shape[0]
        if q == 19:
            lattice = "D3Q19"
        elif q == 27:
            lattice = "D3Q27"
        else:
            raise ValueError(
                f"Cannot auto-detect lattice from f.shape[0]={q}; "
                f"expected 19 (D3Q19) or 27 (D3Q27)"
            )
    lattice_u = lattice.upper()
    if lattice_u not in ("D3Q19", "D3Q27"):
        raise ValueError(
            f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}"
        )

    expected_q = 19 if lattice_u == "D3Q19" else 27
    if f.ndim != 4 or f.shape[0] != expected_q:
        raise ValueError(
            f"{lattice_u} populations must have shape ({expected_q}, nz, ny, nx), "
            f"got {tuple(f.shape)}"
        )

    nz, ny = f.shape[1], f.shape[2]
    device = f.device

    ux_in, uy_in, uz_in = _airy_wave_velocity_3d(
        nz, ny,
        step=wave_params.step,
        u_mean=wave_params.u_mean,
        wave_amp=wave_params.wave_amp,
        wave_period=wave_params.wave_period,
        wave_k=wave_params.wave_k,
        water_depth=wave_params.water_depth,
        z_bed=wave_params.z_bed,
        device=device,
    )

    if lattice_u == "D3Q19":
        f = zou_he_inlet_velocity_profile_19(f, ux_in, uy_in, uz_in)
    else:
        f = zou_he_inlet_velocity_profile_27(f, ux_in, uy_in, uz_in)

    if wave_params.apply_outlet:
        if lattice_u == "D3Q19":
            from .boundaries3d import zou_he_outlet_pressure_3d
            f = zou_he_outlet_pressure_3d(f, rho_out=wave_params.rho_out)
        else:
            from .boundaries_d3q27 import zou_he_outlet_pressure_27
            f = zou_he_outlet_pressure_27(f, rho_out=wave_params.rho_out)

    return f
