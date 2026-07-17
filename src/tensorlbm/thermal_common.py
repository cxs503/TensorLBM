"""Common thermal LBM module — composable heat-conduction step.

Extracts the double-distribution-function (DDF) thermal LBM into a reusable
module that can be composed with *any* collision / turbulence / multiphase
solver.  The thermal field evolves on a D3Q7 passive-scalar lattice, while
the momentum field can be D3Q19 or D3Q27.

Design principles
-----------------
* **No solver hot-path changes** — the common module is a standalone library
  of composable step functions.  Existing ``thermal.py`` / ``thermal3d.py``
  are untouched.
* **Same tau_eff interface** — buoyancy coupling uses the Guo force scheme,
  consistent with the existing thermal modules.
* **Conjugate heat transfer** — fluid–solid interface heat exchange via
  harmonic-mean effective conductivity, extracted from ``conjugate_ht.py``.

Public API
----------
* :func:`thermal_step` — one DDF thermal step (collision + streaming + buoyancy)
* :func:`conjugate_ht_step` — one CHT step (solid diffusion + interface coupling)
* :func:`thermal_equilibrium_3d` — D3Q7 equilibrium
* :func:`thermal_collide_bgk_3d` — BGK collision on D3Q7
* :func:`thermal_stream_3d` — periodic streaming on D3Q7
* :func:`thermal_macroscopic_3d` — recover T from g
* :func:`apply_buoyancy_3d` — Boussinesq buoyancy force on D3Q19/D3Q27

References
----------
He, X., Chen, S., & Doolen, G. D. (1998).
    A novel thermal model for the lattice Boltzmann method in incompressible
    limit. *J. Comput. Phys.* 146(1), 282–300.
"""
from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "C_D3Q7",
    "W_D3Q7",
    "thermal_equilibrium_3d",
    "thermal_collide_bgk_3d",
    "thermal_stream_3d",
    "thermal_macroscopic_3d",
    "apply_buoyancy_3d",
    "thermal_step",
    "conjugate_ht_step",
]

# ---------------------------------------------------------------------------
# D3Q7 lattice constants (shared with thermal3d.py)
# ---------------------------------------------------------------------------

C_D3Q7 = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=torch.int64,
)

W_D3Q7 = torch.tensor(
    [1.0 / 4.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0],
    dtype=torch.float32,
)

_CS2_D3Q7 = 1.0 / 4.0  # sound speed squared for D3Q7

_stream_thermal_cache: dict[
    tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


@functools.cache
def _c_thermal(device: torch.device) -> torch.Tensor:
    return C_D3Q7.to(device)


@functools.cache
def _w_thermal(device: torch.device) -> torch.Tensor:
    return W_D3Q7.to(device)


# ---------------------------------------------------------------------------
# D3Q7 thermal lattice operators
# ---------------------------------------------------------------------------


def thermal_equilibrium_3d(
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute the D3Q7 equilibrium temperature distribution.

    g_i^eq = w_i * T * (1 + 4 * (cx_i*ux + cy_i*uy + cz_i*uz))

    Args:
        T:  Temperature field, shape ``(nz, ny, nx)``.
        ux: x-velocity, shape ``(nz, ny, nx)``.
        uy: y-velocity, shape ``(nz, ny, nx)``.
        uz: z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        Equilibrium distribution, shape ``(7, nz, ny, nx)``.
    """
    device = T.device
    c = _c_thermal(device)
    w = _w_thermal(device).view(7, 1, 1, 1)
    cx = c[:, 0].view(7, 1, 1, 1).float()
    cy = c[:, 1].view(7, 1, 1, 1).float()
    cz = c[:, 2].view(7, 1, 1, 1).float()
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    return w * T.unsqueeze(0) * (1.0 + 4.0 * cu)


def thermal_collide_bgk_3d(
    g: torch.Tensor,
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau_T: float,
) -> torch.Tensor:
    """BGK collision for the D3Q7 temperature distribution.

    Args:
        g:     Temperature distribution, shape ``(7, nz, ny, nx)``.
        T:     Macroscopic temperature, shape ``(nz, ny, nx)``.
        ux:    x-velocity, shape ``(nz, ny, nx)``.
        uy:    y-velocity, shape ``(nz, ny, nx)``.
        uz:    z-velocity, shape ``(nz, ny, nx)``.
        tau_T: Thermal relaxation time (τ_T > 0.5).

    Returns:
        Post-collision distribution, shape ``(7, nz, ny, nx)``.
    """
    geq = thermal_equilibrium_3d(T, ux, uy, uz)
    return g - (g - geq) / tau_T


def thermal_stream_3d(g: torch.Tensor) -> torch.Tensor:
    """Periodic streaming for the D3Q7 temperature distribution.

    Args:
        g: Temperature distribution, shape ``(7, nz, ny, nx)``.

    Returns:
        Streamed distribution of the same shape.
    """
    nz, ny, nx = g.shape[1], g.shape[2], g.shape[3]
    device = g.device
    c = _c_thermal(device)

    cache_key = (nz, ny, nx, device.type, device.index)
    if cache_key not in _stream_thermal_cache:
        z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(7, device=device).view(7, 1, 1, 1).expand(7, nz, ny, nx)
        z_idx = z_src.view(7, nz, 1, 1).expand(7, nz, ny, nx)
        y_idx = y_src.view(7, 1, ny, 1).expand(7, nz, ny, nx)
        x_idx = x_src.view(7, 1, 1, nx).expand(7, nz, ny, nx)
        _stream_thermal_cache[cache_key] = (q_idx, z_idx, y_idx, x_idx)

    q_idx, z_idx, y_idx, x_idx = _stream_thermal_cache[cache_key]
    return g[q_idx, z_idx, y_idx, x_idx]


def thermal_macroscopic_3d(g: torch.Tensor) -> torch.Tensor:
    """Recover the macroscopic temperature from D3Q7 distributions.

    T = Σ_i g_i

    Args:
        g: Temperature distribution, shape ``(7, nz, ny, nx)``.

    Returns:
        Temperature field, shape ``(nz, ny, nx)``.
    """
    return g.sum(dim=0)


# ---------------------------------------------------------------------------
# Buoyancy force (Boussinesq) for D3Q19 / D3Q27
# ---------------------------------------------------------------------------


def apply_buoyancy_3d(
    f: torch.Tensor,
    T: torch.Tensor,
    T_ref: float,
    beta: float,
    g_y: float = -1.0,
    *,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    """Apply Boussinesq buoyancy body force to a D3Q19 or D3Q27 distribution.

    F_y = ρ * β * (T - T_ref) * g_y

    Applied via the first-order Guo scheme:
        f_i ← f_i + w_i * 3 * cy_i * F_y

    Args:
        f:      Momentum distribution, shape ``(Q, nz, ny, nx)``.
        T:      Temperature field, shape ``(nz, ny, nx)``.
        T_ref:  Reference temperature.
        beta:   Thermal expansion coefficient.
        g_y:    Gravitational acceleration in y (negative = downward).
        lattice: ``"D3Q19"`` or ``"D3Q27"`` (case-insensitive).

    Returns:
        Updated distribution, same shape as *f*.
    """
    lattice_u = lattice.upper()
    if lattice_u == "D3Q19":
        from .d3q19 import C as C3D, W as W3D, macroscopic3d

        q = 19
        c = C3D.to(f.device).float()
        w = W3D.to(f.device).float()
        rho, _, _, _ = macroscopic3d(f)
    elif lattice_u == "D3Q27":
        from .d3q27 import C as C27, W as W27, macroscopic27

        q = 27
        c = C27.to(f.device).float()
        w = W27.to(f.device).float()
        rho, _, _, _ = macroscopic27(f)
    else:
        raise ValueError(f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}")

    F_y = -rho * beta * (T - T_ref) * g_y
    cy = c[:, 1].view(q, 1, 1, 1)
    w_view = w.view(q, 1, 1, 1)
    return f + w_view * 3.0 * cy * F_y.unsqueeze(0)


# ---------------------------------------------------------------------------
# Combined thermal step
# ---------------------------------------------------------------------------


def thermal_step(
    f: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    tau_T: float = 0.8,
    lattice: str = "D3Q19",
    T_ref: float = 1.0,
    beta: float = 0.0,
    g_y: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One composable DDF thermal LBM step (collision + streaming + buoyancy).

    This function is designed to be inserted into *any* time loop alongside
    an arbitrary collision / turbulence / multiphase solver::

        for step in range(n_steps):
            f = collide_any(f, tau)      # any collision
            f = stream(f)                 # any streaming
            f, g, T = thermal_step(f, g, mask, tau_T=0.8, lattice="D3Q19")

    The thermal distribution *g* evolves on D3Q7; the momentum distribution
    *f* can be D3Q19 or D3Q27.  Buoyancy coupling (Boussinesq) is applied
    to *f* when ``beta > 0``.

    Args:
        f:      Momentum distribution, shape ``(Q, nz, ny, nx)``.
        g:      Thermal distribution (D3Q7), shape ``(7, nz, ny, nx)``.
        mask:   Optional solid mask for velocity zeroing, shape ``(nz, ny, nx)``.
        tau_T:  Thermal relaxation time (τ_T > 0.5).
        lattice: Momentum lattice — ``"D3Q19"`` or ``"D3Q27"``.
        T_ref:  Reference temperature for Boussinesq approximation.
        beta:   Thermal expansion coefficient (0 = no buoyancy).
        g_y:    Gravitational acceleration in y.

    Returns:
        ``(f_updated, g_updated, T_updated)`` — updated momentum distribution,
        thermal distribution, and temperature field.
    """
    lattice_u = lattice.upper()
    if lattice_u == "D3Q19":
        from .d3q19 import macroscopic3d

        rho, ux, uy, uz = macroscopic3d(f)
    elif lattice_u == "D3Q27":
        from .d3q27 import macroscopic27

        rho, ux, uy, uz = macroscopic27(f)
    else:
        raise ValueError(f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}")

    # Zero velocity in solid cells
    if mask is not None:
        ux = ux.masked_fill(mask, 0.0)
        uy = uy.masked_fill(mask, 0.0)
        uz = uz.masked_fill(mask, 0.0)

    # Recover temperature
    T = thermal_macroscopic_3d(g)

    # Buoyancy coupling (T → f)
    if beta != 0.0:
        f = apply_buoyancy_3d(f, T, T_ref=T_ref, beta=beta, g_y=g_y, lattice=lattice)

    # Thermal collision + streaming
    g = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
    g = thermal_stream_3d(g)

    # Recover updated temperature
    T = thermal_macroscopic_3d(g)

    return f, g, T


# ---------------------------------------------------------------------------
# Conjugate heat transfer step
# ---------------------------------------------------------------------------


def conjugate_ht_step(
    T_fluid: torch.Tensor,
    T_solid: torch.Tensor,
    mask_solid: torch.Tensor,
    *,
    alpha_s: float = 0.1,
    k_ratio: float = 1.0,
    Q_source: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One conjugate heat transfer step (solid diffusion + interface coupling).

    Extracted from ``conjugate_ht.py`` and generalised for 2-D and 3-D fields.

    1. **Solid diffusion**: explicit Euler step on the heat-conduction
       equation in the solid domain.
    2. **Interface coupling**: temperature and heat-flux continuity at the
       fluid–solid interface via harmonic-mean effective conductivity.

    Args:
        T_fluid:    Fluid temperature field.
        T_solid:    Solid temperature field.
        mask_solid: Boolean mask (True = solid cell).
        alpha_s:    Solid thermal diffusivity (lattice units).
        k_ratio:    Conductivity ratio k_s / k_f.
        Q_source:   Uniform volumetric heat source in solid cells.

    Returns:
        ``(T_fluid_updated, T_solid_updated)``.
    """
    ndim = T_solid.ndim

    # --- Solid diffusion step ---
    if ndim == 2:
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=T_solid.dtype, device=T_solid.device,
        ).view(1, 1, 3, 3)
        T_4d = T_solid.unsqueeze(0).unsqueeze(0)
        lap = F.conv2d(T_4d, kernel, padding=1).squeeze(0).squeeze(0)
    elif ndim == 3:
        # 3-D 7-point Laplacian
        lap = (
            torch.roll(T_solid, 1, 0) + torch.roll(T_solid, -1, 0)
            + torch.roll(T_solid, 1, 1) + torch.roll(T_solid, -1, 1)
            + torch.roll(T_solid, 1, 2) + torch.roll(T_solid, -1, 2)
            - 6.0 * T_solid
        )
    else:
        raise ValueError(f"T_solid must be 2-D or 3-D, got {ndim}-D")

    T_s_new = T_solid + alpha_s * lap
    if Q_source != 0.0:
        T_s_new = T_s_new + Q_source
    T_s_new = torch.where(mask_solid, T_s_new, T_solid)

    # --- Interface coupling ---
    # Detect fluid cells adjacent to solid
    solid_f = mask_solid.float()
    if ndim == 2:
        neighbour_kernel = torch.ones(1, 1, 3, 3, device=mask_solid.device, dtype=torch.float32)
        neighbour_kernel[0, 0, 1, 1] = 0.0
        solid_4d = solid_f.unsqueeze(0).unsqueeze(0)
        neighbour_solid = F.conv2d(solid_4d, neighbour_kernel, padding=1).squeeze(0).squeeze(0) > 0.0
    else:
        neighbour_solid = (
            (torch.roll(solid_f, 1, 0) + torch.roll(solid_f, -1, 0)
             + torch.roll(solid_f, 1, 1) + torch.roll(solid_f, -1, 1)
             + torch.roll(solid_f, 1, 2) + torch.roll(solid_f, -1, 2))
            > 0.0
        )

    is_interface_fluid = neighbour_solid & ~mask_solid
    is_interface_solid = neighbour_solid & mask_solid

    w = k_ratio / (1.0 + k_ratio)
    T_int = (1.0 - w) * T_fluid + w * T_s_new

    T_fluid_new = torch.where(is_interface_fluid, T_int, T_fluid)
    T_solid_new = torch.where(is_interface_solid, T_int, T_s_new)

    return T_fluid_new, T_solid_new
