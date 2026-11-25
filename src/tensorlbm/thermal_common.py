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
import math
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
    # Boundary-condition helpers
    "thermal_dirichlet_wall_3d",
    "thermal_adiabatic_walls_3d",
    "thermal_flux_wall_pair_3d",
    "thermal_fixed_temp_mask_3d",
    # Diagnostics
    "nusselt_hot_wall_3d",
    "wall_heat_rate_3d",
    "nusselt_cylinder_3d",
    "heat_flux_at_interface_3d",
    # Benchmark runners (common-module integrated)
    "run_thermal_cavity_common",
    "run_heated_cylinder_common",
    "run_conjugate_ht_common",
    "run_rayleigh_benard_common",
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
        from .d3q19 import C as C3D
        from .d3q19 import W as W3D
        from .d3q19 import macroscopic3d

        q = 19
        c = C3D.to(f.device).float()
        w = W3D.to(f.device).float()
        rho, _, _, _ = macroscopic3d(f)
    elif lattice_u == "D3Q27":
        from .d3q27 import C as C27
        from .d3q27 import W as W27
        from .d3q27 import macroscopic27

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
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=T_solid.dtype,
            device=T_solid.device,
        ).view(1, 1, 3, 3)
        T_4d = T_solid.unsqueeze(0).unsqueeze(0)
        lap = F.conv2d(T_4d, kernel, padding=1).squeeze(0).squeeze(0)
    elif ndim == 3:
        # 3-D 7-point Laplacian
        lap = (
            torch.roll(T_solid, 1, 0)
            + torch.roll(T_solid, -1, 0)
            + torch.roll(T_solid, 1, 1)
            + torch.roll(T_solid, -1, 1)
            + torch.roll(T_solid, 1, 2)
            + torch.roll(T_solid, -1, 2)
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
        neighbour_solid = (
            F.conv2d(solid_4d, neighbour_kernel, padding=1).squeeze(0).squeeze(0) > 0.0
        )
    else:
        neighbour_solid = (
            torch.roll(solid_f, 1, 0)
            + torch.roll(solid_f, -1, 0)
            + torch.roll(solid_f, 1, 1)
            + torch.roll(solid_f, -1, 1)
            + torch.roll(solid_f, 1, 2)
            + torch.roll(solid_f, -1, 2)
        ) > 0.0

    is_interface_fluid = neighbour_solid & ~mask_solid
    is_interface_solid = neighbour_solid & mask_solid

    w = k_ratio / (1.0 + k_ratio)
    T_int = (1.0 - w) * T_fluid + w * T_s_new

    T_fluid_new = torch.where(is_interface_fluid, T_int, T_fluid)
    T_solid_new = torch.where(is_interface_solid, T_int, T_s_new)

    return T_fluid_new, T_solid_new


# ---------------------------------------------------------------------------
# Thermal boundary-condition helpers (D3Q7)
# ---------------------------------------------------------------------------


def thermal_dirichlet_wall_3d(
    g: torch.Tensor,
    T_wall: float,
    wall: str,
    *, method: str = "equilibrium",
) -> torch.Tensor:
    """Apply a Dirichlet (fixed-temperature) condition on one domain wall.

    Apply after streaming. The default resets all populations to the
    zero-velocity wall equilibrium. For pure diffusion at a uniform wall,
    "odd_reflection" preserves the population arriving from the interior
    and mirrors its departure from equilibrium into the opposite direction.
    This places the prescribed temperature on the wall node.

    Args:
        g:       Thermal distribution, shape ``(7, nz, ny, nx)``.
        T_wall:  Wall temperature.
        wall:    One of ``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, ``"z-"``, ``"z+"``.

    Returns:
        Updated distribution (same shape).
    """
    if wall not in ("x-", "x+", "y-", "y+", "z-", "z+"):
        raise ValueError(f"wall must be x-/x+/y-/y+/z-/z+, got {wall!r}")
    if method not in ("equilibrium", "odd_reflection"):
        raise ValueError("method must be equilibrium or odd_reflection")
    axis = "xyz".index(wall[0])
    index = 0 if wall[1] == "-" else -1
    g_new = g.clone()
    face = g_new.select(3-axis, index)
    w = _w_thermal(g.device).to(g.dtype)
    face[:] = (w*T_wall)[:, None, None]
    if method == "odd_reflection":
        positive = 1+2*axis
        arriving, reflected = (positive+1, positive) if index == 0 else (positive, positive+1)
        incoming_value = g.select(3-axis, index)[arriving]
        face[arriving] = incoming_value
        face[reflected] = 2*w[arriving]*T_wall - incoming_value
    return g_new


def thermal_adiabatic_walls_3d(
    g: torch.Tensor, walls: list[str], *, method: str = "copy",
) -> torch.Tensor:
    """Apply insulating conditions after streaming a D3Q7 field.

    "copy" copies the neighboring interior slab onto each selected face.
    For pure diffusion, "even_reflection" places the zero-flux plane on the
    wall nodes. It mirrors the population arriving from inside the domain
    into the opposite direction. Each axis updates its own normal pair,
    including at edges and corners shared by multiple insulating walls.
    """
    if method not in ("copy", "even_reflection"):
        raise ValueError("method must be copy or even_reflection")
    g_new = g.clone()
    for wall in walls:
        if wall not in ("x-", "x+", "y-", "y+", "z-", "z+"):
            raise ValueError(f"wall must be x-/x+/y-/y+/z-/z+, got {wall!r}")
        axis = "xyz".index(wall[0])
        index, neighbour = (0, 1) if wall[1] == "-" else (-1, -2)
        face = g_new.select(3-axis, index)
        if method == "copy":
            face[:] = g.select(3-axis, neighbour)
        else:
            positive = 1+2*axis
            arriving, reflected = (positive+1, positive) if index == 0 else (positive, positive+1)
            face[reflected] = g.select(3-axis, index)[arriving]
    return g_new


def thermal_flux_wall_pair_3d(
    g: torch.Tensor, axis: str, inward_flux: tuple[float | torch.Tensor, float | torch.Tensor], *,
    dt: float = 1., spacing: float = 1., volumetric_heat_capacity: float = 1.,
) -> torch.Tensor:
    """Impose inward heat fluxes on two opposite half-way walls.

    Apply immediately after periodic streaming in a cell-centered D3Q7
    diffusion solve. The two fluxes (lower, upper) are positive into the
    domain in W/m². Each flux may be a scalar or a real tensor with the exact
    face shape in the remaining (z, y, x) dimension order. For time-varying
    loads, supply the average inward flux over the streaming step. Supply dt
    in seconds, normal cell spacing in meters and volumetric heat capacity
    in J/(m³ K), or use consistent lattice units. Each incoming population reflects the outgoing one from its own
    face and adds flux*dt/(spacing*capacity) to the boundary cell.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    if g.ndim != 4 or g.shape[0] != 7 or min(g.shape[1:]) < 1 or not g.is_floating_point():
        raise ValueError("provide real floating D3Q7 populations shaped (7,nz,ny,nx)")
    if len(inward_flux) != 2:
        raise ValueError("provide two inward fluxes")
    if not all(math.isfinite(v) and v > 0 for v in (dt, spacing, volumetric_heat_capacity)):
        raise ValueError("dt, spacing and volumetric heat capacity must be positive and finite")
    direction = "xyz".index(axis)
    positive = 1+2*direction
    negative = positive+1
    dim = 2-direction
    face_shape = g[positive].select(dim, 0).shape
    fluxes = []
    for flux in inward_flux:
        if isinstance(flux, torch.Tensor):
            if flux.is_complex() or (flux.ndim and flux.shape != face_shape):
                raise ValueError(f"expected a real scalar or flux tensor of shape {tuple(face_shape)}")
            flux = flux.to(device=g.device, dtype=g.dtype)
            finite = torch.isfinite(flux).all().item()
        else:
            finite = math.isfinite(flux)
        if not finite:
            raise ValueError("inward flux must be finite")
        fluxes.append(flux)
    scale = dt/(spacing*volumetric_heat_capacity)
    updated = g.clone()
    # Periodic streaming put each outgoing population on the opposite face.
    updated[positive].select(dim,0).copy_(g[negative].select(dim,-1)+fluxes[0]*scale)
    updated[negative].select(dim,-1).copy_(g[positive].select(dim,0)+fluxes[1]*scale)
    return updated


def thermal_fixed_temp_mask_3d(
    g: torch.Tensor,
    T_fixed: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Set a fixed temperature on arbitrary cells (e.g. cylinder surface).

    Args:
        g:       Thermal distribution, shape ``(7, nz, ny, nx)``.
        T_fixed: Temperature to impose.
        mask:    Boolean mask ``(nz, ny, nx)`` of fixed-T cells.

    Returns:
        Updated distribution (same shape).
    """
    T_field = torch.full_like(g[0], T_fixed)
    geq = thermal_equilibrium_3d(
        T_field,
        torch.zeros_like(T_field),
        torch.zeros_like(T_field),
        torch.zeros_like(T_field),
    )
    return torch.where(mask.unsqueeze(0), geq, g)


# ---------------------------------------------------------------------------
# Diagnostics: Nusselt number and heat flux
# ---------------------------------------------------------------------------


def _wall_temperature_difference(T, wall, order):
    axes = {"x": 2, "y": 1, "z": 0}
    if wall not in ("x-", "x+", "y-", "y+", "z-", "z+"):
        raise ValueError(f"unknown hot wall {wall!r}")
    normal = T.movedim(axes[wall[0]], -1)
    L = normal.shape[-1] - 1
    if L < order:
        raise ValueError(f"wall-normal direction needs at least {order + 1} nodes")
    boundary, first, second = (0, 1, 2) if wall[1] == "-" else (-1, -2, -3)
    grad = normal[..., boundary] - normal[..., first]
    if order == 2:
        grad = grad + 0.5 * (grad - (normal[..., first] - normal[..., second]))
    return grad, axes[wall[0]]


def wall_heat_rate_3d(
    T: torch.Tensor, conductivity: float, wall: str, *,
    spacing_zyx: tuple[float, float, float], order: int = 2,
) -> float:
    """Outward conductive heat rate on one face of a wall-node Cartesian grid.

    Supply T in K with shape (nz, ny, nx), constant isotropic conductivity
    in W/(m K), and uniform spacings (dz, dy, dx) in meters. A one-sided
    normal difference and composite trapezoidal integration over the face
    give watts, positive when heat leaves the domain.
    """
    if T.ndim != 3 or min(T.shape) < 2 or not T.is_floating_point():
        raise ValueError("provide a real floating 3D temperature field with at least two nodes per axis")
    if len(spacing_zyx) != 3 or not all(math.isfinite(h) and h > 0 for h in spacing_zyx):
        raise ValueError("provide three positive finite spacings in (dz, dy, dx) order")
    if not math.isfinite(conductivity) or conductivity < 0:
        raise ValueError("conductivity must be finite and nonnegative")
    if order not in (1, 2):
        raise ValueError("wall gradient order must be 1 or 2")
    difference, axis = _wall_temperature_difference(T, wall, order)
    tangential = [h for i,h in enumerate(spacing_zyx) if i != axis]
    flux = -conductivity*difference/spacing_zyx[axis]
    strips = torch.trapezoid(flux, dx=tangential[1], dim=1)
    return torch.trapezoid(strips, dx=tangential[0], dim=0).item()


def nusselt_hot_wall_3d(
    T: torch.Tensor,
    T_hot: float,
    T_cold: float,
    wall: str = "x-",
    *,
    order: int = 1,
) -> float:
    """Average Nusselt number at a hot wall.

    Nu = - (∂T/∂n) * L / ΔT

    Uses a one-sided finite difference at the wall.

    Args:
        T:       Temperature field ``(nz, ny, nx)``.
        T_hot:   Hot wall temperature.
        T_cold:  Cold wall temperature.
        wall:    Hot face (x, y or z followed by - or +).
        order:   Wall-normal difference order on a uniform, wall-node grid (1 or 2).

    Returns:
        Average Nusselt number (float).
    """
    if order not in (1, 2):
        raise ValueError("wall gradient order must be 1 or 2")
    dT = T_hot - T_cold
    if abs(dT) < 1e-12:
        return 0.0
    grad, axis = _wall_temperature_difference(T, wall, order)
    L = T.shape[axis] - 1
    nu = float((grad * L / dT).mean().item())
    return nu


def nusselt_cylinder_3d(
    T: torch.Tensor,
    solid: torch.Tensor,
    T_cyl: float,
    T_inf: float,
    alpha: float,
    D: float,
) -> float:
    """Average Nusselt number around a heated cylinder.

    Nu = - (∂T/∂n) * D / (T_cyl - T_inf)

    Approximates the normal temperature gradient at solid-fluid interface
    cells using the nearest-fluid-cell temperature difference.

    Args:
        T:      Temperature field ``(nz, ny, nx)``.
        solid:  Boolean solid mask.
        T_cyl:  Cylinder surface temperature.
        T_inf:  Free-stream temperature.
        alpha:  Thermal diffusivity (lattice units).
        D:      Cylinder diameter (lattice units).

    Returns:
        Average Nusselt number (float).
    """
    dT = T_cyl - T_inf
    if abs(dT) < 1e-12:
        return 0.0
    # Find interface fluid cells (fluid cells adjacent to solid)
    solid_f = solid.float()
    neighbour_solid = (
        torch.roll(solid_f, 1, 2)
        + torch.roll(solid_f, -1, 2)
        + torch.roll(solid_f, 1, 1)
        + torch.roll(solid_f, -1, 1)
    ) > 0.0
    is_interface = neighbour_solid & ~solid
    if not is_interface.any():
        return 0.0
    # Temperature gradient magnitude at interface (central diff approx)
    dTdx = (torch.roll(T, -1, 2) - torch.roll(T, 1, 2)) / 2.0
    dTdy = (torch.roll(T, -1, 1) - torch.roll(T, 1, 1)) / 2.0
    grad_mag = torch.sqrt(dTdx**2 + dTdy**2)
    # Only interface cells
    grad_interface = grad_mag[is_interface]
    nu = float((grad_interface * D / dT).mean().item())
    return nu


def heat_flux_at_interface_3d(
    T: torch.Tensor,
    solid: torch.Tensor,
    alpha_f: float,
    alpha_s: float,
) -> dict[str, float]:
    """Compute heat flux on both sides of the fluid-solid interface.

    Returns the fluid-side and solid-side heat flux magnitudes and their
    relative difference (continuity error).

    Args:
        T:       Temperature field ``(nz, ny, nx)``.
        solid:   Boolean solid mask.
        alpha_f: Fluid thermal diffusivity.
        alpha_s: Solid thermal diffusivity.

    Returns:
        Dict with ``q_fluid``, ``q_solid``, ``flux_continuity_error``.
    """
    solid_f = solid.float()
    # Fluid cells adjacent to solid
    nb_solid = (
        torch.roll(solid_f, 1, 2)
        + torch.roll(solid_f, -1, 2)
        + torch.roll(solid_f, 1, 1)
        + torch.roll(solid_f, -1, 1)
    ) > 0.0
    is_fluid_int = nb_solid & ~solid
    is_solid_int = nb_solid & solid
    # Gradients
    dTdx = (torch.roll(T, -1, 2) - torch.roll(T, 1, 2)) / 2.0
    dTdy = (torch.roll(T, -1, 1) - torch.roll(T, 1, 1)) / 2.0
    grad_mag = torch.sqrt(dTdx**2 + dTdy**2)
    q_f = float((alpha_f * grad_mag[is_fluid_int]).mean().item()) if is_fluid_int.any() else 0.0
    q_s = float((alpha_s * grad_mag[is_solid_int]).mean().item()) if is_solid_int.any() else 0.0
    err = abs(q_f - q_s) / max(abs(q_f + q_s) / 2.0, 1e-12) if (q_f + q_s) > 0 else 0.0
    return {
        "q_fluid": q_f,
        "q_solid": q_s,
        "flux_continuity_error": err,
    }


# ---------------------------------------------------------------------------
# Benchmark 1: Thermal cavity (de Vahl Davis) — SDAA:28
# ---------------------------------------------------------------------------


def run_thermal_cavity_common(
    device: str | torch.device,
    nx: int = 100,
    ny: int = 100,
    nz: int = 4,
    Ra: float = 1e4,
    Pr: float = 0.71,
    n_steps: int = 8000,
    *,
    nusselt_order: int = 1,
) -> dict[str, object]:
    """Differentially heated cavity using common modules.

    Hot wall at x=0 (T=1), cold wall at x=nx-1 (T=0).
    All other walls: no-slip + adiabatic.
    Buoyancy via Boussinesq approximation. nusselt_order selects the wall
    gradient stencil used for both the recorded history and the final value.

    Reference: de Vahl Davis (1983), Nu ≈ 2.0 at Ra=10^4.
    """
    device = torch.device(device)
    T_hot, T_cold = 1.0, 0.0
    delta_T = T_hot - T_cold
    L = float(nx - 1)

    # Lattice parameters from Ra, Pr
    tau_T = 0.8
    alpha = (tau_T - 0.5) / 3.0  # thermal diffusivity
    nu = alpha * Pr  # kinematic viscosity
    tau = 3.0 * nu + 0.5  # momentum relaxation time
    beta = Ra * nu * alpha / (L**3 * delta_T)

    # Wall mask (all 6 faces)
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = wall[:, :, -1] = True
    wall[:, 0, :] = wall[:, -1, :] = True
    wall[0, :, :] = wall[-1, :, :] = True

    # Initial fields: linear T profile, zero velocity
    _, _, x_idx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rho = torch.ones((nz, ny, nx), device=device)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    T = T_hot - delta_T * x_idx / L

    from .boundaries3d import bounce_back_cells_3d
    from .d3q19 import equilibrium3d, macroscopic3d
    from .solver3d import collide_bgk3d, stream3d

    f = equilibrium3d(rho, ux, uy, uz, device=device)
    g = thermal_equilibrium_3d(T, ux, uy, uz)

    nu_hist = []
    for step in range(n_steps):
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(wall, 0.0)
        uy = uy.masked_fill(wall, 0.0)
        uz = uz.masked_fill(wall, 0.0)

        T = thermal_macroscopic_3d(g)

        # --- FIX: apply buoyancy AFTER collision with Guo factor ---
        # The old code applied buoyancy BEFORE collision, which caused the
        # collision to multiply the force by (1-1/tau).  For tau<1 this is
        # NEGATIVE, reversing the buoyancy direction (hot fluid sinks!).
        # Correct Guo forcing: Delta_f = (1-1/(2*tau)) * w*3*cy*F_y, applied
        # to the POST-collision distribution.
        f_pre = f.clone()  # save pre-collision for BB
        f = collide_bgk3d(f, tau)  # collision first
        guo_factor = 1.0 - 1.0 / (2.0 * tau)  # Guo (2002) forcing factor
        f_buoy = apply_buoyancy_3d(  # compute force increment
            f,
            T,
            T_ref=0.5,
            beta=beta,
            g_y=-1.0,
            lattice="D3Q19",
        )
        f = f + guo_factor * (f_buoy - f)  # add scaled force post-collision

        # Bounce-back + streaming (use pre-collision f for half-way BB)
        f = bounce_back_cells_3d(f, wall, f_pre=f_pre)
        f = stream3d(f)
        f = bounce_back_cells_3d(f, wall)

        # Thermal collision + streaming + BCs
        g = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        g = thermal_stream_3d(g)
        g = thermal_dirichlet_wall_3d(g, T_hot, "x-")
        g = thermal_dirichlet_wall_3d(g, T_cold, "x+")
        g = thermal_adiabatic_walls_3d(g, ["y-", "y+", "z-", "z+"])

        if step % 200 == 0 or step == n_steps - 1:
            T = thermal_macroscopic_3d(g)
            nu_val = nusselt_hot_wall_3d(T, T_hot, T_cold, wall="x-", order=nusselt_order)
            nu_hist.append(nu_val)

    T_final = thermal_macroscopic_3d(g)
    nu_final = nusselt_hot_wall_3d(T_final, T_hot, T_cold, wall="x-", order=nusselt_order)
    return {
        "nusselt": nu_final,
        "nusselt_order": nusselt_order,
        "nusselt_ref": 2.0,
        "nusselt_history": nu_hist,
        "Ra": Ra,
        "Pr": Pr,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_steps": n_steps,
        "T_field": T_final.cpu(),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Conjugate heat transfer — SDAA:30
# ---------------------------------------------------------------------------


def run_conjugate_ht_common(
    device: str | torch.device,
    nx: int = 200,
    ny: int = 80,
    nz: int = 4,
    Pr: float = 0.71,
    n_steps: int = 6000,
) -> dict[str, object]:
    """Channel flow with heated solid block using common modules.

    Fluid: Pr=0.71, Solid: thermal_diff=10*nu.
    Measures temperature at interface and heat flux continuity.

    Target: heat flux continuity error < 10%.
    """
    device = torch.device(device)
    u_in = 0.08
    D_block = 24.0
    nu = 0.02
    tau = 3.0 * nu + 0.5
    tau_T = 0.8
    alpha_f = (tau_T - 0.5) / 3.0
    alpha_s = 10.0 * nu  # solid diffusivity = 10*nu
    T_hot_block, T_inf = 1.0, 0.0

    # Solid block in the channel centre
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    block_x0, block_x1 = nx * 0.35, nx * 0.35 + D_block
    block_y0, block_y1 = ny * 0.5 - D_block / 2, ny * 0.5 + D_block / 2
    solid_block = (xx >= block_x0) & (xx < block_x1) & (yy >= block_y0) & (yy < block_y1)
    solid = solid_block.unsqueeze(0).expand(nz, ny, nx).clone()

    # Channel walls (top/bottom)
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    wall[:, -1, :] = True

    import functools

    from .boundaries3d import far_field_bc_3d
    from .d3q19 import equilibrium3d, macroscopic3d
    from .lbm_step_correct import lbm_step_correct
    from .solver3d import collide_bgk3d

    far_field_fn = functools.partial(
        far_field_bc_3d, bc_config={"far_field_faces": ["x-", "x+"], "periodic_faces": ["z-", "z+"]}
    )

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    ux0[wall] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    T0 = torch.full((nz, ny, nx), T_inf, device=device)
    T0[solid] = T_hot_block
    g = thermal_equilibrium_3d(T0, torch.zeros_like(T0), torch.zeros_like(T0), torch.zeros_like(T0))

    flux_err_hist = []
    for step in range(n_steps):
        # Momentum step
        all_solid = solid | wall
        f = lbm_step_correct(f, collide_bgk3d, tau, all_solid, u_in, far_field_fn)

        # Thermal step (fluid D3Q7)
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(all_solid, 0.0)
        uy = uy.masked_fill(all_solid, 0.0)
        uz = uz.masked_fill(all_solid, 0.0)
        T = thermal_macroscopic_3d(g)
        g = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        g = thermal_stream_3d(g)
        # Fixed T on solid block
        g = thermal_fixed_temp_mask_3d(g, T_hot_block, solid)
        # Adiabatic on channel walls
        g = thermal_adiabatic_walls_3d(g, ["y-", "y+"])

        if step % 200 == 0 or step == n_steps - 1:
            T = thermal_macroscopic_3d(g)
            flux = heat_flux_at_interface_3d(T, solid, alpha_f, alpha_s)
            flux_err_hist.append(flux["flux_continuity_error"])

    T_final = thermal_macroscopic_3d(g)
    flux_final = heat_flux_at_interface_3d(T_final, solid, alpha_f, alpha_s)
    return {
        "flux_continuity_error": flux_final["flux_continuity_error"],
        "q_fluid": flux_final["q_fluid"],
        "q_solid": flux_final["q_solid"],
        "flux_error_history": flux_err_hist,
        "Pr": Pr,
        "alpha_s": alpha_s,
        "alpha_f": alpha_f,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_steps": n_steps,
        "T_field": T_final.cpu(),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Heated cylinder — SDAA:29
# ---------------------------------------------------------------------------


def run_heated_cylinder_common(
    device: str | torch.device,
    D: float = 48.0,
    Re: float = 200.0,
    Pr: float = 0.71,
    n_steps: int = 6000,
    nx: int = 320,
    ny: int = 160,
    nz: int = 4,
) -> dict[str, object]:
    """Heated cylinder in cross-flow using common modules.

    Cylinder at T=1, free-stream fluid at T=0.
    Far-field BC for momentum, Dirichlet T on cylinder surface.

    Reference: Kruger (2017), Nu ≈ 6.5 at Re=200, Pr=0.71.
    """
    device = torch.device(device)
    R = D / 2.0
    cx, cy = nx * 0.25, ny * 0.5
    # Ensure cylinder fits within domain (avoid SDAA crash from OOB access)
    cx = max(cx, R + 2.0)
    cy = max(cy, R + 2.0)
    cy = min(cy, ny - R - 2.0)
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    tau_T = 0.8
    alpha = (tau_T - 0.5) / 3.0
    T_cyl, T_inf = 1.0, 0.0

    # Cylinder solid mask
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= R**2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()

    import functools

    from .boundaries3d import far_field_bc_3d
    from .d3q19 import equilibrium3d, macroscopic3d
    from .lbm_step_correct import lbm_step_correct
    from .solver3d import collide_bgk3d

    far_field_fn = functools.partial(
        far_field_bc_3d, bc_config={"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    )

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    T0 = torch.full((nz, ny, nx), T_inf, device=device)
    T0[solid] = T_cyl
    g = thermal_equilibrium_3d(T0, torch.zeros_like(T0), torch.zeros_like(T0), torch.zeros_like(T0))

    nu_hist = []
    for step in range(n_steps):
        # Momentum step via common lbm_step_correct
        f = lbm_step_correct(
            f,
            collide_bgk3d,
            tau,
            solid,
            u_in,
            far_field_fn,
        )
        # Thermal step
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(solid, 0.0)
        uy = uy.masked_fill(solid, 0.0)
        uz = uz.masked_fill(solid, 0.0)
        T = thermal_macroscopic_3d(g)
        g = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        g = thermal_stream_3d(g)
        # Dirichlet T on cylinder
        g = thermal_fixed_temp_mask_3d(g, T_cyl, solid)
        # Far-field T = T_inf on y walls
        g = thermal_dirichlet_wall_3d(g, T_inf, "y-")
        g = thermal_dirichlet_wall_3d(g, T_inf, "y+")

        if step % 200 == 0 or step == n_steps - 1:
            T = thermal_macroscopic_3d(g)
            nu_val = nusselt_cylinder_3d(T, solid, T_cyl, T_inf, alpha, D)
            nu_hist.append(nu_val)

    T_final = thermal_macroscopic_3d(g)
    nu_final = nusselt_cylinder_3d(T_final, solid, T_cyl, T_inf, alpha, D)
    return {
        "nusselt": nu_final,
        "nusselt_ref": 6.5,
        "nusselt_history": nu_hist,
        "Re": Re,
        "Pr": Pr,
        "D": D,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_steps": n_steps,
        "T_field": T_final.cpu(),
    }


# ---------------------------------------------------------------------------
# Benchmark 4: Rayleigh-Benard convection — SDAA:31
# ---------------------------------------------------------------------------


def run_rayleigh_benard_common(
    device: str | torch.device,
    nx: int = 100,
    ny: int = 50,
    nz: int = 4,
    Ra: float = 1e4,
    Pr: float = 0.71,
    n_steps: int = 8000,
) -> dict[str, object]:
    """Rayleigh-Benard convection using common modules.

    Hot bottom wall (y=0, T=1), cold top wall (y=ny-1, T=0).
    Periodic in x and z.
    Buoyancy via Boussinesq approximation.

    Reference: critical Ra_c=1708. At Ra=10^4 > Ra_c, convection occurs.
    Target: Nu > 1 (convection detected).
    """
    device = torch.device(device)
    T_hot, T_cold = 1.0, 0.0
    delta_T = T_hot - T_cold
    H = float(ny - 1)  # cavity height

    tau_T = 0.8
    alpha = (tau_T - 0.5) / 3.0
    nu = alpha * Pr
    tau = 3.0 * nu + 0.5
    beta = Ra * nu * alpha / (H**3 * delta_T)

    # Wall mask: only top and bottom (no-slip)
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    wall[:, -1, :] = True

    from .boundaries3d import bounce_back_cells_3d
    from .d3q19 import equilibrium3d, macroscopic3d
    from .solver3d import collide_bgk3d, stream3d

    # Initial: linear T profile + small perturbation to trigger convection
    _, y_idx, _ = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rho = torch.ones((nz, ny, nx), device=device)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    T = T_hot - delta_T * y_idx / H
    # Small random perturbation to break symmetry
    T = T + 0.01 * torch.rand_like(T)

    f = equilibrium3d(rho, ux, uy, uz, device=device)
    g = thermal_equilibrium_3d(T, ux, uy, uz)

    nu_hist = []
    for step in range(n_steps):
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(wall, 0.0)
        uy = uy.masked_fill(wall, 0.0)
        uz = uz.masked_fill(wall, 0.0)

        T = thermal_macroscopic_3d(g)
        # Buoyancy: hot fluid rises (positive y direction)
        f = apply_buoyancy_3d(f, T, T_ref=0.5, beta=beta, g_y=-1.0, lattice="D3Q19")
        # Momentum: collision + bounce-back + streaming
        f = collide_bgk3d(f, tau)
        f = bounce_back_cells_3d(f, wall, f_pre=f)
        f = stream3d(f)
        f = bounce_back_cells_3d(f, wall)

        # Thermal: collision + streaming + Dirichlet top/bottom
        g = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        g = thermal_stream_3d(g)
        g = thermal_dirichlet_wall_3d(g, T_hot, "y-")
        g = thermal_dirichlet_wall_3d(g, T_cold, "y+")
        # Periodic in x, z (streaming is already periodic)

        if step % 200 == 0 or step == n_steps - 1:
            T = thermal_macroscopic_3d(g)
            # Nu = heat flux at hot wall / (k * dT/H)
            grad_hot = T[:, 0, :] - T[:, 1, :]
            nu_val = float((-grad_hot * H / delta_T).mean().item())
            nu_hist.append(nu_val)

    T_final = thermal_macroscopic_3d(g)
    grad_hot = T_final[:, 0, :] - T_final[:, 1, :]
    nu_final = float((-grad_hot * H / delta_T).mean().item())
    convection_detected = nu_final > 1.05  # Nu > 1 means convection
    return {
        "nusselt": nu_final,
        "nusselt_ref": 1.0,  # conduction baseline
        "nusselt_history": nu_hist,
        "Ra": Ra,
        "Pr": Pr,
        "Ra_critical": 1708.0,
        "convection_detected": convection_detected,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_steps": n_steps,
        "T_field": T_final.cpu(),
    }
