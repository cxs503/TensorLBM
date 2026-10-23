"""3D thermal lattice Boltzmann model using a D3Q7 temperature lattice.

Implements a double-distribution-function (DDF) thermal LBM in which the
hydrodynamics are solved on D3Q19 and the temperature field is solved on a
passive-scalar D3Q7 lattice following He et al. (1998) and Peng et al. (2003).
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .boundaries3d import bounce_back_cells_3d
from .d3q19 import C as C3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .solver3d import collide_bgk3d, stream3d

__all__ = [
    "C_D3Q7",
    "W_D3Q7",
    "ThermalCavity3DConfig",
    "equilibrium_thermal_3d",
    "collide_thermal_bgk_3d",
    "stream_thermal_3d",
    "macroscopic_thermal_3d",
    "apply_temperature_boundaries_3d",
    "apply_buoyancy_force_3d",
    "run_thermal_cavity_3d",
]

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

_stream_thermal_3d_cache: dict[
    tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


@functools.cache
def _c_thermal_3d(device: torch.device) -> torch.Tensor:
    return C_D3Q7.to(device)


@functools.cache
def _w_thermal_3d(device: torch.device) -> torch.Tensor:
    return W_D3Q7.to(device)


@functools.cache
def _d3q19_constants(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return C3D.to(device).float(), W3D.to(device).float()


def equilibrium_thermal_3d(
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute the D3Q7 equilibrium temperature distribution.

    Args:
        T: Temperature field of shape ``(nz, ny, nx)``.
        ux: x-velocity field of shape ``(nz, ny, nx)``.
        uy: y-velocity field of shape ``(nz, ny, nx)``.
        uz: z-velocity field of shape ``(nz, ny, nx)``.

    Returns:
        Equilibrium temperature distribution of shape ``(7, nz, ny, nx)``.
    """
    device = T.device
    c = _c_thermal_3d(device)
    w = _w_thermal_3d(device).view(7, 1, 1, 1)
    cx = c[:, 0].view(7, 1, 1, 1).float()
    cy = c[:, 1].view(7, 1, 1, 1).float()
    cz = c[:, 2].view(7, 1, 1, 1).float()
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    return w * T.unsqueeze(0) * (1.0 + 4.0 * cu)


def collide_thermal_bgk_3d(
    g: torch.Tensor,
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau_T: float,
) -> torch.Tensor:
    """Apply the BGK collision operator to the D3Q7 temperature field.

    Args:
        g: Temperature distribution of shape ``(7, nz, ny, nx)``.
        T: Temperature field of shape ``(nz, ny, nx)``.
        ux: x-velocity field of shape ``(nz, ny, nx)``.
        uy: y-velocity field of shape ``(nz, ny, nx)``.
        uz: z-velocity field of shape ``(nz, ny, nx)``.
        tau_T: Thermal relaxation time.

    Returns:
        Post-collision temperature distribution of shape ``(7, nz, ny, nx)``.
    """
    geq = equilibrium_thermal_3d(T, ux, uy, uz)
    return g - (g - geq) / tau_T


def stream_thermal_3d(g: torch.Tensor) -> torch.Tensor:
    """Stream the D3Q7 temperature distribution with periodic indexing.

    Args:
        g: Temperature distribution of shape ``(7, nz, ny, nx)``.

    Returns:
        Streamed distribution of the same shape.
    """
    nz, ny, nx = g.shape[1], g.shape[2], g.shape[3]
    device = g.device
    c = _c_thermal_3d(device)

    cache_key = (nz, ny, nx, device.type, device.index)
    if cache_key not in _stream_thermal_3d_cache:
        z_src = (torch.arange(nz, device=device).unsqueeze(0) - c[:, 2].unsqueeze(1)) % nz
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(7, device=device).view(7, 1, 1, 1).expand(7, nz, ny, nx)
        z_idx = z_src.view(7, nz, 1, 1).expand(7, nz, ny, nx)
        y_idx = y_src.view(7, 1, ny, 1).expand(7, nz, ny, nx)
        x_idx = x_src.view(7, 1, 1, nx).expand(7, nz, ny, nx)
        _stream_thermal_3d_cache[cache_key] = (q_idx, z_idx, y_idx, x_idx)

    q_idx, z_idx, y_idx, x_idx = _stream_thermal_3d_cache[cache_key]
    return g[q_idx, z_idx, y_idx, x_idx]


def macroscopic_thermal_3d(g: torch.Tensor) -> torch.Tensor:
    """Recover the macroscopic temperature field.

    Args:
        g: Temperature distribution of shape ``(7, nz, ny, nx)``.

    Returns:
        Temperature field ``T = Σ_i g_i`` with shape ``(nz, ny, nx)``.
    """
    return g.sum(dim=0)


def apply_buoyancy_force_3d(
    f: torch.Tensor,
    T: torch.Tensor,
    T_ref: float,
    beta: float,
    g_y: float = -1.0,
) -> torch.Tensor:
    """Apply a Boussinesq buoyancy force to a D3Q19 distribution field.

    Args:
        f: D3Q19 distribution tensor of shape ``(19, nz, ny, nx)``.
        T: Temperature field of shape ``(nz, ny, nx)``.
        T_ref: Reference temperature.
        beta: Thermal expansion coefficient.
        g_y: Dimensionless gravitational acceleration in the y-direction.

    Returns:
        Updated D3Q19 distribution tensor of shape ``(19, nz, ny, nx)``.
    """
    c, w = _d3q19_constants(f.device)
    rho, _, _, _ = macroscopic3d(f)
    F_y = -rho * beta * (T - T_ref) * g_y
    cy = c[:, 1].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    return f + w_view * 3.0 * cy * F_y.unsqueeze(0)


@dataclass(frozen=True)
class ThermalCavity3DConfig:
    """Configuration for the 3D differentially heated cavity benchmark.

    Args:
        nx: Number of lattice nodes in x.
        ny: Number of lattice nodes in y.
        nz: Number of lattice nodes in z.
        ra: Rayleigh number.
        pr: Prandtl number.
        n_steps: Number of time steps to run.
        device: Torch device string.
    """

    nx: int = 32
    ny: int = 32
    nz: int = 32
    ra: float = 1e4
    pr: float = 0.71
    n_steps: int = 500
    device: str = "cpu"


def _thermal_wall_mask(nz: int, ny: int, nx: int, device: torch.device) -> torch.Tensor:
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall_mask[:, :, 0] = True
    wall_mask[:, :, -1] = True
    wall_mask[:, 0, :] = True
    wall_mask[:, -1, :] = True
    wall_mask[0, :, :] = True
    wall_mask[-1, :, :] = True
    return wall_mask


def _apply_temperature_boundaries_3d(
    g: torch.Tensor,
    T_hot: float,
    T_cold: float,
) -> torch.Tensor:
    """Apply fixed-temperature and insulated boundaries to a D3Q7 field.

    Adiabatic walls (y = 0/ny-1, z = 0/nz-1): half-way bounce-back on the
    wall-normal populations, unwrapping the wrap-around of the periodic
    ``stream_thermal_3d``.  The q4 population leaving y=0 reappears at
    y=ny-1 (post-stream ``g[4, :, -1, :]``), so it is read back from there
    and reflected into q3 at y=0 (symmetrically for the other three walls).
    Each wall pair exchanges energy in exact antisymmetry, so the net heat
    flux through the adiabatic walls is identically zero.  The pre-fix rule
    copied the whole distribution of the adjacent interior slab -- a
    zero-gradient condition referencing interior arrivals instead of the
    wall's own outgoing populations -- which acted as a spurious heat
    source/sink (2026-09 evidence, 32^3 fp64 budget: ~1e-2/step already in
    pure diffusion with wall-normal structure, -7.8e-3/step in developed
    Ra=1e3 convection, -0.46/step at Ra=1e4, worsening under refinement).

    Isothermal walls (x = 0 / x = nx-1): the whole slab is frozen to the
    u=0 equilibrium ``w_i * T_wall`` (wall on the node), and the
    fluid-pointing population (q1 at the hot wall, q2 at the cold wall)
    receives the non-equilibrium carried by the first fluid column, taken
    zero-sum from the resting population q0: the slab sum stays
    ``sum(w) * T_wall`` while the effective isothermal surface sits on the
    wall node (2D analogue: PR #300).
    """
    w = W_D3Q7.to(device=g.device, dtype=g.dtype)
    w1, w2 = float(w[1]), float(w[2])
    # non-equilibrium carried by the pointing population at the first fluid
    # column (read from the pre-BC field; that column is untouched by every
    # wall rule below)
    T1 = g[:, :, :, 1].sum(dim=0)
    Tn2 = g[:, :, :, -2].sum(dim=0)
    neq_left = g[1, :, :, 1] - w1 * T1
    neq_right = g[2, :, :, -2] - w2 * Tn2
    g = g.clone()
    # unwrap: the wall's own outgoing distribution after the periodic stream
    # wrapped it around to the opposite wall -- snapshot before any write
    refl_z0 = g[6, -1, :, :].clone()  # = g_pre[6, 0, :, :]   (z=0, -z)
    refl_zn = g[5, 0, :, :].clone()  # = g_pre[5, nz-1, :, :] (+z)
    refl_y0 = g[4, :, -1, :].clone()  # = g_pre[4, :, 0, :]   (y=0, -y)
    refl_yn = g[3, :, 0, :].clone()  # = g_pre[3, :, ny-1, :] (+y)
    # half-way bounce-back reflection back onto the emitting wall
    g[5, 0, :, :] = refl_z0
    g[6, -1, :, :] = refl_zn
    g[3, :, 0, :] = refl_y0
    g[4, :, -1, :] = refl_yn
    # isothermal slabs: u=0 equilibrium for every population
    g[:, :, :, 0] = w.view(7, 1, 1) * T_hot
    g[:, :, :, -1] = w.view(7, 1, 1) * T_cold
    # pull the effective isothermal surface onto the wall node: give the
    # fluid-pointing population the bulk non-equilibrium it carries in
    # steady conduction (w * tau_T * dT/dx), taken zero-sum from q0
    g[1, :, :, 0] = g[1, :, :, 0] + neq_left
    g[0, :, :, 0] = g[0, :, :, 0] - neq_left
    g[2, :, :, -1] = g[2, :, :, -1] + neq_right
    g[0, :, :, -1] = g[0, :, :, -1] - neq_right
    return g


# physics 命名空间兼容公共入口（与 2D thermal.apply_temperature_boundaries 同构）
apply_temperature_boundaries_3d = _apply_temperature_boundaries_3d


def run_thermal_cavity_3d(config: ThermalCavity3DConfig) -> dict[str, object]:
    """Run a small 3D differentially heated cavity benchmark.

    Args:
        config: Thermal cavity configuration.

    Returns:
        Dictionary containing the average hot-wall Nusselt number and config.
    """
    device = torch.device(config.device)
    nz, ny, nx = config.nz, config.ny, config.nx
    T_hot = 1.5
    T_cold = 0.5
    delta_T = T_hot - T_cold
    tau_T = 0.8
    alpha = (tau_T - 0.5) / 3.0
    nu = alpha * config.pr
    tau = 3.0 * nu + 0.5
    length = max(float(nx - 1), 1.0)
    beta = config.ra * nu * alpha / (length**3 * delta_T)

    wall_mask = _thermal_wall_mask(nz, ny, nx, device)
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
    T = T_hot - delta_T * x_idx / max(float(nx - 1), 1.0)
    f = equilibrium3d(rho, ux, uy, uz)
    g = equilibrium_thermal_3d(T, ux, uy, uz)
    g = _apply_temperature_boundaries_3d(g, T_hot=T_hot, T_cold=T_cold)

    for _ in range(config.n_steps):
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(wall_mask, 0.0)
        uy = uy.masked_fill(wall_mask, 0.0)
        uz = uz.masked_fill(wall_mask, 0.0)

        T = macroscopic_thermal_3d(g)
        g = collide_thermal_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        g = stream_thermal_3d(g)
        g = _apply_temperature_boundaries_3d(g, T_hot=T_hot, T_cold=T_cold)
        T = macroscopic_thermal_3d(g)

        f = apply_buoyancy_force_3d(f, T, T_ref=1.0, beta=beta)
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)
        f = bounce_back_cells_3d(f, wall_mask)

    T = macroscopic_thermal_3d(g)
    grad_hot = (T[:, :, 0] - T[:, :, 1]) * length / delta_T
    nusselt = float(grad_hot.mean().item())
    return {"nusselt": nusselt, "config": asdict(config)}
