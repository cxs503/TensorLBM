"""CSF (Continuum Surface Force) model for surface tension in LBM.

Implements the Brackbill et al. (1992) CSF model where surface tension is
treated as a volumetric force concentrated at the interface:

    F_sigma = sigma * kappa * delta_s * n_hat

where:
    sigma = surface tension coefficient
    kappa = interface curvature
    delta_s = interface delta function (approximated by |grad alpha|)
    n_hat = interface normal

For LBM, the force is incorporated via the Guo forcing scheme:
    S_i = w_i * (3 * (c_i . F) + 9/2 * (c_i . u)(c_i . F) - 3/2 * (u . F)) / cs^4

References
----------
- Brackbill, Kothe & Zemach (1992) "A continuum surface force method for
  modeling surface tension", JCP 100, 335
- Guo, Zheng & Shi (2002) "Discrete lattice effects on the forcing term in
  the lattice Boltzmann method", PRE 65, 046308
- Tölke, Frey & Krafczyk (2006) "Simulation of flow in complex geometries
  using the lattice Boltzmann method with surface tension", IJMF
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

from .d2q9 import C as C2Q9
from .vof_interface_capture import compute_interface_normal, compute_curvature

__all__ = [
    "csf_force",
    "csf_force_lbm",
    "guo_force_term",
    "surface_tension_step",
    "capillary_time_step",
    "laplace_pressure_test",
]

_CS2 = 1.0 / 3.0  # cs^2 for D2Q9


def csf_force(
    alpha: torch.Tensor,
    sigma: float,
    periodic: bool = False,
    curvature_method: str = "height_function",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CSF surface tension force from volume fraction field.

    F = sigma * kappa * |grad alpha| * n_hat

    Args:
        alpha: Volume fraction field (ny, nx)
        sigma: Surface tension coefficient
        periodic: Periodic boundary conditions
        curvature_method: Method for curvature computation

    Returns:
        (fx, fy): Force components (ny, nx)
    """
    # Compute interface normal
    n_x, n_y = compute_interface_normal(alpha, periodic, smooth=True)

    # Compute curvature
    kappa = compute_curvature(alpha, periodic, method=curvature_method)

    # Interface delta function: |grad alpha|
    if periodic:
        dalpha_dx = (torch.roll(alpha, -1, dims=1) - torch.roll(alpha, 1, dims=1)) / 2.0
        dalpha_dy = (torch.roll(alpha, -1, dims=0) - torch.roll(alpha, 1, dims=0)) / 2.0
    else:
        dalpha_dx = torch.zeros_like(alpha)
        dalpha_dy = torch.zeros_like(alpha)
        dalpha_dx[:, 1:-1] = (alpha[:, 2:] - alpha[:, :-2]) / 2.0
        dalpha_dy[1:-1, :] = (alpha[2:, :] - alpha[:-2, :]) / 2.0

    delta_s = torch.sqrt(dalpha_dx**2 + dalpha_dy**2 + 1e-12)

    # CSF force
    fx = sigma * kappa * delta_s * n_x
    fy = sigma * kappa * delta_s * n_y

    return fx, fy


def csf_force_lbm(
    alpha: torch.Tensor,
    sigma: float,
    rho: torch.Tensor,
    periodic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute CSF force in lattice units, scaled by density.

    For two-phase flows, the force per unit volume is:
        F = sigma * kappa * n_hat * delta_s

    The acceleration is F / rho.

    Args:
        alpha: Volume fraction field
        sigma: Surface tension coefficient (lattice units)
        rho: Density field
        periodic: Periodic boundary conditions

    Returns:
        (ax, ay): Acceleration components
    """
    fx, fy = csf_force(alpha, sigma, periodic)

    # Acceleration = force / density
    rho_safe = torch.clamp(rho, min=1e-6)
    ax = fx / rho_safe
    ay = fy / rho_safe

    return ax, ay


def guo_force_term(
    f: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute Guo forcing term for LBM.

    S_i = w_i * [3(c_i.F) + 9/2(c_i.u)(c_i.F) - 3/2(u.F)] / cs^4

    With cs^2 = 1/3 for D2Q9, cs^4 = 1/9.

    Args:
        f: Distribution function (9, ny, nx)
        u: x-velocity (ny, nx)
        v: y-velocity (ny, nx)
        fx: x-force (ny, nx)
        fy: y-force (ny, nx)
        tau: Relaxation time

    Returns:
        Source term S_i (9, ny, nx)
    """
    device = f.device
    dtype = f.dtype

    # D2Q9 velocities and weights
    c = C2Q9.to(device=device, dtype=dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                     device=device, dtype=dtype).view(9, 1, 1)

    # c_i . F
    c_dot_F = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0)

    # c_i . u
    c_dot_u = cx * u.unsqueeze(0) + cy * v.unsqueeze(0)

    # u . F
    u_dot_F = (u * fx + v * fy).unsqueeze(0)

    # Guo source term
    S = w * (3.0 * c_dot_F + 4.5 * c_dot_u * c_dot_F - 1.5 * u_dot_F)

    # Scale by (1 - 1/(2*tau))
    S = (1.0 - 0.5 / tau) * S

    return S


def surface_tension_step(
    f: torch.Tensor,
    alpha: torch.Tensor,
    sigma: float,
    tau: float,
    periodic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply surface tension force to LBM distribution.

    This function computes the CSF force and adds the Guo forcing term
    to the distribution function.

    Args:
        f: Distribution function (9, ny, nx)
        alpha: Volume fraction field (ny, nx)
        sigma: Surface tension coefficient
        tau: Relaxation time
        periodic: Periodic boundary conditions

    Returns:
        (f_new, alpha_new): Updated distribution and volume fraction
    """
    # Compute macroscopic velocity
    c = C2Q9.to(device=f.device, dtype=f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-6)
    u = (f * cx).sum(dim=0) / rho_safe
    v = (f * cy).sum(dim=0) / rho_safe

    # Compute CSF force
    fx, fy = csf_force(alpha, sigma, periodic)

    # Add Guo forcing term
    S = guo_force_term(f, u, v, fx, fy, tau)
    f_new = f + S

    return f_new, alpha


def capillary_time_step(
    sigma: float,
    rho: float,
    dx: float,
    cfl: float = 0.5,
) -> float:
    """Compute capillary time step constraint.

    dt_cap = cfl * sqrt(rho * dx^3 / sigma)

    This is the capillary time step limit for explicit surface tension.

    Args:
        sigma: Surface tension coefficient
        rho: Fluid density
        dx: Grid spacing
        cfl: CFL number (default 0.5)

    Returns:
        Maximum stable time step
    """
    return cfl * (rho * dx**3 / sigma)**0.5


def laplace_pressure_test(
    R: float,
    sigma: float,
    rho_in: float = 1.0,
    rho_out: float = 0.1,
) -> float:
    """Compute analytical Laplace pressure jump for a circular droplet.

    Delta_p = sigma / R (2D) or 2*sigma/R (3D)

    Args:
        R: Droplet radius
        sigma: Surface tension coefficient
        rho_in: Inner density
        rho_out: Outer density

    Returns:
        Pressure jump Delta_p
    """
    # 2D: Delta_p = sigma / R
    return sigma / R


def test_surface_tension_conservation(
    f: torch.Tensor,
    alpha: torch.Tensor,
    sigma: float,
    tau: float,
) -> dict:
    """Test conservation properties of surface tension model.

    Checks:
    1. Total mass conservation
    2. Total momentum conservation (in absence of external forces)
    3. Volume fraction bounded in [0, 1]

    Args:
        f: Distribution function
        alpha: Volume fraction field
        sigma: Surface tension coefficient
        tau: Relaxation time

    Returns:
        Dictionary of conservation metrics
    """
    # Mass
    mass_before = float(f.sum())

    # Apply surface tension
    f_new, alpha_new = surface_tension_step(f, alpha, sigma, tau)

    mass_after = float(f_new.sum())

    # Momentum
    c = C2Q9.to(device=f.device, dtype=f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    jx_before = float((f * cx).sum())
    jy_before = float((f * cy).sum())
    jx_after = float((f_new * cx).sum())
    jy_after = float((f_new * cy).sum())

    # Volume fraction bounds
    alpha_min = float(alpha_new.min())
    alpha_max = float(alpha_new.max())

    return {
        "mass_before": mass_before,
        "mass_after": mass_after,
        "mass_drift": mass_after - mass_before,
        "mass_drift_rel": (mass_after - mass_before) / max(abs(mass_before), 1e-10),
        "jx_before": jx_before,
        "jx_after": jx_after,
        "jy_before": jy_before,
        "jy_after": jy_after,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_bounded": 0.0 <= alpha_min and alpha_max <= 1.0,
    }
