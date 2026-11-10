"""Contact angle measurement for SC/D2Q9 two-phase LBM.

The contact angle is measured from the mid-density level set of a sessile
droplet on a solid wall.  The operational definition is:

    ρ(x) = ρ_m = (ρ_l + ρ_v) / 2

The contact angle θ is the angle, measured inside the liquid, between
the level set and the wall plane.

**Circle-fit method:**
For a sessile droplet that forms a spherical cap, the level set is
approximately circular.  Fitting a circle to the level set points gives
the center (x_c, y_c) and radius R.  The contact angle is then:

    cos(θ) = (y_c - y_wall) / R
    θ = arccos((y_c - y_wall) / R)

This is the Kasa (1976) algebraic circle fit, which is robust and
insensitive to point ordering.

**Wall pseudopotential control:**
In the SC model, the contact angle is controlled by the wall pseudo-
potential ψ_w.  The relationship is approximately:

    cos(θ) ≈ (ψ_w - ψ_v) / (ψ_l - ψ_v)

where ψ_l and ψ_v are the bulk liquid and vapour pseudopotentials.
This is an empirical calibration, not an exact formula.

References
----------
Kasa (1976) IEEE Trans. Instrum. Meas. 25, 8 (algebraic circle fit).
Benzi et al. (2016) Phys. Rep. 654, 1 (contact angle in pseudopotential LBM).
Huang et al. (2020) J. Comput. Phys. 407, 109234 (wetting in SC LBM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .sc_core import SCConfig, SCState, init_droplet, run_simulation


@dataclass
class ContactAngleResult:
    """Result of contact angle measurement."""
    theta_deg: float  # Contact angle in degrees
    theta_rad: float  # Contact angle in radians
    R_circle: float  # Fitted circle radius
    center: Tuple[float, float]  # Fitted circle center (x, y)
    n_points: int  # Number of level set points used
    wall_psi: float  # Wall pseudopotential used
    theta_target: float | None  # Target angle (if specified)
    error_deg: float | None  # |θ - θ_target| if target specified
    pass_angle: bool  # Error < 3° if target specified


def fit_circle_algebraic(points: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic circle fit (Kasa 1976).

    Given N points (x_i, y_i), fit a circle (x - x_c)² + (y - y_c)² = R²
    by minimizing Σ [(x_i - x_c)² + (y_i - y_c)² - R²]².

    This is a linear least squares problem in the variables:
        A = -2 x_c, B = -2 y_c, C = x_c² + y_c² - R²

    The equation becomes: x² + y² + A x + B y + C = 0

    Args:
        points: (N, 2) array of (x, y) coordinates.

    Returns:
        (x_c, y_c, R) center and radius of fitted circle.
    """
    x = points[:, 0]
    y = points[:, 1]
    N = len(x)

    # Build linear system: [x, y, 1] @ [A, B, C]^T = -(x² + y²)
    A_mat = np.column_stack([x, y, np.ones(N)])
    b_vec = -(x**2 + y**2)

    # Solve least squares
    result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    coeffs = result[0]
    A, B, C = coeffs

    # Extract center and radius
    x_c = -A / 2
    y_c = -B / 2
    R = math.sqrt(x_c**2 + y_c**2 - C)

    return x_c, y_c, R


def extract_level_set(
    rho: torch.Tensor,
    rho_mid: float,
    wall_y: float,
    wall_side: str = "bottom",
    band_width: float = 1.5,
) -> np.ndarray:
    """Extract mid-density level set points near a wall.

    Finds cells where ρ ≈ ρ_mid (within band_width) and is near the wall.

    Args:
        rho: Density field (ny, nx).
        rho_mid: Mid-density value (ρ_l + ρ_v) / 2.
        wall_y: y-coordinate of the wall.
        wall_side: "bottom" or "top".
        band_width: Half-width of density band for level set.

    Returns:
        (N, 2) array of (x, y) level set points.
    """
    ny, nx = rho.shape
    device = rho.device

    # Find cells near mid-density
    density_mask = torch.abs(rho - rho_mid) < band_width

    # Find cells near the wall (within 30% of domain height)
    y = torch.arange(ny, device=device, dtype=rho.dtype)
    if wall_side == "bottom":
        wall_mask = y < (0.3 * ny)
    else:
        wall_mask = y > (0.7 * ny)

    # Combined mask
    mask = density_mask & wall_mask.unsqueeze(1)

    # Extract coordinates
    indices = torch.nonzero(mask, as_tuple=False)
    if len(indices) == 0:
        return np.zeros((0, 2))

    points = indices[:, [1, 0]].cpu().numpy().astype(float)  # (x, y)
    return points


def measure_contact_angle(
    state: SCState,
    wall_y: float,
    wall_side: str = "bottom",
    rho_mid: float = 1.06,
    band_width: float = 1.5,
    theta_target: float | None = None,
) -> ContactAngleResult:
    """Measure contact angle from a sessile droplet.

    Extracts the mid-density level set near the wall, fits a circle,
    and computes the contact angle from the circle geometry.

    Args:
        state: SC state with a sessile droplet on a wall.
        wall_y: y-coordinate of the wall (half-way bounce-back location).
        wall_side: "bottom" or "top".
        rho_mid: Mid-density for level set extraction.
        band_width: Half-width of density band.
        theta_target: Target contact angle for verification (degrees).

    Returns:
        ContactAngleResult with measured angle and diagnostics.
    """
    rho = state.rho

    # Extract level set points
    points = extract_level_set(rho, rho_mid, wall_y, wall_side, band_width)

    if len(points) < 5:
        # Not enough points for circle fit
        return ContactAngleResult(
            theta_deg=float('nan'),
            theta_rad=float('nan'),
            R_circle=float('nan'),
            center=(float('nan'), float('nan')),
            n_points=len(points),
            wall_psi=state.config.wall_psi,
            theta_target=theta_target,
            error_deg=None,
            pass_angle=False,
        )

    # Fit circle
    x_c, y_c, R = fit_circle_algebraic(points)

    # Compute contact angle
    # For bottom wall: cos(θ) = (y_c - y_wall) / R
    # For top wall: cos(θ) = (y_wall - y_c) / R
    if wall_side == "bottom":
        cos_theta = (y_c - wall_y) / R
    else:
        cos_theta = (wall_y - y_c) / R

    # Clamp to [-1, 1] for numerical safety
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)

    # Check against target
    error_deg = None
    pass_angle = False
    if theta_target is not None:
        error_deg = abs(theta_deg - theta_target)
        pass_angle = error_deg < 3.0

    return ContactAngleResult(
        theta_deg=theta_deg,
        theta_rad=theta_rad,
        R_circle=R,
        center=(x_c, y_c),
        n_points=len(points),
        wall_psi=state.config.wall_psi,
        theta_target=theta_target,
        error_deg=error_deg,
        pass_angle=pass_angle,
    )


def init_sessile_droplet(
    config: SCConfig,
    R: float,
    wall_psi: float,
    rho_l: float = 1.96,
    rho_v: float = 0.16,
    interface_width: float = 3.0,
) -> SCState:
    """Initialize a sessile droplet on a bottom wall.

    The droplet is a half-circle of liquid sitting on the wall at y=0.
    The wall pseudopotential controls the contact angle.

    Args:
        config: Simulation config (must have wall_bottom=True).
        R: Droplet radius.
        wall_psi: Wall pseudopotential for wetting control.
        rho_l: Liquid density.
        rho_v: Vapour density.
        interface_width: Interface width.

    Returns:
        Initial SCState.
    """
    device = torch.device(config.device)
    dtype = config.dtype
    nx, ny = config.nx, config.ny

    # Center the droplet on the bottom wall
    center = (nx / 2.0, R + 1.0)  # Slightly above wall

    # Create coordinate grid
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Distance from center
    dx = X - center[0]
    dy = Y - center[1]
    r = torch.sqrt(dx**2 + dy**2)

    # Density profile: tanh interface, only upper half (y > 0)
    rho = rho_v + (rho_l - rho_v) * 0.5 * (1.0 + torch.tanh((R - r) / interface_width))

    # Zero out below wall (y < 1)
    rho = torch.where(y.unsqueeze(1) < 1, torch.tensor(rho_v, device=device, dtype=dtype), rho)

    # Zero initial velocity
    ux = torch.zeros(ny, nx, device=device, dtype=dtype)
    uy = torch.zeros(ny, nx, device=device, dtype=dtype)

    # Equilibrium populations
    from .sc_core import equilibrium
    f = equilibrium(rho, ux, uy)

    # Update config with wall settings
    config.wall_bottom = True
    config.wall_psi = wall_psi

    return SCState(
        f=f,
        rho=rho,
        ux=ux,
        uy=uy,
        config=config,
        step=0,
    )
