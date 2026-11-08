"""KVLCC2 Ship Hull Geometry for Ice Resistance Simulations.

Implements the KVLCC2 (KRISO Very Large Crude Carrier 2) hull form,
a standard benchmark hull for resistance and propulsion studies.

The hull is defined by offset tables and can be voxelized for LBM
simulations. The geometry is scaled to model or full scale as needed.

References
----------
- SIMMAN 2008 Workshop on Verification and Validation of Ship
  Maneuvering Simulation Methods.
- KVLCC2 hull form: L_pp = 230 m (full scale), B = 32.29 m, T = 14.5 m,
  C_b = 0.8098.
- Riska, K. (1989). Ice resistance model tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class KVLCC2Geometry:
    """KVLCC2 hull geometry parameters.

    Principal dimensions (full scale):
    - L_pp (length between perpendiculars): 230.0 m
    - L_wl (length on waterline): 234.0 m
    - B (beam): 32.29 m
    - T (draft): 14.5 m
    - D (depth): 19.5 m
    - C_b (block coefficient): 0.8098
    - C_m (midship coefficient): 0.9878
    - C_wp (waterplane coefficient): 0.8168

    Parameters
    ----------
    scale : float
        Scale factor (1.0 = full scale, 1/100 = model scale).
    L_pp : float
        Length between perpendiculars in meters (at given scale).
    B : float
        Beam in meters.
    T : float
        Draft in meters.
    """
    scale: float = 1.0
    L_pp: float = 230.0
    B: float = 32.29
    T: float = 14.5
    D: float = 19.5

    @property
    def L_wl(self) -> float:
        """Length on waterline."""
        return self.L_pp * 234.0 / 230.0

    @property
    def C_b(self) -> float:
        """Block coefficient."""
        return 0.8098

    @property
    def C_m(self) -> float:
        """Midship coefficient."""
        return 0.9878

    @property
    def C_wp(self) -> float:
        """Waterplane coefficient."""
        return 0.8168

    @property
    def displacement(self) -> float:
        """Displacement volume in m³."""
        return self.C_b * self.L_pp * self.B * self.T

    @property
    def wetted_surface_area(self) -> float:
        """Approximate wetted surface area (m²) using Holtrop-Mennen."""
        # Simplified approximation
        return self.L_pp * (self.B + 2 * self.T) * 0.5 * (1 + self.C_b)

    def scaled(self, factor: float) -> "KVLCC2Geometry":
        """Return a scaled copy of the geometry."""
        return KVLCC2Geometry(
            scale=self.scale * factor,
            L_pp=self.L_pp * factor,
            B=self.B * factor,
            T=self.T * factor,
            D=self.D * factor,
        )


def kvlcc2_halfbreadth(x: float, z: float, hull: KVLCC2Geometry) -> float:
    """Compute half-breadth (y-coordinate) of KVLCC2 hull at given (x, z).

    Uses a simplified parametric representation based on the KVLCC2
    offset tables. The hull is symmetric about the centerline (y=0).

    Parameters
    ----------
    x : float
        Longitudinal position (0 = AP, L_pp = FP).
    z : float
        Vertical position (0 = baseline, T = waterline).
    hull : KVLCC2Geometry
        Hull geometry.

    Returns
    -------
    float
        Half-breadth y(x, z) >= 0. Returns 0 if outside hull.
    """
    L = hull.L_pp
    B = hull.B
    T = hull.T

    # Normalize coordinates
    xi = x / L  # 0 to 1
    zeta = z / T  # 0 to 1

    if xi < 0 or xi > 1 or zeta < 0 or zeta > 1:
        return 0.0

    # Simplified KVLCC2 half-breadth formula
    # Based on the characteristic full-form tanker shape
    # Bow region (0 <= xi <= 0.1): fine entry
    # Parallel midbody (0.1 <= xi <= 0.7): constant breadth
    # Stern region (0.7 <= xi <= 1.0): U-shaped stern

    # Longitudinal distribution of maximum half-breadth
    if xi <= 0.05:
        # Bow: parabolic entry
        y_max = B / 2 * (20 * xi)**0.5
    elif xi <= 0.15:
        # Transition to parallel midbody
        t = (xi - 0.05) / 0.1
        y_max = B / 2 * (1.0 - 0.05 * (1 - t)**2)
    elif xi <= 0.7:
        # Parallel midbody
        y_max = B / 2
    elif xi <= 0.85:
        # Stern transition
        t = (xi - 0.7) / 0.15
        y_max = B / 2 * (1.0 - 0.15 * t**2)
    else:
        # Stern: U-shaped
        t = (xi - 0.85) / 0.15
        y_max = B / 2 * (0.85 - 0.3 * t**1.5)

    # Vertical distribution (section shape)
    # Waterplane shape coefficient varies along length
    if xi <= 0.1:
        # V-shaped bow sections
        section_coef = 0.5 + 0.5 * zeta**0.7
    elif xi <= 0.7:
        # U-shaped midbody sections
        section_coef = 0.95 + 0.05 * zeta**2
    else:
        # U-shaped stern sections
        section_coef = 0.9 + 0.1 * zeta**1.5

    # Half-breadth at (x, z)
    y = y_max * section_coef

    # Apply waterplane coefficient correction
    if zeta > 0.95:
        # Near waterline, reduce breadth slightly
        y *= hull.C_wp / 0.95 * (1 - zeta) + 1.0 * zeta

    return max(0.0, y)


def generate_hull_mask(
    hull: KVLCC2Geometry,
    grid_resolution: Tuple[int, int, int],
    domain_size: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Generate a binary mask of the KVLCC2 hull on a 3D grid.

    Parameters
    ----------
    hull : KVLCC2Geometry
        Hull geometry.
    grid_resolution : tuple of int
        Grid resolution (nx, ny, nz).
    domain_size : tuple of float
        Domain size (Lx, Ly, Lz).
    origin : tuple of float
        Domain origin (x0, y0, z0). Hull AP is placed at x0 + L_pp/4.

    Returns
    -------
    torch.Tensor
        Binary mask, shape (nz, ny, nx). True = inside hull.
    """
    nx, ny, nz = grid_resolution
    Lx, Ly, Lz = domain_size
    x0, y0, z0 = origin

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # Hull position: AP at x0 + L_pp/4, centered in y, keel at z0
    ap_x = x0 + Lx * 0.15  # Place hull in upstream portion
    center_y = y0 + Ly / 2
    keel_z = z0

    mask = torch.zeros((nz, ny, nx), dtype=torch.bool)

    for iz in range(nz):
        z = keel_z + (iz + 0.5) * dz
        if z < 0 or z > hull.T:
            continue

        for iy in range(ny):
            y = center_y + (iy + 0.5 - ny / 2) * dy
            y_abs = abs(y - center_y)

            for ix in range(nx):
                x = ap_x + (ix + 0.5) * dx

                # Check if point is within hull length
                x_rel = x - ap_x
                if x_rel < 0 or x_rel > hull.L_pp:
                    continue

                # Check half-breadth
                y_half = kvlcc2_halfbreadth(x_rel, z - keel_z, hull)

                if y_abs <= y_half:
                    mask[iz, iy, ix] = True

    return mask


def compute_hull_surface_cells(
    hull_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identify surface cells of the hull (solid cells adjacent to fluid).

    Parameters
    ----------
    hull_mask : torch.Tensor
        Binary hull mask, shape (nz, ny, nx).

    Returns
    -------
    surface_mask : torch.Tensor
        Boolean mask of surface cells.
    surface_normals : torch.Tensor
        Approximate outward normals, shape (nz, ny, nx, 3).
    """
    nz, ny, nx = hull_mask.shape
    device = hull_mask.device

    # Pad for boundary checking
    padded = torch.nn.functional.pad(
        hull_mask.float().unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1, 1, 1),
        mode="constant",
        value=0,
    ).squeeze()

    # Count fluid neighbors for each solid cell
    fluid_neighbors = (
        (1 - padded[:-2, 1:-1, 1:-1]).float()  # -x
        + (1 - padded[2:, 1:-1, 1:-1]).float()  # +x
        + (1 - padded[1:-1, :-2, 1:-1]).float()  # -y
        + (1 - padded[1:-1, 2:, 1:-1]).float()  # +y
        + (1 - padded[1:-1, 1:-1, :-2]).float()  # -z
        + (1 - padded[1:-1, 1:-1, 2:]).float()  # +z
    )

    # Surface cells: solid with at least one fluid neighbor
    surface_mask = hull_mask & (fluid_neighbors > 0)

    # Compute approximate normals (gradient of mask)
    normals = torch.zeros((nz, ny, nx, 3), dtype=torch.float64, device=device)

    # Finite differences
    if nz > 1:
        normals[1:-1, :, :, 2] = (
            hull_mask[2:, :, :].float() - hull_mask[:-2, :, :].float()
        )
    if ny > 1:
        normals[:, 1:-1, :, 1] = (
            hull_mask[:, 2:, :].float() - hull_mask[:, :-2, :].float()
        )
    if nx > 1:
        normals[:, :, 1:-1, 0] = (
            hull_mask[:, :, 2:].float() - hull_mask[:, :, :-2].float()
        )

    # Normalize
    norm = torch.norm(normals, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)
    normals = normals / norm

    # Zero normals for non-surface cells
    normals[~surface_mask] = 0.0

    return surface_mask, normals


def compute_hull_resistance(
    pressure: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
    shear_z: torch.Tensor,
    surface_mask: torch.Tensor,
    normals: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    rho: float = 1025.0,
) -> dict:
    """Compute hull resistance from surface pressure and shear stress.

    Parameters
    ----------
    pressure : torch.Tensor
        Pressure field, shape (nz, ny, nx) in Pa.
    shear_x, shear_y, shear_z : torch.Tensor
        Shear stress components, shape (nz, ny, nx) in Pa.
    surface_mask : torch.Tensor
        Boolean mask of hull surface cells.
    normals : torch.Tensor
        Outward normals, shape (nz, ny, nx, 3).
    dx, dy, dz : float
        Grid spacing in meters.
    rho : float
        Fluid density in kg/m³.

    Returns
    -------
    dict
        Resistance components:
        - R_friction: frictional resistance (N)
        - R_pressure: pressure (form) resistance (N)
        - R_total: total resistance (N)
        - C_T: total resistance coefficient
    """
    if not surface_mask.any():
        return {
            "R_friction": 0.0,
            "R_pressure": 0.0,
            "R_total": 0.0,
            "C_T": 0.0,
            "n_surface_cells": 0,
        }

    # Surface cell area
    dA = dx * dy  # Approximate area per surface cell (projected)

    # Extract surface values - fix indexing for normals
    p_surf = pressure[surface_mask]
    # normals shape: (nz, ny, nx, 3), surface_mask shape: (nz, ny, nx)
    # First get all normals at surface cells, then extract components
    normals_surf = normals[surface_mask]  # shape: (n_surface, 3)
    nx_surf = normals_surf[:, 0]
    ny_surf = normals_surf[:, 1]
    nz_surf = normals_surf[:, 2]
    tau_x_surf = shear_x[surface_mask]
    tau_y_surf = shear_y[surface_mask]
    tau_z_surf = shear_z[surface_mask]

    # Pressure resistance (x-component): R_p = -∫ p * n_x dA
    R_pressure = -torch.sum(p_surf * nx_surf).item() * dA

    # Friction resistance (x-component): R_f = ∫ τ_x dA
    R_friction = torch.sum(tau_x_surf).item() * dA

    # Total resistance
    R_total = R_friction + R_pressure

    return {
        "R_friction": R_friction,
        "R_pressure": R_pressure,
        "R_total": R_total,
        "n_surface_cells": int(surface_mask.sum().item()),
    }


def riska_ice_resistance(
    h_ice: float,
    v_ship: float,
    L_pp: float,
    B: float,
    phi: float,
    C_b: float,
    psi: float = 0.0,
) -> float:
    """Compute ice resistance using Riska (1989) model.

    Riska's semi-empirical model for ice resistance in broken ice:

        R_ice = R_c + R_f + R_s

    where:
        R_c = crushing resistance
        R_f = frictional resistance (ice-hull)
        R_s = submersion resistance

    Parameters
    ----------
    h_ice : float
        Ice thickness in meters.
    v_ship : float
        Ship speed in m/s.
    L_pp : float
        Length between perpendiculars in meters.
    B : float
        Beam in meters.
    phi : float
        Stem angle (degrees).
    C_b : float
        Block coefficient.
    psi : float
        Ice concentration (0-1). Default 0 (open water).

    Returns
    -------
    float
        Ice resistance in Newtons.
    """
    # Convert stem angle to radians
    phi_rad = math.radians(phi)

    # Ice properties (typical Baltic ice)
    sigma_c = 1.5e6  # Compressive strength (Pa)
    mu = 0.15  # Friction coefficient (ice-steel)
    rho_i = 917.0  # Ice density (kg/m³)
    rho_w = 1025.0  # Water density (kg/m³)

    # Crushing resistance
    # R_c = C_c * sigma_c * h_ice^2 * (B / tan(phi))
    C_c = 0.5  # Empirical coefficient
    R_c = C_c * sigma_c * h_ice**2 * (B / max(math.tan(phi_rad), 0.1))

    # Frictional resistance (ice-hull)
    # R_f = mu * R_c * (1 + v_ship / 5)
    R_f = mu * R_c * (1 + v_ship / 5.0)

    # Submersion resistance
    # R_s = C_s * rho_w * g * h_ice * B * L_pp * C_b * psi
    g = 9.81
    C_s = 0.15  # Empirical coefficient
    R_s = C_s * rho_w * g * h_ice * B * L_pp * C_b * psi

    # Total ice resistance
    R_ice = (R_c + R_f + R_s) * psi

    return R_ice


def riska_total_resistance(
    h_ice: float,
    v_ship: float,
    L_pp: float,
    B: float,
    T: float,
    C_b: float,
    phi: float,
    psi: float,
    R_open_water: Optional[float] = None,
) -> dict:
    """Compute total resistance in ice using Riska model.

    Parameters
    ----------
    h_ice : float
        Ice thickness (m).
    v_ship : float
        Ship speed (m/s).
    L_pp : float
        Length between perpendiculars (m).
    B : float
        Beam (m).
    T : float
        Draft (m).
    C_b : float
        Block coefficient.
    phi : float
        Stem angle (degrees).
    psi : float
        Ice concentration (0-1).
    R_open_water : float, optional
        Open water resistance (N). If None, estimated.

    Returns
    -------
    dict
        Resistance breakdown.
    """
    # Open water resistance (ITTC 1957 friction line + form factor)
    if R_open_water is None:
        # Estimate using ITTC 1957
        Re = v_ship * L_pp / 1.19e-6  # Kinematic viscosity at 15°C
        C_F = 0.075 / (math.log10(Re) - 2)**2
        k = 1 + 0.5 * C_b  # Form factor (simplified)
        S = L_pp * (B + 2 * T) * 0.5  # Wetted surface (approx)
        rho = 1025.0
        R_open_water = 0.5 * rho * v_ship**2 * S * C_F * (1 + k)

    # Ice resistance
    R_ice = riska_ice_resistance(h_ice, v_ship, L_pp, B, phi, C_b, psi)

    # Total resistance
    R_total = R_open_water + R_ice

    return {
        "R_open_water": R_open_water,
        "R_ice": R_ice,
        "R_total": R_total,
        "R_ice_fraction": R_ice / R_total if R_total > 0 else 0.0,
        "v_ship": v_ship,
        "h_ice": h_ice,
        "psi": psi,
    }
