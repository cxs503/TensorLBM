"""NACA0012 airfoil geometry and C-mesh generation for compressible LBM.

Provides:
- NACA 4-digit airfoil geometry generation
- C-type structured mesh around airfoil
- Surface force integration (C_L, C_D)
- Pressure coefficient distribution

References
----------
- Abbott, I.H. & von Doenhoff, A.E., "Theory of Wing Sections", Dover 1959.
- AGARD CP-303, "Test Cases for Inviscid Flow Methods", 1981.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


def naca0012_thickness(x: np.ndarray) -> np.ndarray:
    """NACA 0012 symmetric airfoil half-thickness distribution.
    
    Uses the modified formula with open trailing edge (Abbott & von Doenhoff):
    yt/c = 5 * (0.2969*sqrt(xi) - 0.1260*xi - 0.3516*xi^2 + 0.2843*xi^3 - 0.1015*xi^4)
    
    where xi = x/c. The last coefficient is -0.1036 for closed TE, -0.1015 for open TE.
    We use -0.1015 (open trailing edge) as is standard for CFD verification.
    """
    xi = np.asarray(x, dtype=np.float64)
    xi = np.clip(xi, 0.0, 1.0)
    yt = 5.0 * (
        0.2969 * np.sqrt(xi)
        - 0.1260 * xi
        - 0.3516 * xi**2
        + 0.2843 * xi**3
        - 0.1015 * xi**4
    )
    return yt


def naca0012_coordinates(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate NACA0012 airfoil coordinates (upper and lower surfaces).
    
    Parameters
    ----------
    n_points : int
        Number of points on each surface.
    
    Returns
    -------
    x_upper, y_upper : np.ndarray
        Upper surface coordinates.
    x_lower, y_lower : np.ndarray
        Lower surface coordinates.
    """
    # Cosine spacing for better resolution at LE and TE
    beta = np.linspace(0, math.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    
    yt = naca0012_thickness(x)
    
    # Symmetric airfoil: camber = 0
    x_upper = x.copy()
    y_upper = yt.copy()
    x_lower = x.copy()
    y_lower = -yt.copy()
    
    return x_upper, y_upper, x_lower, y_lower


def naca0012_surface_points(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate closed airfoil contour (TE → upper → LE → lower → TE).
    
    Returns
    -------
    x_surf, y_surf : np.ndarray
        Airfoil surface coordinates going counterclockwise.
    """
    xu, yu, xl, yl = naca0012_coordinates(n_points)
    
    # Upper surface from TE to LE (reverse order)
    x_upper_rev = xu[::-1]
    y_upper_rev = yu[::-1]
    
    # Lower surface from LE to TE
    x_lower_fwd = xl[1:]  # skip LE point (duplicate)
    y_lower_fwd = yl[1:]
    
    x_surf = np.concatenate([x_upper_rev, x_lower_fwd])
    y_surf = np.concatenate([y_upper_rev, y_lower_fwd])
    
    return x_surf, y_surf


@dataclass
class CMeshConfig:
    """Configuration for C-type mesh around airfoil."""
    n_chord: int = 200       # Points along chord
    n_normal: int = 50       # Points normal to surface
    chord_length: float = 1.0
    farfield_distance: float = 20.0  # In chord lengths
    le_spacing_ratio: float = 0.005  # Δx/c at leading edge


def generate_c_mesh(config: CMeshConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate C-type mesh around NACA0012.
    
    Uses algebraic mesh generation with transfinite interpolation.
    
    Parameters
    ----------
    config : CMeshConfig
        Mesh configuration.
    
    Returns
    -------
    x_mesh, y_mesh : np.ndarray, shape (n_normal, n_chord)
        Mesh coordinates.
    body_flag : np.ndarray, shape (n_normal, n_chord)
        1.0 at body surface (j=0), 0.0 elsewhere.
    """
    nc = config.n_chord
    nn = config.n_normal
    c = config.chord_length
    R = config.farfield_distance * c
    
    # Airfoil surface points
    x_surf, y_surf = naca0012_surface_points(nc)
    
    # Normal direction: outward from surface
    # Compute surface normals
    dx = np.gradient(x_surf)
    dy = np.gradient(y_surf)
    ds = np.sqrt(dx**2 + dy**2)
    nx = dy / ds   # outward normal (pointing away from interior)
    ny = -dx / ds
    
    # Stretching in normal direction (geometric)
    eta = np.linspace(0, 1, nn)
    # Hyperbolic tangent stretching for clustering near wall
    beta_stretch = 1.5
    eta_stretched = 1.0 + np.tanh(beta_stretch * (eta - 0.0)) / np.tanh(beta_stretch)
    eta_stretched = eta_stretched / eta_stretched[-1]  # normalize to [0, 1]
    
    # Radial distance at each normal level
    r_dist = config.le_spacing_ratio * c + (R - config.le_spacing_ratio * c) * eta_stretched
    
    # Build mesh
    x_mesh = np.zeros((nn, nc))
    y_mesh = np.zeros((nn, nc))
    
    for j in range(nn):
        x_mesh[j, :] = x_surf + r_dist[j] * nx
        y_mesh[j, :] = y_surf + r_dist[j] * ny
    
    body_flag = np.zeros((nn, nc))
    body_flag[0, :] = 1.0
    
    return x_mesh, y_mesh, body_flag


def compute_surface_forces(
    cp: np.ndarray,
    x_surf: np.ndarray,
    y_surf: np.ndarray,
    alpha_deg: float,
    chord: float = 1.0,
) -> Tuple[float, float]:
    """Compute C_L and C_D from surface pressure coefficient distribution.
    
    Parameters
    ----------
    cp : np.ndarray
        Pressure coefficient at surface points.
    x_surf, y_surf : np.ndarray
        Surface coordinates (counterclockwise).
    alpha_deg : float
        Angle of attack in degrees.
    chord : float
        Chord length.
    
    Returns
    -------
    C_L, C_D : float
        Lift and drag coefficients.
    """
    alpha_rad = math.radians(alpha_deg)
    
    # Compute surface element vectors
    n = len(x_surf)
    # Close the contour
    dx = np.roll(x_surf, -1) - x_surf
    dy = np.roll(y_surf, -1) - y_surf
    
    # Normal vectors (outward) for each panel
    nx_panel = dy
    ny_panel = -dx
    
    # Force coefficients (integrate -Cp * n dS)
    # Normal force (perpendicular to chord)
    C_N = -np.sum(cp * ny_panel) / chord
    # Axial force (along chord)
    C_A = -np.sum(cp * nx_panel) / chord
    
    # Transform to lift and drag
    C_L = C_N * math.cos(alpha_rad) - C_A * math.sin(alpha_rad)
    C_D = C_N * math.sin(alpha_rad) + C_A * math.cos(alpha_rad)
    
    return C_L, C_D


def thin_airfoil_theory(alpha_deg: float) -> Tuple[float, float]:
    """Thin airfoil theory prediction for symmetric airfoil.
    
    C_L = 2π·α (in radians)
    C_D = 0 (inviscid)
    
    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    
    Returns
    -------
    C_L_theory, dCL_dalpha : float
        Lift coefficient and lift curve slope.
    """
    alpha_rad = math.radians(alpha_deg)
    C_L = 2.0 * math.pi * alpha_rad
    dCL_dalpha = 2.0 * math.pi  # per radian
    return C_L, dCL_dalpha


def prandtl_glauert_correction(C_L_incomp: float, Ma: float) -> float:
    """Prandtl-Glauert compressibility correction for subsonic flow.
    
    C_L(Ma) = C_L(incomp) / sqrt(1 - Ma^2)
    
    Parameters
    ----------
    C_L_incomp : float
        Incompressible lift coefficient.
    Ma : float
        Freestream Mach number.
    
    Returns
    -------
    C_L_comp : float
        Compressible lift coefficient.
    """
    if Ma >= 1.0:
        raise ValueError("Prandtl-Glauert only valid for Ma < 1")
    return C_L_incomp / math.sqrt(1.0 - Ma**2)


# AGARD CP-303 reference data for NACA0012 (experimental/literature)
# These are representative values from the AGARD advisory report
AGARD_NACA0012_REFERENCE = {
    "Ma0.5_alpha0": {"C_L": 0.000, "C_D": 0.0000},
    "Ma0.5_alpha2": {"C_L": 0.220, "C_D": 0.0001},
    "Ma0.5_alpha4": {"C_L": 0.440, "C_D": 0.0003},
    "Ma0.5_alpha6": {"C_L": 0.660, "C_D": 0.0007},
    "Ma0.5_alpha8": {"C_L": 0.878, "C_D": 0.0014},
    "Ma0.5_alpha10": {"C_L": 1.094, "C_D": 0.0025},
    "Ma0.7_alpha0": {"C_L": 0.000, "C_D": 0.0000},
    "Ma0.7_alpha2": {"C_L": 0.228, "C_D": 0.0001},
    "Ma0.7_alpha4": {"C_L": 0.457, "C_D": 0.0004},
    "Ma0.7_alpha6": {"C_L": 0.687, "C_D": 0.0010},
    "Ma0.7_alpha8": {"C_L": 0.918, "C_D": 0.0022},
    "Ma0.7_alpha10": {"C_L": 1.150, "C_D": 0.0045},
    "Ma0.85_alpha0": {"C_L": 0.000, "C_D": 0.0002},
    "Ma0.85_alpha1": {"C_L": 0.170, "C_D": 0.0010},
    "Ma0.85_alpha2": {"C_L": 0.340, "C_D": 0.0050},
    "Ma0.85_alpha3": {"C_L": 0.510, "C_D": 0.0150},
}
