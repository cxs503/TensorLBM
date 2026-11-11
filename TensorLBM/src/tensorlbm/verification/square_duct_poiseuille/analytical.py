"""Analytical reference solutions for square duct Poiseuille flow.

Reference: Shah, R. K. & London, A. L. (1978).
"Laminar Flow Forced Convection in Ducts", Academic Press.
- Square duct (aspect ratio α = 1): f·Re = 56.908 ≈ 56.91
  (Table 49, p. 103; also Eq. 49 with α=1)

The exact series solution for fully developed laminar flow in a rectangular duct
gives f·Re as a function of aspect ratio. For a square duct (α = b/a = 1):

    f·Re = 56.908

This is the canonical reference value used throughout this benchmark.
"""

import numpy as np
from typing import Tuple


def shah_london_fRe(aspect_ratio: float = 1.0) -> float:
    """Return f·Re from Shah & London (1978) for given aspect ratio.

    For square duct (aspect_ratio = 1.0): f·Re = 56.908

    Parameters
    ----------
    aspect_ratio : float
        Ratio of shorter side to longer side (0 < α ≤ 1).

    Returns
    -------
    fRe : float
        Product of Darcy friction factor and Reynolds number.

    Reference
    ---------
    Shah & London (1978), Table 49, p. 103.
    """
    if abs(aspect_ratio - 1.0) < 1e-6:
        return 56.908
    # General rectangular duct approximation (Shah & London Eq. 49)
    α = aspect_ratio
    fRe = 24.0 * (1.0 - 1.3553*α + 1.9467*α**2 - 1.7012*α**3 + 0.9564*α**4 - 0.2537*α**5)
    return fRe


def square_duct_velocity_profile(
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    dpdx: float,
    mu: float,
    n_terms: int = 50,
) -> np.ndarray:
    """Analytical velocity profile for square duct Poiseuille flow.

    The exact solution is a Fourier series:

        u(y,z) = (16 a² / (μ π³)) (-dp/dx) Σ_{n odd} [(-1)^{(n-1)/2} / n³]
                 [1 - cosh(nπy/a) / cosh(nπ/2)] cos(nπz/a)

    where y ∈ [-a/2, a/2], z ∈ [-a/2, a/2], and a is the duct side length.

    Parameters
    ----------
    y, z : np.ndarray
        Coordinate arrays (meshgrid-compatible).
    a : float
        Duct side length.
    dpdx : float
        Pressure gradient (negative for flow in +x direction).
    mu : float
        Dynamic viscosity.
    n_terms : int
        Number of Fourier terms (odd n only).

    Returns
    -------
    u : np.ndarray
        Velocity field.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    u = np.zeros_like(y, dtype=float)
    for k in range(n_terms):
        n = 2 * k + 1  # odd integers: 1, 3, 5, ...
        sign = (-1) ** k
        term = (sign / n**3) * (1.0 - np.cosh(n * np.pi * y / a) / np.cosh(n * np.pi / 2)) * np.cos(n * np.pi * z / a)
        u += term

    prefactor = 16.0 * a**2 / (mu * np.pi**3) * (-dpdx)
    return prefactor * u


def square_duct_mean_velocity(
    a: float,
    dpdx: float,
    mu: float,
    n_terms: int = 100,
) -> float:
    """Compute mean velocity by integrating the analytical profile.

    Parameters
    ----------
    a : float
        Duct side length.
    dpdx : float
        Pressure gradient.
    mu : float
        Dynamic viscosity.
    n_terms : int
        Number of Fourier terms.

    Returns
    -------
    u_mean : float
        Cross-section averaged velocity.
    """
    # Use high-resolution quadrature
    ny, nz = 200, 200
    y = np.linspace(-a/2, a/2, ny)
    z = np.linspace(-a/2, a/2, nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    u = square_duct_velocity_profile(Y, Z, a, dpdx, mu, n_terms=n_terms)
    return float(np.mean(u))


def square_duct_max_velocity(
    a: float,
    dpdx: float,
    mu: float,
    n_terms: int = 100,
) -> float:
    """Maximum velocity (at centerline) for square duct.

    Parameters
    ----------
    a : float
        Duct side length.
    dpdx : float
        Pressure gradient.
    mu : float
        Dynamic viscosity.
    n_terms : int
        Number of Fourier terms.

    Returns
    -------
    u_max : float
        Centerline velocity.
    """
    # At center (y=0, z=0):
    u_max = 0.0
    for k in range(n_terms):
        n = 2 * k + 1
        sign = (-1) ** k
        term = sign / n**3 * (1.0 - 1.0 / np.cosh(n * np.pi / 2))
        u_max += term

    prefactor = 16.0 * a**2 / (mu * np.pi**3) * (-dpdx)
    return prefactor * u_max
