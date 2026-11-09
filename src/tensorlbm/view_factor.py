"""View factor (configuration factor) analytical solutions for radiation heat transfer.

This module provides exact/closed-form view factor calculations for common
geometries used in radiation enclosure analysis. All view factors satisfy:
- Reciprocity: A_i * F_ij = A_j * F_ji
- Summation rule: Σ_j F_ij = 1 (for closed enclosures)

Geometries implemented:
- Parallel equal rectangles (aligned, directly opposed) - numerical integration
- Perpendicular rectangles with common edge - numerical integration
- Differential element to rectangle (numerical integration)
- Parallel coaxial disks - analytical formula
- Infinite plate inside cylindrical enclosure

References
----------
Modest, M. F. (2013). Radiative Heat Transfer (3rd ed.). Elsevier.
Howell, J. R., Siegel, R., & Menguc, M. P. (2010). Thermal Radiation Heat
    Transfer (5th ed.). CRC Press.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "vf_parallel_rectangles",
    "vf_perpendicular_rectangles",
    "vf_differential_to_rectangle",
    "vf_parallel_coaxial_disks",
    "vf_infinite_plate_cylinder",
    "normalize_view_factor_matrix",
    "enforce_reciprocity",
    "ViewFactorEnclosure",
]


def vf_parallel_rectangles(X: float, Y: float, n_quad: int = 32) -> float:
    """View factor between two parallel, directly opposed equal rectangles.

    The rectangles have dimensions (a × b) separated by distance c.
    Dimensionless parameters: X = a/c, Y = b/c.

    Uses Gauss-Legendre quadrature for numerical integration.

    Reference value: For X=Y=1 (unit squares separated by unit distance),
    F_12 ≈ 0.1998 (Howell Table 5.1).

    Parameters
    ----------
    X : float
        Ratio a/c where a is one side length and c is separation distance.
    Y : float
        Ratio b/c where b is the other side length and c is separation distance.
    n_quad : int
        Number of quadrature points per dimension (default 32).

    Returns
    -------
    float
        View factor F_12 (dimensionless, 0 to 1).
    """
    if X <= 0 or Y <= 0:
        raise ValueError(f"X and Y must be positive, got X={X}, Y={Y}")

    # Gauss-Legendre quadrature
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    
    # Transform to rectangle coordinates
    x1 = 0.5 * X * pts
    y1 = 0.5 * Y * pts
    wx1 = 0.5 * X * wts
    wy1 = 0.5 * Y * wts
    
    x2 = 0.5 * X * pts
    y2 = 0.5 * Y * pts
    wx2 = 0.5 * X * wts
    wy2 = 0.5 * Y * wts
    
    # Double integral over both rectangles
    F = 0.0
    for i in range(n_quad):
        for j in range(n_quad):
            for k in range(n_quad):
                for l in range(n_quad):
                    dx = x2[k] - x1[i]
                    dy = y2[l] - y1[j]
                    r2 = dx*dx + dy*dy + 1.0
                    r = math.sqrt(r2)
                    dF = 1.0 / (math.pi * r2 * r2)
                    F += wx1[i] * wy1[j] * wx2[k] * wy2[l] * dF
    
    F /= (X * Y)
    
    return max(0.0, min(1.0, F))


def vf_perpendicular_rectangles(X: float, Y: float, n_quad: int = 32) -> float:
    """View factor between two perpendicular rectangles sharing a common edge.

    The rectangles have dimensions (a × c) and (b × c) perpendicular to each
    other, sharing edge of length c.
    Dimensionless parameters: X = a/c, Y = b/c.

    Uses numerical integration.

    Parameters
    ----------
    X : float
        Ratio a/c.
    Y : float
        Ratio b/c.
    n_quad : int
        Number of quadrature points per dimension.

    Returns
    -------
    float
        View factor F_12 (dimensionless, 0 to 1).
    """
    if X <= 0 or Y <= 0:
        raise ValueError(f"X and Y must be positive, got X={X}, Y={Y}")

    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    
    x1 = 0.5 * X * (pts + 1)
    z1 = 0.5 * (pts + 1)
    wx1 = 0.5 * X * wts
    wz1 = 0.5 * wts
    
    y2 = 0.5 * Y * (pts + 1)
    z2 = 0.5 * (pts + 1)
    wy2 = 0.5 * Y * wts
    wz2 = 0.5 * wts
    
    F = 0.0
    for i in range(n_quad):
        for j in range(n_quad):
            for k in range(n_quad):
                for l in range(n_quad):
                    dx = 0 - x1[i]
                    dy = y2[k] - 0
                    dz = z2[l] - z1[j]
                    r2 = dx*dx + dy*dy + dz*dz
                    r = math.sqrt(r2)
                    cos1 = abs(dy) / r
                    cos2 = abs(dx) / r
                    dF = cos1 * cos2 / (math.pi * r2)
                    F += wx1[i] * wz1[j] * wy2[k] * wz2[l] * dF
    
    F /= X
    
    return max(0.0, min(1.0, F))


def vf_differential_to_rectangle(
    a: float, b: float, c: float, n_quad: int = 32
) -> float:
    """View factor from a differential element to a parallel rectangle.

    The differential element dA1 is located at distance c from a rectangle
    of dimensions (a × b), directly opposed and parallel.

    Uses Gauss-Legendre quadrature.

    Parameters
    ----------
    a : float
        Rectangle dimension in x [m].
    b : float
        Rectangle dimension in y [m].
    c : float
        Separation distance [m].
    n_quad : int
        Number of quadrature points per dimension (default 32).

    Returns
    -------
    float
        View factor F_d1-2 (dimensionless, 0 to 1).
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"a, b, c must be positive, got a={a}, b={b}, c={c}")

    x_pts, x_wts = np.polynomial.legendre.leggauss(n_quad)
    y_pts, y_wts = np.polynomial.legendre.leggauss(n_quad)

    x = 0.5 * a * x_pts
    y = 0.5 * b * y_pts
    wx = 0.5 * a * x_wts
    wy = 0.5 * b * y_wts

    F = 0.0
    c2 = c * c
    for i in range(n_quad):
        for j in range(n_quad):
            r2 = x[i] ** 2 + y[j] ** 2 + c2
            F += wx[i] * wy[j] * (c / r2)

    F /= math.pi
    return max(0.0, min(1.0, F))


def vf_parallel_coaxial_disks(R_i: float, R_j: float, S: float) -> float:
    """View factor between two parallel coaxial disks.

    Disk i has radius R_i, disk j has radius R_j, separated by distance S.

    Exact formula (Howell Table 5.1, case 1).

    Parameters
    ----------
    R_i : float
        Radius of disk 1 [m].
    R_j : float
        Radius of disk 2 [m].
    S : float
        Separation distance between disks [m].

    Returns
    -------
    float
        View factor F_12 from disk 1 to disk 2.
    """
    if R_i <= 0 or R_j <= 0 or S <= 0:
        raise ValueError(f"R_i, R_j, S must be positive")

    ri = R_i / S
    rj = R_j / S

    X = 1.0 + (1.0 + rj**2) / ri**2
    F12 = 0.5 * (X - math.sqrt(X**2 - 4.0 * (rj / ri) ** 2))

    return max(0.0, min(1.0, F12))


def vf_infinite_plate_cylinder(radius: float, height: float) -> float:
    """View factor from base plate to cylindrical wall in a finite cylinder.

    Parameters
    ----------
    radius : float
        Cylinder radius [m].
    height : float
        Cylinder height [m].

    Returns
    -------
    float
        View factor from base plate to cylindrical wall.
    """
    if radius <= 0 or height <= 0:
        raise ValueError(f"radius and height must be positive")

    F_base_to_top = vf_parallel_coaxial_disks(radius, radius, height)
    F_base_to_wall = 1.0 - F_base_to_top

    return max(0.0, min(1.0, F_base_to_wall))


def normalize_view_factor_matrix(
    F: np.ndarray,
    areas: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """Normalize a view factor matrix to satisfy summation and reciprocity.

    Applies iterative normalization using alternating projections.

    Parameters
    ----------
    F : np.ndarray
        Initial view factor matrix (n × n).
    areas : np.ndarray, optional
        Surface areas (n,). If None, assumes equal areas.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Normalized view factor matrix.
    """
    n = F.shape[0]
    F = F.copy().astype(np.float64)
    np.fill_diagonal(F, 0.0)

    if areas is None:
        areas = np.ones(n)

    for iteration in range(max_iter):
        F_old = F.copy()

        # Enforce reciprocity
        for i in range(n):
            for j in range(i + 1, n):
                AiFij = areas[i] * F[i, j]
                AjFji = areas[j] * F[j, i]
                avg_product = 0.5 * (AiFij + AjFji)
                F[i, j] = avg_product / areas[i]
                F[j, i] = avg_product / areas[j]

        # Enforce summation
        row_sums = F.sum(axis=1)
        for i in range(n):
            if row_sums[i] > 1e-15:
                scale = 1.0 / row_sums[i]
                F[i, :] *= scale
                for j in range(n):
                    if i != j:
                        F[j, i] = F[i, j] * areas[i] / areas[j]

        if np.max(np.abs(F - F_old)) < tol:
            break

    return F


def enforce_reciprocity(
    F: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    """Enforce reciprocity relation A_i F_ij = A_j F_ji."""
    n = F.shape[0]
    F_out = F.copy().astype(np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            AiFij = areas[i] * F[i, j]
            AjFji = areas[j] * F[j, i]
            avg = 0.5 * (AiFij + AjFji)
            F_out[i, j] = avg / areas[i]
            F_out[j, i] = avg / areas[j]

    return F_out


class ViewFactorEnclosure:
    """Container for a radiation enclosure with view factors."""

    def __init__(
        self,
        areas: Sequence[float],
        view_factors: np.ndarray | None = None,
        normalize: bool = True,
    ):
        self.areas = np.asarray(areas, dtype=np.float64)
        self.n_surfaces = len(self.areas)

        if view_factors is not None:
            self.F = np.asarray(view_factors, dtype=np.float64)
            if self.F.shape != (self.n_surfaces, self.n_surfaces):
                raise ValueError(
                    f"view_factors shape {self.F.shape} does not match "
                    f"({self.n_surfaces}, {self.n_surfaces})"
                )
            if normalize:
                self.F = normalize_view_factor_matrix(self.F, self.areas)
        else:
            self.F = np.zeros((self.n_surfaces, self.n_surfaces), dtype=np.float64)

    def set_view_factor(self, i: int, j: int, value: float) -> None:
        """Set view factor F_ij and enforce reciprocity for F_ji."""
        self.F[i, j] = value
        self.F[j, i] = value * self.areas[i] / self.areas[j]

    def check_summation(self, tol: float = 1e-9) -> float:
        """Check summation rule: max |Σ_j F_ij - 1|."""
        row_sums = self.F.sum(axis=1)
        return float(np.max(np.abs(row_sums - 1.0)))

    def check_reciprocity(self, tol: float = 1e-9) -> float:
        """Check reciprocity: max |A_i F_ij - A_j F_ji|."""
        max_violation = 0.0
        for i in range(self.n_surfaces):
            for j in range(self.n_surfaces):
                AiFij = self.areas[i] * self.F[i, j]
                AjFji = self.areas[j] * self.F[j, i]
                max_violation = max(max_violation, abs(AiFij - AjFji))
        return float(max_violation)

    def to_torch(self):
        """Convert view factor matrix to torch tensor."""
        import torch
        return torch.from_numpy(self.F)

    def to_list(self) -> list[list[float]]:
        """Convert to nested list."""
        return self.F.tolist()
