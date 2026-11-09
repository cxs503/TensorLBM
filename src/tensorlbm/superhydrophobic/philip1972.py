"""Philip (1972) analytical solution for effective slip length on
periodic striped superhydrophobic surfaces.

Reference
---------
Philip, J.R. (1972). "Flows meeting a striped surface with alternating
slip and no-slip." Z. Angew. Math. Phys. (ZAMP), 23, 960-968.

For a surface with periodic stripes of width *a* (gas/liquid interface,
free-slip) alternating with width *b* (solid ridge, no-slip), the
effective slip length for flow parallel to the stripes is:

    b_eff = (a / π) · ln(sec(π φ / 2))

where φ = a / (a + b) is the gas-area (fractional slip) ratio.

For flow perpendicular to the stripes, Philip gives:

    b_eff_perp = (a / (2π)) · ln(sec(π φ / 2))

i.e. exactly half the parallel value.

This module provides:
- ``gas_area_fraction(a, b)``: compute φ
- ``philip_slip_length(a, b, direction='parallel')``: b_eff in physical units
- ``philip_slip_length_dimensionless(phi)``: b_eff / a as a function of φ only
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

NumericType = Union[float, np.ndarray]


def gas_area_fraction(a: NumericType, b: NumericType) -> NumericType:
    """Compute the gas-area fraction φ = a / (a + b).

    Parameters
    ----------
    a : float or array
        Width of the gas-liquid interface (free-slip stripe).
    b : float or array
        Width of the solid ridge (no-slip stripe).

    Returns
    -------
    phi : float or array
        Gas-area fraction, 0 ≤ φ ≤ 1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("Stripe widths a, b must be non-negative.")
    if np.any(a + b == 0):
        raise ValueError("a + b must be positive (non-zero period).")
    return a / (a + b)


def philip_slip_length_dimensionless(phi: NumericType) -> NumericType:
    """Effective slip length normalised by stripe width a: b_eff / a.

    Uses Philip (1972) formula for flow parallel to stripes:

        b_eff / a = (1/π) · ln(sec(π φ / 2))

    Parameters
    ----------
    phi : float or array
        Gas-area fraction, 0 ≤ φ ≤ 1.

    Returns
    -------
    b_eff_over_a : float or array
        Dimensionless effective slip length.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if np.any(phi < 0) or np.any(phi > 1):
        raise ValueError("Gas fraction φ must be in [0, 1].")
    # sec(π φ / 2) = 1 / cos(π φ / 2)
    # ln(sec(x)) = -ln(cos(x))
    arg = np.pi * phi / 2.0
    cos_val = np.cos(arg)
    # Guard against φ → 1 where cos → 0
    cos_val = np.clip(cos_val, 1e-15, None)
    return (1.0 / np.pi) * np.log(1.0 / cos_val)


def philip_slip_length(
    a: NumericType,
    b: NumericType,
    direction: str = "parallel",
) -> NumericType:
    """Philip (1972) effective slip length for striped superhydrophobic surface.

    Parameters
    ----------
    a : float or array
        Width of gas-liquid interface (free-slip stripe) [length units].
    b : float or array
        Width of solid ridge (no-slip stripe) [length units].
    direction : str, optional
        'parallel' (default) or 'perpendicular' to stripes.

    Returns
    -------
    b_eff : float or array
        Effective slip length [same units as a, b].

    Notes
    -----
    For parallel flow:  b_eff = (a/π) · ln(sec(πφ/2))
    For perpendicular:  b_eff = (a/(2π)) · ln(sec(πφ/2))
    """
    phi = gas_area_fraction(a, b)
    b_eff_over_a = philip_slip_length_dimensionless(phi)
    a_arr = np.asarray(a, dtype=np.float64)

    if direction == "parallel":
        return a_arr * b_eff_over_a
    elif direction == "perpendicular":
        return a_arr * b_eff_over_a / 2.0
    else:
        raise ValueError(f"direction must be 'parallel' or 'perpendicular', got '{direction}'")


def philip_slip_length_lattice(
    a_lu: int,
    b_lu: int,
    direction: str = "parallel",
) -> float:
    """Philip (1972) slip length in lattice units for LBM simulations.

    Parameters
    ----------
    a_lu : int
        Free-slip stripe width in lattice units.
    b_lu : int
        No-slip stripe width in lattice units.
    direction : str
        'parallel' or 'perpendicular'.

    Returns
    -------
    b_eff_lu : float
        Effective slip length in lattice units.
    """
    return float(philip_slip_length(float(a_lu), float(b_lu), direction=direction))
