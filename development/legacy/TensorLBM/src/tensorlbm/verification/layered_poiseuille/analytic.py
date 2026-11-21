"""Analytic steady solutions for (layered) plane-Poiseuille flow.

Motivation / cross-validation target
------------------------------------
CFD-GO/TCLB ships a "developed flow" capability in
``models/multiphase/d3q27_pf_velocity`` that initialises the whole domain with
the analytic fully-developed profile and then measures the error of the run
through phase-weighted integrals.  The two formulas used by TCLB (verbatim,
``Dynamics.c.Rt:265-271``) are::

    if (developedFlow > 0.1) {
        U = 6.0 * Uavg * Y*(HEIGHT - Y)/(HEIGHT*HEIGHT);
        V = 0.0; W = 0.0;
    } else if ( developedPipeFlow_X > 0.1 ){
        U = 2.0 * Uavg * (1 - pow( (sqrt(pow((Y-pipeCentre_Y),2)+pow((Z-pipeCentre_Z),2)) / pipeRadius),2));
        V = 0.0; W = 0.0;
    }

with the settings ``HEIGHT`` / ``Uavg`` / ``developedFlow`` declared in
``Dynamics.R:219-221``.  Both are *plane/axisymmetric single-viscosity*
profiles.  The multiphase models then layer a second component (different
viscosity) on top of this initialisation.

This module provides
  1. ``parabolic_profile`` -- exact port of the TCLB ``developedFlow`` formula;
  2. ``pipe_profile``      -- exact port of the TCLB ``developedPipeFlow_X`` formula;
  3. ``layered_profile``   -- the general N-layer steady solution of

         nu_j u_j''(y) = -g ,      u(0) = u(H) = 0 ,
         u_j = u_{j+1} ,  nu_j u_j' = nu_{j+1} u_{j+1}'  at every interface

     which is the exact reference for the layered case that TCLB validates
     against.  It is solved as an exact 2N x 2N linear system (no iterative
     fitting), which keeps the reference independent of the LBM solver.

All quantities are in lattice units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# TCLB developedFlow / developedPipeFlow (single viscosity)
# --------------------------------------------------------------------------- #
def parabolic_profile(y: np.ndarray, height: float, u_avg: float) -> np.ndarray:
    """TCLB ``developedFlow``: U = 6*Uavg*Y*(HEIGHT-Y)/HEIGHT**2.

    Parameters
    ----------
    y : wall-normal positions (0 at the wall, ``height`` at the opposite wall)
    height : channel height in lattice units
    u_avg : cross-section averaged velocity

    Notes
    -----
    The peak velocity of this profile is ``1.5 * u_avg`` and its cross-section
    average is exactly ``u_avg`` (integral of 6Y(H-Y)/H^2 over [0,H] is 1).
    """
    y = np.asarray(y, dtype=float)
    return 6.0 * u_avg * y * (height - y) / (height * height)


def pipe_profile(r: np.ndarray, radius: float, u_avg: float) -> np.ndarray:
    """TCLB ``developedPipeFlow_X``: U = 2*Uavg*(1-(r/R)**2)."""
    r = np.asarray(r, dtype=float)
    return 2.0 * u_avg * (1.0 - (r / radius) ** 2)


def poiseuille_peak_from_force(g: float, height: float, nu: float) -> float:
    """Peak velocity of a body-force driven plane channel: g*H**2/(8*nu)."""
    return g * height * height / (8.0 * nu)


def poiseuille_force_from_peak(u_peak: float, height: float, nu: float) -> float:
    """Body force that yields ``u_peak`` in a plane channel of height ``height``."""
    return 8.0 * nu * u_peak / (height * height)


# --------------------------------------------------------------------------- #
# N-layer steady solution
# --------------------------------------------------------------------------- #
@dataclass
class LayeredSolution:
    """Exact steady solution of the layered (variable-viscosity) channel.

    Attributes
    ----------
    height : channel height (wall-to-wall) in lattice units
    interfaces : interior interface positions, ascending, in (0, height)
    nus : kinematic viscosity of each layer (len == len(interfaces)+1)
    g : body force (acceleration, lattice units) driving the flow along x
    coeffs : (n_layers, 3) array of (a_j, b_j, c_j) with u_j = a_j y^2 + b_j y + c_j
    """

    height: float
    interfaces: Sequence[float]
    nus: Sequence[float]
    g: float
    coeffs: np.ndarray = field(repr=False)

    # -- evaluation -------------------------------------------------------- #
    def layer_index(self, y: np.ndarray) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return np.searchsorted(np.asarray(self.interfaces, dtype=float), y, side="right")

    def u(self, y: np.ndarray) -> np.ndarray:
        """Streamwise velocity u_x(y) (exact)."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        j = np.clip(self.layer_index(y), 0, len(self.coeffs) - 1)
        a, b, c = self.coeffs[j, 0], self.coeffs[j, 1], self.coeffs[j, 2]
        return a * y * y + b * y + c

    def du_dy(self, y: np.ndarray) -> np.ndarray:
        """Velocity gradient u_x'(y) (exact, piecewise)."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        j = np.clip(self.layer_index(y), 0, len(self.coeffs) - 1)
        return 2.0 * self.coeffs[j, 0] * y + self.coeffs[j, 1]

    def shear_stress(self, y: np.ndarray) -> np.ndarray:
        """Kinematic shear stress nu * du/dy (continuous across interfaces)."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        j = np.clip(self.layer_index(y), 0, len(self.coeffs) - 1)
        nu = np.asarray(self.nus, dtype=float)[j]
        return nu * self.du_dy(y)

    # -- integral metrics (TCLB updateMyGlobals analogue) ------------------ #
    def u_peak(self) -> float:
        dy = self.height / 8192.0
        return float(np.max(self.u(np.linspace(0.0, self.height, 8193)[1:-1])))

    def y_peak(self) -> float:
        ys = np.linspace(0.0, self.height, 65537)[1:-1]
        return float(ys[int(np.argmax(self.u(ys)))])

    def flow_rate_layer(self, layer: int, n: int = 4096) -> float:
        """Volume flux per unit width through layer ``layer`` (integral of u dy)."""
        edges = [0.0, *self.interfaces, self.height]
        lo, hi = edges[layer], edges[layer + 1]
        ys = np.linspace(lo, hi, n + 1)
        vals = self.u(ys)
        return float(np.trapezoid(vals, ys)) if hasattr(np, "trapezoid") else float(
            np.trapz(vals, ys)
        )

    def flow_rate(self) -> float:
        return float(sum(self.flow_rate_layer(j) for j in range(len(self.nus))))

    def mean_velocity(self) -> float:
        return self.flow_rate() / self.height


def solve_layered(height: float, interfaces: Sequence[float], nus: Sequence[float], g: float) -> LayeredSolution:
    """Exact steady solution of the N-layer body-force driven channel.

    Unknowns are ``(b_j, c_j)`` for every layer with ``u_j(y) = a_j y^2 + b_j y + c_j``
    and ``a_j = -g/(2 nu_j)``.  Equations:

    * no slip at both walls                            (2 equations)
    * velocity continuity at each interior interface   (N-1)
    * kinematic shear-stress continuity at interfaces  (N-1)

    total ``2N`` unknowns / ``2N`` equations, solved with a dense LU solve.
    """
    interfaces = np.asarray(interfaces, dtype=float)
    nus = np.asarray(nus, dtype=float)
    n_layers = len(nus)
    if n_layers != len(interfaces) + 1:
        raise ValueError("need len(nus) == len(interfaces) + 1")
    if not np.all(np.diff(np.concatenate(([0.0], interfaces, [height]))) > 0):
        raise ValueError("interfaces must be strictly inside (0, height)")
    if np.any(nus <= 0):
        raise ValueError("viscosities must be positive")

    a = -g / (2.0 * nus)

    # index of unknown b_j / c_j -> 2*j, 2*j+1
    A = np.zeros((2 * n_layers, 2 * n_layers), dtype=float)
    rhs = np.zeros(2 * n_layers, dtype=float)
    row = 0

    # wall at y = 0 (only layer 0 touches it)
    A[row, 1] = 1.0
    rhs[row] = 0.0
    row += 1

    # wall at y = height (only last layer touches it)
    A[row, 2 * (n_layers - 1)] = height
    A[row, 2 * (n_layers - 1) + 1] = 1.0
    rhs[row] = -a[-1] * height * height
    row += 1

    for j in range(n_layers - 1):
        y_i = interfaces[j]
        # u_j(y_i) - u_{j+1}(y_i) = 0
        A[row, 2 * j] = y_i
        A[row, 2 * j + 1] = 1.0
        A[row, 2 * (j + 1)] = -y_i
        A[row, 2 * (j + 1) + 1] = -1.0
        rhs[row] = -(a[j] - a[j + 1]) * y_i * y_i
        row += 1

        # nu_j (2 a_j y_i + b_j) - nu_{j+1} (2 a_{j+1} y_i + b_{j+1}) = 0
        A[row, 2 * j] = nus[j]
        A[row, 2 * (j + 1)] = -nus[j + 1]
        rhs[row] = -(nus[j] * 2.0 * a[j] - nus[j + 1] * 2.0 * a[j + 1]) * y_i
        row += 1

    sol = np.linalg.solve(A, rhs)
    coeffs = np.empty((n_layers, 3), dtype=float)
    coeffs[:, 0] = a
    coeffs[:, 1] = sol[0::2]
    coeffs[:, 2] = sol[1::2]
    return LayeredSolution(
        height=float(height), interfaces=tuple(float(v) for v in interfaces),
        nus=tuple(float(v) for v in nus), g=float(g), coeffs=coeffs,
    )


# --------------------------------------------------------------------------- #
# Lattice geometry helpers (half-way bounce-back wall position)
# --------------------------------------------------------------------------- #
def wall_normal_coordinates(ny: int) -> np.ndarray:
    """Wall-normal coordinate of every fluid node for a ``ny``-row channel.

    Rows ``0`` and ``ny-1`` are solid.  Half-way bounce-back places the wall at
    half a lattice spacing inside the outermost solid row, i.e. at
    ``y_wall = 0.5`` and ``y_wall = ny - 1.5`` in lattice-node coordinates;
    shifting the origin to the lower wall gives coordinates ``y - 0.5`` and a
    wall-to-wall height of ``ny - 2``.  This is the geometry fact the
    "geometry check" of this case verifies numerically before any error is
    quoted (see :func:`fit_effective_height`).
    """
    return np.arange(1, ny - 1, dtype=float) - 0.5


def fit_effective_height(y_wall: np.ndarray, u: np.ndarray, g: float, nu: float) -> float:
    """Recover the effective wall-to-wall height from a single-phase profile.

    Given ``u(y) = g/(2 nu) y (H_eff - y)``, the least-squares value of
    ``H_eff`` at every sample is ``H_eff = y + 2 nu u / (g y)``; the median over
    the interior samples is returned.  A deviation from ``ny - 2`` larger than
    the predeclared tolerance falsifies the geometry assumption and stops the
    analysis before any accuracy number is quoted.
    """
    y_wall = np.asarray(y_wall, dtype=float)
    u = np.asarray(u, dtype=float)
    ok = (y_wall > 1.0) & (y_wall < y_wall.max() - 1.0) & (np.abs(u) > 0)
    est = y_wall[ok] + 2.0 * nu * u[ok] / (g * y_wall[ok])
    return float(np.median(est))

def layered_profile(y: np.ndarray, height: float, interfaces: Sequence[float],
                    nus: Sequence[float], g: float) -> np.ndarray:
    """Convenience wrapper: exact layered profile u_x(y) for the given layer stack."""
    return solve_layered(float(height), interfaces, nus, float(g)).u(y)
