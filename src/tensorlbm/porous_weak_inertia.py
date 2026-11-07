"""Weak-inertia (Forchheimer prefix) drag closure for periodic cylinder arrays.

Motivation
----------
For a periodic array of cylinders driven by a constant body force ``a`` the
steady *resistance* of the medium is

.. math::  \\frac{a}{U_s} = \\frac{\\mu}{K} \\left[ 1 + c_1 Re + c_2 Re^2 + O(Re^3) \\right]

with ``U_s`` the superficial (Darcy) velocity and ``Re = U_s d / \\nu`` based on
the cylinder diameter.  For a *centro-symmetric* microstructure the
weak-inertia theorem (:cite:`mei1991`, :cite:`wodie1991`, :cite:`kochladd1997`)
states that the linear coefficient vanishes identically, ``c_1 = 0``, i.e. the
first inertial correction is quadratic in ``Re`` (equivalently *cubic* in the
velocity).  The classical Forchheimer ``U^2`` term is therefore *not* the
leading correction of a centro-symmetric array at ``Re -> 0``.

This module contains, in one place:

* **geometry** -- boolean solid masks for a square array and for a triangular
  ("hex") array realised as a two-disc rectangular periodic cell.  Geometry
  only; no physics kernel.
* **analytic references** -- the dilute-lattice logarithmic law (slope of
  ``4*pi/f`` against ``ln(1/sqrt(phi))`` is exactly one, an analytic
  consequence of the 2D Stokeslet), the Lamb-Oseen cylinder drag, and the
  weak-inertia law ``R = 1 + c_2 Re^2``.
* **driver** -- a float64 D2Q9 TRT driver that reuses the library physics
  kernels (:func:`tensorlbm.solver.collide_trt`, :func:`tensorlbm.solver.stream`,
  :func:`tensorlbm.boundary.boundaries.bounce_back_cells`,
  :func:`tensorlbm.d2q9.equilibrium`/:func:`~tensorlbm.d2q9.macroscopic`) and
  adds only the Guo (2002) second-order body-force source term.
* **estimators** -- polynomial expansion fits with covariance, Ma -> 0
  extrapolation, and dilute-limit log-law fits.

The point of the module is that the *reference* used to judge an LBM campaign
can be produced from first principles in-house (analytic asymptotic structure
+ self-consistent extrapolation) instead of being digitised from a published
figure.

Units
-----
Everything is in lattice units (``dx = dt = 1``), ``c_s^2 = 1/3``.  The Mach
number quoted below is the *interstitial* one, ``Ma = u_fluid / c_s``.

References
----------
* Mei, C.C. & Auriault, J.-L. (1991) The effect of weak inertia on flow through
  a porous medium. *J. Fluid Mech.* 222, 647-663.
* Wodie, A. & Levy, T. (1991) Correction non lineaire de la loi de Darcy.
  *C. R. Acad. Sci. Paris II* 312, 157-161.
* Koch, D.L. & Ladd, A.J.C. (1997) Moderate Reynolds number flows through
  periodic and random arrays of aligned cylinders. *J. Fluid Mech.* 349, 31-66.
* Sangani, A.S. & Acrivos, A. (1982) Slow flow past periodic arrays of
  cylinders with application to heat transfer. *Int. J. Multiphase Flow* 8,
  193-206.
* Hasimoto, H. (1959) On the periodic fundamental solutions of the Stokes
  equations and their application to viscous flow past a cubic array of
  spheres. *J. Fluid Mech.* 5, 317-328.
* Guo, Z., Zheng, C. & Shi, B. (2002) Discrete lattice effects on the forcing
  term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308.
* Lamb, H. (1932) *Hydrodynamics*, 6th ed., Sec. 343 (Oseen cylinder drag).
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .boundary.boundaries import bounce_back_cells
from .d2q9 import _c_on, _w_for, equilibrium, macroscopic
from .solver import collide_trt, stream

__all__ = [
    "GEOMETRIES",
    "ArrayGeometry",
    "LAMBD_SEVERITY",
    "build_array",
    "dilute_log_law_fit",
    "dilute_log_law_slope",
    "expansion_fit",
    "free_intercept_fit",
    "extrapolate_origin",
    "guo_source_term",
    "lbm_step",
    "hex_array_mask",
    "lamb_oseen_cd",
    "resistance_ratio",
    "run_fixed_steps",
    "run_to_steady",
    "power_law_extrapolate",
    "tail_extrapolate",
    "solve_for_reynolds",
    "square_array_mask",
    "wide_horizon_fit",
    "weak_inertia_ratio",
]

CS = 1.0 / math.sqrt(3.0)
SQRT3 = math.sqrt(3.0)
EULER_GAMMA = 0.5772156649015329
GEOMETRIES = ("square", "hex")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrayGeometry:
    """A periodic array of circular cylinders described by one unit cell.

    Attributes:
        kind: ``"square"`` or ``"hex"``.
        n: base grid resolution (edge length of the square cell, or the
            ``x``-period of the hex cell).
        phi_target: design solid fraction handed to the mask builder.
        nx, ny: cell size in lattice nodes.
        phi_actual: realised solid fraction of the boolean mask (always
            slightly below ``phi_target`` because of pixelisation).
        radius: nominal disc radius in lattice units.
        diameter: nominal cylinder diameter ``2*radius`` in lattice units; this
            is the length scale entering ``Re = U_s d / nu``.
        n_solid: number of solid nodes.
        mask: boolean tensor of shape ``(ny, nx)``, ``True`` == solid.
    """

    kind: str
    n: int
    phi_target: float
    nx: int
    ny: int
    phi_actual: float
    radius: float
    diameter: float
    n_solid: int
    mask: torch.Tensor = field(repr=False)

    @property
    def fluid_fraction(self) -> float:
        """Volumetric fluid fraction of the discretised mask."""
        return 1.0 - self.phi_actual

    @property
    def cell_length(self) -> float:
        """Lattice length of one periodic cell *per disc*.

        ``sqrt(cell area / number of discs)``: ``n`` for the square array,
        ``sqrt(nx*ny/2)`` for the two-disc hex cell.  This is the length the
        Sangani & Acrivos pressure-form drag ``f = L^2 / ((1 - phi) K)`` is
        normalised by, so it is the right scale for any comparison with their
        table (and with ``benchmarks/verified/permeability``).
        """
        n_discs = 1 if self.kind == "square" else 2
        return math.sqrt(self.nx * self.ny / n_discs)

    def meta(self) -> dict[str, Any]:
        """JSON-serialisable description (mask itself excluded)."""
        return {
            "kind": self.kind,
            "n": self.n,
            "nx": self.nx,
            "ny": self.ny,
            "phi_target": self.phi_target,
            "phi_actual": self.phi_actual,
            "radius": self.radius,
            "diameter": self.diameter,
            "cell_length": self.cell_length,
            "n_solid": self.n_solid,
            "mask_rule": "periodic folded distance <= radius, float64, disc centres per cell",
        }


def _folded_coord(m: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Coordinates folded to ``[-m/2, m/2)`` so that images share a cell."""
    idx = torch.arange(m, dtype=dtype, device=device)
    return torch.where(idx < m / 2.0, idx, idx - m)


def _min_dist2(
    nx: int,
    ny: int,
    centres: Sequence[tuple[float, float]],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Squared distance to the nearest lattice image of any disc centre."""
    xs = _folded_coord(nx, dtype, device)
    ys = _folded_coord(ny, dtype, device)
    best = torch.full((ny, nx), float("inf"), dtype=dtype, device=device)
    for cx, cy in centres:
        # fold the centre into the same window as the coordinates
        dx = torch.remainder(xs - cx + nx / 2.0, float(nx)) - nx / 2.0
        dy = torch.remainder(ys - cy + ny / 2.0, float(ny)) - ny / 2.0
        d2 = dy[:, None] ** 2 + dx[None, :] ** 2
        best = torch.minimum(best, d2)
    return best


def _cylinder_mask(
    nx: int,
    ny: int,
    centres: Sequence[tuple[float, float]],
    radius: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean solid mask for discs of given radius at periodic images."""
    device = torch.device(device)
    d2 = _min_dist2(nx, ny, centres, dtype, device)
    return d2 <= radius * radius


def square_array_mask(
    n: int,
    phi: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Solid mask of a square array of cylinders, one disc per ``n x n`` cell.

    ``phi = pi R^2 / n^2`` fixes the nominal radius ``R = n sqrt(phi/pi)``.
    """
    radius = n * math.sqrt(phi / math.pi)
    return _cylinder_mask(n, n, [(n / 2.0, n / 2.0)], radius, dtype=dtype, device=device)


def hex_array_mask(
    n: int,
    phi: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Solid mask of a triangular ("hex") array as a two-disc rectangular cell.

    The cell is ``nx x ny`` with ``nx = n`` and ``ny = round(n*sqrt(3))`` and
    holds two discs, at ``(0, 0)`` and ``(n/2, ny/2)``.  Per disc the cell area
    is ``n^2 sqrt(3)/2``, i.e. a triangular lattice of spacing ``n``; the
    nominal radius follows from ``phi = 2 pi R^2 / (nx ny)``.
    """
    nx, ny = n, int(round(n * SQRT3))
    radius = math.sqrt(phi * nx * ny / (2.0 * math.pi))
    return _cylinder_mask(
        nx, ny, [(0.0, 0.0), (nx / 2.0, ny / 2.0)], radius, dtype=dtype, device=device
    )


def build_array(
    kind: str,
    n: int,
    phi: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> ArrayGeometry:
    """Build a validated :class:`ArrayGeometry`.

    Raises:
        ValueError: unknown ``kind`` or a geometry whose disc does not fit in
            its cell.
    """
    if kind == "square":
        mask = square_array_mask(n, phi, dtype=dtype, device=device)
        nx = ny = n
        radius = n * math.sqrt(phi / math.pi)
    elif kind == "hex":
        mask = hex_array_mask(n, phi, dtype=dtype, device=device)
        nx, ny = n, int(round(n * SQRT3))
        radius = math.sqrt(phi * nx * ny / (2.0 * math.pi))
    else:  # pragma: no cover - defensive
        raise ValueError(f"kind must be one of {GEOMETRIES}, got {kind!r}")
    n_solid = int(mask.sum().item())
    phi_actual = n_solid / float(nx * ny)
    if n_solid == 0 or n_solid == nx * ny:
        raise ValueError(
            f"{kind} array at n={n}, phi={phi}: mask is degenerate "
            f"(n_solid={n_solid} of {nx * ny})"
        )
    return ArrayGeometry(
        kind=kind,
        n=n,
        phi_target=phi,
        nx=nx,
        ny=ny,
        phi_actual=phi_actual,
        radius=radius,
        diameter=2.0 * radius,
        n_solid=n_solid,
        mask=mask,
    )


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------

#: Lamb's constant: ``0.5 - gamma + ln 8``.
LAMB_CONST = 0.5 - EULER_GAMMA + math.log(8.0)  # = 2.00222...

#: Meander LAMBD_SEVERITY is kept for backwards-compatible imports.
LAMBD_SEVERITY = LAMB_CONST


def lamb_oseen_cd(re: float) -> float:
    """Lamb-Oseen drag coefficient of a circular cylinder, per unit length.

    ``C_D = 8 pi / (Re (LAMB_CONST - ln Re))`` with ``Re = U d / nu``.
    Valid for ``Re << 1`` only, and singular at ``Re = exp(LAMB_CONST) ~ 7.4``:
    in two dimensions the steady low-Reynolds-number cylinder drag has *no*
    finite unbounded-flow limit (Stokes paradox), so this reference carries a
    logarithmic outer-scale dependence and must be matched to the confinement
    scale before it can be used quantitatively.
    """
    if re <= 0.0:
        raise ValueError("Re must be positive")
    denom = LAMB_CONST - math.log(re)
    if denom <= 0.0:
        return float("inf")
    return 8.0 * math.pi / (re * denom)


def dilute_log_law_slope(phis: Iterable[float], drags: Iterable[float]) -> dict[str, float]:
    """Fit the dilute-lattice logarithmic law ``4 pi / f = ln(1/sqrt(phi)) + C``.

    The *slope* is an analytic result: for a dilute periodic array the drag per
    unit length obeys ``f = 4 pi / (ln(1/sqrt(phi)) + O(1))`` because the outer
    scale of the two-dimensional Stokeslet field is the cell size and
    ``ln(cell/radius) = ln(1/sqrt(phi)) + O(1)``.  The slope is therefore a
    parameter-free analytic anchor (independent of any tabulated constant).

    Args:
        phis: solid fractions (ascending, all small, ``phi <= 0.1``).
        drags: matching dimensionless drags ``f = F / (mu U)``.

    Returns:
        Dict with ``slope``, ``intercept``, ``rms``, ``max_abs_resid`` and the
        ``n`` used.
    """
    ph = [float(p) for p in phis]
    dr = [float(f) for f in drags]
    if len(ph) != len(dr) or len(ph) < 2:
        raise ValueError("need matching phis/drags with at least two points")
    x = [math.log(1.0 / math.sqrt(p)) for p in ph]
    y = [4.0 * math.pi / f for f in dr]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
    return {
        "slope": slope,
        "intercept": intercept,
        "rms": math.sqrt(sum(r * r for r in resid) / n),
        "max_abs_resid": max(abs(r) for r in resid),
        "n": n,
    }


def dilute_log_law_fit(
    phis: Sequence[float], drags: Sequence[float]
) -> dict[str, float]:
    """Fit the dilute-lattice law with the analytic log coefficient held at one.

    For a dilute periodic array the drag per unit length of a cylinder diverges
    logarithmically with the ratio of the cell size to the cylinder radius,

    .. math::  4 \\pi / f = \\ln(1/\\sqrt{\\phi}) + C + A \\phi + O(\\phi^2),

    and the coefficient of ``ln(1/sqrt(phi))`` is *exactly* one -- it is the
    prefactor of the two-dimensional Stokeslet logarithm and needs no tabulated
    constant.  Fixing that coefficient and fitting only ``(C, A)`` therefore
    tests the analytic structure itself; ``A`` absorbs the leading
    non-logarithmic lattice correction, which is what a finite-``phi`` window
    sees.  A free-slope fit is returned as well so that the size of the O(phi)
    shift is visible.

    Args:
        phis: solid fractions (ascending, ``phi <= 0.1``).
        drags: matching pressure-form drags ``f = L^2/((1-phi)K)``.

    Returns:
        Dict with ``intercept``, ``slope_phi``, ``max_rel_resid``, ``rms``, the
        free-slope diagnostics ``slope_free``/``intercept_free`` and ``n``.
    """
    import numpy as np

    ph = np.asarray([float(v) for v in phis], dtype=float)
    f = np.asarray([float(v) for v in drags], dtype=float)
    if ph.size != f.size or ph.size < 3:
        raise ValueError("dilute_log_law_fit needs matching phis/drags with >= 3 points")
    x = np.log(1.0 / np.sqrt(ph))
    y = 4.0 * np.pi / f
    # constrained: y - x = C + A phi
    basis = np.column_stack([np.ones_like(ph), ph])
    sol, *_ = np.linalg.lstsq(basis, y - x, rcond=None)
    resid = (y - x) - basis @ sol
    free = dilute_log_law_slope(ph.tolist(), f.tolist())
    return {
        "intercept": float(sol[0]),
        "slope_phi": float(sol[1]),
        "max_rel_resid": float(np.max(np.abs(resid) / np.abs(y))),
        "rms": float(np.sqrt(float(resid @ resid) / ph.size)),
        "slope_free": float(free["slope"]),
        "intercept_free": float(free["intercept"]),
        "n": int(ph.size),
    }


def weak_inertia_ratio(re: float | Sequence[float], c2: float) -> Any:
    """Weak-inertia drag ratio ``R(Re) = 1 + c2 Re^2`` (``c1 = 0``)."""
    if isinstance(re, (int, float)):
        return 1.0 + c2 * float(re) ** 2
    return [1.0 + c2 * float(r) ** 2 for r in re]


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def _polyfit(x: Sequence[float], y: Sequence[float], degree: int) -> dict[str, Any]:
    """Least-squares polynomial fit with coefficient covariance (numpy only)."""
    import numpy as np

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size or xa.size <= degree:
        raise ValueError("need len(x) == len(y) > degree")
    design = np.vander(xa, degree + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(design, ya, rcond=None)
    resid = ya - design @ coeffs
    dof = max(xa.size - (degree + 1), 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:  # pragma: no cover - pathological design
        cov = np.full((degree + 1, degree + 1), float("nan"))
    return {
        "coeffs": [float(c) for c in coeffs],
        "se": [float(math.sqrt(max(cov[i, i], 0.0))) for i in range(degree + 1)],
        "rms": float(math.sqrt(float(resid @ resid) / xa.size)),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "n": int(xa.size),
        "design_condition": float(np.linalg.cond(design)),
    }


def expansion_fit(re: Sequence[float], ratio: Sequence[float], *, degree: int = 2) -> dict[str, Any]:
    """Fit ``R(Re) - 1 = c1 Re + c2 Re^2 (+ ...)`` on a Reynolds window.

    Returns:
        Dict with ``c1``, ``c2``, their standard errors, ``c1_share`` (the
        linear term's contribution relative to the quadratic term at the top of
        the window, the machine-checkable form of "``c1`` matches the
        literature value zero"), and fit-quality diagnostics.
    """
    re_l = [float(r) for r in re]
    excess = [float(r) - 1.0 for r in ratio]
    fit = _polyfit(re_l, excess, degree)
    # _polyfit returns an intercept column even though the model has none; the
    # intercept is reported as a diagnostic (it must be zero for a clean fit).
    c1 = fit["coeffs"][1]
    c2 = fit["coeffs"][2] if degree >= 2 else 0.0
    re_max = max(re_l)
    quad = abs(c2) * re_max**2
    lin = abs(c1) * re_max
    fit.update(
        {
            "c1": c1,
            "c2": c2,
            "intercept": fit["coeffs"][0],
            "se_c1": fit["se"][1],
            "se_c2": fit["se"][2] if degree >= 2 else float("nan"),
            "re_max": re_max,
            "c1_share": (lin / quad) if quad > 0.0 else float("inf"),
            "c1_sigma_significance": (
                abs(c1) / fit["se"][1] if fit["se"][1] > 0.0 else float("inf")
            ),
            "degree": degree,
        }
    )
    return fit


def free_intercept_fit(
    re: Sequence[float], ratio: Sequence[float], *, degree: int = 2
) -> dict[str, Any]:
    """Fit ``R(Re) = c0 + c1 Re + c2 Re^2`` with a *free* intercept.

    Unlike :func:`expansion_fit` (which fits ``R - 1`` and therefore forces the
    zero-Reynolds resistance to unity) this fit estimates the offset ``c0`` from
    the data, so a systematic resistance offset is observable.  Both fits are
    reported: ``c1`` and ``c2`` from :func:`expansion_fit` (the weak-inertia
    expansion) and ``c0 - 1`` from here (the Stokes-limit gate).
    """
    re_l = [float(r) for r in re]
    rat = [float(r) for r in ratio]
    fit = _polyfit(re_l, rat, degree)
    re_max = max(re_l)
    quad = abs(fit["coeffs"][2]) * re_max**2 if degree >= 2 else float("nan")
    return {
        "c0": fit["coeffs"][0],
        "c1": fit["coeffs"][1],
        "c2": fit["coeffs"][2] if degree >= 2 else float("nan"),
        "se_c0": fit["se"][0],
        "se_c1": fit["se"][1],
        "se_c2": fit["se"][2] if degree >= 2 else float("nan"),
        "c1_share": abs(fit["coeffs"][1]) * re_max / quad if quad and quad > 0 else float("inf"),
        "rms": fit["rms"],
        "max_abs_resid": fit["max_abs_resid"],
        "design_condition": fit["design_condition"],
        "n": fit["n"],
    }


def extrapolate_origin(
    x: Sequence[float], y: Sequence[float], *, degree: int = 2
) -> dict[str, Any]:
    """Polynomial extrapolation of ``y(x)`` to ``x = 0`` with residuals.

    Used for the Ma -> 0 limit.  ``residual`` is the relative difference
    between the fitted value at the origin and the fitted value at the
    smallest measured abscissa, i.e. the size of the correction the
    extrapolation removes; ``rms`` and ``max_abs_resid`` are the fit residuals.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    fit = _polyfit(xs, ys, degree)
    y0 = fit["coeffs"][0]
    xmin = min(xs)
    y_at_min = sum(c * xmin**k for k, c in enumerate(fit["coeffs"]))
    return {
        "value_at_origin": y0,
        "value_at_xmin": y_at_min,
        "relative_correction": abs(y_at_min - y0) / abs(y0) if y0 else float("inf"),
        "coeffs": fit["coeffs"],
        "se": fit["se"],
        "rms": fit["rms"],
        "max_abs_resid": fit["max_abs_resid"],
        "degree": degree,
        "n": fit["n"],
    }


def wide_horizon_fit(
    re: Sequence[float], ratio: Sequence[float], *, degree: int = 2
) -> dict[str, Any]:
    """Alias of :func:`expansion_fit` used for fits on the wider Re horizon."""
    return expansion_fit(re, ratio, degree=degree)


def resistance_ratio(
    a: float, u_superficial: float, *, a_ref: float, u_ref: float
) -> float:
    """Drag (resistance) ratio ``R = (a/U)|_case / (a/U)|_reference``."""
    if u_superficial <= 0.0 or u_ref <= 0.0:
        raise ValueError("superficial velocities must be positive")
    return (a / u_superficial) / (a_ref / u_ref)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


#: D2Q9 opposite-direction table (same order as ``tensorlbm.d2q9.C``).
_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)


@functools.cache
def _tables(
    dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(c_ix, c_iy, w)`` in the requested dtype, shaped for broadcasting."""
    c = _c_on(device).to(dtype)
    w = _w_for(dtype, device)
    return c[:, 0].view(9, 1, 1), c[:, 1].view(9, 1, 1), w.view(9, 1, 1)


@functools.cache
def _opposite(device: torch.device) -> torch.Tensor:
    """Opposite-direction index table (cached)."""
    return torch.tensor(_OPP, device=device)


def guo_source_term(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    a: float,
    tau: float,
    *,
    mass_conserving: bool = True,
) -> torch.Tensor:
    """Guo (2002) second-order body-force source term for D2Q9.

    ``S_i = (1 - 1/(2 tau)) w_i rho [3 (c_ix - u_x) + 9 (c_ix u_x + c_iy u_y) c_ix] a``

    The velocity ``u`` that enters ``S`` and the equilibrium must be the
    *half-force-corrected* hydrodynamic velocity ``u* = sum_i f_i c_i / rho + F/(2 rho)``;
    pairing the prefactored source with the plain moment velocity is a common
    mistake and injects only ``(1 - 1/(2 tau)) a`` per step (measured: a factor
    2.03 too little flow at ``tau = 1``, and a spurious ``tau`` dependence).
    :func:`lbm_step` couples the two correctly.

    ``sum_i S_i = 0`` holds identically (opposite directions cancel), and
    ``sum_i S_i c_i = (1 - 1/(2 tau)) rho a``; with the ``F/2`` velocity
    correction this makes the discrete force exactly ``rho a``.
    """
    cix, ciy, w = _tables(rho.dtype, rho.device)
    pref = (1.0 - 0.5 / tau) * a
    term = 3.0 * (cix - ux.unsqueeze(0)) + 9.0 * (cix * ux.unsqueeze(0) + ciy * uy.unsqueeze(0)) * cix
    src = pref * w * rho.unsqueeze(0) * term
    if mass_conserving:
        src = src - w * src.sum(dim=0, keepdim=True)
    return src


def lbm_step(
    f: torch.Tensor,
    solid: torch.Tensor,
    tau: float,
    a: float,
    *,
    scheme: str = "split",
) -> torch.Tensor:
    """One D2Q9 TRT step with second-order (Guo) body forcing on fluid nodes.

    Two forcing schemes are provided:

    ``scheme="split"`` (default, used by the campaign)
        collision -> streaming -> unit-prefactor momentum kick on fluid nodes ->
        bounce-back.  This is exactly the loop of the verified
        ``benchmarks/verified/permeability`` case, which is the anchor: at
        ``phi = 0.30, d = 26`` it reproduces the reference permeability to
        5e-4 relative (``K = 10.014`` lattice units).  The measured velocity is
        the plain moment velocity.
    ``scheme="guo"``
        collision with the Guo (2002) source inserted, using the half-force
        corrected velocity ``u* = M/rho + F/(2 rho)`` in both the equilibrium
        and the source, then streaming, then bounce-back.  Formally second order
        in the force, but it does *not* reproduce the verified anchor: at the
        same ``phi = 0.30, d = 26`` cell it gives ``K = 10.54`` (+5.3 % versus
        the anchored chain, i.e. the two schemes differ by more than the
        ``d``-convergence band of the anchor at that resolution).  Resolving
        which wall position is right needs its own grid study, so the campaign
        stays on the anchored split scheme; the Guo path is kept for
        reproducibility of that comparison.

    The TRT anti-symmetric rate uses the magic parameter ``Lambda = 3/16``.

    Args:
        f: distribution of shape ``(9, ny, nx)``.
        solid: boolean solid mask ``(ny, nx)``.
        tau: TRT ``tau_plus``; ``nu = (tau - 0.5)/3``.
        a: streamwise body-force acceleration (per unit fluid mass).

    Returns:
        The distribution after one step.
    """
    dtype = f.dtype
    device = f.device
    cix, ciy, w = _tables(dtype, device)
    fluid = ~solid
    fluid_b = fluid.unsqueeze(0)
    opp = _opposite(device)

    if scheme == "split":
        f_post = _trt_collide(f, tau, cix, ciy, w, opp)
        f_stream = stream(f_post)
        if a != 0.0:
            kick = 3.0 * a * w * f_stream.sum(dim=0).unsqueeze(0) * cix
            f_stream = torch.where(fluid_b, f_stream + kick, f_stream)
        return bounce_back_cells(f_stream, solid)

    if scheme == "guo":
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-30)
        a_field = a * fluid.to(dtype)
        ux = (f * cix).sum(dim=0) / rho_safe + 0.5 * a_field
        uy = (f * ciy).sum(dim=0) / rho_safe
        cu = cix * ux.unsqueeze(0) + ciy * uy.unsqueeze(0)
        u_sq = ux * ux + uy * uy
        feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
        f_post = _trt_collide(f, tau, cix, ciy, w, opp, feq=feq)
        if a != 0.0:
            src = guo_source_term(rho, ux, uy, a, tau)
            f_post = torch.where(fluid_b, f_post + src, f_post)
        return bounce_back_cells(stream(f_post), solid)

    raise ValueError(f"unknown forcing scheme {scheme!r}; use 'split' or 'guo'")


def _trt_collide(
    f: torch.Tensor,
    tau: float,
    cix: torch.Tensor,
    ciy: torch.Tensor,
    w: torch.Tensor,
    opp: torch.Tensor,
    *,
    feq: torch.Tensor | None = None,
) -> torch.Tensor:
    """TRT collision with the magic parameter ``Lambda = 3/16``."""
    if feq is None:
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-30)
        ux = (f * cix).sum(dim=0) / rho_safe
        uy = (f * ciy).sum(dim=0) / rho_safe
        cu = cix * ux.unsqueeze(0) + ciy * uy.unsqueeze(0)
        u_sq = ux * ux + uy * uy
        feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    tau_minus = 0.5 + (3.0 / 16.0) / (tau - 0.5)
    f_plus = 0.5 * (f + f[opp])
    f_minus = 0.5 * (f - f[opp])
    feq_plus = 0.5 * (feq + feq[opp])
    feq_minus = 0.5 * (feq - feq[opp])
    return f - (f_plus - feq_plus) / tau - (f_minus - feq_minus) / tau_minus


def _hydro_velocity(
    f: torch.Tensor,
    a: float = 0.0,
    solid: torch.Tensor | None = None,
    *,
    half_force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Velocity ``u = sum_i f_i c_i / rho (+ F/(2 rho))``.

    The half-force term belongs to the Guo scheme only; the split scheme's kick
    already enters the moments, and its use is what the anchored permeability
    benchmark does.
    """
    cix, ciy, _ = _tables(f.dtype, f.device)
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-30)
    ux = (f * cix).sum(dim=0) / rho_safe
    uy = (f * ciy).sum(dim=0) / rho_safe
    if half_force and a != 0.0 and solid is not None:
        fluid = ~solid
        ux = torch.where(fluid, ux + 0.5 * a, ux)
    return rho, ux, uy


@dataclass
class SteadyState:
    """Outcome of one steady-state LBM solve."""

    f: torch.Tensor = field(repr=False)
    u_fluid: float
    u_superficial: float
    u_max: float
    rho_mean: float
    ma: float
    steps: int
    drift: float
    converged: bool
    wall_s: float
    u_raw: float = float("nan")
    tail_ratio: float = float("nan")
    monitor: str = "drift"

    def meta(self) -> dict[str, Any]:
        return {
            "u_fluid": self.u_fluid,
            "u_superficial": self.u_superficial,
            "u_raw": self.u_raw,
            "u_max": self.u_max,
            "ma": self.ma,
            "rho_mean": self.rho_mean,
            "steps": self.steps,
            "drift": self.drift,
            "tail_ratio": self.tail_ratio,
            "monitor": self.monitor,
            "converged": self.converged,
            "wall_s": self.wall_s,
        }


def tail_extrapolate(u0: float, u1: float, u2: float) -> tuple[float, float]:
    """Aitken/Richardson extrapolation of a geometric time tail.

    Given three equally spaced samples ``u0, u1, u2`` of a signal that relaxes
    as ``u_k = u_inf + A q^k``, returns ``(u_inf, q)``.  ``q`` is clamped to
    ``[-0.999, 0.999]`` to keep the estimate finite for noisy samples.
    """
    d1 = u1 - u0
    d2 = u2 - u1
    if d1 == 0.0 or not (d1 * d2 > 0.0):
        return u2, float("nan")
    q = d2 / d1
    if not (-0.999 < q < 0.999):
        return u2, q
    return u2 + d2 * q / (1.0 - q), q


def power_law_extrapolate(
    times: Sequence[float],
    values: Sequence[float],
    *,
    p_bounds: tuple[float, float] = (0.3, 3.0),
    n_scan: int = 271,
) -> dict[str, float]:
    """Extrapolate an *algebraic* time tail ``u(t) = u_inf + A t^{-p}``.

    Two-dimensional Stokes relaxation in a periodic cell is not exponential: the
    residuals of a forced solve are potential-flow-like and decay algebraically.
    (Measured on the ``phi = 0.40`` cell: the Aitken geometric ratio of
    successive sample differences exceeds one, so an Aitken estimate is
    inadmissible.)  This routine scans the decay exponent ``p`` and, for each
    ``p``, solves the remaining *linear* least-squares problem for
    ``(u_inf, A)``.

    Returns:
        Dict with ``u_inf``, ``p``, ``rms`` (fit residual) and ``n``.
    """
    import numpy as np

    t = np.asarray(times, dtype=float)
    u = np.asarray(values, dtype=float)
    if t.size != u.size or t.size < 4:
        raise ValueError("power_law_extrapolate needs >= 4 samples")
    t = t - t[0] + 1.0  # shift to avoid t <= 0
    best: dict[str, float] = {"u_inf": float(u[-1]), "p": float("nan"), "rms": float("inf"), "n": int(t.size)}
    for pp in np.linspace(p_bounds[0], p_bounds[1], n_scan):
        basis = np.column_stack([np.ones_like(t), t ** (-pp)])
        sol, *_ = np.linalg.lstsq(basis, u, rcond=None)
        resid = u - basis @ sol
        rms = float(np.sqrt(float(resid @ resid) / t.size))
        if rms < best["rms"]:
            best = {"u_inf": float(sol[0]), "p": float(pp), "rms": rms, "n": int(t.size)}
    return best


def run_to_steady(
    geom: ArrayGeometry,
    tau: float,
    a: float,
    *,
    f_init: torch.Tensor | None = None,
    u_init: float | None = None,
    dtype: torch.dtype = torch.float64,
    max_steps: int = 400_000,
    sample_every: int = 50,
    drift_tol: float = 1e-9,
    drift_span: int = 20,
    repeats: int = 3,
    extrap_tol: float = 1e-10,
    device: torch.device | str = "cpu",
) -> SteadyState:
    """March a forced periodic array to a steady state and report moments.

    Args:
        geom: array geometry (mask fixed for the whole run).
        tau: TRT ``tau_plus``; viscosity ``nu = (tau - 1/2)/3``.  The TRT magic
            parameter is left at the library default ``3/16`` so that the
            bounce-back wall location is independent of ``nu`` (important for
            the Mach-number sweep, which must vary ``nu`` at fixed ``Re``).
        a: streamwise body-force acceleration (lattice units).
        f_init: optional warm start (distribution of a previous solution).
        u_init: optional uniform initial velocity (ignored when ``f_init`` is
            given); a good guess removes most of the diffusive transient.
        max_steps: hard step cap.
        sample_every: sampling stride for the steady-state monitor.
        drift_tol: relative change of ``u_fluid`` over ``drift_span`` samples
            below which the run counts as steady.
        drift_span: number of samples in the trailing drift window.
        repeats: consecutive passing evaluations required.
        extrap_tol: relative stability threshold of the Aitken-extrapolated
            steady value; the design of a periodic Stokes solve relaxes with a
            long geometric tail (measured ratio ~0.8 per 500 steps), so
            extrapolating the tail is far cheaper than waiting for the raw
            monitor to reach ``drift_tol``.
        device: torch device.

    Returns:
        :class:`SteadyState`; ``converged`` is ``False`` if the cap was hit.
    """
    device = torch.device(device)
    solid = geom.mask.to(device)
    fluid = ~solid
    fluid_b = fluid.unsqueeze(0)
    n_fluid = int(fluid.sum().item())

    if f_init is not None:
        f = f_init.to(dtype=dtype, device=device).clone()
    else:
        u0 = float(u_init or 0.0)
        rho0 = torch.ones((geom.ny, geom.nx), dtype=dtype, device=device)
        ux0 = torch.full_like(rho0, u0)
        f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=device)

    t0 = time.perf_counter()
    hist: list[float] = []
    passes = 0
    steady_step = 0
    drift = float("nan")
    step = 0
    converged = False
    monitor = "not_converged"
    q = float("nan")
    u_inf = float("nan")
    u_prev_ext = float("nan")
    while step < max_steps:
        for _ in range(sample_every):
            f = lbm_step(f, solid, tau, a)
        step += sample_every
        _, uxh, _ = _hydro_velocity(f, a, solid)
        hist.append(float(uxh[fluid].mean().item()))
        if len(hist) > drift_span:
            prev = hist[-1 - drift_span]
            cur = hist[-1]
            drift = abs(cur - prev) / max(abs(cur), 1e-300)
        extrap_ok = False
        if len(hist) >= 8 and len(hist) % 2 == 0:
            times = [float(sample_every * k) for k in range(1, len(hist) + 1)]
            ext = power_law_extrapolate(times[-min(len(times), 400):], hist[-min(len(hist), 400):])
            if math.isfinite(ext["u_inf"]):
                u_inf = ext["u_inf"]
                q = ext["p"]
                if math.isfinite(u_prev_ext) and abs(u_inf - u_prev_ext) <= extrap_tol * abs(u_inf):
                    extrap_ok = True
                u_prev_ext = u_inf
        if (len(hist) > drift_span and drift < drift_tol) or extrap_ok:
            passes += 1
            if passes >= repeats:
                converged = True
                steady_step = step
                monitor = "extrapolation" if extrap_ok else "drift"
                break
        else:
            passes = 0
    else:
        monitor = "not_converged"

    # moments of the final state
    rho, uxh, uyh = _hydro_velocity(f, a, solid)
    u_raw = float(uxh[fluid].mean().item())
    u_fluid = u_raw
    if math.isfinite(u_inf) and abs(q) < 0.999:
        u_fluid = u_inf
    u_sup = u_fluid * n_fluid / float(geom.nx * geom.ny)
    u_max = float(torch.sqrt(uxh[fluid] ** 2 + uyh[fluid] ** 2).max().item())
    return SteadyState(
        f=f,
        u_fluid=u_fluid,
        u_superficial=u_sup,
        u_max=u_max,
        rho_mean=float(rho.mean().item()),
        ma=abs(u_fluid) / CS,
        steps=steady_step if converged else step,
        drift=drift,
        converged=converged,
        wall_s=time.perf_counter() - t0,
        u_raw=u_raw,
        tail_ratio=q,
        monitor=monitor,
    )


def run_fixed_steps(
    geom: ArrayGeometry,
    tau: float,
    a: float,
    steps: int,
    *,
    dtype: torch.dtype = torch.float64,
    sample_every: int = 100,
    device: torch.device | str = "cpu",
) -> SteadyState:
    """Run exactly ``steps`` steps from rest — the *common-protocol* driver.

    Steady-state detection is deliberately absent.  For a Stokes-like problem
    the response to a step body force is linear, ``u(t) = a G(t)``, so two runs
    of the *same* geometry, viscosity and duration share an identical
    *relative* transient and the resistance ratio

    ``R = (a/u)|_Re / (a/u)|_anchor``

    is time-invariant to second order even when neither run is fully steady.
    That is what makes a 1e-4-level inertial signal measurable with a 1e-6
    level transient: the transient is common-mode and cancels in the ratio.

    The remaining relative drift (last decile of the sample history) is
    reported as an uncertainty proxy.

    Args:
        geom: array geometry.
        tau: TRT ``tau_plus`` (``nu = (tau - 0.5)/3``).
        a: streamwise body-force acceleration.
        steps: exact number of lattice steps.
        dtype: torch dtype (float64 recommended).
        sample_every: sampling stride of the drift monitor.
        device: torch device.

    Returns:
        :class:`SteadyState` with ``monitor="fixed_protocol"`` and
        ``converged=True`` when the requested step count completed.
    """
    device = torch.device(device)
    solid = geom.mask.to(device)
    fluid = ~solid
    fluid_b = fluid.unsqueeze(0)
    n_fluid = int(fluid.sum().item())
    rho0 = torch.ones((geom.ny, geom.nx), dtype=dtype, device=device)
    z = torch.zeros_like(rho0)
    f = equilibrium(rho0, z, z, device=device)

    t0 = time.perf_counter()
    hist: list[float] = []
    for step in range(1, steps + 1):
        f = lbm_step(f, solid, tau, a)
        if step % sample_every == 0:
            _, uxh, _ = _hydro_velocity(f, a, solid)
            hist.append(float(uxh[fluid].mean().item()))

    rho, uxh, uyh = _hydro_velocity(f, a, solid)
    u_fluid = float(uxh[fluid].mean().item())
    n_tail = max(len(hist) // 10, 2)
    drift = abs(hist[-1] - hist[-n_tail]) / max(abs(hist[-1]), 1e-300) if len(hist) > n_tail else float("nan")
    return SteadyState(
        f=f,
        u_fluid=u_fluid,
        u_superficial=u_fluid * n_fluid / float(geom.nx * geom.ny),
        u_max=float(torch.sqrt(uxh[fluid] ** 2 + uyh[fluid] ** 2).max().item()),
        rho_mean=float(rho.mean().item()),
        ma=abs(u_fluid) / CS,
        steps=steps,
        drift=drift,
        converged=True,
        wall_s=time.perf_counter() - t0,
        u_raw=u_fluid,
        tail_ratio=float("nan"),
        monitor="fixed_protocol",
    )


def solve_for_reynolds(
    geom: ArrayGeometry,
    re_target: float,
    *,
    nu: float,
    tau: float | None = None,
    f_init: torch.Tensor | None = None,
    a_guess: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 6,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Find the body force ``a`` that lands ``Re = U_s d / nu`` on target.

    ``Re`` is linear in ``a`` in the Stokes regime, so a secant iteration
    converges in two or three steady solves.  Each solve is warmed from its
    predecessor, which keeps the total cost close to a single solve.

    Returns:
        Dict with ``a``, ``re``, the :class:`SteadyState` (key ``"state"``) and
        the wall time of the whole iteration.
    """
    tau = (3.0 * nu + 0.5) if tau is None else tau
    d = geom.diameter
    u_sup_target = re_target * nu / d
    u_fluid_target = u_sup_target / geom.fluid_fraction
    a = a_guess if a_guess is not None else 1e-6
    history: list[tuple[float, float]] = []
    state: SteadyState | None = None
    t0 = time.perf_counter()
    for _ in range(max_iter):
        state = run_to_steady(
            geom,
            tau,
            a,
            f_init=f_init,
            u_init=u_fluid_target,
            **run_kwargs,
        )
        f_init = state.f
        history.append((a, state.u_superficial))
        re_now = state.u_superficial * d / nu
        if re_now > 0.0 and abs(re_now - re_target) / re_target <= tol:
            break
        # secant step on a (resistance is linear in a at low Re)
        if len(history) >= 2 and history[-1][1] != history[-2][1]:
            slope = (history[-1][1] - history[-2][1]) / (history[-1][0] - history[-2][0])
            a = a + (u_sup_target - history[-1][1]) / slope
        else:
            a = a * (u_sup_target / state.u_superficial) if state.u_superficial > 0 else a * 2.0
        a = max(a, 1e-12)
    assert state is not None
    return {
        "a": a,
        "re": state.u_superficial * d / nu,
        "re_target": re_target,
        "nu": nu,
        "tau": tau,
        "state": state,
        "iterations": len(history),
        "wall_s": time.perf_counter() - t0,
    }