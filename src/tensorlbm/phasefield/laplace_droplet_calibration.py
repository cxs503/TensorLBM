"""Analytic Young-Laplace calibration of a free-energy-LBM droplet.

This module owns the *measurement and calibration protocol* for the curved-interface
Laplace law of the free-energy binary-fluid model implemented in
:mod:`tensorlbm.phasefield.free_energy_pressure` (the sailfish transplant).  It does
not restate the transport kernel: the case runner calls
``initial_condition`` / ``collide_and_stream`` / ``macroscopic`` /
``pressure_tensor_measured`` from there, so any change to the solver is picked up here.

Why a separate protocol is needed
---------------------------------
The flat-interface companion (:anchored ``XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE``)
grades ``P_nn = p_coex`` across a planar interface.  A curved interface adds two things
that the planar protocol cannot see:

1. **A pressure jump.**  For a 2-D (cylinder) droplet the capillary law is
   ``p_in - p_out = sigma / R`` -- one principal curvature, *not* ``2 sigma / R``.
   The jump has to be read from the *bulk* pressures, i.e. from regions where the
   Korteweg term has decayed, otherwise the overweighted ``phi lap(phi)`` band
   contaminates ``Delta p``.
2. **A radius definition.**  A diffuse interface has no unique radius.  Three
   independent definitions are produced here and reported side by side:

   * ``R_contour``   -- azimuthally averaged ``phi = 0`` level set (equimolar /
     ``phi = 0`` surface), by linear interpolation between radial bins;
   * ``R_fit``       -- ``atanh(phi(r)) = (R - r)/xi`` linear regression over the
     interface band (exact for the model tanh profile, gives ``xi`` as a bonus);
   * ``R_sigma``     -- the surface of tension, ``integral r (P_rr - P_tt) dr /
     integral (P_rr - P_tt) dr`` (Rowlinson & Widom 1982).  This is the radius the
     Laplace law refers to in the sharp-interface limit.

Analytic anchors (``f = A(-phi^2/2 + phi^4/4) + kappa |grad phi|^2 / 2``)
-----------------------------------------------------------------------
``xi = sqrt(2 kappa / A)``, ``sigma = (2/3) sqrt(2 kappa A) = (4/3) kappa / xi``,
``phi(r) = -tanh((r - R)/xi)`` (``phi = +1`` inside the droplet).

The identity ``(2/3) sqrt(2 kappa A) == (4/3) kappa / xi`` is *checked numerically*
(:func:`sigma_from_gradient_energy` integrates ``kappa phi'^2`` over the analytic
profile) rather than taken on faith, so a later mismatch on a simulated field is a
statement about the simulation and not about the post-processing.

Pressure-tensor route (exact, no radius needed)
-----------------------------------------------
For a radially symmetric field the radial momentum balance of the free-energy
pressure tensor is ``dP_rr/dr + (P_rr - P_tt)/r = 0``, hence

    ``p_in - p_out = integral_0^inf (P_rr - P_tt)/r dr``                 (capillary)
    ``sigma       = integral_0^inf (P_rr - P_tt) dr``                    (local tension)

Both integrals are computed from the same radial anisotropy profile, which makes the
Laplace residual ``Delta p - sigma/R`` testable without assuming the interface is
infinitely thin: for a *thin* interface ``sigma/R`` reproduces the capillary integral,
and the difference is the finite-``xi``/``R`` correction (reported, not hidden).

All quantities are float64 by default; the module never mutates the populations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .free_energy_pressure import (
    FreeEnergyBinaryParams,
    collide_and_stream,
    initial_condition,
    macroscopic,
    mean_pressure,
    periodic_distance,
    pressure_tensor_analytic,
    pressure_tensor_measured,
    grad_lap,
    tanh_profile,
)

__all__ = [
    "DropletCase",
    "DropletMeasures",
    "LaplaceCriteria",
    "sigma_analytic",
    "xi_analytic",
    "sigma_from_gradient_energy",
    "droplet_phi0",
    "radial_average",
    "interface_series",
    "radius_from_contour",
    "radius_and_width_from_tanh_fit",
    "polar_pressure_components",
    "radial_anisotropy_profile",
    "tension_integrals",
    "bulk_pressure_jump",
    "run_droplet_case",
    "grade_case",
    "grade_report",
]


# --------------------------------------------------------------------------- #
# analytic anchors
# --------------------------------------------------------------------------- #
def xi_analytic(A: float, kappa: float) -> float:
    """Interface width ``sqrt(2 kappa / A)``."""
    _require_positive(A=A, kappa=kappa)
    return math.sqrt(2.0 * kappa / A)


def sigma_analytic(A: float, kappa: float) -> float:
    """Surface tension ``(2/3) sqrt(2 kappa A) == (4/3) kappa / xi``."""
    _require_positive(A=A, kappa=kappa)
    return (2.0 / 3.0) * math.sqrt(2.0 * kappa * A)


def sigma_from_gradient_energy(
    A: float, kappa: float, *, xi_span: float = 12.0, n: int = 240001
) -> float:
    """Integrate ``kappa (d phi/dx)^2`` over the analytic tanh profile.

    ``phi = tanh(x/xi)``, ``d phi/dx = (1 - phi^2)/xi``, so
    ``sigma = (kappa/xi) * integral sech^4`` = ``(4/3) kappa / xi``.  The integral is
    evaluated by the trapezoid rule on a wide window (``|x| <= xi_span * xi``) so the
    closed form is verified numerically rather than assumed.
    """
    xi = xi_analytic(A, kappa)
    if n < 3:
        raise ValueError("n must be >= 3 for the trapezoid rule")
    x = torch.linspace(-xi_span * xi, xi_span * xi, n, dtype=torch.float64)
    dphi = 1.0 - torch.tanh(x / xi) ** 2
    weight = (x[1] - x[0]).item()
    return float((kappa * (dphi / xi) ** 2).sum().item() * weight)


# --------------------------------------------------------------------------- #
# configuration / result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DropletCase:
    """One static-droplet calibration case (all lengths in lattice units)."""

    N: int
    R: float
    A: float = 0.1
    kappa: float = 0.2
    Gamma: float = 0.05
    tau_f: float = 1.0
    tau_phi: float = 1.0
    steps: int = 2000
    sample_every: int = 50
    bulk_offset_xi: float = 3.0
    bin_width: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.N, int) or self.N < 16:
            raise ValueError("N must be an integer >= 16")
        _require_positive(R=self.R, A=self.A, kappa=self.kappa, tau_f=self.tau_f, tau_phi=self.tau_phi)
        if self.Gamma <= 0.0:
            raise ValueError("Gamma must be positive")
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not 0.0 < self.bulk_offset_xi < 0.25 * self.N / xi_analytic(self.A, self.kappa):
            raise ValueError("bulk_offset_xi leaves no bulk region inside the box")

    @property
    def xi(self) -> float:
        return xi_analytic(self.A, self.kappa)

    @property
    def sigma(self) -> float:
        return sigma_analytic(self.A, self.kappa)

    @property
    def params(self) -> FreeEnergyBinaryParams:
        return FreeEnergyBinaryParams(
            A=self.A,
            kappa=self.kappa,
            Gamma=self.Gamma,
            tau_f=self.tau_f,
            tau_phi=self.tau_phi,
        )

    def as_dict(self) -> dict:
        return {
            "N": self.N,
            "R": self.R,
            "A": self.A,
            "kappa": self.kappa,
            "Gamma": self.Gamma,
            "tau_f": self.tau_f,
            "tau_phi": self.tau_phi,
            "steps": self.steps,
            "sample_every": self.sample_every,
            "bulk_offset_xi": self.bulk_offset_xi,
            "bin_width": self.bin_width,
            "xi": self.xi,
            "sigma_analytic": self.sigma,
            "mobility": self.params.mobility,
        }


@dataclass
class DropletMeasures:
    """Measured quantities of one steady droplet state (all plain floats)."""

    R_contour: float
    R_fit: float
    xi_fit: float
    R_sigma: float
    p_in: float
    p_out: float
    dp_bulk: float
    dp_tensor: float
    sigma_integral: float
    u_max: float
    u_rms: float
    sigma_analytic: float
    xi_analytic: float
    radius_over_xi: float
    phi_drift: float
    rho_drift: float
    profile: list = field(default_factory=list)
    history: list = field(default_factory=list)

    @property
    def sigma_capillary(self) -> float:
        """``sigma`` recovered from ``Delta p * R_fit`` (the calibration KPI)."""
        return self.dp_bulk * self.R_fit

    @property
    def sigma_capillary_contour(self) -> float:
        return self.dp_bulk * self.R_contour

    @property
    def sigma_capillary_surface_of_tension(self) -> float:
        return self.dp_bulk * self.R_sigma

    def errors(self) -> dict:
        s = self.sigma_analytic
        return {
            "sigma_capillary_fit": (self.sigma_capillary - s) / s,
            "sigma_capillary_contour": (self.sigma_capillary_contour - s) / s,
            "sigma_capillary_surface_of_tension": (
                self.sigma_capillary_surface_of_tension - s
            )
            / s,
            "sigma_integral": (self.sigma_integral - s) / s,
            "dp_tensor_laplace": (
                self.dp_tensor - self.sigma_analytic / self.R_sigma
            )
            / (self.sigma_analytic / self.R_sigma),
            "xi_fit": (self.xi_fit - self.xi_analytic) / self.xi_analytic,
            "R_fit_vs_contour_cells": self.R_fit - self.R_contour,
        }

    def as_dict(self) -> dict:
        out = {
            "R_contour": self.R_contour,
            "R_fit": self.R_fit,
            "xi_fit": self.xi_fit,
            "R_sigma": self.R_sigma,
            "p_in": self.p_in,
            "p_out": self.p_out,
            "dp_bulk": self.dp_bulk,
            "dp_tensor": self.dp_tensor,
            "sigma_integral": self.sigma_integral,
            "sigma_capillary_fit": self.sigma_capillary,
            "sigma_capillary_contour": self.sigma_capillary_contour,
            "sigma_capillary_surface_of_tension": self.sigma_capillary_surface_of_tension,
            "sigma_analytic": self.sigma_analytic,
            "xi_analytic": self.xi_analytic,
            "radius_over_xi": self.radius_over_xi,
            "u_max": self.u_max,
            "u_rms": self.u_rms,
            "phi_drift": self.phi_drift,
            "rho_drift": self.rho_drift,
            "errors": self.errors(),
        }
        return out


@dataclass(frozen=True)
class LaplaceCriteria:
    """Pre-declared acceptance bands (owner standard: >15 % is unacceptable)."""

    owner_band: float = 0.15
    stretch_band: float = 0.03
    sigma_integral_band: float = 0.15
    dp_tensor_band: float = 0.15
    xi_band: float = 0.15
    contour_consistency_cells: float = 0.35
    spurious_velocity_over_cs: float = 0.01
    conservation_drift: float = 1.0e-8

    @property
    def cs(self) -> float:
        return math.sqrt(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# input validation helpers
# --------------------------------------------------------------------------- #
def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")


def _validate_series(series: torch.Tensor, name: str = "series") -> None:
    """Validate a 1-D radial reduction (as produced by :func:`radial_average`)."""
    if not isinstance(series, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if series.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not series.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _validate_field(field: torch.Tensor, name: str = "field") -> None:
    if not isinstance(field, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if field.ndim != 2:
        raise ValueError(f"{name} must be 2-D (ny, nx)")
    if not field.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(field).all().item()):
        raise ValueError(f"{name} must contain only finite values")


# --------------------------------------------------------------------------- #
# geometry: droplet field + radial reduction
# --------------------------------------------------------------------------- #
def droplet_phi0(N: int, R: float, xi: float, *, center: float | None = None) -> torch.Tensor:
    """Return ``phi = -tanh((r - R)/xi)`` (``phi = +1`` inside) on an ``N x N`` grid."""
    if not isinstance(N, int) or N < 8:
        raise ValueError("N must be an integer >= 8")
    _require_positive(R=R, xi=xi)
    c = 0.5 * (N - 1) if center is None else float(center)
    grid = torch.arange(N, dtype=torch.float64)
    dx = periodic_distance(grid.view(1, -1).expand(N, N).contiguous(), c, float(N))
    dy = periodic_distance(grid.view(-1, 1).expand(N, N).contiguous(), c, float(N))
    r = torch.sqrt(dx * dx + dy * dy)
    return -tanh_profile(r, R, xi)


def _radius_grid(N: int, *, center: float | None = None) -> torch.Tensor:
    c = 0.5 * (N - 1) if center is None else float(center)
    grid = torch.arange(N, dtype=torch.float64)
    dx = periodic_distance(grid.view(1, -1).expand(N, N).contiguous(), c, float(N))
    dy = periodic_distance(grid.view(-1, 1).expand(N, N).contiguous(), c, float(N))
    return torch.sqrt(dx * dx + dy * dy), dx, dy


def radial_average(
    field: torch.Tensor, radius: torch.Tensor, *, bin_width: float = 0.25, r_max: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Azimuthal mean of ``field`` on radial bins ``[k dr, (k+1) dr)``.

    Returns ``(r_centers, mean, counts)``; bins with no sample are dropped (periodic
    corner geometry cannot fill them evenly, and dropping keeps the mean unbiased).
    """
    _validate_field(field)
    _validate_field(radius, "radius")
    if field.shape != radius.shape:
        raise ValueError("field and radius must share a shape")
    if not (math.isfinite(bin_width) and bin_width > 0.0):
        raise ValueError("bin_width must be finite and positive")
    r_max = float(radius.max().item()) if r_max is None else float(r_max)
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")
    nbins = int(math.floor(r_max / bin_width)) + 1
    idx = torch.clamp((radius / bin_width).floor().to(torch.int64), 0, nbins - 1)
    sums = torch.zeros(nbins, dtype=torch.float64)
    sums.index_add_(0, idx.reshape(-1), field.reshape(-1).to(torch.float64))
    counts = torch.zeros(nbins, dtype=torch.float64)
    counts.index_add_(0, idx.reshape(-1), torch.ones_like(field.reshape(-1), dtype=torch.float64))
    keep = counts > 0
    centers = (torch.arange(nbins, dtype=torch.float64) + 0.5) * bin_width
    return centers[keep], (sums[keep] / counts[keep]), counts[keep]


def interface_series(
    phi: torch.Tensor, radius: torch.Tensor, *, bin_width: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthally averaged profile ``(r_centers, phi_mean)`` of a single droplet."""
    return radial_average(phi, radius, bin_width=bin_width)[:2]


def radius_from_contour(r: torch.Tensor, phi_mean: torch.Tensor) -> float:
    """Radius of the ``phi = 0`` level set, linearly interpolated between bins."""
    _validate_series(r, "r")
    _validate_series(phi_mean, "phi_mean")
    if r.numel() != phi_mean.numel() or r.numel() < 3:
        raise ValueError("r and phi_mean must be equally sized with >= 3 bins")
    if not torch.all(r[1:] > r[:-1]):
        raise ValueError("r must be strictly increasing")
    sign = torch.sign(phi_mean)
    crossings: list[float] = []
    for i in range(phi_mean.numel() - 1):
        if sign[i] == 0.0:
            crossings.append(float(r[i]))
            continue
        if sign[i] * sign[i + 1] < 0.0:
            # phi ~ 1 corresponds to the inner phase, so this is the outward crossing.
            t = float(phi_mean[i] / (phi_mean[i] - phi_mean[i + 1]))
            crossings.append(float(r[i] + t * (r[i + 1] - r[i])))
    if not crossings:
        raise ValueError("no phi = 0 crossing found in the radial profile")
    return min(crossings)


def radius_and_width_from_tanh_fit(
    r: torch.Tensor, phi_mean: torch.Tensor, *, band: float = 0.9
) -> tuple[float, float]:
    """Fit ``atanh(phi) = (R - r)/xi`` in the interface band; return ``(R, xi)``.

    ``atanh`` is linear in ``r`` for the model's tanh profile, so the fit is an exact
    two-parameter least squares -- no iterative optimiser, no seed dependence.
    """
    if not 0.0 < band < 1.0:
        raise ValueError("band must lie in (0, 1)")
    _validate_series(r, "r")
    _validate_series(phi_mean, "phi_mean")
    if r.numel() != phi_mean.numel() or r.numel() < 3:
        raise ValueError("r and phi_mean must be equally sized with >= 3 bins")
    mask = phi_mean.abs() <= band
    if int(mask.sum().item()) < 3:
        raise ValueError("fewer than 3 bins inside the interface band; refine bin_width")
    x = r[mask]
    y = torch.atanh(phi_mean[mask])
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum().item())
    if sxx <= 0.0:
        raise ValueError("degenerate radial sampling in the interface band")
    slope = float(((x - xm) * (y - ym)).sum().item()) / sxx
    if slope == 0.0:
        raise ValueError("degenerate tanh fit (zero slope)")
    xi = -1.0 / slope
    # y = (R - r)/xi  =>  intercept a = R/xi  =>  R = xi * a = r_mean + xi * y_mean
    R = float(xm.item()) + xi * float(ym.item())
    return R, xi


# --------------------------------------------------------------------------- #
# pressure-tensor geometry
# --------------------------------------------------------------------------- #
def polar_pressure_components(
    Pxx: torch.Tensor, Pyy: torch.Tensor, Pxy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radial/tangential components ``(P_rr, P_tt)`` of a symmetric 2-D tensor.

    ``P_tt = P_xx + P_yy - P_rr`` (the trace is rotation invariant), which avoids
    carrying a second trigonometric projection and makes ``P_rr + P_tt == P_xx + P_yy``
    exact up to round-off -- asserted in the tests.
    """
    for name, t in (("Pxx", Pxx), ("Pyy", Pyy), ("Pxy", Pxy), ("dx", dx), ("dy", dy)):
        _validate_field(t, name)
    if any(t.shape != Pxx.shape for t in (Pyy, Pxy, dx, dy)):
        raise ValueError("all pressure components and offsets must share a shape")
    r = torch.sqrt(dx * dx + dy * dy)
    safe = torch.where(r > 0.0, r, torch.ones_like(r))
    ex = torch.where(r > 0.0, dx / safe, torch.zeros_like(r))
    ey = torch.where(r > 0.0, dy / safe, torch.zeros_like(r))
    Prr = Pxx * ex * ex + Pyy * ey * ey + 2.0 * Pxy * ex * ey
    Ptt = Pxx + Pyy - Prr
    return Prr, Ptt


def radial_anisotropy_profile(
    Prr: torch.Tensor,
    Ptt: torch.Tensor,
    radius: torch.Tensor,
    *,
    bin_width: float = 0.25,
    r_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthal mean of ``P_rr - P_tt`` on radial bins (``(r, anis)``)."""
    if Prr.shape != Ptt.shape or Prr.shape != radius.shape:
        raise ValueError("Prr, Ptt and radius must share a shape")
    return radial_average(Prr - Ptt, radius, bin_width=bin_width, r_max=r_max)[:2]


def tension_integrals(
    r: torch.Tensor, anis: torch.Tensor, *, r_lo: float, r_hi: float, bin_width: float = 0.25
) -> dict:
    """Integrate the radial pressure anisotropy across **one** interface.

    ``sigma = integral (P_rr - P_tt) dr`` and
    ``p_in - p_out = integral (P_rr - P_tt)/r dr`` (radial momentum balance),
    both restricted to ``r_lo <= r <= r_hi`` so exactly one interface is crossed.
    The surface-of-tension radius is the first moment of the same kernel.

    The window must be one-sided: integrating a full diameter crosses two interfaces
    and returns ``2 sigma`` (a defect that shipped once in the SOY xval), so
    ``r_hi <= 2 * r_lo``-style misuse is not detectable numerically -- instead the
    caller declares the window and :func:`tension_integrals` refuses a window whose
    kernel has two separated lobes (a sufficient condition for double counting).
    """
    _validate_series(r, "r")
    _validate_series(anis, "anis")
    if r.numel() != anis.numel() or r.numel() < 3:
        raise ValueError("r and anis must be equally sized with >= 3 bins")
    if not (math.isfinite(r_lo) and math.isfinite(r_hi) and 0.0 <= r_lo < r_hi):
        raise ValueError("need finite 0 <= r_lo < r_hi")
    mask = (r >= r_lo) & (r <= r_hi)
    if int(mask.sum().item()) < 3:
        raise ValueError("integration window holds fewer than 3 bins")
    rr = r[mask]
    aa = anis[mask]
    if not bool(torch.isfinite(aa).all().item()):
        raise ValueError("non-finite anisotropy inside the integration window")
    weight = bin_width
    sigma = float(aa.sum().item() * weight)
    dp = float((aa / rr).sum().item() * weight)
    # the continuum kernel kappa |grad phi|^2 is non-negative; the discrete field can
    # carry small negative round-off/oscillation lobes, which must not leak into the
    # first moment (they would drag R_sigma outside the window)
    kernel = torch.clamp(aa, min=0.0)
    denom = float(kernel.sum().item() * weight)
    if denom <= 1e-300:
        raise ValueError("no positive anisotropy inside the window: no interface there")
    R_sigma = float((rr * kernel).sum().item() / kernel.sum().item())
    # double-count guard: a window spanning two interfaces contains two separated peaks
    # of the kernel with a deep trough between them
    peak = float(kernel.max().item())
    if peak > 0.0:
        above = kernel > 0.05 * peak
        runs = int(
            (above.to(torch.int64)[1:] - above.to(torch.int64)[:-1]).clamp(min=0).sum().item()
        ) + int(above[0].item())
        if runs > 1:
            raise ValueError(
                "anisotropy kernel has more than one lobe inside the window -> the window "
                "crosses more than one interface and sigma would be double counted"
            )
    flips = 0  # kept for artifact compatibility: sign changes of the raw kernel
    return {
        "sigma_integral": sigma,
        "dp_tensor": dp,
        "R_surface_of_tension": R_sigma,
        "window": [float(r_lo), float(r_hi)],
        "n_bins": int(mask.sum().item()),
        "lobe_sign_flips": flips,
    }


def bulk_pressure_jump(
    pressure: torch.Tensor,
    radius: torch.Tensor,
    R: float,
    xi: float,
    *,
    offset_xi: float = 3.0,
    r_max: float | None = None,
) -> tuple[float, float, float]:
    """Return ``(p_in, p_out, dp)`` from the bulk bands around a droplet.

    The inner mask is ``r <= R - offset_xi * xi`` and the outer mask is
    ``R + offset_xi * xi <= r <= r_max``; both must be non-empty, otherwise the
    measurement is meaningless and an exception (not ``nan``) is raised.
    """
    _validate_field(pressure, "pressure")
    if pressure.shape != radius.shape:
        raise ValueError("pressure and radius must share a shape")
    _require_positive(R=R, xi=xi, offset_xi=offset_xi)
    r_max = float(radius.max().item()) if r_max is None else float(r_max)
    inner = radius <= R - offset_xi * xi
    outer = (radius >= R + offset_xi * xi) & (radius <= r_max)
    if int(inner.sum().item()) == 0:
        raise ValueError("inner bulk band is empty (droplet too small / box too tight)")
    if int(outer.sum().item()) == 0:
        raise ValueError("outer bulk band is empty (box too small for offset_xi * xi)")
    p_in = float(pressure[inner].mean().item())
    p_out = float(pressure[outer].mean().item())
    return p_in, p_out, p_in - p_out


# --------------------------------------------------------------------------- #
# case runner
# --------------------------------------------------------------------------- #
def _field_drift(field: torch.Tensor, reference: float) -> float:
    return abs(float(field.mean().item()) - reference)


def run_droplet_case(case: DropletCase, *, dtype: torch.dtype = torch.float64) -> DropletMeasures:
    """Time-step one static droplet to a steady state and measure the Laplace law.

    The populations are advanced by the production kernel
    (``collide_and_stream`` from :mod:`free_energy_pressure`); nothing here relaxes
    the physics.  Every reported number is measured -- the analytic ``sigma``/``xi``
    are carried along for grading only.
    """
    p = case.params
    phi0 = droplet_phi0(case.N, case.R, case.xi).to(dtype)
    f, g = initial_condition(phi0, p)
    radius, dx, dy = _radius_grid(case.N)
    radius = radius.to(dtype)
    dx, dy = dx.to(dtype), dy.to(dtype)
    r_max = 0.5 * case.N - 1.0  # stay clear of the periodic corners
    phi_ref = float(phi0.mean().item())
    rho_ref = float(f.sum(dim=0).mean().item())

    history: list[dict] = []
    with torch.no_grad():
        for step in range(1, case.steps + 1):
            f, g = collide_and_stream(f, g, p)
            if not bool(torch.isfinite(f).all().item() and torch.isfinite(g).all().item()):
                raise FloatingPointError(f"free-energy LBM diverged at step {step}")
            if step % case.sample_every == 0 or step == case.steps:
                rho, ux, uy, phi = macroscopic(f, g)
                pr = mean_pressure(f)
                p_in, p_out, dp = bulk_pressure_jump(
                    pr, radius, case.R, case.xi, offset_xi=case.bulk_offset_xi, r_max=r_max
                )
                speed = torch.sqrt(ux * ux + uy * uy)
                history.append(
                    {
                        "step": step,
                        "dp_bulk": dp,
                        "sigma_capillary_at_R_target": dp * case.R,
                        "u_max": float(speed.max().item()),
                        "u_rms": float(torch.sqrt((speed * speed).mean()).item()),
                    }
                )

    rho, ux, uy, phi = macroscopic(f, g)
    pr = mean_pressure(f)
    Pxx, Pyy, Pxy = pressure_tensor_measured(f)
    # identical bin grid for the phase profile and the pressure anisotropy, so the
    # per-bin table written to the artifact is aligned bin-by-bin
    r_centers, phi_mean = radial_average(
        phi, radius, bin_width=case.bin_width, r_max=r_max
    )[:2]
    R_contour = radius_from_contour(r_centers, phi_mean)
    R_fit, xi_fit = radius_and_width_from_tanh_fit(r_centers, phi_mean)
    Prr, Ptt = polar_pressure_components(Pxx, Pyy, Pxy, dx, dy)
    r_bins, anis = radial_anisotropy_profile(
        Prr, Ptt, radius, bin_width=case.bin_width, r_max=r_max
    )
    if r_centers.numel() != r_bins.numel():
        raise RuntimeError("phase and pressure bin grids disagree")
    r_lo = max(0.0, R_fit - 4.0 * case.xi)
    r_hi = min(r_max, R_fit + 4.0 * case.xi)
    tens = tension_integrals(r_bins, anis, r_lo=r_lo, r_hi=r_hi, bin_width=case.bin_width)
    p_in, p_out, dp_bulk = bulk_pressure_jump(
        pr, radius, R_fit, case.xi, offset_xi=case.bulk_offset_xi, r_max=r_max
    )
    speed = torch.sqrt(ux * ux + uy * uy)
    return DropletMeasures(
        R_contour=R_contour,
        R_fit=R_fit,
        xi_fit=xi_fit,
        R_sigma=tens["R_surface_of_tension"],
        p_in=p_in,
        p_out=p_out,
        dp_bulk=dp_bulk,
        dp_tensor=tens["dp_tensor"],
        sigma_integral=tens["sigma_integral"],
        u_max=float(speed.max().item()),
        u_rms=float(torch.sqrt((speed * speed).mean()).item()),
        sigma_analytic=case.sigma,
        xi_analytic=case.xi,
        radius_over_xi=R_fit / case.xi,
        phi_drift=_field_drift(phi, phi_ref),
        rho_drift=_field_drift(rho, rho_ref),
        profile=[
            {"r": float(rr), "phi": float(pp), "anis": float(aa)}
            for rr, pp, aa in zip(r_bins.tolist(), phi_mean.tolist(), anis.tolist())
        ],
        history=history,
    )


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def grade_case(
    case: DropletCase, measures: DropletMeasures, criteria: LaplaceCriteria | None = None
) -> dict:
    """Grade one case against the pre-declared bands.

    Returns a per-criterion table; ``passed`` is the conjunction (no aggregate
    cherry-picking).  ``sigma_capillary_fit`` is the headline KPI, the other routes
    are consistency checks on the same state.
    """
    criteria = criteria or LaplaceCriteria()
    err = measures.errors()
    checks = {
        "C1_sigma_capillary_vs_analytic": {
            "value": err["sigma_capillary_fit"],
            "abs_value": abs(err["sigma_capillary_fit"]),
            "band": criteria.owner_band,
            "stretch_band": criteria.stretch_band,
            "passed": abs(err["sigma_capillary_fit"]) <= criteria.owner_band,
            "within_stretch": abs(err["sigma_capillary_fit"]) <= criteria.stretch_band,
        },
        "C2_sigma_integral_vs_analytic": {
            "value": err["sigma_integral"],
            "abs_value": abs(err["sigma_integral"]),
            "band": criteria.sigma_integral_band,
            "passed": abs(err["sigma_integral"]) <= criteria.sigma_integral_band,
        },
        "C3_tensor_capillary_vs_sigma_over_R": {
            "value": err["dp_tensor_laplace"],
            "abs_value": abs(err["dp_tensor_laplace"]),
            "band": criteria.dp_tensor_band,
            "passed": abs(err["dp_tensor_laplace"]) <= criteria.dp_tensor_band,
        },
        "C4_interface_width_recovered": {
            "value": err["xi_fit"],
            "abs_value": abs(err["xi_fit"]),
            "band": criteria.xi_band,
            "passed": abs(err["xi_fit"]) <= criteria.xi_band,
        },
        "C5_radius_definition_consistency": {
            "value": err["R_fit_vs_contour_cells"],
            "abs_value": abs(err["R_fit_vs_contour_cells"]),
            "band": criteria.contour_consistency_cells,
            "passed": abs(err["R_fit_vs_contour_cells"]) <= criteria.contour_consistency_cells,
        },
        "C6_spurious_current": {
            "value": measures.u_max / criteria.cs,
            "abs_value": measures.u_max / criteria.cs,
            "band": criteria.spurious_velocity_over_cs,
            "passed": measures.u_max / criteria.cs <= criteria.spurious_velocity_over_cs,
        },
        "C7_conservation": {
            "value": max(measures.phi_drift, measures.rho_drift),
            "abs_value": max(measures.phi_drift, measures.rho_drift),
            "band": criteria.conservation_drift,
            "passed": max(measures.phi_drift, measures.rho_drift) <= criteria.conservation_drift,
        },
    }
    return {
        "case": case.as_dict(),
        "measures": measures.as_dict(),
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
    }


def grade_report(
    rows: Sequence[dict], criteria: LaplaceCriteria | None = None
) -> dict:
    """Aggregate per-case rows: verdict, worst error, and the calibration plateau.

    The verdict is ``PASS`` iff every row passes.  ``radius_trend`` reports the
    capillary-tension estimate as a function of ``R/xi`` so the finite-``xi`` trend
    is visible instead of being averaged away.
    """
    if not rows:
        raise ValueError("no rows to grade")
    criteria = criteria or LaplaceCriteria()
    worst = max(abs(r["measures"]["errors"]["sigma_capillary_fit"]) for r in rows)
    worst_integral = max(abs(r["measures"]["errors"]["sigma_integral"]) for r in rows)
    worst_tensor = max(abs(r["measures"]["errors"]["dp_tensor_laplace"]) for r in rows)
    trend = sorted(
        (
            {
                "R_over_xi": r["measures"]["radius_over_xi"],
                "sigma_capillary_rel_error": r["measures"]["errors"]["sigma_capillary_fit"],
                "sigma_integral_rel_error": r["measures"]["errors"]["sigma_integral"],
            }
            for r in rows
        ),
        key=lambda row: row["R_over_xi"],
    )
    return {
        "verdict": "PASS" if all(r["passed"] for r in rows) else "FAIL",
        "owner_band": criteria.owner_band,
        "stretch_band": criteria.stretch_band,
        "worst_abs_rel_error_sigma_capillary": worst,
        "worst_abs_rel_error_sigma_integral": worst_integral,
        "worst_abs_rel_error_dp_tensor": worst_tensor,
        "within_owner_band": worst <= criteria.owner_band,
        "within_stretch_band": worst <= criteria.stretch_band,
        "radius_trend": trend,
        "failing_checks": [
            f"{r['case'].get('case_id', '?')}:{name}"
            for r in rows
            for name, c in r["checks"].items()
            if not c["passed"]
        ],
    }