"""Free-surface LBM discrete dispersion relation, phase-error budget and phase-calibration operator.

Why this module exists
----------------------
The wave-tank campaigns (``benchmarks/verified/wave_tank``,
``docs/P49_N01_STOKES_WAVE_TANK.md``) reported phase-speed errors of **7 % … 19 %**
against the closed form ``c^2 = (g/k) tanh(k h)``, while the same solver measured on
a *periodic* box reproduced the closed form to ~0.1 %.  Two candidate explanations
survive that juxtaposition and they have completely different remedies:

* **H_solver** — the free-surface lattice-Boltzmann model itself transports gravity
  waves with a 7-19 % wrong phase speed (a discretisation defect of the interface /
  pressure model).  Remedy: change the collision term or the equilibrium, i.e. a
  *phase-calibration operator*.
* **H_estimator** — the short-baseline two-point estimator used on the wavemaker
  cases is catastrophically amplified by the residual standing wave, so the
  *measured* k (hence c) carries a bias that does not vanish as the probe spacing
  shrinks.  Remedy: fix the estimator, not the solver.

This module supplies the instrumentation that discriminates them:

1. :func:`lattice_linear_modes` — the **exact** eigenfrequencies of the linearised
   isothermal D2Q9 BGK operator from the von Neumann (Fourier) analysis of the full
   9-population system.  No approximation beyond linearity.
2. :func:`phase_error_budget` — a closed-form map from lattice/interface
   discretisation parameters onto the gravity-wave phase speed, one term per
   mechanism, so each mechanism is separately testable.
3. :func:`standing_wave_estimator_bias` — **closed form** for the apparent
   phase-speed error of a local phase-slope (two-point / short-baseline) estimator
   in a field with a residual reflection of ratio ``r``.  For ``r = 0.1`` the bias
   reaches +-18 % and does *not* vanish as the probe spacing shrinks — the
   quantitative statement of H_estimator.
4. :func:`subcell_interface_pressure` — the **phase-calibration operator** proper:
   the Körner anti-bounce-back interface reconstruction imposes ``p = p_atm`` at the
   *interface cell centre*, which biases the effective water depth by ``O(dx)``.
   The operator imposes it at the sub-cell surface instead
   (``p = p_atm + rho_l g (fill - 1/2) dx``) — local and momentum conserving.  It is
   wired into :func:`tensorlbm.free_surface_lbm_2d.koerner_step_2d` behind
   ``interface_pressure_policy="subcell"`` (default ``"const"`` = legacy).
5. :func:`run_linear_wave` — a horizontally periodic linear-wave rig that measures
   the *intrinsic* discrete phase speed, group speed and amplitude decay of the
   solver with a whole-wavelength least-squares estimator (no wavemaker, no
   reflection, no short-baseline assumption).

Conventions
-----------
All quantities are in lattice units (``dx = dt = 1``), ``y`` points **up**, the
still-water level is ``y = h`` and the bed ``y = 0``.  ``g`` is the magnitude of the
lattice gravity (applied as ``gy = -g``).  Frequencies are per time step.

References
----------
Körner, C., Thürey, N., Ulrich, R. (2005). *Lattice Boltzmann model for free surface
    flows.* J. Comput. Phys. 202, 68.
Thürey, N. (2007). *A lattice Boltzmann method for free surface flows.* PhD thesis,
    Univ. Erlangen.
Lallemand, P., Luo, L.-S. (2000). *Theory of the lattice Boltzmann method: dispersion,
    dissipation, isotropy, Galilean invariance, and stability.* Phys. Rev. E 61, 6546.
Dean, R. G., Dalrymple, R. A. (1991). *Water Wave Mechanics for Engineers and
    Scientists*, §4 — ``omega^2 = g k tanh(k h)``, depth and group-speed relations.
Landau, L. D., Lifshitz, E. M. (1987). *Fluid Mechanics* 2nd ed., §62 — viscous
    damping of the interface, ``gamma = 2 nu k^2``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "C",
    "W",
    "CS2",
    "lattice_linear_modes",
    "acoustic_mode",
    "lattice_phase_speed_error",
    "continuous_omega",
    "continuous_phase_speed",
    "continuous_group_speed",
    "depth_sensitivity",
    "laplace_phase_shift",
    "PhaseErrorBudget",
    "phase_error_budget",
    "standing_wave_estimator_bias",
    "bias_at",
    "bias_scan",
    "subcell_interface_pressure",
    "calibrate_interface_offset",
    "LinearWaveCase",
    "run_linear_wave",
    "fit_mode_2d",
    "fit_traveling_wave",
    "fit_two_wave",
    "two_point_phase_speed",
    "group_speed_from_sweep",
    "synthetic_standing_wave_bias",
]

#: D2Q9 lattice velocities (same ordering as :mod:`tensorlbm.d2q9`).
C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=float,
)
#: D2Q9 weights.
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=float)
#: Lattice sound speed squared.
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# 1. exact lattice dispersion: von Neumann analysis of the linearised D2Q9 BGK
# ---------------------------------------------------------------------------
def _equilibrium_projector(weights: np.ndarray = W) -> np.ndarray:
    """Matrix ``P`` with ``P @ f = f_eq`` for the *linearised* isothermal D2Q9.

    Around ``rho0 = 1, u0 = 0`` the linear equilibrium is
    ``f_a^eq = w_a (drho + 3 c_a . du)`` with ``drho = sum f`` and
    ``du_i = sum_a c_ai f_a`` (momentum at ``rho0 = 1``), so ``P`` has rank 3.
    """
    P = np.zeros((9, 9), dtype=complex)
    for a in range(9):
        P[a, :] = weights[a]  # drho column
        P[a, :] += 3.0 * weights[a] * (C[a, 0] * C[:, 0] + C[a, 1] * C[:, 1])  # du columns
    return P


def lattice_linear_modes(kx: float, ky: float, tau: float, dt: float = 1.0) -> np.ndarray:
    """Exact complex angular frequencies of the linearised D2Q9 BGK operator.

    Fourier ansatz ``f_a(x, t) = f_a exp(i (k . x - omega t))`` gives the per-step
    eigenvalue problem

    ``lambda f_a = e^{-i theta_a} [(1 - 1/tau) f_a + (1/tau) f_a^eq]``,
    ``theta_a = k . c_a``,

    with ``omega = i ln(lambda) / dt``.  Returns the nine ``omega`` (per step): the
    two shear branches, the entropy branch and the sound/pressure pair.  Exact
    within the linearised isothermal model; no small-``k`` expansion.
    """
    if tau <= 0.5:
        raise ValueError(f"BGK requires tau > 0.5, got {tau}")
    theta = kx * C[:, 0] + ky * C[:, 1]
    E = np.exp(-1j * theta)
    P = _equilibrium_projector()
    L = np.diag(E) @ ((1.0 - 1.0 / tau) * np.eye(9) + (1.0 / tau) * P)
    lam = np.linalg.eigvals(L)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = 1j * np.log(lam) / dt
    return np.asarray(omega, dtype=complex)


def acoustic_mode(kx: float, ky: float, tau: float, dt: float = 1.0) -> complex:
    """The sound (pressure) branch of :func:`lattice_linear_modes` with ``Re omega > 0``."""
    omega = lattice_linear_modes(kx, ky, tau, dt)
    kk = math.hypot(kx, ky)
    target = math.sqrt(CS2) * kk
    best = None
    for w in omega:
        if w.real <= 0.0:
            continue
        if best is None or abs(w.real - target) < abs(best.real - target):
            best = w
    if best is None:  # pragma: no cover - eigvals always contain a positive pair
        raise RuntimeError("no positive-frequency sound branch found")
    return complex(best)


def lattice_phase_speed_error(k: float, tau: float, angle: float = 0.0) -> dict:
    """Acoustic-branch phase speed / damping from the exact lattice modes.

    ``c = Re(omega)/k``; used as the solver-side term of the phase budget.
    """
    kx, ky = k * math.cos(angle), k * math.sin(angle)
    w = acoustic_mode(kx, ky, tau)
    c_lat = w.real / k
    return {
        "k": float(k),
        "angle": float(angle),
        "omega_re": float(w.real),
        "omega_im": float(w.imag),
        "c_lattice": float(c_lat),
        "c_continuum": math.sqrt(CS2),
        "rel_error": float(c_lat / math.sqrt(CS2) - 1.0),
        "damping": float(-w.imag),
    }


# ---------------------------------------------------------------------------
# 2. continuum reference
# ---------------------------------------------------------------------------
def continuous_omega(k: float, h: float, g: float) -> float:
    """``omega^2 = g k tanh(k h)`` (linear gravity waves)."""
    return math.sqrt(max(g * k * math.tanh(k * h), 0.0))


def continuous_phase_speed(k: float, h: float, g: float) -> float:
    """``c = omega/k`` — the closed form the solver is judged against."""
    return continuous_omega(k, h, g) / k


def continuous_group_speed(k: float, h: float, g: float) -> float:
    """``c_g = domega/dk`` in closed form (Dean & Dalrymple §4)."""
    c = continuous_phase_speed(k, h, g)
    x = 2.0 * k * h
    # x / sinh(x) without overflow: -> 1 as x -> 0, -> 0 as x -> inf
    ratio = 1.0 if x < 1e-8 else (0.0 if x > 700.0 else x / math.sinh(x))
    return c * 0.5 * (1.0 + ratio)


def depth_sensitivity(k: float, h: float) -> float:
    """``d ln c / d ln h`` for a gravity wave — the interface-depth lever arm.

    ``c^2 = (g/k) tanh(kh)`` gives exactly ``d ln c / d ln h = k h / sinh(2 k h)``,
    so this coefficient converts the ``O(dx)`` interface-depth bias into a
    phase-speed error.
    """
    return k * h / math.sinh(2.0 * k * h)


def laplace_phase_shift(k: float, h: float, g: float, sigma: float, rho: float = 1.0) -> float:
    """Relative phase-speed change produced by a Laplace (surface-tension) term.

    ``omega^2 = g k tanh(kh) + sigma k^3 / rho`` gives
    ``c/c0 = sqrt(1 + sigma k^2 / (rho g tanh(kh)))``; the returned value is that
    root minus one.  A **negative** ``sigma`` (a numerically admissible calibration
    coefficient, not a physical surface tension) slows short waves more than long
    ones — the ``O(k^2)`` shape needed to cancel a lattice short-wave phase excess.
    """
    base = g * math.tanh(k * h)
    if base <= 0.0:  # pragma: no cover - defensive
        return 0.0
    return math.sqrt(max(1.0 + sigma * k * k / (rho * base), 0.0)) - 1.0


# ---------------------------------------------------------------------------
# 3. phase-error budget (one term per mechanism)
# ---------------------------------------------------------------------------
@dataclass
class PhaseErrorBudget:
    """Decomposition of the free-surface LBM phase-speed error per mechanism.

    All error entries are *relative* (``c/c_cont - 1``), additive to leading order.
    """

    k: float
    h: float
    g: float
    tau: float
    cells_per_wavelength: float
    interface_offset_cells: float
    sigma_cal: float
    mach: float
    e_lattice: float
    e_interface: float
    e_laplace: float
    e_viscous: float
    e_total: float
    c_cont: float
    c_predicted: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def phase_error_budget(
    k: float,
    h: float,
    g: float,
    tau: float,
    *,
    dx: float = 1.0,
    interface_offset_cells: float = 0.0,
    sigma_cal: float = 0.0,
    rho: float = 1.0,
    lattice_coefficient: float = 0.125,
) -> PhaseErrorBudget:
    """Closed-form phase-speed error budget of the free-surface LBM.

    Mechanisms (each derived in the campaign report):

    ``e_lattice``
        second-order lattice term of the incompressible response,
        ``C2 * (k dx)^2``, ``C2`` identified against the exact lattice modes and
        the periodic-rig measurements (see ``artifacts/``); reported as an
        identified coefficient, not a free fit parameter.
    ``e_interface``
        ``O(dx)`` effective-depth bias of the interface pressure reconstruction,
        converted with the exact lever arm :func:`depth_sensitivity`.
    ``e_laplace``
        the calibration operator's own ``O(k^2)`` term (:func:`laplace_phase_shift`).
    ``e_viscous``
        ``-1/2 (nu k^2 / omega)^2`` — real part of the viscous correction (the
        imaginary part is the amplitude decay ``2 nu k^2``).
    """
    if k <= 0.0 or h <= 0.0:
        raise ValueError("k and h must be positive")
    nu = CS2 * (tau - 0.5)
    c_cont = continuous_phase_speed(k, h, g)
    omega = continuous_omega(k, h, g)
    e_lattice = lattice_coefficient * (k * dx) ** 2
    e_interface = depth_sensitivity(k, h) * (interface_offset_cells * dx) / h
    e_laplace = laplace_phase_shift(k, h, g, sigma_cal, rho)
    e_viscous = -0.5 * (nu * k * k / omega) ** 2
    e_total = e_lattice + e_interface + e_laplace + e_viscous
    return PhaseErrorBudget(
        k=float(k),
        h=float(h),
        g=float(g),
        tau=float(tau),
        cells_per_wavelength=float(2.0 * math.pi / (k * dx)),
        interface_offset_cells=float(interface_offset_cells),
        sigma_cal=float(sigma_cal),
        mach=float(c_cont / math.sqrt(CS2)),
        e_lattice=float(e_lattice),
        e_interface=float(e_interface),
        e_laplace=float(e_laplace),
        e_viscous=float(e_viscous),
        e_total=float(e_total),
        c_cont=float(c_cont),
        c_predicted=float(c_cont * (1.0 + e_total)),
    )


# ---------------------------------------------------------------------------
# 4. why a two-point estimator reports 7-19 %: closed-form estimator bias
# ---------------------------------------------------------------------------
def standing_wave_estimator_bias(
    k: float,
    reflection: float,
    *,
    dx: float = 1.0,
    x: float | None = None,
    n_points: int = 2,
) -> dict:
    """Apparent phase-speed error of a local phase-slope (short-baseline) estimator.

    A partially standing field is ``Z(x) = A e^{-ikx} + B e^{+ikx}`` with
    ``r = |B|/|A|``.  An estimator that infers ``k`` from the *local* phase slope
    (two-point difference ``d arg Z / dx`` or its ``dx -> 0`` limit) returns

    ``k_app = k - 2 k r (cos phi + r) / (1 + 2 r cos phi + r^2)``,  ``phi = 2 k x``,

    hence ``c_app = omega / k_app`` and the relative bias ``c_app/c - 1``.  Two facts
    matter:

    * the bias does **not** vanish as ``dx -> 0`` — it is a property of the field,
      not of the differencing, so refining the probe spacing cannot remove it;
    * for ``r = 0.1`` it spans about ``-22 % … +18 %`` depending on where the probes
      sit relative to the standing-wave antinodes — the exact range the wave-tank
      campaign reported.

    Returns the bias at the requested station plus its min/max over a wavelength and
    the reflection ratio needed to reproduce a 10 % / 19 % observed bias.
    """
    if reflection < 0.0:
        raise ValueError("reflection must be non-negative")
    r = float(reflection)
    xx = 0.0 if x is None else float(x)
    scan = bias_scan(k, r)
    return {
        "k": float(k),
        "reflection": r,
        "dx": float(dx),
        "n_points": int(n_points),
        "x": xx,
        "bias_at_x": float(bias_at(k, r, 2.0 * k * xx)),
        "bias_min": float(min(scan)),
        "bias_max": float(max(scan)),
        "bias_span": float(max(scan) - min(scan)),
        "r_for_10pct_bias": _reflection_for_bias(0.10, k),
        "r_for_19pct_bias": _reflection_for_bias(0.19, k),
    }


def bias_at(k: float, r: float, phi: float) -> float:
    """Relative phase-speed bias ``c_app/c - 1`` at standing-wave phase ``phi``."""
    denom = 1.0 + 2.0 * r * math.cos(phi) + r * r
    k_app = k - 2.0 * k * r * (math.cos(phi) + r) / denom
    return k / k_app - 1.0


def bias_scan(k: float, r: float, n: int = 400) -> list[float]:
    """Bias over one wavelength of probe positions (``phi = 2 k x`` in ``[0, 2pi]``)."""
    return [bias_at(k, r, 2.0 * math.pi * i / n) for i in range(n)]


def _reflection_for_bias(target: float, k: float) -> float:
    """Smallest ``r`` whose worst-case local-slope estimator bias reaches ``target``."""
    lo, hi = 0.0, 0.98
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if max(bias_scan(k, mid, n=64)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 5. the phase-calibration operator (sub-cell interface pressure)
# ---------------------------------------------------------------------------
def subcell_interface_pressure(fill: Any, g: float, *, limit: float = 0.1) -> Any:
    """Calibration operator: density offset of the sub-cell interface pressure.

    The Körner anti-bounce-back reconstruction imposes ``p = p_atm`` at the *centre*
    of an interface cell, while the physical surface sits at
    ``y_s = y_c - 1/2 + fill`` (lattice units, ``y`` up).  The resulting ``O(dx)``
    bias of the effective water depth is

    ``p(y_c) = p_atm + rho_l g (fill - 1/2) dx``  =>
    ``drho = 3 g (fill - 1/2) dx``  (``p = c_s^2 rho``, ``c_s^2 = 1/3``).

    Adding this offset to the ABB gas density puts the atmospheric pressure on the
    *sub-cell* surface.  ``limit`` clips the offset to the same +-0.1 band used for
    the Laplace term (Thürey 2007), keeping the reconstruction positive.

    Works for numpy arrays and torch tensors (returns the same type).
    """
    scale = 3.0 * float(g)
    try:
        import torch

        if isinstance(fill, torch.Tensor):
            return (scale * (fill - 0.5)).clamp(-limit, limit)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    arr = np.asarray(fill, dtype=float)
    return np.clip(scale * (arr - 0.5), -limit, limit)


def calibrate_interface_offset(
    measurements: Sequence[dict],
    *,
    h: float,
    dx: float = 1.0,
) -> dict:
    """Identify the interface-depth offset from measured phase-speed errors.

    ``measurements`` are dicts with ``k`` and ``rel_error`` (``c_meas/c_cont - 1``).
    The offset is the least-squares solution of ``rel_error_k = dsens(k) offset / h``.
    """
    if not measurements:  # pragma: no cover - defensive
        raise ValueError("need at least one measurement")
    ks = np.array([m["k"] for m in measurements], dtype=float)
    errs = np.array([m["rel_error"] for m in measurements], dtype=float)
    basis = np.array([depth_sensitivity(k, h) / h for k in ks], dtype=float)
    denom = float(np.dot(basis, basis))
    offset = float(np.dot(basis, errs) / denom) if denom > 0 else 0.0
    resid = errs - offset * basis
    return {
        "offset_cells": offset,
        "dx": float(dx),
        "rms_residual": float(np.sqrt(np.mean(resid**2))),
        "n": int(len(ks)),
        "per_k": [
            {"k": float(k), "rel_error": float(e), "predicted": float(offset * b)}
            for k, e, b in zip(ks, errs, basis)
        ],
    }


# ---------------------------------------------------------------------------
# 6. periodic linear-wave rig (the clean measurement)
# ---------------------------------------------------------------------------
@dataclass
class LinearWaveCase:
    """Discretisation of a horizontally periodic linear gravity wave."""

    h: float = 20.0                     # still-water depth [cells]
    kh: float = 1.5                     # relative depth -> k = kh / h
    amplitude: float = 0.05             # wave amplitude a [cells]
    c_target: float = 0.04              # target phase speed -> sets the lattice g
    tau: float = 0.51                   # BGK relaxation time
    wavelengths: int = 2                # wavelengths resolved in the periodic box
    headroom: int = 10                  # gas cells above the still-water level
    n_periods: float = 4.0              # integration length in wave periods
    warmup_periods: float = 2.0         # window excluded from the analysis
    record_per_period: int = 32         # surface-elevation samples per period
    interface_pressure_policy: str = "const"   # "const" (legacy) | "subcell"
    sigma_cal: float = 0.0              # calibration Laplace coefficient
    interface_force_alpha: float = 0.0  # calibration gravity-weight blend (see module doc)
    budget_offset_cells: float = 0.0    # interface-depth offset used by the budget
    g_override: float | None = None     # fixed lattice gravity (group-speed sweeps)
    rho_gas: float = 1.0
    rho_liquid: float = 1.0

    # -- derived ------------------------------------------------------------
    @property
    def k(self) -> float:
        lam = self.wavelengths * 2.0 * math.pi * self.h / self.kh  # box length
        return 2.0 * math.pi * self.wavelengths / lam

    @property
    def nx(self) -> int:
        return int(round(2.0 * math.pi * self.wavelengths / self.k))

    @property
    def ny(self) -> int:
        return 1 + int(math.ceil(self.h + self.headroom))

    def gravity(self) -> float:
        """Lattice gravity: ``c_target``-derived, unless ``g_override`` is set.

        A group-speed sweep must hold ``g`` and ``h`` fixed while ``k`` varies;
        the default derivation (``g = c^2 k / tanh(kh)``) would hold the *phase*
        speed fixed instead and make ``d c / d k`` meaningless.
        """
        if self.g_override is not None:
            return float(self.g_override)
        return self.c_target**2 * self.k / math.tanh(self.k * self.h)

    def summary(self) -> dict:
        k = self.k
        g = self.gravity()
        return {
            "h": self.h,
            "kh": self.kh,
            "k": k,
            "nx": self.nx,
            "ny": self.ny,
            "cells_per_wavelength": 2.0 * math.pi / k,
            "amplitude": self.amplitude,
            "steepness": self.amplitude * k,
            "tau": self.tau,
            "nu": CS2 * (self.tau - 0.5),
            "g_lattice": g,
            "c_target": self.c_target,
            "c_continuum": continuous_phase_speed(k, self.h, g),
            "interface_pressure_policy": self.interface_pressure_policy,
            "sigma_cal": self.sigma_cal,
            "interface_force_alpha": self.interface_force_alpha,
            "n_periods": self.n_periods,
        }


def _surface_elevation(flags: Any, mass: Any, h: float) -> np.ndarray:
    """Free-surface elevation from the liquid-column integral (sub-cell exact)."""
    import torch

    from .free_surface_lbm_2d import GAS

    fill = torch.where(flags == GAS, torch.zeros_like(mass), mass)
    return (fill.sum(dim=0) - h).detach().cpu().numpy().astype(float)

# ---------------------------------------------------------------------------
# 7. estimators (whole-wavelength 2-D mode fit; no short-baseline assumption)
# ---------------------------------------------------------------------------
def _mode_amplitude(
    eta: np.ndarray, t: np.ndarray, x: np.ndarray, omega: float, k: float
) -> complex:
    """Complex amplitude ``<eta exp(-i (omega t - k x))>`` of a single mode."""
    phase = omega * t[:, None] - k * x[None, :]
    return complex(np.mean(eta * np.exp(-1j * phase)))


def fit_mode_2d(
    eta: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    omega0: float,
    k0: float,
    *,
    half_width: float = 0.35,
    levels: int = 5,
    points: int = 13,
) -> dict:
    """Least-squares single-mode fit of ``eta(x, t)`` over the whole box.

    Maximises ``|<eta exp(-i(omega t - k x))>|`` over ``(omega, k)`` by nested grid
    search around ``(omega0, k0)``.  Unlike a two-point phase difference this uses
    every column and every sample, so it has no ``1/|sin k ds|`` conditioning
    amplification.
    """
    best = (0.0, omega0, k0, 0j)
    lo_w, hi_w = omega0 * (1.0 - half_width), omega0 * (1.0 + half_width)
    lo_k, hi_k = k0 * (1.0 - half_width), k0 * (1.0 + half_width)
    for _ in range(levels):
        ws = np.linspace(lo_w, hi_w, points)
        ks = np.linspace(lo_k, hi_k, points)
        best_local = None
        for w in ws:
            for kk in ks:
                z = _mode_amplitude(eta, t, x, w, kk)
                if best_local is None or abs(z) > abs(best_local[3]):
                    best_local = (float(w), float(w), float(kk), z)
        assert best_local is not None
        _, w, kk, z = best_local
        if abs(z) > best[0]:
            best = (abs(z), w, kk, z)
        dw = (hi_w - lo_w) / (points - 1)
        dk = (hi_k - lo_k) / (points - 1)
        lo_w, hi_w = w - 2 * dw, w + 2 * dw
        lo_k, hi_k = max(kk - 2 * dk, 0.1 * k0), kk + 2 * dk
    amp, w, kk, z = best
    return {
        "omega": float(w),
        "k": float(kk),
        "amplitude": float(2.0 * amp),
        "phase": float(np.angle(z)),
        "mode_amplitude": float(amp),
    }


def fit_traveling_wave(
    eta: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    omega0: float,
    k0: float,
    *,
    n_iter: int = 8,
) -> dict:
    """Iterative spatial/temporal phasor fit of a (damped) travelling wave.

    Alternates two one-dimensional phase-slope fits,

    ``k = -d arg <eta e^{-i omega t}> / dx``   and   ``omega = d arg <eta e^{+i k x}> / dt``,

    which average over *every* column / sample and are immune to amplitude decay
    (a decay only biases a magnitude, never a phase slope).  Converges to the
    dominant travelling component in a few iterations.

    Also decomposes the field into forward/backward components at the fitted
    ``(omega, k)``, returning the residual standing-wave ratio ``r`` — which is
    exactly the input of :func:`standing_wave_estimator_bias`, so the pair turns
    the two-point estimator discussion into a measured quantity.
    """
    omega, k = float(omega0), float(k0)
    for _ in range(n_iter):
        zt = np.mean(eta * np.exp(-1j * omega * t)[:, None], axis=0)
        k = -float(np.polyfit(x, np.unwrap(np.angle(zt)), 1)[0])
        zx = np.mean(eta * np.exp(1j * k * x)[None, :], axis=1)
        omega = float(np.polyfit(t, np.unwrap(np.angle(zx)), 1)[0])
    zt = np.mean(eta * np.exp(-1j * omega * t)[:, None], axis=0)
    a_fwd = complex(np.mean(zt * np.exp(1j * k * x)))
    a_bwd = complex(np.mean(zt * np.exp(-1j * k * x)))
    return {
        "omega": float(omega),
        "k": float(k),
        "amplitude": float(2.0 * abs(a_fwd)),
        "phase": float(np.angle(a_fwd)),
        "a_forward": float(abs(a_fwd)),
        "a_backward": float(abs(a_bwd)),
        "reflection_ratio": float(abs(a_bwd) / abs(a_fwd)) if abs(a_fwd) > 0 else float("inf"),
    }


def fit_two_wave(
    eta: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    omega0: float,
    k0: float,
    *,
    half_width: float = 0.35,
    levels: int = 5,
    points: int = 13,
) -> dict:
    """Unbiased fit of a forward + backward wave pair ``(omega, k)``.

    Model: ``eta(x,t) = Re[A e^{i(omega t - k x)} + B e^{i(omega t + k x)}]``, linear
    in ``(A, B)`` for a given ``(omega, k)``.  The pair is found by minimising the
    least-squares residual on a nested grid around ``(omega0, k0)``.

    Why this matters: any single-wave estimator (including a whole-box 2-D mode
    fit) is biased by ``O(|B|/|A|)`` when a counter-propagating component of a few
    percent is present — the same size as the effect being measured.  Solving for
    the counter-wave removes that bias and returns its amplitude as a diagnostic
    (``reflection_ratio``), which feeds :func:`standing_wave_estimator_bias`.
    """
    nt, nx = eta.shape
    y = eta.ravel()

    def _fit(omega: float, k: float) -> tuple[float, complex, complex]:
        th_f = omega * t[:, None] - k * x[None, :]
        th_b = omega * t[:, None] + k * x[None, :]
        cols = [
            np.cos(th_f).ravel(),
            (-np.sin(th_f)).ravel(),
            np.cos(th_b).ravel(),
            (-np.sin(th_b)).ravel(),
        ]
        G = np.empty((4, 4))
        b = np.empty(4)
        for i in range(4):
            b[i] = cols[i] @ y
            for j in range(i, 4):
                G[i, j] = G[j, i] = cols[i] @ cols[j]
        coef = np.linalg.solve(G + 1e-12 * np.eye(4), b)
        resid = float(y @ y - coef @ b)
        return resid, complex(coef[0], coef[1]), complex(coef[2], coef[3])

    lo_w, hi_w = omega0 * (1.0 - half_width), omega0 * (1.0 + half_width)
    lo_k, hi_k = k0 * (1.0 - half_width), k0 * (1.0 + half_width)
    best = (float("inf"), float(omega0), float(k0), 0j, 0j)
    for _ in range(levels):
        local = None
        for w in np.linspace(lo_w, hi_w, points):
            for kk in np.linspace(lo_k, hi_k, points):
                r, A, B = _fit(w, kk)
                if local is None or r < local[0]:
                    local = (r, float(w), float(kk), A, B)
        assert local is not None
        if local[0] < best[0]:
            best = local
        dw = (hi_w - lo_w) / (points - 1)
        dk = (hi_k - lo_k) / (points - 1)
        lo_w, hi_w = local[1] - 1.5 * dw, local[1] + 1.5 * dw
        lo_k, hi_k = max(local[2] - 1.5 * dk, 0.2 * k0), local[2] + 1.5 * dk
    resid, w, kk, A, B = best
    total = float(np.sum((eta - eta.mean()) ** 2))
    return {
        "omega": float(w),
        "k": float(kk),
        "amplitude": float(2.0 * abs(A)),
        "phase": float(np.angle(A)),
        "a_forward": float(abs(A)),
        "a_backward": float(abs(B)),
        "reflection_ratio": float(abs(B) / abs(A)) if abs(A) > 0 else float("inf"),
        "rms_residual": float(math.sqrt(max(resid, 0.0) / eta.size)),
        "explained_fraction": float(1.0 - resid / total) if total > 0 else float("nan"),
    }


def run_linear_wave(case: LinearWaveCase, verbose: bool = False) -> dict:
    """Integrate one periodic linear wave and measure the discrete dispersion.

    Initial condition: Airy (linear) kinematics on a hydrostatically stratified
    density field (``rho = 1 + 3 g (h - y)`` below the surface), which removes the
    acoustic start-up transient documented in
    ``docs/P49_N01_STOKES_WAVE_TANK.md`` pitfall 2.

    Diagnostics (all computed from the *field*, never from a two-point difference):

    * ``k_measured`` / ``omega_measured`` — whole-box single-mode fit;
    * ``c_measured`` / ``c_rel_error`` against ``c^2 = (g/k) tanh(kh)``;
    * ``amplitude`` and the fitted temporal decay rate ``gamma`` (``d ln a/dt``),
      from which the amplitude after 100 periods is predicted;
    * liquid-mass drift of the Körner ledger, and the runtime.
    """
    import time

    import torch

    from .d2q9 import equilibrium
    from .free_surface_lbm_2d import (
        GAS,
        init_flags_from_fill_2d,
        init_mass_from_fill_2d,
        koerner_step_2d,
        total_liquid_inventory_2d,
    )

    torch.set_default_dtype(torch.float32)
    k = case.k
    g = case.gravity()
    h = case.h
    nx, ny = case.nx, case.ny
    omega0 = continuous_omega(k, h, g)
    period = 2.0 * math.pi / omega0

    y = torch.arange(ny, dtype=torch.float32)
    x = torch.arange(nx, dtype=torch.float32)
    YY = y.unsqueeze(1).expand(ny, nx)
    XX = x.unsqueeze(0).expand(ny, nx)
    eta_ic = case.amplitude * torch.cos(k * XX)
    # sub-cell fill of every cell in a column whose surface is at y = h + eta
    fill = (h + eta_ic - (YY - 0.5)).clamp(0.0, 1.0)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags, case.rho_liquid)

    active = flags != GAS
    u_prof = case.amplitude * omega0 * torch.cosh(k * (YY - 0.0)) / math.sinh(k * h)
    w_prof = case.amplitude * omega0 * torch.sinh(k * (YY - 0.0)) / math.sinh(k * h)
    zero = torch.zeros_like(XX)
    # theta = k x - omega t; at t = 0:  eta = a cos(kx),  u = a w cosh/sinh cos(kx),
    # w = a w sinh/sinh sin(kx)  (the vertical velocity must be in phase with the
    # surface rate: w(h) = d eta/dt = a w sin(kx))
    u = torch.where(active, u_prof * torch.cos(k * XX), zero)
    v = torch.where(active, w_prof * torch.sin(k * XX), zero)
    rho = torch.where(
        active,
        (1.0 + 3.0 * g * (h - YY)).clamp(min=case.rho_gas),
        torch.full_like(XX, case.rho_gas),
    )
    f = equilibrium(rho, u, v)

    m0 = float(total_liquid_inventory_2d(f, mass, flags))
    eta_offset = float(_surface_elevation(flags, mass, h).mean())
    record_every = max(int(round(period / case.record_per_period)), 1)
    n_steps = int(round(case.n_periods * period))
    warm = int(round(case.warmup_periods * period))

    times: list[float] = []
    records: list[np.ndarray] = []
    t_start = time.time()
    for step in range(1, n_steps + 1):
        f, mass, flags = koerner_step_2d(
            f,
            mass,
            flags,
            solid,
            tau=case.tau,
            gy=-g,
            rho_liquid=case.rho_liquid,
            rho_gas=case.rho_gas,
            surface_tension=case.sigma_cal,
            interface_pressure_policy=case.interface_pressure_policy,
            interface_force_alpha=case.interface_force_alpha,
        )
        if step % record_every == 0 and step > warm:
            times.append(float(step))
            records.append(_surface_elevation(flags, mass, h) - eta_offset)
        if verbose and step % max(int(period), 1) == 0:
            print(
                f"    step {step}/{n_steps} t/T={step / period:.2f} "
                f"({time.time() - t_start:.0f}s)",
                flush=True,
            )

    S = np.asarray(records, dtype=float)          # (nt, nx)
    t_arr = np.asarray(times, dtype=float)
    x_arr = np.arange(nx, dtype=float)
    fit = fit_two_wave(S, t_arr, x_arr, omega0, k)
    fit_tw = fit_traveling_wave(S, t_arr, x_arr, omega0, k)
    c_cont = continuous_phase_speed(k, h, g)
    c_meas = fit["omega"] / fit["k"]

    # -- amplitude decay: split the steady window into per-period sub-windows --
    amps: list[tuple[float, float]] = []
    n_win = max(int(math.floor(case.n_periods - case.warmup_periods)), 1)
    for i in range(n_win):
        lo = t_arr[0] + i * period
        hi = lo + period
        m = (t_arr >= lo) & (t_arr < hi)
        if m.sum() < 8:
            continue
        sub = fit_mode_2d(S[m], t_arr[m], x_arr, fit["omega"], fit["k"], levels=2, points=11)
        amps.append((float(np.mean(t_arr[m])), sub["amplitude"]))
    gamma = float("nan")
    gamma_r2 = float("nan")
    if len(amps) >= 2:
        tt = np.array([a[0] for a in amps], dtype=float)
        aa = np.array([a[1] for a in amps], dtype=float)
        good = aa > 0
        if good.sum() >= 2:
            slope, intercept = np.polyfit(tt[good], np.log(aa[good]), 1)
            gamma = float(-slope)
            pred = intercept + slope * tt[good]
            ss_res = float(np.sum((np.log(aa[good]) - pred) ** 2))
            ss_tot = float(np.sum((np.log(aa[good]) - np.mean(np.log(aa[good]))) ** 2))
            gamma_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    nu = CS2 * (case.tau - 0.5)
    gamma_theory = 2.0 * nu * k * k
    amp_check = fit_mode_2d(
        S[t_arr >= t_arr[-1] - period], t_arr[t_arr >= t_arr[-1] - period],
        x_arr, fit["omega"], fit["k"], levels=2, points=11,
    )
    m_drift = (float(total_liquid_inventory_2d(f, mass, flags)) - m0) / m0

    return {
        "case": case.summary(),
        "discrete": {
            "omega_measured": fit["omega"],
            "k_measured": fit["k"],
            "c_measured": float(c_meas),
            "c_continuum": float(c_cont),
            "c_rel_error": float(c_meas / c_cont - 1.0),
            "k_rel_error": float(fit["k"] / k - 1.0),
            "omega_rel_error": float(fit["omega"] / omega0 - 1.0),
            "reflection_ratio_residual": fit["reflection_ratio"],
            "fit_explained_fraction": fit["explained_fraction"],
            "fit_rms_residual": fit["rms_residual"],
            "single_wave_cross_check": {
                "k": fit_tw["k"],
                "omega": fit_tw["omega"],
                "c_rel_error": float(fit_tw["omega"] / fit_tw["k"] / c_cont - 1.0),
            },
        },
        "amplitude": {
            "initial": float(case.amplitude),
            "final_first_period": float(amps[0][1]) if amps else float("nan"),
            "final_last_period": float(amp_check["amplitude"]),
            "window_series": amps,
            "gamma_measured": gamma,
            "gamma_theory_viscous": float(gamma_theory),
            "gamma_fit_r2": gamma_r2,
            "amplitude_after_100_periods": float(
                math.exp(-gamma * 100.0 * period) if gamma == gamma else float("nan")
            ),
        },
        "mass": {"relative_drift_total": m_drift},
        "budget": phase_error_budget(
            k,
            h,
            g,
            case.tau,
            interface_offset_cells=case.budget_offset_cells,
            sigma_cal=case.sigma_cal,
        ).as_dict(),
        "runtime_s": float(time.time() - t_start),
        "n_steps": int(n_steps),
        "field": {"t": t_arr.tolist(), "eta": S.tolist()},
    }


# ---------------------------------------------------------------------------
# 8. derived quantities: group speed and the estimator-bias reproduction
# ---------------------------------------------------------------------------
def group_speed_from_sweep(measurements: Sequence[dict], k0: float) -> dict:
    """Group speed of the *discrete* system from three measured phase speeds.

    With ``c(k)`` sampled at ``k0 - dk, k0, k0 + dk``,

    ``c_g = domega/dk = c + k dc/dk  ~=  c0 + k0 (c_+ - c_-) / (2 dk)``.

    Returns the discrete and closed-form group speeds, their relative difference
    and the k-triplet used.  ``measurements`` are dicts with ``k`` and ``c``.
    """
    ms = sorted(measurements, key=lambda m: m["k"])
    if len(ms) < 3:
        raise ValueError("need at least three (k, c) pairs for a central difference")
    lo, mid, hi = ms[len(ms) // 2 - 1], ms[len(ms) // 2], ms[len(ms) // 2 + 1]
    dk = 0.5 * (hi["k"] - lo["k"])
    dc = (hi["c"] - lo["c"]) / (hi["k"] - lo["k"])
    c_g = mid["c"] + mid["k"] * dc
    c_g_cont = continuous_group_speed(k0, ms[0]["h"], ms[0]["g"])
    return {
        "k_triplet": [float(lo["k"]), float(mid["k"]), float(hi["k"])],
        "dk": float(dk),
        "c_phase": float(mid["c"]),
        "c_group_measured": float(c_g),
        "c_group_continuum": float(c_g_cont),
        "c_group_rel_error": float(c_g / c_g_cont - 1.0),
    }


def synthetic_standing_wave_bias(
    k: float,
    reflection: float,
    *,
    nx: int = 256,
    dx: float = 1.0,
    probe_spacing_wavelengths: float = 0.17,
    n_stations: int = 64,
) -> dict:
    """Numerically reproduce the finite-spacing two-point estimator bias.

    Builds the exact partially standing field ``Z(x) = e^{-ikx} + r e^{+ikx}`` on a
    grid and applies the two-point phase-slope estimator used by the wave tank,

    ``k_app = |arg(Z(x+ds)/Z(x))| / ds``,  ``ds = probe_spacing_wavelengths * lambda``,

    scanning the probe station over one wavelength.  This both validates the
    ``dx -> 0`` closed form of :func:`standing_wave_estimator_bias` (which it
    approaches as ``ds -> 0``) and gives the *envelope* the wave tank could have
    reported for a given residual reflection.
    """
    lam = 2.0 * math.pi / k
    ds = probe_spacing_wavelengths * lam
    x = np.arange(nx, dtype=float) * dx

    def _z(xx: np.ndarray) -> np.ndarray:
        return np.exp(-1j * k * xx) + reflection * np.exp(1j * k * xx)

    stations = np.linspace(0.0, lam, n_stations, endpoint=False)
    biases = []
    for x1 in stations:
        z1 = _z(np.array([x1]))[0]
        z2 = _z(np.array([x1 + ds]))[0]
        if abs(z1) < 1e-12 or abs(z2) < 1e-12:  # pragma: no cover - nodes of a standing wave
            continue
        k_app = abs(float(np.angle(z2 / z1))) / ds
        biases.append(k / k_app - 1.0)
    info = standing_wave_estimator_bias(k, reflection, dx=ds)
    return {
        "k": float(k),
        "reflection": float(reflection),
        "probe_spacing_cells": float(ds),
        "probe_spacing_wavelengths": float(probe_spacing_wavelengths),
        "n_stations": int(len(biases)),
        "bias_two_point_min": float(min(biases)),
        "bias_two_point_max": float(max(biases)),
        "bias_two_point_span": float(max(biases) - min(biases)),
        "closed_form_min_dx0": float(info["bias_min"]),
        "closed_form_max_dx0": float(info["bias_max"]),
        "field_samples": int(x.size),
    }


def two_point_phase_speed(
    eta: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    omega: float,
    *,
    spacing_wavelengths: float = 0.17,
    station_wavelengths: tuple[float, float] = (0.20, 0.40),
    n_stations: int = 5,
) -> dict:
    """Reproduce the wave-tank two-point phase-speed estimator on a field.

    The estimator the tank campaign used (``benchmarks/verified/wave_tank``):

    1. demodulate every column at the *theoretical* ``omega`` → phasor ``Z(x)``;
    2. take the phase difference between two probes spaced ``ds`` apart;
    3. ``k = |d arg Z| / ds``, ``c = omega / k``.

    ``ds`` and the probe station are given in wavelengths, defaulting to the tank
    geometry (``ds = 0.17 lambda``, probes 0.2-0.4 ``lambda`` downstream).  The
    estimator is run over several stations so the *envelope* the campaign could
    report is visible — that envelope is the point of the exercise: it is not a
    property of the solver.

    Returns ``{'k_apparent', 'c_apparent', 'c_hat', 'rel_error', 'stations'}`` with
    ``c_hat = omega / k_true`` (the continuum reference for the same omega).
    """
    zt = np.mean(eta * np.exp(-1j * omega * t)[:, None], axis=0)
    k_true = float(np.polyfit(x, np.unwrap(np.angle(zt)), 1)[0] * -1.0)
    lam = 2.0 * math.pi / k_true
    ds = spacing_wavelengths * lam
    rows = []
    for frac in np.linspace(station_wavelengths[0], station_wavelengths[1], n_stations):
        x1 = frac * lam
        z1 = np.interp(x1, x, zt.real) + 1j * np.interp(x1, x, zt.imag)
        z2 = np.interp(x1 + ds, x, zt.real) + 1j * np.interp(x1 + ds, x, zt.imag)
        if abs(z1) < 1e-14 or abs(z2) < 1e-14:  # pragma: no cover - standing-wave node
            continue
        k_app = abs(float(np.angle(z2 / z1))) / ds
        if k_app < 1e-12:  # pragma: no cover - degenerate field (no travelling wave)
            continue
        rows.append(
            {
                "station_wavelengths": float(frac),
                "k_apparent": float(k_app),
                "c_apparent": float(omega / k_app),
                "rel_error": float(k_true / k_app - 1.0),
            }
        )
    errs = [r["rel_error"] for r in rows]
    return {
        "k_true_from_field": k_true,
        "wavelength": float(lam),
        "probe_spacing_cells": float(ds),
        "probe_spacing_wavelengths": float(spacing_wavelengths),
        "stations": rows,
        "rel_error_min": float(min(errs)) if errs else float("nan"),
        "rel_error_max": float(max(errs)) if errs else float("nan"),
        "rel_error_span": float(max(errs) - min(errs)) if errs else float("nan"),
    }


# ---------------------------------------------------------------------------
# 9. Interface layer law (p495w4-n04, m13946)
# ---------------------------------------------------------------------------
@dataclass
class TwoTermLaw:
    """Two-term error law for free-surface LBM (m13946).
    
    err(kh,h) = -c1·ξ/h · L1(kh) + c2·(kΔx)²
    
    where L1(kh) = kh/sinh(2kh) is the depth sensitivity.
    
    Physical interpretation:
    - Term 1: Interface layer mass deficit from tanh(d/ξ) profile
    - Term 2: Lattice discretization error (O(k²))
    
    The interface layer effectively reduces the water depth by c1·ξ cells,
    causing a phase speed error proportional to L1(kh)·(c1·ξ/h).
    """
    c1_xi: float  # interface layer mass deficit [cells]
    c2: float     # lattice coefficient
    rms_residual: float  # fit quality
    
    def predict(self, k: float, h: float, dx: float = 1.0) -> float:
        """Predict phase speed error."""
        L1 = depth_sensitivity(k, h)
        return -self.c1_xi / h * L1 + self.c2 * (k * dx) ** 2
    
    def interface_term(self, k: float, h: float) -> float:
        """Interface layer contribution."""
        L1 = depth_sensitivity(k, h)
        return -self.c1_xi / h * L1
    
    def lattice_term(self, k: float, h: float, dx: float = 1.0) -> float:
        """Lattice discretization contribution."""
        return self.c2 * (k * dx) ** 2


def fit_two_term_law(measurements: list) -> TwoTermLaw:
    """Fit two-term law to measurement data.
    
    Each measurement must have: kh, h, c_rel_error
    """
    n = len(measurements)
    if n < 2:
        raise ValueError("Need at least 2 measurements")
    
    A = np.zeros((n, 2))
    b = np.zeros(n)
    
    for i, m in enumerate(measurements):
        kh = m["kh"]
        h = m["h"]
        k = kh / h
        err = m["c_rel_error"]
        L1 = depth_sensitivity(k, h)
        A[i, 0] = -L1 / h
        A[i, 1] = k ** 2
        b[i] = err
    
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c1_xi, c2 = result
    
    predicted = A @ result
    rms = float(np.sqrt(np.mean((b - predicted) ** 2)))
    
    return TwoTermLaw(c1_xi=float(c1_xi), c2=float(c2), rms_residual=rms)
