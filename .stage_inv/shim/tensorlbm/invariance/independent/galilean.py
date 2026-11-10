"""Galilean booster and Mach-number defect analysis.

The continuous Navier-Stokes equations are Galilean invariant: if ``(u, p)`` is a
solution then so is ``(u + U, p - U.x d rho/dt)`` for any constant ``U`` (in the
incompressible limit the pressure simply acquires a linear ramp).  The *discrete*
lattice Boltzmann equation is **not** exactly invariant, because the lattice
velocities ``c_i`` are fixed and the equilibrium is a truncated Maxwellian.  The
residual is a polynomial in the Mach number, and on general grounds the leading
term is quadratic in ``Ma`` - the same order at which the D2Q9 equilibrium first
differs from a Maxwellian in its third moment.  This module

* applies the boost to a state (equilibrium re-evaluation, i.e. an exact
  superposition of the uniform flow on the fluctuating field),
* synchronises the boundary data (inlet velocity, moving walls) with the boost so
  that the *whole problem*, boundary conditions included, is transformed,
* measures the residual defect and fits ``defect = c * Ma^p`` so that the exponent
  can be *falsified* rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import CS, Simulation, equilibrium, macroscopic


# --- boost -------------------------------------------------------------------
def boosted_equilibrium(
    rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, boost: np.ndarray
) -> np.ndarray:
    """Equilibrium of the boosted state ``(rho, u + U)``.

    Superimposing a constant velocity on a kinetic state is done through the
    equilibrium distribution because a *rigid* shift of the discrete populations
    would require interpolation between lattice nodes (that interpolation is a
    different, larger error source).  The equilibrium boost is exact for a
    uniform field, which is why :func:`uniform_flow_fixed_point` is a meaningful
    sanity check.
    """
    return equilibrium(rho, ux + boost[0], uy + boost[1])


def boost_simulation(sim: Simulation, boost: np.ndarray) -> None:
    """Boost the field *and* every prescribed velocity of ``sim``."""
    rho, ux, uy = macroscopic(sim.f)
    sim.f = boosted_equilibrium(rho, ux, uy, boost)
    sim.inlet_velocity = np.asarray(sim.inlet_velocity) + np.asarray(boost)
    # a body force is a force, not a velocity: it must NOT be boosted


def momentum_mach(boost: np.ndarray) -> float:
    """Mach number of a boost, ``|U| / c_s``."""
    return float(np.linalg.norm(boost) / CS)


def uniform_flow_fixed_point(
    n: int = 48, tau: float = 0.6, boost: tuple[float, float] = (0.05, 0.0), steps: int = 400
) -> dict:
    """A uniform flow must be an exact fixed point of the lattice BGK equation.

    Periodic box, no body force.  The only dynamics allowed is the (exactly zero)
    change of the uniform state; everything else is round-off.  Returns the
    relative drift of the velocity field and the mass drift.
    """
    from .lattice import Simulation as _Sim

    sim = _Sim(n, n, tau)
    u0 = np.array(boost, dtype=np.float64)
    sim.f = equilibrium(np.ones((n, n)), np.full((n, n), u0[0]), np.full((n, n), u0[1]))
    rho0, ux0, uy0 = macroscopic(sim.f)
    for _ in range(steps):
        sim.step()
    rho1, ux1, uy1 = macroscopic(sim.f)
    return {
        "steps": steps,
        "mach": momentum_mach(u0),
        "velocity_drift_relative": float(
            max(np.abs(ux1 - ux0).max(), np.abs(uy1 - uy0).max()) / np.linalg.norm(u0)
        ),
        "density_drift_relative": float(np.abs(rho1 - rho0).max()),
    }


# --- defect metrics ----------------------------------------------------------
def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sqrt(np.sum((a - b) ** 2))
    den = np.sqrt(np.sum(b ** 2))
    if den == 0.0:
        return float(num)
    return float(num / den)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares fit of ``y = c x^p`` in log-log space (x, y > 0)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return {"exponent": float("nan"), "coefficient": float("nan"), "r2": float("nan")}
    lx = np.log(x[ok])
    ly = np.log(y[ok])
    p, logc = np.polyfit(lx, ly, 1)
    pred = p * lx + logc
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"exponent": float(p), "coefficient": float(np.exp(logc)), "r2": float(r2)}


def fit_ma2(mas: np.ndarray, defects: np.ndarray) -> dict:
    """Fit ``defect = c * Ma^2`` (fixed exponent) and compare with a free power law.

    The fixed-exponent fit is the *declared* law of this verification face; the
    free fit is reported next to it so that a departure from ``p = 2`` cannot be
    hidden.
    """
    mas = np.asarray(mas, dtype=np.float64)
    defects = np.asarray(defects, dtype=np.float64)
    ok = defects > 0
    c = float("nan")
    resid = float("nan")
    if ok.any():
        basis = mas[ok] ** 2
        c = float(np.sum(basis * defects[ok]) / np.sum(basis * basis))
        resid = float(np.sqrt(np.mean((defects[ok] - c * basis) ** 2)))
    free = fit_power_law(mas, np.abs(defects))
    return {
        "c_ma2": c,
        "residual_ma2": resid,
        "free_exponent": free["exponent"],
        "free_r2": free["r2"],
        "declared_law": "defect = c * Ma^2",
    }


# --- strict translation test -------------------------------------------------
def _spectral_vorticity(ux: np.ndarray, uy: np.ndarray, k: float) -> np.ndarray:
    nx = ux.shape[1]
    ky = 2.0 * np.pi * np.fft.fftfreq(ux.shape[0], d=1.0)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0)
    u_hat = np.fft.fft2(ux)
    v_hat = np.fft.fft2(uy)
    return 1j * (kx[None, :] * v_hat - ky[:, None] * u_hat)


def translation_phase(
    ux: np.ndarray, uy: np.ndarray, mode: tuple[int, int] = (1, 1)
) -> float:
    """Phase of the Fourier mode ``mode`` of the vorticity, in radians."""
    wh = _spectral_vorticity(ux, uy, 1.0)
    return float(np.angle(wh[mode]))


class DriftTracker:
    """Translation of a periodic field, measured *relative to its own initial state*.

    For ``omega = -2 A cos(k (x - a)) cos(k (y - b))`` the phases of the spectral
    modes ``(k, k)`` and ``(k, -k)`` are ``-k (a + b)`` and ``-k (a - b)``.  Taking
    the phase *difference* with respect to the first sample removes the constant
    ``pi`` branch of the cosine, which is otherwise the dominant error of a naive
    absolute-phase estimator.
    """

    def __init__(self, shape: tuple[int, int], k: float) -> None:
        self.shape = shape
        self.k = float(k)
        self._ref: tuple[complex, complex] | None = None

    def __call__(self, ux: np.ndarray, uy: np.ndarray) -> tuple[float, float]:
        ny, nx = ux.shape
        wh = _spectral_vorticity(ux, uy, self.k)
        c_pp = complex(wh[1, 1])
        c_pm = complex(wh[ny - 1, 1])
        if self._ref is None:
            self._ref = (c_pp, c_pm)
            return 0.0, 0.0
        ref_pp, ref_pm = self._ref
        d_pp = float(np.angle(c_pp * np.conj(ref_pp)))
        d_pm = float(np.angle(c_pm * np.conj(ref_pm)))
        p_pp = -d_pp / self.k
        p_pm = -d_pm / self.k
        return 0.5 * (p_pp + p_pm), 0.5 * (p_pp - p_pm)


def unwrap(track: list[float], period: float) -> np.ndarray:
    """Unwrap a sequence of phases sampled modulo ``2*pi/k`` (i.e. modulo ``period``)."""
    arr = np.asarray(track, dtype=np.float64)
    if arr.size == 0:
        return arr
    out = arr.copy()
    for i in range(1, arr.size):
        d = out[i] - out[i - 1]
        d -= period * np.round(d / period)
        out[i] = out[i - 1] + d
    return out