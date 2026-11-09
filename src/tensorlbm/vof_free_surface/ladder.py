"""Three-tier verification ladder for the VOF free-surface solver.

Tier definitions and anchors (all analytic, no tabulated "reference" values):

* **L1 - Ritter (1892) dam break.**  Frictionless shallow-water release of a
  column of depth ``h0`` onto a dry bed.  Analytic depth/velocity and front
  position ``xf = 2 sqrt(g h0) t``; the ladder also audits liquid-volume
  conservation of the VOF transport.
* **L2 - second-order Stokes wave (Dean & Dalrymple 1991).**
  ``eta = a cos(theta) + (k a^2/4) (3 - sigma^2)/sigma^3 cos(2 theta)``,
  ``sigma = tanh(kh)``, ``omega^2 = g k tanh(kh)``; deep-water limit is used
  here, so the second-order bound term reduces to ``k a^2/2``.  Phase speed and
  free-surface shape are measured against the analytic solution.
* **L3 - Lamb (1932) capillary-wave damping.**  Deep-water free-surface wave
  with ``omega^2 = g k + sigma k^3 / rho`` and viscous amplitude decay rate
  ``2 nu k^2`` (Lamb 1932, Sec. 348).  The frequency and the decay rate are
  extracted from the simulated free surface and compared with the theory.

Every returned number is measured on the simulated fields; nothing is
hard-coded, so a regression in the solver shows up as a change of these values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .solver import VOFChannelFlow, VOFConfig

__all__ = [
    "capillary_damping_lamb",
    "dam_break_ritter_profile",
    "run_capillary_damping",
    "run_dam_break_ritter",
    "run_stokes2_wave",
    "stokes2_free_surface",
]


# --------------------------------------------------------------------------- #
# L1 - Ritter dam break
# --------------------------------------------------------------------------- #
def dam_break_ritter_profile(
    x: np.ndarray, t: float, *, x0: float, h0: float, gravity: float = 9.81
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ritter (1892) shallow-water solution on a dry, frictionless bed."""
    if t <= 0.0:
        raise ValueError("t must be positive")
    c0 = math.sqrt(gravity * h0)
    xi = np.asarray(x, dtype=float) - x0
    h = np.zeros_like(xi)
    u = np.zeros_like(xi)
    reservoir = xi <= -c0 * t
    rarefaction = (xi > -c0 * t) & (xi < 2.0 * c0 * t)
    h[reservoir] = h0
    h[rarefaction] = (2.0 * c0 - xi[rarefaction] / t) ** 2 / (9.0 * gravity)
    u[rarefaction] = (2.0 / 3.0) * (xi[rarefaction] / t + c0)
    front = x0 + 2.0 * c0 * t
    return h, u, front


def run_dam_break_ritter(
    *,
    nx: int = 400,
    h0: float = 0.10,
    aspect: float = 20.0,
    t_end: float | None = None,
    n_samples: int = 8,
    bottom_noslip: bool = False,
    gravity: float = 9.81,
    cfl: float = 0.30,
    sigma: float = 0.0,
) -> dict[str, object]:
    """L1: release a column of water and compare the front with Ritter.

    ``aspect`` is ``length_x / h0``; the shallow-water limit needs a long, thin
    domain, so the aspect ratio is an explicit, swept parameter.
    """
    length_x = aspect * h0
    # the ceiling must stay well above the column: a lid at 3 h0 turns the
    # release into a lid-driven flow and the front no longer follows Ritter
    length_y = 6.0 * h0
    dx = length_x / nx
    ny = int(round(length_y / dx))
    cfg = VOFConfig(
        nx=nx,
        ny=ny,
        length_x=length_x,
        length_y=ny * dx,
        nu_liquid=1.0e-6,
        nu_gas=1.5e-5,
        sigma=max(sigma, 1.0e-9),
        surface_tension=False,
        gravity=gravity,
        bottom_noslip=bottom_noslip,
        cfl=cfl,
    )
    x0 = 0.5 * length_x
    xc = (np.arange(nx) + 0.5) * cfg.dx
    yc = (np.arange(ny) + 0.5) * cfg.dy
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    F = ((X < x0) & (Y < h0)).astype(float)
    solver = VOFChannelFlow(cfg, F)
    if t_end is None:
        t_end = 4.0 * math.sqrt(h0 / gravity)
    times = np.linspace(t_end / n_samples, t_end, n_samples)
    rows: list[dict[str, float]] = []
    ci = math.sqrt(gravity * h0)
    for target in times:
        solver.run(target)
        height = solver.surface_height()
        threshold = max(2.0 * cfg.dy, 0.02 * h0)
        wet = np.nonzero(height > threshold)[0]
        x_front = xc[wet[-1]] if wet.size else 0.0
        _, _, x_front_ref = dam_break_ritter_profile(
            np.array([x_front]), solver.t, x0=x0, h0=h0, gravity=gravity
        )
        # analytic depth profile, evaluated at the same stations
        h_ref, _, _ = dam_break_ritter_profile(xc, solver.t, x0=x0, h0=h0, gravity=gravity)
        depth = height
        wet_ref = h_ref > 1.0e-4 * h0
        depth_error = float(
            np.sqrt(np.mean((depth[wet_ref] - h_ref[wet_ref]) ** 2)) / h0
        )
        rows.append(
            {
                "t": float(solver.t),
                "t_star": float(solver.t * math.sqrt(gravity / h0)),
                "x_front_sim": float(x_front),
                "x_front_ritter": float(x_front_ref),
                "front_error_pct": float(100.0 * abs(x_front - x_front_ref) / x_front_ref),
                "depth_rms_error_pct_h0": 100.0 * depth_error,
                "mass_error_pct": 100.0 * solver.mass_error(),
            }
        )
    longest = max(rows, key=lambda r: r["t_star"])
    return {
        "tier": "L1_ritter_dam_break",
        "configuration": {
            "nx": nx,
            "ny": ny,
            "h0": h0,
            "aspect_ratio": aspect,
            "bottom_noslip": bottom_noslip,
            "dx": cfg.dx,
            "cfl": cfl,
            "t_end": solver.t,
            "steps": solver.time_steps,
        },
        "samples": rows,
        "metrics": {
            "max_front_error_pct": max(r["front_error_pct"] for r in rows),
            "final_front_error_pct": rows[-1]["front_error_pct"],
            "max_depth_rms_error_pct_h0": max(r["depth_rms_error_pct_h0"] for r in rows),
            "max_mass_error_pct": max(abs(r["mass_error_pct"]) for r in rows),
            "final_t_star": longest["t_star"],
        },
        "criteria": {
            "front_error_below_3pct": max(r["front_error_pct"] for r in rows) < 3.0,
            "mass_error_below_1pct": max(abs(r["mass_error_pct"]) for r in rows) < 1.0,
        },
    }


# --------------------------------------------------------------------------- #
# L2 - second-order Stokes wave
# --------------------------------------------------------------------------- #
def stokes2_free_surface(
    x: np.ndarray, t: float, *, a: float, k: float, h: float, gravity: float = 9.81
) -> tuple[np.ndarray, float, float]:
    """Second-order Stokes free surface (Dean & Dalrymple 1991, eq. 3.28/3.29)."""
    sigma = math.tanh(k * h)
    omega = math.sqrt(gravity * k * sigma)
    theta = k * x - omega * t
    amp2 = 0.25 * k * a * a * (3.0 - sigma * sigma) / sigma**3
    eta = a * np.cos(theta) + amp2 * np.cos(2.0 * theta)
    return eta, omega, amp2


def _stokes2_velocity(
    X: np.ndarray, Y: np.ndarray, t: float, *, a: float, k: float, h: float, gravity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-order Stokes kinematics above a flat bed at ``y = 0``."""
    sigma = math.tanh(k * h)
    omega = math.sqrt(gravity * k * sigma)
    theta = k * X - omega * t
    z = Y - h
    uk = a * omega * np.cosh(k * (z + h)) / sigma * np.cos(theta)
    uk2 = 0.75 * k * a * a * omega * np.cosh(2.0 * k * (z + h)) / sigma**4 * np.cos(2.0 * theta)
    wk = a * omega * np.sinh(k * (z + h)) / sigma * np.sin(theta)
    wk2 = 0.75 * k * a * a * omega * np.sinh(2.0 * k * (z + h)) / sigma**4 * np.sin(2.0 * theta)
    eta, _, _ = stokes2_free_surface(X[:, 0], t, a=a, k=k, h=h, gravity=gravity)
    return uk + uk2, wk + wk2, eta


def run_stokes2_wave(
    *,
    nx: int = 360,
    n_wavelengths: float = 4.0,
    wavelength: float = 2.0,
    amplitude: float = 0.05,
    depth: float = 2.0,
    periods: float = 3.0,
    n_samples: int = 24,
    cells_per_wavelength: float = 60.0,
    gravity: float = 9.81,
) -> dict[str, object]:
    """L2: propagate a second-order Stokes wave and compare with the theory."""
    length_x = n_wavelengths * wavelength
    dx = wavelength / cells_per_wavelength
    nx = int(round(length_x / dx))
    length_y = depth + 1.5 * amplitude
    ny = int(round(length_y / dx))
    cfg = VOFConfig(
        nx=nx,
        ny=ny,
        length_x=length_x,
        length_y=ny * dx,
        nu_liquid=1.0e-6,
        nu_gas=1.5e-5,
        sigma=0.0728,
        surface_tension=False,
        gravity=gravity,
        bottom_noslip=False,
        cfl=0.30,
    )
    k = 2.0 * math.pi / wavelength
    xc = (np.arange(nx) + 0.5) * cfg.dx
    yc = (np.arange(ny) + 0.5) * cfg.dy
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    eta0, omega, _ = stokes2_free_surface(xc, 0.0, a=amplitude, k=k, h=depth, gravity=gravity)
    F = (Y < (depth + eta0)[:, None]).astype(float)
    solver = VOFChannelFlow(cfg, F)
    u0, w0, _ = _stokes2_velocity(X, Y, 0.0, a=amplitude, k=k, h=depth, gravity=gravity)
    # y-face velocities: average the cell-centred analytic field
    solver.u = 0.5 * (u0 + np.roll(u0, 1, axis=0))
    solver.v[:, 1:-1] = 0.5 * (w0[:, 1:] + w0[:, :-1])
    period = 2.0 * math.pi / omega
    t_end = periods * period
    times = np.linspace(t_end / n_samples, t_end, n_samples)
    phase: list[float] = []
    phi_t: list[float] = []
    eta_errors: list[float] = []
    speed_errors: list[float] = []
    c0 = omega / k
    for target in times:
        solver.run(target)
        eta_sim = solver.surface_height() - depth
        eta_ref, _, _ = stokes2_free_surface(
            xc - c0 * solver.t, 0.0, a=amplitude, k=k, h=depth, gravity=gravity
        )
        eta_errors.append(float(np.sqrt(np.mean((eta_sim - eta_ref) ** 2)) / amplitude))
        spec = np.fft.rfft(eta_sim)
        idx = int(round(k * length_x / (2.0 * math.pi)))
        phase.append(float(np.angle(spec[idx])))
        phi_t.append(float(solver.t))
    phase_u = np.unwrap(np.asarray(phase))
    # phase increases as omega t with a negative sign in angle(exp(i(kx - omega t)))
    omega_meas = float(-np.polyfit(np.asarray(phi_t), phase_u, 1)[0])
    c_meas = abs(omega_meas) / k
    return {
        "tier": "L2_stokes2_wave",
        "configuration": {
            "nx": nx,
            "ny": ny,
            "wavelength": wavelength,
            "amplitude": amplitude,
            "depth": depth,
            "kh": k * depth,
            "ka": k * amplitude,
            "cells_per_wavelength": cells_per_wavelength,
            "periods": periods,
            "steps": solver.time_steps,
            "dx": cfg.dx,
        },
        "metrics": {
            "phase_speed_analytic": c0,
            "phase_speed_measured": c_meas,
            "phase_speed_error_pct": abs(c_meas - c0) / c0 * 100.0,
            "omega_analytic": omega,
            "omega_measured": abs(omega_meas),
            "eta_rms_error_pct_amplitude": 100.0 * max(eta_errors),
            "eta_rms_error_final_pct_amplitude": 100.0 * eta_errors[-1],
            "mass_error_pct": 100.0 * solver.mass_error(),
        },
        "criteria": {
            "phase_speed_error_below_5pct": abs(c_meas - c0) / c0 < 0.05,
            "eta_error_below_8pct": max(eta_errors) < 0.08,
        },
    }


# --------------------------------------------------------------------------- #
# L3 - Lamb capillary-wave damping
# --------------------------------------------------------------------------- #
def capillary_damping_lamb(
    wavelength: float, viscosity: float, *, gravity: float = 9.81,
    rho: float = 1000.0, sigma: float = 0.0728,
) -> tuple[float, float]:
    """Lamb (1932) deep-water capillary-gravity wave: ``(omega, decay_rate)``."""
    k = 2.0 * math.pi / wavelength
    omega = math.sqrt(gravity * k + sigma * k**3 / rho)
    return omega, 2.0 * viscosity * k * k


def run_capillary_damping(
    *,
    wavelength: float = 0.015,
    amplitude: float = 0.0005,
    depth: float = 0.06,
    n_wavelengths: float = 6.0,
    cells_per_wavelength: float = 60.0,
    viscosity: float = 1.0e-5,
    sigma: float = 0.0728,
    rho: float = 1000.0,
    gravity: float = 9.81,
    periods: float = 4.0,
    n_samples: int = 30,
) -> dict[str, object]:
    """L3: capillary wave damping rate and frequency against Lamb's theory."""
    length_x = n_wavelengths * wavelength
    dx = wavelength / cells_per_wavelength
    nx = int(round(length_x / dx))
    length_y = depth + 1.5 * amplitude
    ny = int(round(length_y / dx))
    cfg = VOFConfig(
        nx=nx,
        ny=ny,
        length_x=length_x,
        length_y=ny * dx,
        rho_liquid=rho,
        nu_liquid=viscosity,
        nu_gas=viscosity,
        sigma=sigma,
        surface_tension=True,
        gravity=gravity,
        bottom_noslip=False,
        cfl=0.30,
        capillary_cfl=0.40,
    )
    k = 2.0 * math.pi / wavelength
    omega, decay = capillary_damping_lamb(
        wavelength, viscosity, gravity=gravity, rho=rho, sigma=sigma
    )
    xc = (np.arange(nx) + 0.5) * cfg.dx
    yc = (np.arange(ny) + 0.5) * cfg.dy
    eta0 = amplitude * np.cos(k * xc)
    F = (yc[:, None] < (depth + eta0)[None, :]).T.astype(float)
    solver = VOFChannelFlow(cfg, F)
    # linear capillary-wave kinematics: u = a omega cos(theta), w = a omega sin(theta)
    u0 = amplitude * omega * np.cos(k * xc)
    w_prof = amplitude * omega * np.exp(k * (yc - depth))
    u_face = 0.5 * (u0 + np.roll(u0, 1, axis=0))
    solver.u = np.broadcast_to(u_face[:, None], (nx, ny)).copy()
    w_face = 0.5 * (w_prof[1:] + w_prof[:-1])
    solver.v[:, 1:-1] = np.sin(k * xc)[:, None] * w_face[None, :]
    period = 2.0 * math.pi / omega
    times = np.linspace(periods * period / n_samples, periods * period, n_samples)
    amps: list[float] = []
    phases: list[float] = []
    mass: list[float] = []
    for target in times:
        solver.run(target)
        eta_sim = solver.surface_height() - depth
        spec = np.fft.rfft(eta_sim)
        idx = int(round(k * length_x / (2.0 * math.pi)))
        amps.append(float(2.0 * abs(spec[idx]) / nx))
        phases.append(float(np.angle(spec[idx])))
        mass.append(100.0 * solver.mass_error())
    t = np.asarray(times)
    amp = np.asarray(amps)
    # fit the exponential decay on the samples where the wave is still resolved
    keep = amp > 0.02 * amplitude
    fit = np.polyfit(t[keep], np.log(amp[keep]), 1)
    decay_meas = float(-fit[0])
    phase_u = np.unwrap(np.asarray(phases))
    omega_meas = float(abs(np.polyfit(t, phase_u, 1)[0]))
    omega_err = abs(omega_meas - omega) / omega * 100.0
    decay_err = abs(decay_meas - decay) / decay * 100.0
    return {
        "tier": "L3_lamb_capillary_damping",
        "configuration": {
            "nx": nx,
            "ny": ny,
            "wavelength": wavelength,
            "amplitude": amplitude,
            "depth": depth,
            "viscosity": viscosity,
            "sigma": sigma,
            "cells_per_wavelength": cells_per_wavelength,
            "periods": periods,
            "steps": solver.time_steps,
            "dx": cfg.dx,
        },
        "samples": {
            "t": t.tolist(),
            "amplitude": amps,
            "decay_ratio_measured_over_lamb": (decay_meas / decay) if decay else None,
        },
        "metrics": {
            "omega_lamb": omega,
            "omega_measured": omega_meas,
            "omega_error_pct": omega_err,
            "decay_lamb": decay,
            "decay_measured": decay_meas,
            "decay_error_pct": decay_err,
            "mass_error_pct": max(abs(m) for m in mass),
        },
        "criteria": {
            "decay_error_below_10pct": decay_err < 10.0,
            "omega_error_below_5pct": omega_err < 5.0,
        },
    }