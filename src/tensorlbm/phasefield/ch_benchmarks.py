"""Executable CH-core acceptance experiments using the production FE steppers.

Linear growth/decay and a freely evolving planar interface are separate gates.
Passing them does not certify curved-interface pressure, capillary dispersion,
variable-density flow, or impact. All streaming is explicit and periodic.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..d2q9 import equilibrium, macroscopic
from ..d3q19 import equilibrium3d
from ..multiphase import free_energy_step, init_free_energy_g
from ..multiphase3d import free_energy_step_3d, init_free_energy_g_3d
from ..solver import stream
from ..solver3d import stream3d
from .operators import central_gradient_2d, central_gradient_3d, laplacian_2d, laplacian_3d


def _operators(dimension):
    if dimension == 2:
        return (
            equilibrium,
            init_free_energy_g,
            free_energy_step,
            stream,
            laplacian_2d,
            central_gradient_2d,
        )
    return (
        equilibrium3d,
        init_free_energy_g_3d,
        free_energy_step_3d,
        stream3d,
        laplacian_3d,
        central_gradient_3d,
    )


@torch.no_grad()
def run_linear_mode(
    *,
    dimension=2,
    n=32,
    axis=0,
    background=0.0,
    Gamma=0.5,
    tau_g=1.0,
    steps=400,
    geq_convention="lee_lin",
):
    """Fit a Fourier mode to an independent continuum CH growth/decay rate.

    The domain is thin in transverse directions; ``axis`` uses x,y,z order.
    kappa scales with n^2: n=16,32,64 samples the same physical wavelength
    and interface-width ratio under diffusive time scaling. The caller should
    scale steps by n^2 when comparing a fixed physical time interval.
    Momentum is deliberately prescribed so *physical* velocity is zero,
    including the half-force correction. This isolates CH, not coupled NS.
    """
    if dimension not in (2, 3) or axis not in range(dimension) or n < 16 or steps < 40:
        raise ValueError("invalid dimension, axis, n (<16), or steps (<40)")
    if not math.isfinite(tau_g) or tau_g <= 0.5:
        raise ValueError("tau_g must be finite and > 0.5")
    feq, init, collide, streaming, lap, gradient = _operators(dimension)
    shape = [2] * dimension
    shape[-axis - 1] = n
    coord_shape = [1] * dimension
    coord_shape[-axis - 1] = n
    k = 2 * math.pi / n
    mode = torch.cos(k * torch.arange(n, dtype=torch.float64)).view(coord_shape).expand(shape)
    A = B = 0.04
    kappa = 0.02 * (n / 32) ** 2
    epsilon = 1e-4
    phi0 = background + epsilon * mode
    zero = torch.zeros_like(phi0)
    mu0 = -A * phi0 + B * phi0**3 - kappa * lap(phi0, boundary="periodic")
    g = init(
        phi0,
        *([zero] * dimension),
        geq_convention=geq_convention,
        mu=mu0 if geq_convention == "lee_lin" else None,
        Gamma=Gamma,
    )
    initial_mass = float(g.sum())
    scale = max(float(phi0.abs().sum()), 1.0)
    history = []
    max_drift = 0.0
    finite = True
    for step in range(steps + 1):
        phi = g.sum(0)
        if not bool(torch.isfinite(g).all()):
            finite = False
            break
        max_drift = max(max_drift, abs(float(phi.sum()) - initial_mass) / scale)
        amplitude = float(2 * ((phi - background) * mode).mean())
        if step % 5 == 0 or step == steps:
            history.append({"step": step, "amplitude": amplitude})
        if step == steps:
            break
        mu = -A * phi + B * phi**3 - kappa * lap(phi, boundary="periodic")
        # F=-phi*grad(mu): set raw u=-F/(2*rho), so corrected physical u=0.
        # The legacy 3D path uses raw velocity, so prescribe exactly zero there.
        velocity = tuple(0.5 * phi * v for v in gradient(mu, boundary="periodic"))
        if dimension == 3 and geq_convention == "swift":
            velocity = (zero,) * dimension
        f = feq(torch.ones_like(phi), *velocity)
        _, g = collide(
            f, g, tau_g=tau_g, A=A, B=B, kappa=kappa, Gamma=Gamma, geq_convention=geq_convention
        )
        g = streaming(g)
    samples = [r for r in history if r["step"] >= steps // 4 and r["amplitude"] > 0]
    complete = finite and step == steps and len(samples) >= 6
    measured = None
    fit_residual = None
    if complete:
        times = np.array([r["step"] for r in samples])
        log_amp = np.log([r["amplitude"] for r in samples])
        measured, intercept = np.polyfit(times, log_amp, 1)
        measured = float(measured)
        fit_residual = float(np.max(np.abs(log_amp - (measured * times + intercept))))
    expected = -Gamma * (tau_g - 0.5) * k**2 * (-A + 3 * B * background**2 + kappa * k**2)
    error = abs(measured - expected) / abs(expected) if complete and expected else None
    amplitude_change = max(abs(r["amplitude"] / epsilon - 1) for r in history)
    checks = {
        "completed_finite": complete,
        "mass_drift": max_drift < 1e-10,
        "rate_or_zero_mobility": (
            error < 0.05 if error is not None else complete and amplitude_change < 1e-8
        ),
        "fit_residual": fit_residual is not None and fit_residual < 1e-4,
    }
    return {
        "case": "linear_ch_mode",
        "scope": "prescribed_zero_physical_velocity",
        "config": {
            "dimension": dimension,
            "n": n,
            "axis": axis,
            "background": background,
            "Gamma": Gamma,
            "tau_g": tau_g,
            "steps": steps,
            "A": A,
            "B": B,
            "kappa": kappa,
            "epsilon": epsilon,
            "geq_convention": geq_convention,
        },
        "analytic_rate": expected,
        "measured_rate": measured,
        "relative_rate_error": error,
        "fit_log_residual": fit_residual,
        "max_amplitude_relative_change": amplitude_change,
        "max_phase_mass_drift": max_drift,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_steps": step,
        "history": history,
    }


@torch.no_grad()
def run_planar_interface(*, xi=3.0, steps=1000, A_base=0.004, geq_convention="lee_lin"):
    """Evolve f AND g with two periodic interfaces; no phase resetting.

    Sigma is the discrete gradient-stress integral on the evolved field,
    compared with the continuum tanh anchor. It is not a measured Laplace
    pressure jump or a capillary-frequency measurement.
    """
    if not math.isfinite(xi) or xi < 2 or steps < 1 or not math.isfinite(A_base) or A_base <= 0:
        raise ValueError("xi must be >= 2; steps and A_base must be positive")
    # Hold kappa fixed: under spatial refinement A scales as dx^2.
    # CH physical time then scales as dx^4 (use steps proportional to xi^4).
    A = B = A_base * (3.0 / xi) ** 2
    kappa = A * xi**2 / 2
    nx = int(math.ceil(32 * xi))
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    phi0 = (
        (torch.tanh((x - nx / 4) / xi) * torch.tanh((3 * nx / 4 - x) / xi))
        .expand(4, nx)
        .contiguous()
    )
    zero = torch.zeros_like(phi0)
    f = equilibrium(torch.ones_like(phi0), zero, zero)
    mu0 = -A * phi0 + B * phi0**3 - kappa * laplacian_2d(phi0, boundary="periodic")
    g = init_free_energy_g(
        phi0,
        geq_convention=geq_convention,
        mu=mu0 if geq_convention == "lee_lin" else None,
        Gamma=0.5,
    )
    mass0 = float(g.sum())
    scale = float(phi0.abs().sum())
    sigma = (2 / 3) * math.sqrt(2 * kappa * A)
    history = []
    finite = True
    max_drift = max_speed = 0.0
    for step in range(steps + 1):
        rho, ux, uy = macroscopic(f)
        phi = g.sum(0)
        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all() and (rho > 0).all()):
            finite = False
            break
        mu = -A * phi + B * phi**3 - kappa * laplacian_2d(phi, boundary="periodic")
        mx, my = central_gradient_2d(mu, boundary="periodic")
        speed = float(
            ((ux - 0.5 * phi * mx / rho) ** 2 + (uy - 0.5 * phi * my / rho) ** 2).sqrt().max()
        )
        drift = abs(float(phi.sum()) - mass0) / scale
        max_drift, max_speed = max(max_drift, drift), max(max_speed, speed)
        gradx, _ = central_gradient_2d(phi, boundary="periodic")
        measured_sigma = float(kappa * gradx.square().mean(0).sum() / 2)
        if step % 50 == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "sigma_gradient_integral": measured_sigma,
                    "profile_linf_change": float((phi - phi0).abs().max()),
                    "max_physical_speed": speed,
                    "phase_mass_drift": drift,
                }
            )
        if step == steps:
            break
        f, g = free_energy_step(
            f,
            g,
            tau_f=1.0,
            tau_g=1.0,
            A=A,
            B=B,
            kappa=kappa,
            Gamma=0.5,
            geq_convention=geq_convention,
        )
        f, g = stream(f), stream(g)
    final = history[-1]
    error = abs(final["sigma_gradient_integral"] - sigma) / sigma
    checks = {
        "completed_finite": finite and step == steps,
        "mass_drift": max_drift < 1e-10,
        "surface_tension_integral": error < 0.1,
        "profile_preserved": final["profile_linf_change"] < 0.1,
        "spurious_velocity": max_speed < 1e-3,
    }
    return {
        "case": "evolved_planar_interface",
        "scope": "coupled_d2q9_periodic_gradient_stress",
        "config": {
            "nx": nx,
            "ny": 4,
            "xi": xi,
            "A": A,
            "B": B,
            "kappa": kappa,
            "Gamma": 0.5,
            "tau_f": 1.0,
            "tau_g": 1.0,
            "steps": steps,
            "geq_convention": geq_convention,
        },
        "sigma_analytic": sigma,
        "sigma_relative_error": error,
        "max_phase_mass_drift": max_drift,
        "max_physical_speed": max_speed,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_steps": step,
        "history": history,
    }
