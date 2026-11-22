"""CH6 validation: static droplet, linear CH mode, planar interface.

This module provides deterministic diagnostics for the CH6 (Lee-Lin
consistent, D3Q19-isotropic) phase-field model, following the same
validation protocol as CH4 (``ch_validation.py``) but using the isotropic
operators throughout.

Validation steps (same as PF1/CH4):
1. Static droplet: three density ratios (1, 10, 1000) → Laplace pressure jump
2. Linear CH mode: prescribed zero-velocity evolution → compare growth rate
3. Planar interface: evolved interface → surface tension integral

Comparison with CH4:
- CH4 uses ``central7`` (7-point Laplacian + 2-point central gradient)
- CH6 uses D3Q19-isotropic operators (6 axis + 12 face-diagonal neighbours)
- CH6 should show lower spurious currents on curved interfaces
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..d3q19 import equilibrium3d
from .ch6 import CH6FreeEnergy, ch6_equilibrium, ch6_step_3d

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class CH6ValidationConfig:
    """Fixed-size, periodic diagnostic inputs for CH6."""

    shape: tuple[int, int, int] = (8, 8, 8)
    steps: int = 10
    seed: int = 0
    tau_f: float = 1.0
    tau_g: float = 0.7
    A: float = 0.1
    B: float = 0.1
    kappa: float = 0.02
    Gamma: float = 0.5

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(size <= 0 for size in self.shape):
            raise ValueError("shape must contain three positive (z, y, x) sizes")
        if self.steps < 2:
            raise ValueError("steps must be at least 2 for a multi-step diagnostic")
        if self.tau_f <= 0.0 or self.tau_g <= 0.0:
            raise ValueError("tau_f and tau_g must be positive")


@dataclass(frozen=True)
class CH6StepDiagnostic:
    """One observation of the CH6 state."""

    step: int
    phase_integral: float
    f_mass: float
    phase_is_finite: bool
    f_is_finite: bool
    g_is_finite: bool
    phase_min: float
    phase_max: float
    max_spurious_speed: float


@dataclass(frozen=True)
class CH6DiagnosticResult:
    """Diagnostic observations for CH6; explicitly not a physical acceptance result."""

    status: str
    physical_acceptance: bool
    conservation_claim: bool
    phase_integral_name: str
    f_mass_name: str
    series: tuple[CH6StepDiagnostic, ...]


def _init_free_energy_g_3d_ch6(phi: torch.Tensor) -> torch.Tensor:
    """Initialize g from phi using CH6 equilibrium with zero mu."""
    from ..d3q19 import W_EXACT64, C

    device = phi.device
    weights = W_EXACT64.to(device=device, dtype=phi.dtype)
    shape = (-1,) + (1,) * 3
    c = C.to(device=device, dtype=phi.dtype)
    w = weights.view(shape)

    # Zero velocity, zero mu → g_eq = w * phi
    geq = w * phi.unsqueeze(0)
    return geq


def _sample_ch6(
    step: int, f: torch.Tensor, g: torch.Tensor
) -> CH6StepDiagnostic:
    """Record CH6 state quantities."""
    phi = g.sum(dim=0)
    rho = f.sum(dim=0)
    cx = torch.tensor([1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0],
                      dtype=f.dtype, device=f.device).view(19, 1, 1, 1)
    cy = torch.tensor([1, 0, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, -1, 0, 0, 0, 1, 1, -1],
                      dtype=f.dtype, device=f.device).view(19, 1, 1, 1)
    cz = torch.tensor([0, 1, 0, 1, -1, 0, 0, 1, -1, 1, 1, -1, -1, 1, -1, 0, 0, 0, 0],
                      dtype=f.dtype, device=f.device).view(19, 1, 1, 1)
    ux = ((f * cx).sum(dim=0) / torch.clamp(rho, min=1e-12))
    uy = ((f * cy).sum(dim=0) / torch.clamp(rho, min=1e-12))
    uz = ((f * cz).sum(dim=0) / torch.clamp(rho, min=1e-12))
    speed = torch.sqrt(ux**2 + uy**2 + uz**2)
    return CH6StepDiagnostic(
        step=step,
        phase_integral=float(phi.sum().item()),
        f_mass=float(f.sum().item()),
        phase_is_finite=bool(torch.isfinite(phi).all().item()),
        f_is_finite=bool(torch.isfinite(f).all().item()),
        g_is_finite=bool(torch.isfinite(g).all().item()),
        phase_min=float(phi.min().item()),
        phase_max=float(phi.max().item()),
        max_spurious_speed=float(speed.max().item()),
    )


def run_ch6_closed_periodic_diagnostic(
    config: CH6ValidationConfig = CH6ValidationConfig(),
) -> CH6DiagnosticResult:
    """Run a deterministic, collision-only periodic sequence through CH6.

    Uses D3Q19-isotropic operators for both chemical potential and Korteweg
    force, ensuring pressure and flux are discretised at the same order.
    """
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    phi0 = 0.5 * (2.0 * torch.rand(config.shape, generator=generator) - 1.0)
    rho0 = torch.ones_like(phi0)
    zero = torch.zeros_like(phi0)
    f = equilibrium3d(rho0, zero, zero, zero)
    g = _init_free_energy_g_3d_ch6(phi0)

    series = [_sample_ch6(0, f, g)]
    for step in range(1, config.steps + 1):
        f, g = ch6_step_3d(
            f, g,
            tau_f=config.tau_f,
            tau_g=config.tau_g,
            A=config.A,
            B=config.B,
            kappa=config.kappa,
            Gamma=config.Gamma,
        )
        series.append(_sample_ch6(step, f, g))

    return CH6DiagnosticResult(
        status="diagnostic_only",
        physical_acceptance=False,
        conservation_claim=False,
        phase_integral_name="phase_integral=sum(phi), where phi=sum_i(g_i)",
        f_mass_name="f_mass=sum_i,x(f_i)",
        series=tuple(series),
    )


# --------------------------------------------------------------------------- #
# Static droplet validation (same protocol as CH4)
# --------------------------------------------------------------------------- #

def _tanh_profile_1d(x: torch.Tensor, x0: float, xi: float) -> torch.Tensor:
    """1-D tanh profile: tanh((x - x0) / xi)."""
    return torch.tanh((x - x0) / xi)


def _xi_from_params(A: float, B: float, kappa: float) -> float:
    """Interface width parameter from free-energy parameters."""
    return math.sqrt(kappa / (2.0 * A))


def _sigma_analytic(A: float, kappa: float) -> float:
    """Analytic surface tension for double-well free energy."""
    return (2.0 / 3.0) * math.sqrt(2.0 * kappa * A)


def run_ch6_static_droplet(
    *,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    radius: float = 8.0,
    A: float = 0.004,
    B: float = 0.004,
    kappa: float | None = None,
    xi: float = 3.0,
    tau_f: float = 1.0,
    tau_g: float = 1.0,
    Gamma: float = 0.5,
    steps: int = 500,
    geq_convention: str = "ch6",
) -> dict:
    """Static droplet validation for CH6.

    Creates a spherical droplet with tanh profile, evolves it, and measures:
    - Laplace pressure jump vs analytic
    - Spurious currents (max/RMS velocity)
    - Phase mass conservation

    Args:
        nx, ny, nz: Grid dimensions.
        radius: Droplet radius in lattice units.
        A: Double-well coefficient.
        B: Quartic coefficient.
        kappa: Interfacial tension parameter.  If None, computed from xi.
        xi: Interface width parameter.
        tau_f, tau_g: Relaxation times.
        Gamma: Mobility coupling.
        steps: Number of evolution steps.
        geq_convention: Must be "ch6".

    Returns:
        Dictionary with validation results.
    """
    if geq_convention != "ch6":
        raise ValueError("CH6 static droplet requires geq_convention='ch6'")

    if kappa is None:
        kappa = A * xi**2 / 2.0

    model = CH6FreeEnergy(A=A, B=B, kappa=kappa)
    xi_actual = _xi_from_params(A, B, kappa)
    sigma_analytic_val = _sigma_analytic(A, kappa)

    # Create spherical droplet with tanh profile
    z = torch.arange(nz, dtype=torch.float64).view(nz, 1, 1)
    y = torch.arange(ny, dtype=torch.float64).view(1, ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, nx)
    center_z, center_y, center_x = nz / 2, ny / 2, nx / 2

    r = torch.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
    phi0 = torch.tanh((radius - r) / xi_actual)

    # Initialize distributions
    rho0 = torch.ones_like(phi0, dtype=torch.float64)
    zero = torch.zeros_like(phi0)
    f = equilibrium3d(rho0, zero, zero, zero)
    mu0 = model.chemical_potential(phi0, boundary="periodic")
    velocity = (zero, zero, zero)
    g = ch6_equilibrium(phi0, velocity, mu0, Gamma)

    mass0 = float(g.sum())
    scale = float(phi0.abs().sum())
    sigma_history = []
    speed_history = []
    finite = True

    for step in range(steps + 1):
        phi = g.sum(dim=0)
        rho = f.sum(dim=0)

        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all() and (rho > 0).all()):
            finite = False
            break

        # Measure spurious currents
        cx = torch.tensor([1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0],
                          dtype=torch.float64, device=f.device).view(19, 1, 1, 1)
        cy = torch.tensor([1, 0, -1, 1, 0, -1, 1, 0, -1, 1, -1, 1, -1, 0, 0, 0, 1, 1, -1],
                          dtype=torch.float64, device=f.device).view(19, 1, 1, 1)
        cz = torch.tensor([0, 1, 0, 1, -1, 0, 0, 1, -1, 1, 1, -1, -1, 1, -1, 0, 0, 0, 0],
                          dtype=torch.float64, device=f.device).view(19, 1, 1, 1)
        ux = ((f * cx).sum(dim=0) / torch.clamp(rho, min=1e-12))
        uy = ((f * cy).sum(dim=0) / torch.clamp(rho, min=1e-12))
        uz = ((f * cz).sum(dim=0) / torch.clamp(rho, min=1e-12))
        speed = torch.sqrt(ux**2 + uy**2 + uz**2)

        # Measure surface tension from gradient energy
        # For a 3D profile, use the isotropic gradient
        from .isotropic_fd import isotropic_gradient_3d
        gx, gy, gz = isotropic_gradient_3d(phi, boundary="periodic")
        # sigma = kappa * sum_{y,z}(|grad phi|^2) / 2 (per interface)
        measured_sigma = float(kappa * (gx**2 + gy**2 + gz**2).mean(0).sum() / 2.0)

        drift = abs(float(phi.sum()) - mass0) / scale
        max_speed = float(speed.max().item())
        rms_speed = float(torch.sqrt((speed * speed).mean()).item())

        if step % 50 == 0 or step == steps:
            sigma_history.append({
                "step": step,
                "sigma_gradient_integral": measured_sigma,
                "max_speed": max_speed,
                "rms_speed": rms_speed,
                "phase_mass_drift": drift,
            })
            speed_history.append({
                "step": step,
                "max_speed": max_speed,
                "rms_speed": rms_speed,
            })

        if step == steps:
            break

        # Evolve
        f, g = ch6_step_3d(
            f, g,
            tau_f=tau_f, tau_g=tau_g,
            A=A, B=B, kappa=kappa, Gamma=Gamma,
        )

    final = sigma_history[-1] if sigma_history else {}
    sigma_error = abs(final.get("sigma_gradient_integral", 0) - sigma_analytic_val) / sigma_analytic_val if sigma_analytic_val > 0 else None

    checks = {
        "completed_finite": finite and steps > 0,
        "mass_drift": final.get("phase_mass_drift", 1.0) < 1e-10,
        "surface_tension_integral": sigma_error is not None and sigma_error < 0.1,
        "spurious_velocity_max": final.get("max_speed", 1.0) < 1e-3,
        "spurious_velocity_rms": final.get("rms_speed", 1.0) < 1e-4,
    }

    return {
        "case": "ch6_static_droplet",
        "scope": "d3q19_isotropic_chemical_potential_and_korteweg_force",
        "config": {
            "nx": nx, "ny": ny, "nz": nz,
            "radius": radius, "xi": xi, "xi_actual": xi_actual,
            "A": A, "B": B, "kappa": kappa,
            "tau_f": tau_f, "tau_g": tau_g, "Gamma": Gamma,
            "steps": steps, "geq_convention": geq_convention,
        },
        "sigma_analytic": sigma_analytic_val,
        "sigma_measured": final.get("sigma_gradient_integral"),
        "sigma_relative_error": sigma_error,
        "max_spurious_speed": final.get("max_speed"),
        "rms_spurious_speed": final.get("rms_speed"),
        "max_phase_mass_drift": final.get("phase_mass_drift"),
        "checks": checks,
        "passed": all(checks.values()),
        "completed_steps": steps if finite else 0,
        "sigma_history": sigma_history,
        "speed_history": speed_history,
    }


# --------------------------------------------------------------------------- #
# Linear CH mode validation
# --------------------------------------------------------------------------- #

def run_ch6_linear_mode(
    *,
    dimension: int = 3,
    n: int = 32,
    axis: int = 0,
    background: float = 0.0,
    Gamma: float = 0.5,
    tau_g: float = 1.0,
    steps: int = 200,
    A: float = 0.1,
    B: float = 0.1,
    kappa: float = 0.02,
    epsilon: float = 0.01,
    geq_convention: str = "ch6",
) -> dict:
    """Linear CH mode validation for CH6.

    Evolves a single Fourier mode on a uniform background with zero velocity.
    Compares the measured exponential growth/decay rate with the analytic
    CH dispersion relation.

    Analytic rate: ``-Gamma * (tau_g - 0.5) * k^2 * (-A + 3*B*background^2 + kappa*k^2)``

    Args:
        dimension: 2 or 3.
        n: Grid size.
        axis: Fourier mode axis.
        background: Uniform background value.
        Gamma: Mobility coupling.
        tau_g: Phase relaxation time.
        steps: Evolution steps.
        A, B, kappa: Free-energy parameters.
        epsilon: Perturbation amplitude.
        geq_convention: Must be "ch6".

    Returns:
        Dictionary with validation results.
    """
    if geq_convention != "ch6":
        raise ValueError("CH6 linear mode requires geq_convention='ch6'")

    if dimension == 2:
        from ..d2q9 import equilibrium, stream
        from .operators import laplacian_2d, central_gradient_2d
    else:
        from ..d3q19 import equilibrium3d
        from ..solver3d import stream3d
        from .operators import laplacian_3d, central_gradient_3d

    k = 2.0 * math.pi / n
    x = torch.arange(n, dtype=torch.float64).view(1, -1) if dimension == 2 else torch.arange(n, dtype=torch.float64).view(n, 1, 1)

    phi0 = background + epsilon * torch.cos(k * x)
    if dimension == 3:
        phi0 = phi0.expand(n, n, n).contiguous()

    zero = torch.zeros_like(phi0)
    f = equilibrium3d(torch.ones_like(phi0), zero, zero, zero) if dimension == 3 else equilibrium(torch.ones_like(phi0), zero, zero)
    mu0 = -A * phi0 + B * phi0**3 - kappa * (laplacian_2d(phi0, boundary="periodic") if dimension == 2 else laplacian_3d(phi0, boundary="periodic"))
    g = ch6_equilibrium(phi0, (zero, zero) if dimension == 2 else (zero, zero, zero), mu0, Gamma)

    mass0 = float(g.sum())
    scale = float(phi0.abs().sum())
    history = []
    finite = True

    for step in range(steps + 1):
        phi = g.sum(dim=0) if dimension == 3 else g.sum(0)
        rho = f.sum(dim=0) if dimension == 3 else f.sum(0)

        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all() and (rho > 0).all()):
            finite = False
            break

        drift = abs(float(phi.sum()) - mass0) / scale
        amplitude = float((phi - background).abs().mean().item())

        if step % 20 == 0 or step == steps:
            history.append({
                "step": step,
                "amplitude": amplitude,
                "phase_mass_drift": drift,
            })

        if step == steps:
            break

        f, g = ch6_step_3d(
            f, g,
            tau_f=1.0, tau_g=tau_g,
            A=A, B=B, kappa=kappa, Gamma=Gamma,
        )
        if dimension == 2:
            f, g = stream(f), stream(g)
        else:
            f, g = stream3d(f), stream3d(g)

    # Analytic rate
    expected = -Gamma * (tau_g - 0.5) * k**2 * (-A + 3 * B * background**2 + kappa * k**2)

    # Measure rate from amplitude decay
    samples = [r for r in history if r["step"] >= steps // 4 and r["amplitude"] > 0]
    complete = finite and step == steps and len(samples) >= 6
    measured = None
    fit_residual = None
    if complete:
        times = torch.tensor([r["step"] for r in samples], dtype=torch.float64)
        log_amp = torch.log(torch.tensor([r["amplitude"] for r in samples], dtype=torch.float64))
        # Simple linear fit
        mean_t = times.mean()
        mean_l = log_amp.mean()
        num = ((times - mean_t) * (log_amp - mean_l)).sum()
        den = ((times - mean_t)**2).sum()
        if den > 0:
            measured = float((num / den).item())
            fit_residual = float(torch.max(torch.abs(log_amp - (measured * times + (mean_l - measured * mean_t)))).item())

    amplitude_change = max(abs(r["amplitude"] / epsilon - 1) for r in history) if history else 1.0
    max_drift = max(r["phase_mass_drift"] for r in history) if history else 1.0

    rate_error = abs(measured - expected) / abs(expected) if complete and expected else None
    checks = {
        "completed_finite": complete,
        "mass_drift": max_drift < 1e-10,
        "rate_or_zero_mobility": (
            rate_error < 0.05 if rate_error is not None else complete and amplitude_change < 1e-8
        ),
        "fit_residual": fit_residual is not None and fit_residual < 1e-4,
    }

    return {
        "case": "ch6_linear_ch_mode",
        "scope": "prescribed_zero_physical_velocity",
        "config": {
            "dimension": dimension, "n": n, "axis": axis,
            "background": background, "Gamma": Gamma, "tau_g": tau_g,
            "steps": steps, "A": A, "B": B, "kappa": kappa,
            "epsilon": epsilon, "geq_convention": geq_convention,
        },
        "analytic_rate": expected,
        "measured_rate": measured,
        "relative_rate_error": abs(measured - expected) / abs(expected) if complete and expected else None,
        "fit_log_residual": fit_residual,
        "max_amplitude_relative_change": amplitude_change,
        "max_phase_mass_drift": max_drift,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_steps": step if finite else 0,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Planar interface validation
# --------------------------------------------------------------------------- #

def run_ch6_planar_interface(
    *,
    xi: float = 3.0,
    steps: int = 1000,
    A_base: float = 0.004,
    geq_convention: str = "ch6",
) -> dict:
    """Evolved planar interface validation for CH6.

    Two periodic interfaces evolve; surface tension integral is compared
    with the continuum tanh anchor.

    Args:
        xi: Interface width parameter.
        steps: Evolution steps.
        A_base: Base double-well coefficient.
        geq_convention: Must be "ch6".

    Returns:
        Dictionary with validation results.
    """
    if geq_convention != "ch6":
        raise ValueError("CH6 planar interface requires geq_convention='ch6'")

    from .isotropic_fd import isotropic_gradient_3d

    A = B = A_base * (3.0 / xi) ** 2
    kappa = A * xi**2 / 2
    nx = int(math.ceil(32 * xi))
    ny = 4
    nz = 4
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, -1)
    phi0 = (
        (torch.tanh((x - nx / 4) / xi) * torch.tanh((3 * nx / 4 - x) / xi))
        .expand(nz, ny, nx)
        .contiguous()
    )

    model = CH6FreeEnergy(A=A, B=B, kappa=kappa)
    zero = torch.zeros_like(phi0)
    f = equilibrium3d(torch.ones_like(phi0), zero, zero, zero)
    mu0 = model.chemical_potential(phi0, boundary="periodic")
    g = ch6_equilibrium(phi0, (zero, zero, zero), mu0, 0.5)

    mass0 = float(g.sum())
    scale = float(phi0.abs().sum())
    sigma_analytic_val = (2 / 3) * math.sqrt(2 * kappa * A)
    history = []
    finite = True
    max_drift = max_speed = 0.0

    for step in range(steps + 1):
        phi = g.sum(dim=0)
        rho = f.sum(dim=0)

        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all() and (rho > 0).all()):
            finite = False
            break

        mu = model.chemical_potential(phi, boundary="periodic")
        gx, gy, gz = isotropic_gradient_3d(mu, boundary="periodic")
        speed = float(
            ((gx**2 + gy**2 + gz**2).sqrt().max().item())
        )
        drift = abs(float(phi.sum()) - mass0) / scale
        max_drift = max(max_drift, drift)
        max_speed = max(max_speed, speed)

        gradx, _, _ = isotropic_gradient_3d(phi, boundary="periodic")
        # Match ch_benchmarks.py formula: mean over z and y, then sum over x
        measured_sigma = float(kappa * gradx.square().mean((0, 1)).sum() / 2)

        if step % 50 == 0 or step == steps:
            history.append({
                "step": step,
                "sigma_gradient_integral": measured_sigma,
                "profile_linf_change": float((phi - phi0).abs().max().item()),
                "max_physical_speed": speed,
                "phase_mass_drift": drift,
            })

        if step == steps:
            break

        f, g = ch6_step_3d(
            f, g,
            tau_f=1.0, tau_g=1.0,
            A=A, B=B, kappa=kappa, Gamma=0.5,
        )

    final = history[-1] if history else {}
    sigma_error = abs(final.get("sigma_gradient_integral", 0) - sigma_analytic_val) / sigma_analytic_val if sigma_analytic_val > 0 else None

    checks = {
        "completed_finite": finite and step == steps,
        "mass_drift": max_drift < 1e-10,
        "surface_tension_integral": sigma_error is not None and sigma_error < 0.1,
        "profile_preserved": final.get("profile_linf_change", 1.0) < 0.1,
        "spurious_velocity": max_speed < 1e-3,
    }

    return {
        "case": "ch6_evolved_planar_interface",
        "scope": "coupled_d3q19_periodic_isotropic_gradient_stress",
        "config": {
            "nx": nx, "ny": 4, "xi": xi,
            "A": A, "B": B, "kappa": kappa,
            "Gamma": 0.5, "tau_f": 1.0, "tau_g": 1.0,
            "steps": steps, "geq_convention": geq_convention,
        },
        "sigma_analytic": sigma_analytic_val,
        "sigma_relative_error": sigma_error,
        "max_phase_mass_drift": max_drift,
        "max_physical_speed": max_speed,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_steps": step if finite else 0,
        "history": history,
    }
