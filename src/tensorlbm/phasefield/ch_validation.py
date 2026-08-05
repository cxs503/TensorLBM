"""Small, deterministic diagnostics for the periodic D3Q19 free-energy step.

This module intentionally exercises the existing collision operator without
turning its observations into a physical validation or a conservation claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..d3q19 import equilibrium3d
from .free_energy import DoubleWellFreeEnergy, force_minus_phi_grad_mu

if TYPE_CHECKING:
    from ..multiphase3d import free_energy_step_3d as _free_energy_step_3d


def free_energy_step_3d(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily call the production operator, avoiding its phasefield import cycle."""
    from ..multiphase3d import free_energy_step_3d as production_step

    return production_step(*args, **kwargs)  # type: ignore[arg-type]


def _init_free_energy_g_3d(phi: torch.Tensor) -> torch.Tensor:
    """Import the production initializer only after package initialization."""
    from ..multiphase3d import init_free_energy_g_3d

    return init_free_energy_g_3d(phi)


@dataclass(frozen=True)
class FreeEnergyCHValidationConfig:
    """Fixed-size, periodic diagnostic inputs for ``free_energy_step_3d``."""

    shape: tuple[int, int, int] = (4, 5, 6)
    steps: int = 2
    seed: int = 0
    tau_f: float = 0.9
    tau_g: float = 0.8
    A: float = 0.1
    B: float = 0.1
    kappa: float = 0.02
    Gamma: float = 0.4

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(size <= 0 for size in self.shape):
            raise ValueError("shape must contain three positive (z, y, x) sizes")
        if self.steps < 2:
            raise ValueError("steps must be at least 2 for a multi-step diagnostic")
        if self.tau_f <= 0.0 or self.tau_g <= 0.0:
            raise ValueError("tau_f and tau_g must be positive")


@dataclass(frozen=True)
class FreeEnergyCHStepDiagnostic:
    """One observation, with each quantity named by its own distribution."""

    step: int
    phase_integral: float
    f_mass: float
    phase_is_finite: bool
    f_is_finite: bool
    g_is_finite: bool
    phase_min: float
    phase_max: float
    f_min: float
    f_max: float
    g_min: float
    g_max: float


@dataclass(frozen=True)
class FreeEnergyCHDiagnosticResult:
    """Diagnostic observations; explicitly not a physical acceptance result."""

    status: str
    physical_acceptance: bool
    conservation_claim: bool
    phase_integral_name: str
    f_mass_name: str
    series: tuple[FreeEnergyCHStepDiagnostic, ...]


def uniform_phase_capillary_force(
    phi: torch.Tensor, *, A: float, B: float, kappa: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the periodic Korteweg force for a supplied uniform phase field.

    The caller owns uniformity selection; this function exposes the common
    free-energy operators used by the production D3Q19 step.
    """
    model = DoubleWellFreeEnergy(A=A, B=B, kappa=kappa)
    mu = model.chemical_potential(phi, boundary="periodic")
    return force_minus_phi_grad_mu(phi, mu, boundary="periodic")


def _sample(step: int, f: torch.Tensor, g: torch.Tensor) -> FreeEnergyCHStepDiagnostic:
    """Record phase and momentum quantities without treating them as aliases."""
    phi = g.sum(dim=0)
    return FreeEnergyCHStepDiagnostic(
        step=step,
        phase_integral=float(phi.sum().item()),
        f_mass=float(f.sum().item()),
        phase_is_finite=bool(torch.isfinite(phi).all().item()),
        f_is_finite=bool(torch.isfinite(f).all().item()),
        g_is_finite=bool(torch.isfinite(g).all().item()),
        phase_min=float(phi.min().item()),
        phase_max=float(phi.max().item()),
        f_min=float(f.min().item()),
        f_max=float(f.max().item()),
        g_min=float(g.min().item()),
        g_max=float(g.max().item()),
    )


def run_closed_periodic_free_energy_diagnostic(
    config: FreeEnergyCHValidationConfig = FreeEnergyCHValidationConfig(),
) -> FreeEnergyCHDiagnosticResult:
    """Run a deterministic, collision-only periodic sequence through the real step.

    ``free_energy_step_3d`` has periodic finite-difference operators internally.
    This runner deliberately performs no boundary, geometry, forcing, streaming,
    or physical-acceptance assessment.  In particular, the recorded phase
    integral is ``sum(phi)`` for ``phi=sum_i(g_i)`` and is not advertised as an
    exact Cahn--Hilliard conservation check.
    """
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    # A nonuniform phase makes the multi-step call exercise the coupled g path.
    phi0 = 0.5 * (2.0 * torch.rand(config.shape, generator=generator) - 1.0)
    rho0 = torch.ones_like(phi0)
    zero = torch.zeros_like(phi0)
    f = equilibrium3d(rho0, zero, zero, zero)
    g = _init_free_energy_g_3d(phi0)

    series = [_sample(0, f, g)]
    for step in range(1, config.steps + 1):
        f, g = free_energy_step_3d(
            f,
            g,
            tau_f=config.tau_f,
            tau_g=config.tau_g,
            A=config.A,
            B=config.B,
            kappa=config.kappa,
            Gamma=config.Gamma,
        )
        series.append(_sample(step, f, g))

    return FreeEnergyCHDiagnosticResult(
        status="diagnostic_only",
        physical_acceptance=False,
        conservation_claim=False,
        phase_integral_name="phase_integral=sum(phi), where phi=sum_i(g_i)",
        f_mass_name="f_mass=sum_i,x(f_i)",
        series=tuple(series),
    )
