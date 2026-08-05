"""Explicit D3Q19 free-energy collision-only evolution adapter.

The adapter is deliberately limited to the production collision operator.  It
neither streams populations nor applies boundary treatment, so its results are
not physical evolution results until those owners are connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..d3q19 import equilibrium3d


D3Q19_POPULATIONS = 19
COLLISION_ONLY_STAGE = "collision_only"
NO_STREAMING_BOUNDARY_WITHHELD = "no_streaming_boundary_withheld"
_PERIODIC_DIFFERENTIAL_OPERATOR = "periodic finite-difference operator internal to production free_energy_step_3d"


def init_free_energy_g_3d(phi: torch.Tensor) -> torch.Tensor:
    """Lazily invoke the real production initializer without an import cycle."""
    from ..multiphase3d import init_free_energy_g_3d as production_initializer

    return production_initializer(phi)


def free_energy_step_3d(
    f: torch.Tensor, g: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily invoke the real production collision operator."""
    from ..multiphase3d import free_energy_step_3d as production_step

    return production_step(f, g, **kwargs)  # type: ignore[arg-type]


def _validate_distribution(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[0] != D3Q19_POPULATIONS:
        raise ValueError(f"{name} must have shape (19, nz, ny, nx)")
    if any(size <= 0 for size in value.shape[1:]):
        raise ValueError(f"{name} spatial dimensions must be positive")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


@dataclass(frozen=True)
class FreeEnergyCollisionOnlyState:
    """The coupled production state: D3Q19 ``(f, g)`` distributions."""

    f: torch.Tensor
    g: torch.Tensor

    def __post_init__(self) -> None:
        _validate_distribution("f", self.f)
        _validate_distribution("g", self.g)
        if self.f.shape != self.g.shape:
            raise ValueError("f and g must have the same shape")
        if self.f.device != self.g.device:
            raise ValueError("f and g must have the same device")
        if self.f.dtype != self.g.dtype:
            raise TypeError("f and g must have the same dtype")


@dataclass(frozen=True)
class FreeEnergyCollisionOnlyConfig:
    """Parameters passed unchanged to the production collision operator."""

    steps: int = 2
    tau_f: float = 1.0
    tau_g: float = 0.7
    A: float = 0.1
    B: float = 0.1
    kappa: float = 0.02
    Gamma: float = 0.5
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    rho_heavy: float | None = None
    rho_light: float | None = None

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.tau_f <= 0.0 or self.tau_g <= 0.0:
            raise ValueError("tau_f and tau_g must be positive")


@dataclass(frozen=True)
class FreeEnergyCollisionOnlyDiagnostic:
    """One explicitly non-acceptance observation of the ``(f, g)`` state."""

    step: int
    phi_integral: float
    f_mass: float
    g_sum: float
    phi_integral_name: str = "phi_integral=sum_x(phi), where phi=sum_i(g_i)"
    f_mass_name: str = "f_mass=sum_i,x(f_i)"
    g_sum_name: str = "g_sum=sum_i,x(g_i)"


@dataclass(frozen=True)
class FreeEnergyCollisionOnlyResult:
    """Collision sequence output, withheld from physical interpretation."""

    state: FreeEnergyCollisionOnlyState
    diagnostics: tuple[FreeEnergyCollisionOnlyDiagnostic, ...]
    stage: str = COLLISION_ONLY_STAGE
    status: str = NO_STREAMING_BOUNDARY_WITHHELD
    physical: bool = False
    differential_operator: str = _PERIODIC_DIFFERENTIAL_OPERATOR


def initialize_free_energy_collision_only_state(phi: torch.Tensor) -> FreeEnergyCollisionOnlyState:
    """Build a production-compatible state from a 3-D phase field.

    ``g`` is initialized by the real ``init_free_energy_g_3d`` production
    initializer; ``f`` is the D3Q19 zero-velocity unit-density equilibrium.
    """
    if not isinstance(phi, torch.Tensor):
        raise TypeError("phi must be a torch.Tensor")
    if phi.ndim != 3:
        raise ValueError("phi must be a 3-D scalar field with shape (nz, ny, nx)")
    if any(size <= 0 for size in phi.shape):
        raise ValueError("phi spatial dimensions must be positive")
    if not phi.is_floating_point():
        raise TypeError("phi must have a floating-point dtype")
    zero = torch.zeros_like(phi)
    return FreeEnergyCollisionOnlyState(
        f=equilibrium3d(torch.ones_like(phi), zero, zero, zero),
        g=init_free_energy_g_3d(phi),
    )


def _diagnostic(step: int, state: FreeEnergyCollisionOnlyState) -> FreeEnergyCollisionOnlyDiagnostic:
    phi = state.g.sum(dim=0)
    return FreeEnergyCollisionOnlyDiagnostic(
        step=step,
        phi_integral=float(phi.sum().item()),
        f_mass=float(state.f.sum().item()),
        g_sum=float(state.g.sum().item()),
    )


def run_free_energy_collision_only(
    state: FreeEnergyCollisionOnlyState,
    config: FreeEnergyCollisionOnlyConfig = FreeEnergyCollisionOnlyConfig(),
) -> FreeEnergyCollisionOnlyResult:
    """Apply only real production collisions and retain the updated ``(f, g)``.

    The production step uses periodic differential operators internally.  This
    runner intentionally adds no streaming and no boundary action; therefore
    ``status`` is always ``no_streaming_boundary_withheld`` and ``physical`` is
    always false.
    """
    if not isinstance(state, FreeEnergyCollisionOnlyState):
        raise TypeError("state must be a FreeEnergyCollisionOnlyState")
    if not isinstance(config, FreeEnergyCollisionOnlyConfig):
        raise TypeError("config must be a FreeEnergyCollisionOnlyConfig")

    current = state
    diagnostics = [_diagnostic(0, current)]
    for step in range(1, config.steps + 1):
        f, g = free_energy_step_3d(
            current.f,
            current.g,
            tau_f=config.tau_f,
            tau_g=config.tau_g,
            A=config.A,
            B=config.B,
            kappa=config.kappa,
            Gamma=config.Gamma,
            gx=config.gx,
            gy=config.gy,
            gz=config.gz,
            rho_heavy=config.rho_heavy,
            rho_light=config.rho_light,
        )
        current = FreeEnergyCollisionOnlyState(f=f, g=g)
        diagnostics.append(_diagnostic(step, current))

    return FreeEnergyCollisionOnlyResult(state=current, diagnostics=tuple(diagnostics))


__all__ = [
    "COLLISION_ONLY_STAGE",
    "D3Q19_POPULATIONS",
    "NO_STREAMING_BOUNDARY_WITHHELD",
    "FreeEnergyCollisionOnlyConfig",
    "FreeEnergyCollisionOnlyDiagnostic",
    "FreeEnergyCollisionOnlyResult",
    "FreeEnergyCollisionOnlyState",
    "initialize_free_energy_collision_only_state",
    "run_free_energy_collision_only",
]
