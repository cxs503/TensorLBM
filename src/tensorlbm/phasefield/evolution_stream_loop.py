"""Real CH collision followed by explicit adapter streaming, with physics withheld.

This R1 loop is the sole owner of state handoff for this sequence: each step
calls the real production ``free_energy_step_3d`` collision through its lazy
adapter wrapper, then calls ``stream_free_energy_adapter``.  It is not a full
physical CH production solver: phase flux, wetting, pressure, and Laplace
observables are deliberately absent or withheld.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .evolution_adapter import FreeEnergyCollisionOnlyState, free_energy_step_3d
from .stream_boundary_contract import (
    PHASE_FLUX_WITHHELD,
    BoundaryPolicy,
    stream_free_energy_adapter,
)

ADAPTER_STREAM_LOOP_STAGE = "collision_then_adapter_stream"


@dataclass(frozen=True)
class FreeEnergyAdapterStreamLoopConfig:
    """Collision parameters plus the only supported adapter boundary policies."""

    steps: int = 2
    boundary: BoundaryPolicy = "periodic"
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
        if self.boundary not in ("periodic", "no_flux"):
            raise ValueError("boundary must be either 'periodic' or 'no_flux'")
        if self.tau_f <= 0.0 or self.tau_g <= 0.0:
            raise ValueError("tau_f and tau_g must be positive")


@dataclass(frozen=True)
class FreeEnergyAdapterStreamLoopDiagnostic:
    """Separate distribution and macroscopic bookkeeping sample."""

    step: int
    phi_integral: float
    f_mass: float
    g_sum: float
    f_distribution_inventory: float
    g_distribution_inventory: float
    distribution_inventory: float
    phi_integral_name: str = "phi_integral=sum_x(phi), where phi=sum_i(g_i)"
    f_mass_name: str = "f_mass=sum_i,x(f_i)"
    g_sum_name: str = "g_sum=sum_i,x(g_i)"
    f_distribution_inventory_name: str = "f_distribution_inventory=sum_i,x(f_i)"
    g_distribution_inventory_name: str = "g_distribution_inventory=sum_i,x(g_i)"
    distribution_inventory_name: str = "distribution_inventory=sum_i,x(f_i)+sum_i,x(g_i)"


@dataclass(frozen=True)
class FreeEnergyAdapterStreamLoopResult:
    """Final coupled state and explicit nonphysical adapter metadata."""

    state: FreeEnergyCollisionOnlyState
    step_states: tuple[FreeEnergyCollisionOnlyState, ...]
    diagnostics: tuple[FreeEnergyAdapterStreamLoopDiagnostic, ...]
    boundary: BoundaryPolicy
    stage: str = ADAPTER_STREAM_LOOP_STAGE
    physical: bool = False
    phase_flux_status: str = PHASE_FLUX_WITHHELD
    phase_flux: None = None


def _diagnostic(
    step: int, state: FreeEnergyCollisionOnlyState
) -> FreeEnergyAdapterStreamLoopDiagnostic:
    f_inventory = float(state.f.sum().item())
    g_inventory = float(state.g.sum().item())
    return FreeEnergyAdapterStreamLoopDiagnostic(
        step=step,
        phi_integral=g_inventory,
        f_mass=f_inventory,
        g_sum=g_inventory,
        f_distribution_inventory=f_inventory,
        g_distribution_inventory=g_inventory,
        distribution_inventory=f_inventory + g_inventory,
    )


def collision_then_adapter_stream(
    state: FreeEnergyCollisionOnlyState,
    config: FreeEnergyAdapterStreamLoopConfig | None = None,
) -> FreeEnergyAdapterStreamLoopResult:
    """Run 2+ real collision → adapter-stream state handoffs.

    The collision callable is the lazy wrapper around the production
    ``multiphase3d.free_energy_step_3d``; no collision is recreated here.  The
    post-collision populations are passed directly to the adapter stream, and
    its output is the only state supplied to the next collision.
    """
    if not isinstance(state, FreeEnergyCollisionOnlyState):
        raise TypeError("state must be a FreeEnergyCollisionOnlyState")
    if config is None:
        config = FreeEnergyAdapterStreamLoopConfig()
    if not isinstance(config, FreeEnergyAdapterStreamLoopConfig):
        raise TypeError("config must be a FreeEnergyAdapterStreamLoopConfig")

    current = state
    step_states: list[FreeEnergyCollisionOnlyState] = []
    diagnostics = [_diagnostic(0, current)]
    for step in range(1, config.steps + 1):
        post_collision_f, post_collision_g = free_energy_step_3d(
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
        streamed = stream_free_energy_adapter(
            post_collision_f, post_collision_g, boundary=config.boundary
        )
        current = FreeEnergyCollisionOnlyState(f=streamed.f, g=streamed.g)
        step_states.append(current)
        diagnostics.append(_diagnostic(step, current))

    return FreeEnergyAdapterStreamLoopResult(
        state=current,
        step_states=tuple(step_states),
        diagnostics=tuple(diagnostics),
        boundary=config.boundary,
    )


def run_free_energy_adapter_stream_loop(
    state: FreeEnergyCollisionOnlyState,
    config: FreeEnergyAdapterStreamLoopConfig | None = None,
) -> FreeEnergyAdapterStreamLoopResult:
    """Named R1 entry point for :func:`collision_then_adapter_stream`."""
    return collision_then_adapter_stream(state, config)


__all__ = [
    "ADAPTER_STREAM_LOOP_STAGE",
    "FreeEnergyAdapterStreamLoopConfig",
    "FreeEnergyAdapterStreamLoopDiagnostic",
    "FreeEnergyAdapterStreamLoopResult",
    "collision_then_adapter_stream",
    "run_free_energy_adapter_stream_loop",
]
