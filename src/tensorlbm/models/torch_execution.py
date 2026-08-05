"""PyTorch-only prebound D3Q19 MRT execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from time import perf_counter
from typing import Callable

import torch

from ..solver3d import collide_mrt3d, stream3d
from .contracts import ModelComposition


@dataclass(frozen=True, slots=True)
class TorchD3Q19MRTPlan:
    """Hot-path plan containing only prebound PyTorch kernels and MRT tau."""

    collision_kernel: Callable[[torch.Tensor, float], torch.Tensor]
    stream_kernel: Callable[[torch.Tensor], torch.Tensor]
    tau: float

    def step(self, f: torch.Tensor) -> torch.Tensor:
        return self.stream_kernel(self.collision_kernel(f, self.tau))


@dataclass(frozen=True, slots=True)
class PlanPerformanceSample:
    """CPU/GPU-neutral timing observation; callers own any performance verdict."""

    direct_median_seconds: float
    plan_median_seconds: float
    ratio: float


def _valid_tau(tau: object) -> float:
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not isfinite(tau) or tau <= 0.5:
        raise ValueError("tau must be a finite scalar > 0.5")
    return float(tau)


def _positive_count(value: object, name: str, *, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def compile_torch_d3q19_mrt_plan(composition: ModelComposition, tau: float) -> TorchD3Q19MRTPlan:
    """Compile the allowed cold-path metadata into a PyTorch-only hot-path plan."""
    if composition.lattice != "D3Q19":
        raise ValueError("Torch D3Q19 MRT plan requires lattice='D3Q19'")
    if composition.collision != "MRT":
        raise ValueError("Torch D3Q19 MRT plan requires collision='MRT'")
    if composition.physics_modules.get("single_phase") != "incompressible":
        raise ValueError("Torch D3Q19 MRT plan requires single_phase='incompressible'")
    return TorchD3Q19MRTPlan(collide_mrt3d, stream3d, _valid_tau(tau))


def measure_plan_overhead(
    plan: TorchD3Q19MRTPlan,
    initial_state: torch.Tensor,
    direct_step: Callable[[torch.Tensor], torch.Tensor],
    *,
    warmup_steps: int = 3,
    measured_steps: int = 10,
    clock: Callable[[], float] = perf_counter,
) -> PlanPerformanceSample:
    """Measure direct and plan steps; the reported ratio deliberately has no pass/fail policy."""
    warmup_steps = _positive_count(warmup_steps, "warmup_steps", allow_zero=True)
    measured_steps = _positive_count(measured_steps, "measured_steps", allow_zero=False)
    direct_state = initial_state
    plan_state = initial_state
    for _ in range(warmup_steps):
        direct_state = direct_step(direct_state)
        plan_state = plan.step(plan_state)
    direct_seconds: list[float] = []
    plan_seconds: list[float] = []
    for _ in range(measured_steps):
        start = clock()
        direct_state = direct_step(direct_state)
        direct_seconds.append(clock() - start)
        start = clock()
        plan_state = plan.step(plan_state)
        plan_seconds.append(clock() - start)
    direct_median_seconds = float(median(direct_seconds))
    plan_median_seconds = float(median(plan_seconds))
    return PlanPerformanceSample(
        direct_median_seconds=direct_median_seconds,
        plan_median_seconds=plan_median_seconds,
        ratio=plan_median_seconds / direct_median_seconds,
    )
