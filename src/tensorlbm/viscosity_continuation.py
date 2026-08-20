"""Smooth resolved-viscosity continuation for stable LBM startup."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .static_block_amr import convective_refined_tau


@dataclass(frozen=True)
class ResolvedReynoldsContinuation:
    """Ramp inverse Reynolds number with zero endpoint derivatives."""

    start_reynolds: float
    target_reynolds: float
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                self.start_reynolds,
                self.target_reynolds,
            )
        ):
            raise ValueError("resolved Reynolds numbers must be finite and positive")
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("continuation steps must satisfy 0 <= start <= end")
        if self.start_reynolds != self.target_reynolds and self.end_step == self.start_step:
            raise ValueError("a non-constant continuation needs a positive ramp duration")

    def reynolds_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative")
        if self.start_reynolds == self.target_reynolds:
            return self.target_reynolds
        if step <= self.start_step:
            return self.start_reynolds
        if step >= self.end_step:
            return self.target_reynolds
        phase = (step - self.start_step) / (self.end_step - self.start_step)
        activation = 0.5 * (1.0 - math.cos(math.pi * phase))
        inverse_reynolds = (
            1.0 - activation
        ) / self.start_reynolds + activation / self.target_reynolds
        return 1.0 / inverse_reynolds

    def tau_by_level(
        self,
        step: int,
        *,
        lattice_speed: float,
        root_hull_length: float,
        levels: int,
    ) -> tuple[float, ...]:
        if lattice_speed <= 0.0 or root_hull_length <= 0.0 or levels < 1:
            raise ValueError("lattice speed, hull length and levels must be positive")
        reynolds = self.reynolds_at(step)
        root_tau = 0.5 + 3.0 * lattice_speed * root_hull_length / reynolds
        result = [root_tau]
        for _ in range(1, levels):
            result.append(convective_refined_tau(result[-1]))
        return tuple(result)


__all__ = ["ResolvedReynoldsContinuation"]
