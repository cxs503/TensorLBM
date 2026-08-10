"""Checkpointable temporal filter for wall pressure-gradient vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WallPressureGradientFilterDiagnostics:
    requested_nodes: int
    valid_nodes: int
    initialized_nodes: int
    newly_initialized_nodes: int
    cleared_nodes: int
    delta_steps: float
    relaxation_fraction: float


class WallPressureGradientFilter:
    """Filter persistent geometry-indexed gradient vectors in physical steps.

    The exact exponential update

    ``g_bar <- exp(-dt/T) g_bar + (1-exp(-dt/T)) g``

    makes the decay invariant to diagnostic cadence for a piecewise-constant
    input.  A newly valid node initializes from its first observation.  An
    invalid node is cleared immediately so stale pressure evidence can never
    keep a non-equilibrium wall selector active.
    """

    def __init__(self, *, time_constant_steps: float) -> None:
        if not math.isfinite(time_constant_steps) or time_constant_steps <= 0.0:
            raise ValueError("time_constant_steps must be finite and positive")
        self.time_constant_steps = float(time_constant_steps)
        self._mean: torch.Tensor | None = None
        self._initialized: torch.Tensor | None = None

    @property
    def mean(self) -> torch.Tensor | None:
        return None if self._mean is None else self._mean.clone()

    @property
    def initialized(self) -> torch.Tensor | None:
        return None if self._initialized is None else self._initialized.clone()

    def reset(self) -> None:
        self._mean = None
        self._initialized = None

    def update(
        self,
        gradient_vector: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        delta_steps: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, WallPressureGradientFilterDiagnostics]:
        """Update filtered vectors and return values, validity and diagnostics."""
        if not gradient_vector.is_floating_point() or gradient_vector.ndim < 1:
            raise ValueError("gradient_vector must be a floating tensor")
        if gradient_vector.shape[-1] != 3:
            raise ValueError("gradient_vector final dimension must contain xyz")
        sample_shape = gradient_vector.shape[:-1]
        if valid is None:
            valid = torch.ones(
                sample_shape,
                dtype=torch.bool,
                device=gradient_vector.device,
            )
        if valid.shape != sample_shape or valid.dtype is not torch.bool:
            raise ValueError("valid must be boolean with the sample shape")
        if valid.device != gradient_vector.device:
            raise ValueError("gradient_vector and valid must share a device")
        if not math.isfinite(delta_steps) or delta_steps <= 0.0:
            raise ValueError("delta_steps must be finite and positive")
        finite = torch.isfinite(gradient_vector).all(dim=-1)
        admissible = valid & finite
        if self._mean is None:
            self._mean = torch.zeros_like(gradient_vector)
            self._initialized = torch.zeros(
                sample_shape,
                dtype=torch.bool,
                device=gradient_vector.device,
            )
        elif (
            self._mean.shape != gradient_vector.shape
            or self._mean.device != gradient_vector.device
            or self._mean.dtype != gradient_vector.dtype
        ):
            raise ValueError("filter state shape, device or dtype changed; call reset first")
        assert self._initialized is not None
        previous_initialized = self._initialized.clone()
        cleared = self._initialized & ~admissible
        self._initialized &= admissible
        self._mean[~admissible] = 0.0

        newly_initialized = admissible & ~self._initialized
        self._mean[newly_initialized] = gradient_vector[newly_initialized]
        self._initialized |= newly_initialized
        continuing = admissible & previous_initialized
        relaxation = 1.0 - math.exp(-float(delta_steps) / self.time_constant_steps)
        if bool(continuing.any()):
            self._mean[continuing] += relaxation * (
                gradient_vector[continuing] - self._mean[continuing]
            )
        output = torch.full_like(gradient_vector, torch.nan)
        output[self._initialized] = self._mean[self._initialized]
        diagnostics = WallPressureGradientFilterDiagnostics(
            requested_nodes=valid.numel(),
            valid_nodes=int(admissible.sum().item()),
            initialized_nodes=int(self._initialized.sum().item()),
            newly_initialized_nodes=int(newly_initialized.sum().item()),
            cleared_nodes=int(cleared.sum().item()),
            delta_steps=float(delta_steps),
            relaxation_fraction=relaxation,
        )
        return output, self._initialized.clone(), diagnostics

    def state_dict(self) -> dict[str, object]:
        initialized = self._mean is not None
        return {
            "schema": "tensorlbm-wall-pressure-gradient-filter-v1",
            "configuration": {
                "time_constant_steps": self.time_constant_steps,
            },
            "state_initialized": initialized,
            "mean": (
                self._mean.detach().to(device="cpu").clone()
                if initialized else None
            ),
            "initialized": (
                self._initialized.detach().to(device="cpu").clone()
                if initialized else None
            ),
        }

    def load_state_dict(
        self,
        state: dict[str, object],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if state.get("schema") != "tensorlbm-wall-pressure-gradient-filter-v1":
            raise ValueError("unsupported wall pressure-gradient filter state")
        if state.get("configuration") != {
            "time_constant_steps": self.time_constant_steps,
        }:
            raise ValueError("pressure-gradient filter configuration mismatch")
        state_initialized = state.get("state_initialized")
        if not isinstance(state_initialized, bool):
            raise ValueError("filter checkpoint lacks state_initialized flag")
        if not state_initialized:
            if state.get("mean") is not None or state.get("initialized") is not None:
                raise ValueError("uninitialised filter checkpoint contains state")
            self.reset()
            return
        mean = state.get("mean")
        initialized = state.get("initialized")
        if not isinstance(mean, torch.Tensor) or not isinstance(
            initialized, torch.Tensor,
        ):
            raise ValueError("initialised filter checkpoint lacks tensors")
        if not mean.is_floating_point() or mean.ndim < 1 or mean.shape[-1] != 3:
            raise ValueError("filter mean checkpoint must contain xyz vectors")
        if initialized.dtype is not torch.bool or initialized.shape != mean.shape[:-1]:
            raise ValueError("filter initialized checkpoint has invalid shape or dtype")
        if not torch.isfinite(mean[initialized]).all():
            raise ValueError("filter checkpoint contains non-finite active means")
        target = torch.device(device) if device is not None else mean.device
        self._mean = mean.detach().to(device=target).clone()
        self._initialized = initialized.detach().to(device=target).clone()


__all__ = [
    "WallPressureGradientFilter",
    "WallPressureGradientFilterDiagnostics",
]
