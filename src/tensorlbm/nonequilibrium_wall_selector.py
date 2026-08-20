"""Fail-closed temporal selector for non-equilibrium wall treatment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NonequilibriumWallSelectionDiagnostics:
    requested_nodes: int
    valid_nodes: int
    eligible_nodes: int
    active_nodes: int
    newly_activated_nodes: int
    newly_deactivated_nodes: int
    low_shear_rejected_nodes: int
    y_plus_rejected_nodes: int


class NonequilibriumWallSelector:
    """Select persistent adverse-gradient nodes without altering wall stress.

    The signed pressure-gradient parameter is positive for an adverse gradient
    along the local tangential flow.  Entry and exit thresholds form a
    Schmitt-trigger hysteresis, while consecutive observations reject isolated
    turbulent spikes.  Invalid, low-shear and out-of-policy ``y+`` samples are
    deactivated immediately because the dimensionless pressure-gradient ratio
    is unreliable there.

    This object only owns selection state.  It deliberately contains no wall
    stress correction law, so a new closure cannot enter a production force
    path merely by enabling the selector.
    """

    def __init__(
        self,
        *,
        enter_threshold: float = 1.0,
        exit_threshold: float = 0.5,
        enter_observations: int = 3,
        exit_observations: int = 3,
        minimum_u_tau: float = 1.0e-8,
        y_plus_lower_bound: float = 30.0,
        y_plus_upper_bound: float = 1000.0,
    ) -> None:
        if not 0.0 <= exit_threshold < enter_threshold:
            raise ValueError("require 0 <= exit threshold < enter threshold")
        if min(enter_observations, exit_observations) < 1:
            raise ValueError("observation counts must be positive")
        if minimum_u_tau <= 0.0:
            raise ValueError("minimum u_tau must be positive")
        if not 0.0 <= y_plus_lower_bound < y_plus_upper_bound:
            raise ValueError("invalid y+ interval")
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.enter_observations = int(enter_observations)
        self.exit_observations = int(exit_observations)
        self.minimum_u_tau = float(minimum_u_tau)
        self.y_plus_lower_bound = float(y_plus_lower_bound)
        self.y_plus_upper_bound = float(y_plus_upper_bound)
        self._active: torch.Tensor | None = None
        self._enter_count: torch.Tensor | None = None
        self._exit_count: torch.Tensor | None = None

    @property
    def active(self) -> torch.Tensor | None:
        """Return a copy of the current selection mask, if initialized."""
        return None if self._active is None else self._active.clone()

    def reset(self) -> None:
        self._active = None
        self._enter_count = None
        self._exit_count = None

    def state_dict(self) -> dict[str, object]:
        """Return a CPU checkpoint without exposing mutable selector state.

        The configuration is part of the state contract: restoring temporal
        hysteresis with different thresholds or observation counts would
        silently change which wall nodes are selected after a restart.
        Uninitialised selectors are serialised explicitly so a checkpoint
        taken before the first diagnostic observation remains reproducible.
        """
        initialized = self._active is not None
        return {
            "schema": "tensorlbm-nonequilibrium-wall-selector-v1",
            "configuration": {
                "enter_threshold": self.enter_threshold,
                "exit_threshold": self.exit_threshold,
                "enter_observations": self.enter_observations,
                "exit_observations": self.exit_observations,
                "minimum_u_tau": self.minimum_u_tau,
                "y_plus_lower_bound": self.y_plus_lower_bound,
                "y_plus_upper_bound": self.y_plus_upper_bound,
            },
            "initialized": initialized,
            "active": (self._active.detach().to(device="cpu").clone() if initialized else None),
            "enter_count": (
                self._enter_count.detach().to(device="cpu").clone() if initialized else None
            ),
            "exit_count": (
                self._exit_count.detach().to(device="cpu").clone() if initialized else None
            ),
        }

    def load_state_dict(
        self,
        state: dict[str, object],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        """Restore an exact hysteresis state, failing closed on mismatch."""
        if state.get("schema") != "tensorlbm-nonequilibrium-wall-selector-v1":
            raise ValueError("unsupported non-equilibrium wall selector state")
        expected_configuration = {
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "enter_observations": self.enter_observations,
            "exit_observations": self.exit_observations,
            "minimum_u_tau": self.minimum_u_tau,
            "y_plus_lower_bound": self.y_plus_lower_bound,
            "y_plus_upper_bound": self.y_plus_upper_bound,
        }
        if state.get("configuration") != expected_configuration:
            raise ValueError("selector checkpoint configuration mismatch")
        initialized = state.get("initialized")
        if not isinstance(initialized, bool):
            raise ValueError("selector checkpoint lacks initialized flag")
        if not initialized:
            if any(
                state.get(key) is not None
                for key in (
                    "active",
                    "enter_count",
                    "exit_count",
                )
            ):
                raise ValueError("uninitialised selector checkpoint contains state")
            self.reset()
            return
        active = state.get("active")
        enter_count = state.get("enter_count")
        exit_count = state.get("exit_count")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                active,
                enter_count,
                exit_count,
            )
        ):
            raise ValueError("initialised selector checkpoint lacks tensors")
        assert isinstance(active, torch.Tensor)
        assert isinstance(enter_count, torch.Tensor)
        assert isinstance(exit_count, torch.Tensor)
        if active.dtype is not torch.bool:
            raise ValueError("selector active state must be boolean")
        if enter_count.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError("selector entry count must be integral")
        if exit_count.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError("selector exit count must be integral")
        if not (active.shape == enter_count.shape == exit_count.shape):
            raise ValueError("selector checkpoint tensors must share a shape")
        if bool((enter_count < 0).any()) or bool((exit_count < 0).any()):
            raise ValueError("selector checkpoint counters must be non-negative")
        if bool((enter_count > self.enter_observations).any()):
            raise ValueError("selector entry counter exceeds configured limit")
        if bool((exit_count > self.exit_observations).any()):
            raise ValueError("selector exit counter exceeds configured limit")
        target = torch.device(device) if device is not None else active.device
        self._active = active.detach().to(device=target).clone()
        self._enter_count = (
            enter_count.detach()
            .to(
                device=target,
                dtype=torch.int32,
            )
            .clone()
        )
        self._exit_count = (
            exit_count.detach()
            .to(
                device=target,
                dtype=torch.int32,
            )
            .clone()
        )

    def update(
        self,
        signed_pressure_gradient_parameter: torch.Tensor,
        u_tau: torch.Tensor,
        y_plus: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, NonequilibriumWallSelectionDiagnostics]:
        """Update and return the selected non-equilibrium wall-node mask."""
        if not signed_pressure_gradient_parameter.is_floating_point():
            raise ValueError("pressure-gradient parameter must be floating point")
        if not (signed_pressure_gradient_parameter.shape == u_tau.shape == y_plus.shape):
            raise ValueError("pressure gradient, u_tau and y+ must share a shape")
        if valid is None:
            valid = torch.ones_like(
                signed_pressure_gradient_parameter,
                dtype=torch.bool,
            )
        if valid.shape != signed_pressure_gradient_parameter.shape:
            raise ValueError("valid mask must share the sample shape")
        if valid.dtype is not torch.bool:
            raise ValueError("valid mask must be boolean")
        devices = {
            signed_pressure_gradient_parameter.device,
            u_tau.device,
            y_plus.device,
            valid.device,
        }
        if len(devices) != 1:
            raise ValueError("all selector inputs must share a device")

        shape = signed_pressure_gradient_parameter.shape
        device = signed_pressure_gradient_parameter.device
        if self._active is None:
            self._active = torch.zeros(shape, dtype=torch.bool, device=device)
            # Int32 avoids observation-counter overflow in long campaigns and
            # has identical cost significance next to the population fields.
            self._enter_count = torch.zeros(shape, dtype=torch.int32, device=device)
            self._exit_count = torch.zeros(shape, dtype=torch.int32, device=device)
        elif self._active.shape != shape or self._active.device != device:
            raise ValueError("selector state shape/device changed; call reset first")
        assert self._enter_count is not None
        assert self._exit_count is not None

        finite = (
            torch.isfinite(signed_pressure_gradient_parameter)
            & torch.isfinite(u_tau)
            & torch.isfinite(y_plus)
        )
        low_shear = u_tau < self.minimum_u_tau
        y_plus_rejected = (y_plus < self.y_plus_lower_bound) | (y_plus > self.y_plus_upper_bound)
        admissible = valid & finite & ~low_shear & ~y_plus_rejected
        previous = self._active.clone()

        # Invalid wall-model evidence fails closed immediately.
        self._active &= admissible
        enter = (
            admissible
            & ~self._active
            & (signed_pressure_gradient_parameter >= self.enter_threshold)
        )
        self._enter_count = torch.where(
            enter,
            torch.clamp(self._enter_count + 1, max=self.enter_observations),
            torch.zeros_like(self._enter_count),
        )
        self._active |= self._enter_count >= self.enter_observations

        exit_candidate = self._active & (signed_pressure_gradient_parameter < self.exit_threshold)
        self._exit_count = torch.where(
            exit_candidate,
            torch.clamp(self._exit_count + 1, max=self.exit_observations),
            torch.zeros_like(self._exit_count),
        )
        self._active &= self._exit_count < self.exit_observations
        self._enter_count = torch.where(
            self._active,
            torch.zeros_like(self._enter_count),
            self._enter_count,
        )

        diagnostics = NonequilibriumWallSelectionDiagnostics(
            requested_nodes=int(valid.numel()),
            valid_nodes=int((valid & finite).sum().item()),
            eligible_nodes=int(enter.sum().item()),
            active_nodes=int(self._active.sum().item()),
            newly_activated_nodes=int((self._active & ~previous).sum().item()),
            newly_deactivated_nodes=int((previous & ~self._active).sum().item()),
            low_shear_rejected_nodes=int((valid & finite & low_shear).sum().item()),
            y_plus_rejected_nodes=int(
                (valid & finite & y_plus_rejected).sum().item(),
            ),
        )
        return self._active.clone(), diagnostics


__all__ = [
    "NonequilibriumWallSelectionDiagnostics",
    "NonequilibriumWallSelector",
]
