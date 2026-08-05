"""Pure, fail-closed paired common-flux accounting transaction.

This module intentionally does not import, call, or mutate a free-surface
solver, populations, phase fields, or PV state.  It only proves that explicitly
provided interface and counterpart ledger deltas have a common net flux.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


class FluxTransactionError(ValueError):
    """The requested pure ledger transaction cannot safely proceed."""


def _require_finite_floating_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise FluxTransactionError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise FluxTransactionError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise FluxTransactionError(f"{name} must contain only finite values")
    return value


def _require_tensor(name: str, value: object, shape: torch.Size) -> torch.Tensor:
    tensor = _require_finite_floating_tensor(name, value)
    if tensor.shape != shape:
        raise FluxTransactionError(f"{name} shape must match both inventories")
    return tensor


def _require_tolerances(atol: float, rtol: float) -> None:
    for name, value in (("atol", atol), ("rtol", rtol)):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise FluxTransactionError(f"{name} tolerance must be finite and non-negative") from error
        if numeric < 0.0 or numeric == float("inf") or numeric != numeric:
            raise FluxTransactionError(f"{name} tolerance must be finite and non-negative")


@dataclass(frozen=True)
class CommonFluxTransaction:
    """Factory for an explicit two-ledger common-flux transaction."""

    active: bool = True
    atol: float = 1.0e-8
    rtol: float = 1.0e-6

    def __post_init__(self) -> None:
        _require_tolerances(self.atol, self.rtol)

    def plan(
        self,
        interface: torch.Tensor,
        counterpart: torch.Tensor,
        interface_delta: torch.Tensor,
        counterpart_delta: torch.Tensor,
    ) -> "PlannedCommonFlux":
        if not self.active:
            raise FluxTransactionError("common-flux transaction must be active")
        interface = _require_finite_floating_tensor("interface", interface)
        _require_tensor("counterpart", counterpart, interface.shape)
        _require_tensor("interface_delta", interface_delta, interface.shape)
        _require_tensor("counterpart_delta", counterpart_delta, interface.shape)
        if interface.device != counterpart.device or interface.device != interface_delta.device or interface.device != counterpart_delta.device:
            raise FluxTransactionError("inventories and deltas must share one device")
        return PlannedCommonFlux(
            interface=interface.clone(),
            counterpart=counterpart.clone(),
            interface_delta=interface_delta.clone(),
            counterpart_delta=counterpart_delta.clone(),
            atol=float(self.atol),
            rtol=float(self.rtol),
        )


@dataclass(frozen=True)
class PlannedCommonFlux:
    """Validated input snapshot; staging has not altered any input buffer."""

    interface: torch.Tensor
    counterpart: torch.Tensor
    interface_delta: torch.Tensor
    counterpart_delta: torch.Tensor
    atol: float
    rtol: float

    def stage(self) -> "StagedCommonFlux":
        """Produce independent candidate inventories without input mutation."""
        return StagedCommonFlux(
            interface_before=self.interface.clone(),
            counterpart_before=self.counterpart.clone(),
            interface_candidate=self.interface + self.interface_delta,
            counterpart_candidate=self.counterpart + self.counterpart_delta,
            interface_flux=self.interface_delta.sum(),
            counterpart_flux=self.counterpart_delta.sum(),
            atol=self.atol,
            rtol=self.rtol,
        )


@dataclass(frozen=True)
class StagedCommonFlux:
    """Independent candidate buffers awaiting an explicit validation gate."""

    interface_before: torch.Tensor
    counterpart_before: torch.Tensor
    interface_candidate: torch.Tensor
    counterpart_candidate: torch.Tensor
    interface_flux: torch.Tensor
    counterpart_flux: torch.Tensor
    atol: float
    rtol: float

    def validate(self) -> "CommonFluxValidation":
        if not bool(torch.isfinite(self.interface_candidate).all()) or not bool(torch.isfinite(self.counterpart_candidate).all()):
            raise FluxTransactionError("staged inventories must remain finite")
        residual = torch.abs(self.interface_flux + self.counterpart_flux)
        scale = torch.maximum(torch.abs(self.interface_flux), torch.abs(self.counterpart_flux))
        tolerance = torch.as_tensor(self.atol, dtype=residual.dtype, device=residual.device) + self.rtol * scale
        return CommonFluxValidation(
            interface_candidate=self.interface_candidate.clone(),
            counterpart_candidate=self.counterpart_candidate.clone(),
            residual=float(residual),
            tolerance=float(tolerance),
            valid=bool(residual <= tolerance),
        )


@dataclass(frozen=True)
class CommonFluxValidation:
    """Validation result. ``commit`` is unavailable unless the pair balances."""

    interface_candidate: torch.Tensor
    counterpart_candidate: torch.Tensor
    residual: float
    tolerance: float
    valid: bool

    def commit(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return independent candidate buffers; never write caller-owned inputs."""
        if not self.valid:
            raise FluxTransactionError(
                f"common-flux residual {self.residual!r} exceeds tolerance {self.tolerance!r}"
            )
        return self.interface_candidate.clone(), self.counterpart_candidate.clone()
