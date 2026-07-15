"""Reusable R1 fully-wetted stationary-voxel flow application.

This module is deliberately a narrow application facade over the existing
D3Q19 MRT kernels.  Setup owns metadata validation and plan binding; the step
loop owns only prebound numerical work and same-phase observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, cast

import torch

from .backends.contracts import DeviceSpec
from .backends.torch_backend import TorchBackend
from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .models.contracts import ModelComposition
from .obstacles import compute_obstacle_forces_3d, compute_obstacle_moments_3d


_R1_DEVICE = DeviceSpec(device="cpu", dtype_name="float32")
_UNSUPPORTED = (
    "free_surface",
    "moving_geometry",
    "physical_control_volume_closure",
    "arbitrary_geometry_physical_accuracy_claim",
)


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


@dataclass(frozen=True, slots=True)
class VoxelBodyGeometry:
    """Static, 3-D boolean voxel body geometry with an x/y/z moment origin."""

    mask: torch.Tensor
    body_id: str
    reference_area: float | None = None
    reference_length: float | None = None
    origin: tuple[float, float, float] | None = None
    resolved_origin: tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mask, torch.Tensor) or self.mask.ndim != 3 or self.mask.dtype is not torch.bool:
            raise ValueError("mask must be a 3D bool torch.Tensor")
        if not bool(self.mask.any().item()):
            raise ValueError("mask must contain at least one solid voxel")
        if not isinstance(self.body_id, str) or not self.body_id:
            raise ValueError("body_id must be a non-empty string")
        # A frozen dataclass does not freeze Tensor storage. Snapshot static
        # geometry at setup so caller-side in-place mutation cannot alter it.
        object.__setattr__(self, "mask", self.mask.clone())
        if self.reference_area is not None:
            object.__setattr__(self, "reference_area", _finite_positive(self.reference_area, "reference_area"))
        if self.reference_length is not None:
            object.__setattr__(self, "reference_length", _finite_positive(self.reference_length, "reference_length"))
        if self.origin is None:
            zyx = self.mask.nonzero(as_tuple=False).to(dtype=torch.float64).mean(dim=0)
            resolved = (float(zyx[2].item()), float(zyx[1].item()), float(zyx[0].item()))
        else:
            if not isinstance(self.origin, tuple) or len(self.origin) != 3:
                raise ValueError("origin must be an (x, y, z) tuple of length 3")
            resolved = tuple(_finite_scalar(value, "origin") for value in self.origin)
        object.__setattr__(self, "resolved_origin", resolved)


@dataclass(frozen=True, slots=True)
class FullyWettedFlowConfig:
    """R1 setup contract: CPU float32, D3Q19 MRT, incompressible single phase."""

    geometry: VoxelBodyGeometry
    composition: ModelComposition
    device_spec: DeviceSpec
    shape: tuple[int, int, int]
    tau: float
    inlet_velocity: float
    steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, VoxelBodyGeometry):
            raise ValueError("geometry must be a VoxelBodyGeometry")
        if not isinstance(self.composition, ModelComposition):
            raise ValueError("composition must be a ModelComposition")
        if self.device_spec != _R1_DEVICE:
            raise ValueError("Fully wetted flow R1 supports only TorchBackend CPU float32")
        if self.shape != tuple(self.geometry.mask.shape):
            raise ValueError("shape must equal geometry.mask.shape")
        if (not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool)
                or not isfinite(self.tau) or self.tau <= 0.5):
            raise ValueError("tau must be finite and > 0.5")
        if (not isinstance(self.inlet_velocity, (int, float)) or isinstance(self.inlet_velocity, bool)
                or not isfinite(self.inlet_velocity) or not 0.0 < self.inlet_velocity < 0.15):
            raise ValueError("inlet_velocity must be finite and in (0, 0.15)")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be an integer >= 1")
        if self.composition.lattice != "D3Q19":
            raise ValueError("Fully wetted flow R1 requires composition.lattice='D3Q19'")
        if self.composition.collision != "MRT":
            raise ValueError("Fully wetted flow R1 requires composition.collision='MRT'")
        if self.composition.physics_modules != {"single_phase": "incompressible"}:
            raise ValueError("Fully wetted flow R1 permits no additional physics modules")
        if self.composition.turbulence is not None:
            raise ValueError("Fully wetted flow R1 requires turbulence=None")
        if self.composition.forcing:
            raise ValueError("Fully wetted flow R1 requires no forcing modules")
        if self.composition.boundaries != ("zou_he_channel", "stationary_bounce_back"):
            raise ValueError("Fully wetted flow R1 requires its fixed channel boundary contract")


@dataclass(frozen=True, slots=True)
class FullyWettedFlowResult:
    """Final R1 fields and same-phase diagnostics, without accuracy claims."""

    density: torch.Tensor
    velocity: torch.Tensor
    force: tuple[float, float, float]
    reaction: tuple[float, float, float]
    moment: tuple[float, float, float]
    status: str
    evidence: Mapping[str, object]


def run_fully_wetted_flow(config: FullyWettedFlowConfig) -> FullyWettedFlowResult:
    """Run a static voxel body through the shared CPU D3Q19 MRT channel path."""
    geometry = config.geometry
    device = torch.device(config.device_spec.device)
    mask = geometry.mask.to(device=device)
    nz, ny, nx = config.shape
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=device)
    plan = TorchBackend().compile_d3q19_mrt(config.composition, float(config.tau), config.device_spec)
    u_in = float(config.inlet_velocity)
    steps = config.steps
    origin_x, origin_y, origin_z = geometry.resolved_origin
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    zero = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, zero, zero, device=device)
    force = (0.0, 0.0, 0.0)
    moment = (0.0, 0.0, 0.0)
    density, ux, uy, uz = macroscopic3d(f)
    status = "COMPLETED"

    for _ in range(steps):
        f = plan.step(f)
        force_tensors = compute_obstacle_forces_3d(f, mask)
        moment_tensors = compute_obstacle_moments_3d(f, mask, origin_x, origin_y, origin_z)
        f = apply_zou_he_channel_boundaries_3d(f, u_in, wall_mask, mask)
        density, ux, uy, uz = macroscopic3d(f)
        finite = bool(torch.isfinite(f).all().item() and torch.isfinite(density).all().item()
                      and torch.isfinite(ux).all().item() and torch.isfinite(uy).all().item()
                      and torch.isfinite(uz).all().item())
        force = cast(tuple[float, float, float], tuple(float(component.item()) for component in force_tensors))
        moment = cast(tuple[float, float, float], tuple(float(component.item()) for component in moment_tensors))
        if not finite:
            status = "FAILED_NONFINITE"
            break

    velocity = torch.stack((ux, uy, uz))
    evidence = {
        "model_identity": {
            "backend": "torch",
            "device": "cpu",
            "dtype": "float32",
            "lattice": "D3Q19",
            "collision": "MRT",
            "single_phase": "incompressible",
            "channel_boundary": "fixed_zou_he_inlet_pressure_outlet_with_stationary_bounce_back",
        },
        "force": {
            "kind": "same_phase_momentum_exchange_diagnostic",
            "phase": "post_stream_pre_bounce_back",
            "implementation": "compute_obstacle_forces_3d",
        },
        "moment": {
            "kind": "same_phase_momentum_exchange_diagnostic",
            "phase": "post_stream_pre_bounce_back",
            "implementation": "compute_obstacle_moments_3d",
            "origin_xyz": geometry.resolved_origin,
        },
        "physical_control_volume": {
            "status": "not_definable",
            "reason": "retained cell-reset boundary operators expose no link-owned control-volume closure",
        },
        "unsupported": _UNSUPPORTED,
    }
    reaction = cast(tuple[float, float, float], tuple(-component for component in force))
    return FullyWettedFlowResult(
        density=density.clone(),
        velocity=velocity.clone(),
        force=force,
        reaction=reaction,
        moment=moment,
        status=status,
        evidence=evidence,
    )


__all__ = [
    "FullyWettedFlowConfig",
    "FullyWettedFlowResult",
    "VoxelBodyGeometry",
    "run_fully_wetted_flow",
]
