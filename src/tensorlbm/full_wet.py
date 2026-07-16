"""Reusable R1 fully-wetted stationary-voxel flow application.

This module is deliberately a narrow application facade over the existing
D3Q19 MRT kernels.  Setup owns metadata validation and plan binding; the step
loop owns only prebound numerical work and same-phase observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
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
_POPULATION_SAMPLE_PHASE = "post_stream_pre_bounce_back"


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def _geometry_ownership_hash(geometry: "VoxelBodyGeometry") -> str:
    """Hash the immutable geometry snapshot which owns an exported state."""
    mask = geometry.mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = sha256()
    digest.update(geometry.body_id.encode("utf-8"))
    digest.update(repr(tuple(mask.shape)).encode("ascii"))
    digest.update(mask.numpy().tobytes())
    return digest.hexdigest()


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
    capture_population_steps: tuple[int, ...] = ()

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
        if not isinstance(self.capture_population_steps, tuple):
            raise ValueError("capture_population_steps must be a tuple of unique ascending step indices")
        if any(not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= self.steps
               for step in self.capture_population_steps):
            raise ValueError("capture_population_steps must contain step indices in [1, steps]")
        if tuple(sorted(self.capture_population_steps)) != self.capture_population_steps or (
                len(set(self.capture_population_steps)) != len(self.capture_population_steps)):
            raise ValueError("capture_population_steps must be unique and ascending")
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


@dataclass(frozen=True, slots=True, init=False)
class D3Q19PopulationSnapshot:
    """Immutable view of detached production ``f`` at post-stream/pre-bounce-back.

    The private payload is cloned at capture and every public ``f`` access
    returns another detached clone.  Neither a later solver update nor a
    consumer's in-place edit can mutate the result's auditable snapshot.
    """

    step_index: int
    sample_phase: str
    _f: torch.Tensor = field(repr=False)
    ownership_hash: str

    def __init__(self, step_index: int, sample_phase: str, f: torch.Tensor, ownership_hash: str) -> None:
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "sample_phase", sample_phase)
        object.__setattr__(self, "_f", f.detach().clone())
        object.__setattr__(self, "ownership_hash", ownership_hash)

    @property
    def f(self) -> torch.Tensor:
        """Return a detached clone so caller mutation cannot alter this record."""
        return self._f.detach().clone()


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
    population_snapshots: tuple[D3Q19PopulationSnapshot, ...] = ()

    @property
    def population_states(self) -> tuple[torch.Tensor, ...]:
        """Real-state observer compatibility view; empty unless explicitly opted in."""
        return tuple(snapshot.f for snapshot in self.population_snapshots)


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
    capture_steps = config.capture_population_steps
    population_ownership_hash = _geometry_ownership_hash(geometry) if capture_steps else ""
    population_snapshots: list[D3Q19PopulationSnapshot] = []
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

    for step_index in range(1, steps + 1):
        f = plan.step(f)
        force_tensors = compute_obstacle_forces_3d(f, mask)
        moment_tensors = compute_obstacle_moments_3d(f, mask, origin_x, origin_y, origin_z)
        # Actual production f after collision+stream and before retained
        # channel/bounce-back updates; no population is reconstructed.
        if step_index in capture_steps:
            population_snapshots.append(D3Q19PopulationSnapshot(
                step_index=step_index,
                sample_phase=_POPULATION_SAMPLE_PHASE,
                f=f.detach().clone(),
                ownership_hash=population_ownership_hash,
            ))
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
        population_snapshots=tuple(population_snapshots),
    )


__all__ = [
    "D3Q19PopulationSnapshot",
    "FullyWettedFlowConfig",
    "FullyWettedFlowResult",
    "VoxelBodyGeometry",
    "run_fully_wetted_flow",
]
