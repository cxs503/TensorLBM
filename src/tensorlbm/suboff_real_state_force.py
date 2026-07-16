"""Link-wise SUBOFF force observations from caller-owned D3Q19 populations.

This observer does not advance, synthesize, reset, or otherwise alter a fluid
state.  It only samples caller-provided post-stream/pre-bounce-back population
windows at wall links compiled from a :class:`GeometryAsset`.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import torch

from .d3q19 import C, OPPOSITE
from .force_observation import ForceObservation
from .marine_geometry import GeometryAsset, compile_d3q19_wall_links
from .marine_resistance_contract import ResistanceForceContract, build_resistance_force_contract


_METHOD = "d3q19_linkwise_momentum_exchange"
_SAMPLE_PHASE = "post_stream_pre_bounce_back"


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


def _direction(value: object) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("direction must be a finite non-zero (x, y, z) tuple")
    if any(isinstance(component, bool) or not isinstance(component, (int, float)) or not isfinite(float(component)) for component in value):
        raise ValueError("direction must be a finite non-zero (x, y, z) tuple")
    result = tuple(float(component) for component in value)
    if not any(result):
        raise ValueError("direction must be a finite non-zero (x, y, z) tuple")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SuboffRealStateForceConfig:
    """Explicit coefficient normalization for a state-backed observation."""

    rho: float = 1000.0
    U: float = 1.0
    reference_area: float = 1.0
    length: float = 1.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("rho", "U", "reference_area", "length"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "direction", _direction(self.direction))


@dataclass(frozen=True, slots=True)
class SuboffRealStateForceWindow:
    """Mean force over caller-owned population windows, never a physical claim."""

    observation: ForceObservation
    contract: ResistanceForceContract
    window_forces: tuple[tuple[float, float, float], ...]
    link_count: int
    physical_validation: bool = False

    def __post_init__(self) -> None:
        if not self.window_forces:
            raise ValueError("window_forces must contain at least one caller-provided state")
        if self.observation.status != "measured" or not self.observation.link_ownership:
            raise ValueError("observation must be a link-owned measured observation")
        if self.contract.status != "measured_candidate" or self.contract.validated:
            raise ValueError("contract must be an unvalidated measured candidate")
        if self.physical_validation:
            raise ValueError("real-state force observations are not physical validation")

    @property
    def windows(self) -> int:
        return len(self.window_forces)


def _validate_state(state: object, asset: GeometryAsset, index: int) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"states[{index}] must be a torch.Tensor")
    expected_shape = (19, *tuple(asset.solid_mask.shape))
    if tuple(state.shape) != expected_shape:
        raise ValueError(f"states[{index}] must have shape {expected_shape}, got {tuple(state.shape)}")
    if not state.dtype.is_floating_point:
        raise TypeError(f"states[{index}] must have a floating-point dtype, got {state.dtype}")
    if state.device != asset.solid_mask.device:
        raise ValueError(f"states[{index}] device {state.device} must equal geometry device {asset.solid_mask.device}")
    if not bool(torch.isfinite(state).all().item()):
        raise ValueError(f"states[{index}] must contain only finite populations")
    return state


def observe_suboff_real_state_force_window(
    asset: GeometryAsset,
    states: Sequence[torch.Tensor],
    *,
    config: SuboffRealStateForceConfig | None = None,
) -> SuboffRealStateForceWindow:
    """Observe a mean body force from N real D3Q19 population states.

    For every compiled solid-to-fluid link ``(q, neighbor)``, this reads
    ``f[opposite[q], *neighbor]`` from each caller-provided state and contributes
    ``-2 f c_q``.  No state is generated or mutated.  The returned contract is
    deliberately an unvalidated ``measured_candidate``, not a physical result.
    """
    if not isinstance(asset, GeometryAsset):
        raise TypeError("asset must be a GeometryAsset")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or not states:
        raise ValueError("states must be a non-empty sequence of D3Q19 population tensors")
    resolved_config = config or SuboffRealStateForceConfig()
    if not isinstance(resolved_config, SuboffRealStateForceConfig):
        raise TypeError("config must be a SuboffRealStateForceConfig")

    links = compile_d3q19_wall_links(asset)
    if links.count <= 0:
        raise ValueError("observer requires at least one compiled wall link")
    validated_states = tuple(_validate_state(state, asset, index) for index, state in enumerate(states))

    device = validated_states[0].device
    directions = links.direction.to(device=device)
    neighbors = links.neighbor_zyx.to(device=device)
    opposite = OPPOSITE.to(device=device, dtype=torch.int64).index_select(0, directions)
    c = C.to(device=device, dtype=validated_states[0].dtype).index_select(0, directions)
    z, y, x = neighbors.unbind(dim=1)

    window_forces: list[tuple[float, float, float]] = []
    for state in validated_states:
        incident = state[opposite, z, y, x]
        force = (-2.0 * incident.unsqueeze(1) * c.to(dtype=state.dtype)).sum(dim=0)
        window_forces.append((float(force[0].item()), float(force[1].item()), float(force[2].item())))
    mean_force = (
        sum(force[0] for force in window_forces) / len(window_forces),
        sum(force[1] for force in window_forces) / len(window_forces),
        sum(force[2] for force in window_forces) / len(window_forces),
    )

    observation = ForceObservation(
        method=_METHOD,
        lattice_id="D3Q19",
        sample_phase=_SAMPLE_PHASE,
        force_on="body",
        origin=asset.origin,
        status="measured",
        force=mean_force,
        link_ownership=links.has_link_ownership,
    )
    ownership = {
        "status": "complete",
        "owner": asset.body_id,
        "owned_links": links.count,
        "geometry_source_hash": asset.source_hash,
    }
    contract = build_resistance_force_contract(
        reference_area=resolved_config.reference_area,
        length=resolved_config.length,
        rho=resolved_config.rho,
        U=resolved_config.U,
        direction=resolved_config.direction,
        method=_METHOD,
        sample_phase=_SAMPLE_PHASE,
        link_ownership=ownership,
        force=mean_force,
    )
    return SuboffRealStateForceWindow(
        observation=observation,
        contract=contract,
        window_forces=tuple(window_forces),
        link_count=links.count,
    )


__all__ = [
    "SuboffRealStateForceConfig",
    "SuboffRealStateForceWindow",
    "observe_suboff_real_state_force_window",
]
