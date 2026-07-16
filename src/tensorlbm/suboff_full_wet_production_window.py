"""Fail-closed production adapter for public full-wet population windows.

The public ``run_fully_wetted_flow`` R1 result currently exposes final
macroscopic fields and diagnostics, but no D3Q19 population states.  This
adapter intentionally does not reconstruct populations from those fields.  It
records the runner result and withholds the force window unless a future public
result explicitly exposes a sequence through ``population_states``.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Mapping, Sequence

import torch

from .full_wet import FullyWettedFlowConfig, run_fully_wetted_flow
from .marine_geometry import GeometryAsset
from .suboff_real_state_force import (
    SuboffRealStateForceConfig,
    SuboffRealStateForceWindow,
    observe_suboff_real_state_force_window,
)

_WITHHELD = "WITHHELD_NO_POPULATION_STATE"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    """Produce metadata only; tensor contents are never substituted for state."""
    if isinstance(value, torch.Tensor):
        return {"tensor": {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}}
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _runner_metadata(result: object) -> dict[str, object]:
    """Keep public runner provenance without claiming hidden implementation state."""
    return {
        "api": "tensorlbm.full_wet.run_fully_wetted_flow",
        "result_type": f"{type(result).__module__}.{type(result).__qualname__}",
        "status": _json_value(getattr(result, "status", None)),
        "force": _json_value(getattr(result, "force", None)),
        "evidence": _json_value(getattr(result, "evidence", None)),
    }


def _public_population_sequence(result: object) -> Sequence[torch.Tensor] | None:
    """Accept only an explicit public state-window field, never inferred fields."""
    states = getattr(result, "population_states", None)
    if states is None:
        return None
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or not states:
        raise ValueError("public population_states must be a non-empty sequence")
    if not all(isinstance(state, torch.Tensor) for state in states):
        raise TypeError("public population_states must contain only torch.Tensor values")
    return states


def _window_record(window: SuboffRealStateForceWindow) -> dict[str, object]:
    return {
        "windows": window.windows,
        "link_count": window.link_count,
        "window_forces": window.window_forces,
        "observation": {
            "method": window.observation.method,
            "lattice_id": window.observation.lattice_id,
            "sample_phase": window.observation.sample_phase,
            "force_on": window.observation.force_on,
            "status": window.observation.status,
            "force": window.observation.force,
            "link_ownership": window.observation.link_ownership,
        },
        "contract": _json_value(window.contract),
        "physical_validation": window.physical_validation,
    }


def run_suboff_full_wet_production_window(
    asset: GeometryAsset,
    config: FullyWettedFlowConfig,
    *,
    force_config: SuboffRealStateForceConfig | None = None,
    runner: Callable[[FullyWettedFlowConfig], object] = run_fully_wetted_flow,
) -> dict[str, Any]:
    """Run public full-wet then observe only explicitly exported real states.

    The default runner is the production public API.  ``runner`` is injectable
    solely for result-interface contract tests.  A result lacking explicit
    ``population_states`` yields ``WITHHELD_NO_POPULATION_STATE``; density,
    velocity, and force diagnostics are never converted into populations.
    """
    if not isinstance(asset, GeometryAsset):
        raise TypeError("asset must be a GeometryAsset")
    if not isinstance(config, FullyWettedFlowConfig):
        raise TypeError("config must be a FullyWettedFlowConfig")
    if not callable(runner):
        raise TypeError("runner must be callable")
    if tuple(asset.solid_mask.shape) != config.shape:
        raise ValueError("asset shape must equal full-wet config.shape")
    if not torch.equal(asset.solid_mask, config.geometry.mask):
        raise ValueError("asset solid_mask must equal full-wet geometry.mask")

    result = runner(config)
    runner_metadata = _runner_metadata(result)
    states = _public_population_sequence(result)
    common: dict[str, Any] = {
        "artifact_kind": "suboff_full_wet_production_window",
        "schema": "suboff-full-wet-production-window-r1",
        "runner": runner_metadata,
        "geometry": {"body_id": asset.body_id, "source_hash": asset.source_hash, "shape_zyx": list(asset.solid_mask.shape)},
        "physical_validation": False,
    }
    if states is None:
        provenance = {
            "population_source": "public_full_wet_result_absent",
            "public_result_contract": "FullyWettedFlowResult exposes density, velocity, force, reaction, moment, status, evidence only",
            "prohibition": "no_population_reconstruction_or_synthetic_state",
        }
        payload = {**common, "status": _WITHHELD, "window_status": _WITHHELD, "force_window": None, "provenance": provenance}
        payload["provenance_hash"] = _canonical_hash(_json_value(payload))
        return payload

    window = observe_suboff_real_state_force_window(asset, states, config=force_config)
    record = _window_record(window)
    provenance = {
        "population_source": "public_full_wet_result.population_states",
        "state_count": len(states),
        "state_kind": "caller-produced_public_runner_population_sequence",
    }
    payload = {**common, "status": "measured_candidate", "window_status": "MEASURED_REAL_POPULATION_STATE", "force_window": record, "provenance": provenance}
    payload["provenance_hash"] = _canonical_hash(_json_value(payload))
    return payload


__all__ = ["run_suboff_full_wet_production_window"]
