"""Cold-path R1 campaign: real full-wet captures into the production force window.

This runner does not alter the full-wet step loop.  It requires the caller to
opt in to two or more ``capture_population_steps``, runs that production path
once, and forwards its detached, post-stream/pre-bounce-back ``f`` snapshots to
the production window adapter.  The output is an unvalidated diagnostic only.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import sqrt
from typing import Any, cast

import torch

from .full_wet import D3Q19PopulationSnapshot, FullyWettedFlowConfig, run_fully_wetted_flow
from .marine_geometry import GeometryAsset
from .suboff_full_wet_production_window import run_suboff_full_wet_production_window
from .suboff_real_state_force import SuboffRealStateForceConfig


_SCHEMA = "suboff-full-wet-force-window-campaign-r1"
_POPULATION_SOURCE = "full_wet_opt_in_production_snapshot"


def _canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _population_hash(snapshot: D3Q19PopulationSnapshot) -> str:
    """Hash the actual exported population bytes and immutable capture metadata."""
    population = snapshot.f.detach().to(device="cpu").contiguous()
    digest = sha256()
    digest.update(b"D3Q19PopulationSnapshot/R1\0")
    digest.update(str(snapshot.step_index).encode("ascii"))
    digest.update(snapshot.sample_phase.encode("utf-8"))
    digest.update(snapshot.ownership_hash.encode("ascii"))
    digest.update(str(tuple(population.shape)).encode("ascii"))
    digest.update(str(population.dtype).encode("ascii"))
    digest.update(population.numpy().tobytes())
    return digest.hexdigest()


def _std_component(forces: tuple[tuple[float, float, float], ...], component: int) -> float:
    values = tuple(force[component] for force in forces)
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def run_suboff_full_wet_force_window_campaign(
    asset: GeometryAsset,
    config: FullyWettedFlowConfig,
    *,
    force_config: SuboffRealStateForceConfig | None = None,
) -> dict[str, Any]:
    """Run a real N-capture diagnostic campaign through the production adapter.

    At least two capture steps are mandatory so this is a window diagnostic,
    rather than a single-state observation.  Steady-state and physical status
    are deliberately withheld regardless of the numerical values observed.
    """
    if not isinstance(asset, GeometryAsset):
        raise TypeError("asset must be a GeometryAsset")
    if not isinstance(config, FullyWettedFlowConfig):
        raise TypeError("config must be a FullyWettedFlowConfig")
    if len(config.capture_population_steps) < 2:
        raise ValueError("campaign requires at least two opt-in population capture steps")

    # Run once, then inject that exact public production result into the public
    # adapter.  This avoids a second solver run and never synthesizes a state.
    result = run_fully_wetted_flow(config)
    snapshots = result.population_snapshots
    if len(snapshots) < 2:
        raise RuntimeError("production full-wet run returned fewer than two requested population snapshots")
    if tuple(snapshot.step_index for snapshot in snapshots) != config.capture_population_steps:
        raise RuntimeError("production full-wet capture schedule was not fulfilled")
    if any(snapshot.sample_phase != "post_stream_pre_bounce_back" for snapshot in snapshots):
        raise RuntimeError("production full-wet snapshots have an unsupported sample phase")

    adapter_result = run_suboff_full_wet_production_window(
        asset, config, force_config=force_config, runner=lambda _: result,
    )
    if adapter_result["status"] != "measured_candidate" or adapter_result["force_window"] is None:
        raise RuntimeError("production window adapter withheld the requested real population window")

    force_window = adapter_result["force_window"]
    window_forces = tuple(
        cast(tuple[float, float, float], tuple(float(value) for value in force))
        for force in force_window["window_forces"]
    )
    if len(window_forces) != len(snapshots):
        raise RuntimeError("production adapter force record count does not match captured snapshots")
    records = tuple({
        "capture_step": snapshot.step_index,
        "sample_phase": snapshot.sample_phase,
        "ownership_hash": snapshot.ownership_hash,
        "population_source": _POPULATION_SOURCE,
        "population_sha256": _population_hash(snapshot),
        "force_on_body": window_forces[index],
    } for index, snapshot in enumerate(snapshots))
    mean_force = tuple(float(value) for value in force_window["observation"]["force"])
    std_force = tuple(_std_component(window_forces, component) for component in range(3))
    provenance = {
        "population_source": _POPULATION_SOURCE,
        "capture_steps": list(config.capture_population_steps),
        "capture_count": len(snapshots),
        "snapshot_hashes": [record["population_sha256"] for record in records],
        "snapshot_ownership_hashes": [snapshot.ownership_hash for snapshot in snapshots],
        "geometry_source_hash": asset.source_hash,
        "adapter_provenance_hash": adapter_result["provenance_hash"],
        "runner_api": "tensorlbm.full_wet.run_fully_wetted_flow",
        "adapter_api": "tensorlbm.suboff_full_wet_production_window.run_suboff_full_wet_production_window",
        "prohibition": "no_population_reconstruction_or_synthetic_state",
    }
    payload: dict[str, Any] = {
        "artifact_kind": "suboff_full_wet_force_window_campaign",
        "schema": _SCHEMA,
        "status": "measured_candidate",
        "physical_validation": False,
        "steady_state_status": "diagnostic_withheld",
        "diagnostic_status": "diagnostic_withheld",
        "geometry": adapter_result["geometry"],
        "runner": adapter_result["runner"],
        "force_window": force_window,
        "force_records": records,
        "sample_windows": {
            "count": len(window_forces),
            "capture_steps": list(config.capture_population_steps),
            "mean_force": mean_force,
            "std_force": std_force,
            "link_count": force_window["link_count"],
        },
        "provenance": provenance,
    }
    payload["provenance_hash"] = _canonical_hash(payload)
    return payload


__all__ = ["run_suboff_full_wet_force_window_campaign"]
