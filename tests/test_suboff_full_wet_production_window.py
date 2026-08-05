"""Fail-closed bridge from the public full-wet runner to the real-state observer."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorlbm.backends.contracts import DeviceSpec
from tensorlbm.full_wet import FullyWettedFlowConfig, VoxelBodyGeometry
from tensorlbm.marine_geometry import GeometryAsset
from tensorlbm.models.contracts import ModelComposition
from tensorlbm.suboff_full_wet_production_window import run_suboff_full_wet_production_window


@dataclass(frozen=True)
class _PublicResultWithoutPopulations:
    density: torch.Tensor
    velocity: torch.Tensor
    force: tuple[float, float, float]
    reaction: tuple[float, float, float]
    moment: tuple[float, float, float]
    status: str
    evidence: dict[str, object]


def _composition() -> ModelComposition:
    return ModelComposition(
        lattice="D3Q19", collision="MRT", turbulence=None, forcing=(),
        boundaries=("zou_he_channel", "stationary_bounce_back"),
        physics_modules={"single_phase": "incompressible"},
    )


def _inputs() -> tuple[GeometryAsset, FullyWettedFlowConfig]:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True
    asset = GeometryAsset(mask, "production-window-tiny", (2.0, 2.0, 2.0), "lattice", "test")
    config = FullyWettedFlowConfig(
        geometry=VoxelBodyGeometry(mask, "production-window-tiny", origin=(2.0, 2.0, 2.0)),
        composition=_composition(), device_spec=DeviceSpec("cpu", "float32"), shape=tuple(mask.shape),
        tau=0.6, inlet_velocity=0.03, steps=1,
    )
    return asset, config


def test_adapter_withholds_when_mocked_public_result_has_no_population_state() -> None:
    asset, config = _inputs()
    fake = _PublicResultWithoutPopulations(
        density=torch.ones(config.shape), velocity=torch.zeros((3, *config.shape)),
        force=(0.0, 0.0, 0.0), reaction=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0),
        status="COMPLETED", evidence={"force": {"kind": "diagnostic"}},
    )

    result = run_suboff_full_wet_production_window(asset, config, runner=lambda _: fake)

    assert result["window_status"] == "WITHHELD_NO_POPULATION_STATE"
    assert result["status"] == "WITHHELD_NO_POPULATION_STATE"
    assert result["force_window"] is None
    assert result["runner"]["status"] == "COMPLETED"
    assert result["provenance"]["population_source"] == "public_full_wet_result_absent"
    assert len(result["provenance_hash"]) == 64


def test_actual_tiny_full_wet_runner_smoke_is_withheld_not_synthetic() -> None:
    asset, config = _inputs()

    result = run_suboff_full_wet_production_window(asset, config)

    assert result["runner"]["status"] == "COMPLETED"
    assert result["window_status"] == "WITHHELD_NO_POPULATION_STATE"
    assert result["force_window"] is None
    assert result["physical_validation"] is False


def test_actual_population_export_is_consumed_as_measured_candidate() -> None:
    asset, config = _inputs()
    config = FullyWettedFlowConfig(
        geometry=config.geometry, composition=config.composition, device_spec=config.device_spec,
        shape=config.shape, tau=config.tau, inlet_velocity=config.inlet_velocity, steps=config.steps,
        capture_population_steps=(1,),
    )

    result = run_suboff_full_wet_production_window(asset, config)

    assert result["status"] == "measured_candidate"
    assert result["window_status"] == "MEASURED_REAL_POPULATION_STATE"
    assert result["force_window"]["windows"] == 1
    assert result["force_window"]["observation"]["sample_phase"] == "post_stream_pre_bounce_back"
    assert result["physical_validation"] is False
