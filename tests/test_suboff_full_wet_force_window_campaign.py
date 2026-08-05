"""Cold-path diagnostic campaign over real full-wet population captures."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.backends.contracts import DeviceSpec
from tensorlbm.full_wet import FullyWettedFlowConfig, VoxelBodyGeometry
from tensorlbm.marine_geometry import GeometryAsset
from tensorlbm.models.contracts import ModelComposition
from tensorlbm.suboff_full_wet_force_window_campaign import run_suboff_full_wet_force_window_campaign


def _composition() -> ModelComposition:
    return ModelComposition(
        lattice="D3Q19", collision="MRT", turbulence=None, forcing=(),
        boundaries=("zou_he_channel", "stationary_bounce_back"),
        physics_modules={"single_phase": "incompressible"},
    )


def _inputs(capture_steps: tuple[int, ...] = (1, 2)) -> tuple[GeometryAsset, FullyWettedFlowConfig]:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True
    asset = GeometryAsset(mask, "campaign-tiny", (2.0, 2.0, 2.0), "lattice", "campaign-test")
    config = FullyWettedFlowConfig(
        geometry=VoxelBodyGeometry(mask, "campaign-tiny", origin=(2.0, 2.0, 2.0)),
        composition=_composition(), device_spec=DeviceSpec("cpu", "float32"), shape=(5, 5, 5),
        tau=0.6, inlet_velocity=0.03, steps=2, capture_population_steps=capture_steps,
    )
    return asset, config


def test_campaign_measures_two_real_full_wet_snapshots_deterministically() -> None:
    asset, config = _inputs()

    first = run_suboff_full_wet_force_window_campaign(asset, config)
    second = run_suboff_full_wet_force_window_campaign(asset, config)

    assert first["artifact_kind"] == "suboff_full_wet_force_window_campaign"
    assert first["schema"] == "suboff-full-wet-force-window-campaign-r1"
    assert first["status"] == "measured_candidate"
    assert first["physical_validation"] is False
    assert first["steady_state_status"] == "diagnostic_withheld"
    assert first["diagnostic_status"] == "diagnostic_withheld"
    assert first["sample_windows"]["count"] == 2
    assert first["sample_windows"]["mean_force"] == pytest.approx(first["force_window"]["observation"]["force"])
    assert len(first["sample_windows"]["std_force"]) == 3
    assert len(first["force_records"]) == 2
    assert [record["capture_step"] for record in first["force_records"]] == [1, 2]
    assert all(record["sample_phase"] == "post_stream_pre_bounce_back" for record in first["force_records"])
    assert all(record["population_source"] == "full_wet_opt_in_production_snapshot" for record in first["force_records"])
    assert all(len(record["population_sha256"]) == 64 for record in first["force_records"])
    assert first["provenance"]["capture_count"] == 2
    assert first["provenance"]["geometry_source_hash"] == asset.source_hash
    assert len(first["provenance_hash"]) == 64
    assert first["sample_windows"] == second["sample_windows"]
    assert first["force_records"] == second["force_records"]
    assert first["provenance_hash"] == second["provenance_hash"]


def test_campaign_fails_closed_without_at_least_two_actual_captures() -> None:
    asset, config = _inputs((1,))

    with pytest.raises(ValueError, match="at least two opt-in population capture steps"):
        run_suboff_full_wet_force_window_campaign(asset, config)
