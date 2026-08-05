"""R1 software-artifact chain from compiled wall links to a force candidate."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.marine_geometry import GeometryAsset
from tensorlbm.suboff_case_definition import SuboffCaseDefinition
from tensorlbm.suboff_full_wet_runtime import (
    SuboffFullWetRuntimeConfig,
    run_suboff_full_wet_runtime,
)


def _tiny_asset() -> GeometryAsset:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True
    return GeometryAsset(
        solid_mask=mask,
        body_id="suboff-r1-tiny-software-body",
        origin=(0.0, 0.0, 0.0),
        units="lattice",
        source_id="r1-test-synthetic-voxel",
    )


def test_runtime_uses_compiled_links_and_explicit_per_link_exchange() -> None:
    result = run_suboff_full_wet_runtime(
        _tiny_asset(),
        case=SuboffCaseDefinition(),
        config=SuboffFullWetRuntimeConfig(samples=2, velocity=(0.04, 0.0, 0.0)),
    )

    assert result["artifact_kind"] == "software_runtime"
    assert result["method"] == "d3q19_linkwise_momentum_exchange"
    assert result["sample_phase"] == "synthetic_population_post_stream_pre_bounce_back"
    assert result["links"]["count"] == 18
    assert result["links"]["ownership"]["status"] == "complete"
    assert len(result["force_series"]) == 2
    assert all(sample["link_count"] == 18 for sample in result["force_series"])
    assert all(len(sample["per_link_momentum_exchange"]) == 18 for sample in result["force_series"])
    assert result["force_observation"]["status"] == "measured"
    assert result["force"] == pytest.approx(result["force_observation"]["force"])
    assert result["status"] == "measured_candidate"
    assert result["Ct"] == pytest.approx(result["contract"]["Ct"])
    assert result["contract"]["status"] == "measured_candidate"
    assert result["contract"]["Ct"] is not None
    assert result["contract"]["validated"] is False
    assert result["reference"]["source_status"] == "withheld"
    assert result["physical_validation"] is False


def test_runtime_force_is_the_sum_of_linkwise_exchange_not_a_proxy_or_reset() -> None:
    result = run_suboff_full_wet_runtime(
        _tiny_asset(), config=SuboffFullWetRuntimeConfig(samples=1, velocity=(0.03, 0.0, 0.0))
    )

    sample = result["force_series"][0]
    expected = tuple(
        sum(link["force_on_body"][axis] for link in sample["per_link_momentum_exchange"])
        for axis in range(3)
    )
    assert sample["force_on_body"] == pytest.approx(expected)
    assert sample["force_on_body"] == pytest.approx(result["force_observation"]["force"])
    assert "cell_reset" not in result["prohibitions"]
    assert result["prohibitions"] == ["no_collision", "no_boundary_update", "no_population_reset"]


def test_runtime_rejects_empty_wall_link_geometry() -> None:
    empty = GeometryAsset(
        solid_mask=torch.zeros((3, 3, 3), dtype=torch.bool), body_id="empty",
        origin=(0.0, 0.0, 0.0), units="lattice", source_id="test",
    )
    with pytest.raises(ValueError, match="at least one compiled wall link"):
        run_suboff_full_wet_runtime(empty)
