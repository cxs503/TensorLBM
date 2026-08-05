"""SUBOFF bare-hull grid convergence study (D3Q19+MRT, diagnostic only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm.suboff_grid_convergence_study import (
    GridConvergenceStudyConfig,
    GridLevel,
    run_suboff_grid_convergence_study,
)


def _small_study_config() -> GridConvergenceStudyConfig:
    """Three tiny grid levels for fast test execution."""
    return GridConvergenceStudyConfig(
        grid_levels=(
            GridLevel("coarse", 16, 8, 8, steps=3, capture_steps=(2, 3)),
            GridLevel("medium", 24, 12, 12, steps=3, capture_steps=(2, 3)),
            GridLevel("fine", 32, 16, 16, steps=3, capture_steps=(2, 3)),
        ),
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_fewer_than_three_grid_levels() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        GridConvergenceStudyConfig(
            grid_levels=(
                GridLevel("a", 16, 8, 8, steps=2, capture_steps=(1, 2)),
                GridLevel("b", 24, 12, 12, steps=2, capture_steps=(1, 2)),
            ),
        )


def test_config_rejects_non_bare_hull() -> None:
    with pytest.raises(ValueError, match="bare_hull"):
        GridConvergenceStudyConfig(
            grid_levels=(
                GridLevel("a", 16, 8, 8, steps=2, capture_steps=(1, 2)),
                GridLevel("b", 24, 12, 12, steps=2, capture_steps=(1, 2)),
                GridLevel("c", 32, 16, 16, steps=2, capture_steps=(1, 2)),
            ),
            hull_type="with_sail",
        )


def test_config_rejects_non_d3q19_mrt() -> None:
    with pytest.raises(ValueError, match="D3Q19"):
        GridConvergenceStudyConfig(
            grid_levels=(
                GridLevel("a", 16, 8, 8, steps=2, capture_steps=(1, 2)),
                GridLevel("b", 24, 12, 12, steps=2, capture_steps=(1, 2)),
                GridLevel("c", 32, 16, 16, steps=2, capture_steps=(1, 2)),
            ),
            lattice="D3Q27",
        )


def test_grid_level_validates_capture_steps() -> None:
    with pytest.raises(ValueError, match="capture_steps"):
        GridLevel("bad", 16, 8, 8, steps=3, capture_steps=(1,))


# ---------------------------------------------------------------------------
# Study execution
# ---------------------------------------------------------------------------

def test_study_runs_three_grid_levels_and_produces_diagnostic_artifact() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    assert artifact["artifact_kind"] == "suboff_grid_convergence_study"
    assert artifact["schema"] == "suboff-grid-convergence-study-r1"
    assert artifact["status"] == "diagnostic_only"
    assert artifact["physical_validation"] is False
    assert len(artifact["grid_levels"]) == 3
    assert len(artifact["Ct_per_level"]) == 3
    assert len(artifact["per_level_results"]) == 3


def test_study_artifact_grid_levels_match_config() -> None:
    config = _small_study_config()
    artifact = run_suboff_grid_convergence_study(config)

    for level_record, grid_level in zip(artifact["grid_levels"], config.grid_levels):
        assert level_record["level_id"] == grid_level.level_id
        assert level_record["nx"] == grid_level.nx
        assert level_record["ny"] == grid_level.ny
        assert level_record["nz"] == grid_level.nz
        assert level_record["steps"] == grid_level.steps
        assert tuple(level_record["capture_steps"]) == grid_level.capture_steps


def test_study_per_level_has_force_time_series_and_measured_candidate() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    for level_result in artifact["per_level_results"]:
        assert level_result["campaign_status"] == "measured_candidate"
        assert level_result["Ct"] is not None
        assert isinstance(level_result["Ct"], float)
        # Force time series: one entry per capture step
        assert len(level_result["force_time_series"]) == 2
        for force in level_result["force_time_series"]:
            assert len(force) == 3
            assert all(isinstance(c, float) for c in force)
        assert level_result["link_count"] > 0
        assert level_result["solid_cells"] > 0


def test_study_ct_per_level_matches_per_level_results() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    for ct_value, level_result in zip(artifact["Ct_per_level"], artifact["per_level_results"]):
        assert ct_value == pytest.approx(level_result["Ct"])


def test_study_convergence_indicator_has_required_fields() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    indicator = artifact["convergence_indicator"]
    assert "relative_ct_changes" in indicator
    assert "ct_trend" in indicator
    assert "max_relative_change" in indicator
    assert "convergence_claim" in indicator
    # relative_ct_changes has N-1 entries for N levels
    assert len(indicator["relative_ct_changes"]) == 2
    # convergence claim is always withheld
    assert indicator["convergence_claim"] == "withheld"
    # trend is one of the allowed values
    assert indicator["ct_trend"] in ("decreasing", "increasing", "non_monotonic")


def test_study_does_not_claim_convergence_or_validation() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    # The artifact must not claim convergence or physical validation
    assert artifact["status"] == "diagnostic_only"
    assert artifact["physical_validation"] is False
    assert artifact["convergence_indicator"]["convergence_claim"] == "withheld"
    assert "converged" not in artifact["status"].lower()
    assert "validated" not in artifact["status"].lower()
    # Per-level results must also be unvalidated
    for level_result in artifact["per_level_results"]:
        assert level_result["campaign_status"] == "measured_candidate"
        assert level_result["physical_validation"] is False


def test_study_is_deterministic() -> None:
    config = _small_study_config()
    first = run_suboff_grid_convergence_study(config)
    second = run_suboff_grid_convergence_study(config)

    assert first["Ct_per_level"] == pytest.approx(second["Ct_per_level"])
    assert first["convergence_indicator"] == second["convergence_indicator"]
    assert first["provenance_hash"] == second["provenance_hash"]


def test_study_provenance_records_runner_and_model_identity() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    provenance = artifact["provenance"]
    assert provenance["runner_api"] == "tensorlbm.suboff_full_wet_force_window_campaign.run_suboff_full_wet_force_window_campaign"
    assert provenance["model_identity"]["lattice"] == "D3Q19"
    assert provenance["model_identity"]["collision"] == "MRT"
    assert provenance["model_identity"]["hull_type"] == "bare_hull"
    assert provenance["prohibition"] == "no_convergence_claim_or_physical_validation"


def test_study_artifact_is_json_serializable() -> None:
    artifact = run_suboff_grid_convergence_study(_small_study_config())

    # Must be fully JSON-serializable for machine readability
    serialized = json.dumps(artifact, sort_keys=True, allow_nan=False)
    deserialized = json.loads(serialized)
    assert deserialized["artifact_kind"] == artifact["artifact_kind"]
    assert deserialized["Ct_per_level"] == artifact["Ct_per_level"]


def test_study_writes_machine_readable_artifact_file(tmp_path: Path) -> None:
    config = _small_study_config()
    artifact = run_suboff_grid_convergence_study(config, output_path=tmp_path / "convergence.json")

    artifact_path = tmp_path / "convergence.json"
    assert artifact_path.exists()
    loaded = json.loads(artifact_path.read_text())
    assert loaded["artifact_kind"] == "suboff_grid_convergence_study"
    assert loaded["status"] == "diagnostic_only"
    assert len(loaded["Ct_per_level"]) == 3
