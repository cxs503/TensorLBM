"""SUBOFF bare-hull time convergence study (D3Q19+MRT, diagnostic only).

TDD tests for the time convergence study runner.  The study runs the SUBOFF
bare-hull D3Q19+MRT validation runner at four or more different step counts on
a fixed grid (48×24×24), collects the measured Ct candidate per time level,
and computes relative-change indicators — but deliberately withholds any
convergence or physical-validation claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm.suboff_time_convergence_study import (
    TimeConvergenceStudyConfig,
    TimeLevel,
    run_suboff_time_convergence_study,
)


# ---------------------------------------------------------------------------
# Small study config factory (fast test execution)
# ---------------------------------------------------------------------------

def _small_study_config() -> TimeConvergenceStudyConfig:
    """Four tiny time levels for fast test execution."""
    return TimeConvergenceStudyConfig(
        time_levels=(
            TimeLevel("t02", n_steps=2, capture_window=1),
            TimeLevel("t04", n_steps=4, capture_window=2),
            TimeLevel("t06", n_steps=6, capture_window=2),
            TimeLevel("t08", n_steps=8, capture_window=2),
        ),
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_fewer_than_four_time_levels() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        TimeConvergenceStudyConfig(
            time_levels=(
                TimeLevel("a", n_steps=2, capture_window=1),
                TimeLevel("b", n_steps=4, capture_window=2),
                TimeLevel("c", n_steps=6, capture_window=2),
            ),
        )


def test_config_rejects_non_bare_hull() -> None:
    with pytest.raises(ValueError, match="bare_hull"):
        TimeConvergenceStudyConfig(
            time_levels=(
                TimeLevel("a", n_steps=2, capture_window=1),
                TimeLevel("b", n_steps=4, capture_window=2),
                TimeLevel("c", n_steps=6, capture_window=2),
                TimeLevel("d", n_steps=8, capture_window=2),
            ),
            hull_type="with_sail",
        )


def test_config_rejects_non_d3q19() -> None:
    with pytest.raises(ValueError, match="D3Q19"):
        TimeConvergenceStudyConfig(
            time_levels=(
                TimeLevel("a", n_steps=2, capture_window=1),
                TimeLevel("b", n_steps=4, capture_window=2),
                TimeLevel("c", n_steps=6, capture_window=2),
                TimeLevel("d", n_steps=8, capture_window=2),
            ),
            lattice="D3Q27",
        )


def test_config_rejects_non_mrt() -> None:
    with pytest.raises(ValueError, match="MRT"):
        TimeConvergenceStudyConfig(
            time_levels=(
                TimeLevel("a", n_steps=2, capture_window=1),
                TimeLevel("b", n_steps=4, capture_window=2),
                TimeLevel("c", n_steps=6, capture_window=2),
                TimeLevel("d", n_steps=8, capture_window=2),
            ),
            collision="BGK",
        )


def test_time_level_validates_n_steps() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        TimeLevel("bad", n_steps=0, capture_window=1)


def test_time_level_validates_capture_window() -> None:
    with pytest.raises(ValueError, match="capture_window"):
        TimeLevel("bad", n_steps=4, capture_window=5)


def test_time_level_validates_capture_window_positive() -> None:
    with pytest.raises(ValueError, match="capture_window"):
        TimeLevel("bad", n_steps=4, capture_window=0)


def test_config_rejects_duplicate_level_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        TimeConvergenceStudyConfig(
            time_levels=(
                TimeLevel("dup", n_steps=2, capture_window=1),
                TimeLevel("dup", n_steps=4, capture_window=2),
                TimeLevel("c", n_steps=6, capture_window=2),
                TimeLevel("d", n_steps=8, capture_window=2),
            ),
        )


# ---------------------------------------------------------------------------
# Study execution
# ---------------------------------------------------------------------------

def test_study_runs_four_time_levels_and_produces_diagnostic_artifact() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    assert artifact["artifact_kind"] == "suboff_time_convergence_study"
    assert artifact["schema"] == "suboff-time-convergence-study-r1"
    assert artifact["status"] == "diagnostic_only"
    assert artifact["physical_validation"] is False
    assert len(artifact["time_levels"]) == 4
    assert len(artifact["Ct_per_level"]) == 4
    assert len(artifact["per_level_results"]) == 4


def test_study_artifact_time_levels_match_config() -> None:
    config = _small_study_config()
    artifact = run_suboff_time_convergence_study(config)

    for level_record, time_level in zip(artifact["time_levels"], config.time_levels):
        assert level_record["level_id"] == time_level.level_id
        assert level_record["n_steps"] == time_level.n_steps
        assert level_record["capture_window"] == time_level.capture_window


def test_study_artifact_records_fixed_grid_shape() -> None:
    config = _small_study_config()
    artifact = run_suboff_time_convergence_study(config)

    grid = artifact["grid_shape"]
    assert grid["nx"] == config.nx
    assert grid["ny"] == config.ny
    assert grid["nz"] == config.nz


def test_study_per_level_has_force_and_ct_time_series_and_measured_candidate() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    for level_result in artifact["per_level_results"]:
        assert level_result["status"] == "measured_candidate"
        assert level_result["physical_validation"] is False
        assert level_result["Ct"] is not None
        assert isinstance(level_result["Ct"], float)
        # Force time series: one entry per step
        assert len(level_result["force_time_series"]) == level_result["n_steps"]
        for sample in level_result["force_time_series"]:
            assert "step" in sample
            assert "fx" in sample
            assert "fy" in sample
            assert "fz" in sample
        # Ct time series: one entry per step
        assert len(level_result["ct_time_series"]) == level_result["n_steps"]
        for sample in level_result["ct_time_series"]:
            assert "step" in sample
            assert "ct" in sample
            assert "ct_fric" in sample
            assert "ct_pres" in sample


def test_study_ct_per_level_matches_per_level_results() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    for ct_value, level_result in zip(artifact["Ct_per_level"], artifact["per_level_results"]):
        assert ct_value == pytest.approx(level_result["Ct"])


def test_study_capture_steps_correct() -> None:
    """The capture_steps field must list the actual step indices in the window."""
    config = _small_study_config()
    artifact = run_suboff_time_convergence_study(config)

    for level_record, time_level in zip(artifact["time_levels"], config.time_levels):
        expected = list(range(
            time_level.n_steps - time_level.capture_window + 1,
            time_level.n_steps + 1,
        ))
        assert level_record["capture_steps"] == expected


def test_study_convergence_indicator_has_required_fields() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    indicator = artifact["convergence_indicator"]
    assert "relative_ct_changes" in indicator
    assert "ct_trend" in indicator
    assert "max_relative_change" in indicator
    assert "convergence_claim" in indicator
    # relative_ct_changes has N-1 entries for N levels
    assert len(indicator["relative_ct_changes"]) == 3
    # convergence claim is always withheld
    assert indicator["convergence_claim"] == "withheld"
    # trend is one of the allowed values
    assert indicator["ct_trend"] in ("decreasing", "increasing", "non_monotonic")


def test_study_does_not_claim_convergence_or_validation() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    # The artifact must not claim convergence or physical validation
    assert artifact["status"] == "diagnostic_only"
    assert artifact["physical_validation"] is False
    assert artifact["convergence_indicator"]["convergence_claim"] == "withheld"
    assert "converged" not in artifact["status"].lower()
    assert "validated" not in artifact["status"].lower()
    # Per-level results must also be unvalidated
    for level_result in artifact["per_level_results"]:
        assert level_result["status"] == "measured_candidate"
        assert level_result["physical_validation"] is False


def test_study_is_deterministic() -> None:
    config = _small_study_config()
    first = run_suboff_time_convergence_study(config)
    second = run_suboff_time_convergence_study(config)

    assert first["Ct_per_level"] == pytest.approx(second["Ct_per_level"])
    assert first["convergence_indicator"] == second["convergence_indicator"]
    assert first["provenance_hash"] == second["provenance_hash"]


def test_study_provenance_records_runner_and_model_identity() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    provenance = artifact["provenance"]
    assert provenance["runner_api"] == (
        "tensorlbm.suboff_validation_runner.run_suboff_d3q19_mrt_validation"
    )
    assert provenance["model_identity"]["lattice"] == "D3Q19"
    assert provenance["model_identity"]["collision"] == "MRT"
    assert provenance["model_identity"]["hull_type"] == "bare_hull"
    assert provenance["prohibition"] == "no_convergence_claim_or_physical_validation"


def test_study_artifact_is_json_serializable() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    # Must be fully JSON-serializable for machine readability
    serialized = json.dumps(artifact, sort_keys=True, allow_nan=False)
    deserialized = json.loads(serialized)
    assert deserialized["artifact_kind"] == artifact["artifact_kind"]
    assert deserialized["Ct_per_level"] == artifact["Ct_per_level"]


def test_study_writes_machine_readable_artifact_file(tmp_path: Path) -> None:
    config = _small_study_config()
    artifact = run_suboff_time_convergence_study(
        config, output_path=tmp_path / "time_convergence.json",
    )

    artifact_path = tmp_path / "time_convergence.json"
    assert artifact_path.exists()
    loaded = json.loads(artifact_path.read_text())
    assert loaded["artifact_kind"] == "suboff_time_convergence_study"
    assert loaded["status"] == "diagnostic_only"
    assert len(loaded["Ct_per_level"]) == 4


def test_study_per_level_runtime_evidence() -> None:
    """Each per-level result must carry runtime evidence from the real run."""
    artifact = run_suboff_time_convergence_study(_small_study_config())

    for level_result in artifact["per_level_results"]:
        runtime = level_result["runtime"]
        assert runtime["completed_steps"] == level_result["n_steps"]
        assert runtime["requested_steps"] == level_result["n_steps"]
        assert runtime["all_populations_finite"] is True
        assert runtime["all_densities_finite"] is True


def test_study_per_level_records_config_snapshot() -> None:
    artifact = run_suboff_time_convergence_study(_small_study_config())

    for level_result in artifact["per_level_results"]:
        cfg = level_result["config"]
        assert cfg["lattice"] == "D3Q19"
        assert cfg["collision"] == "MRT"
        assert cfg["boundary"] == "bounce_back"
        assert cfg["hull_type"] == "bare_hull"
