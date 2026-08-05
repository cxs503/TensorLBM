"""SUBOFF bare-hull domain convergence study (D3Q19+MRT, diagnostic only).

TDD tests for the domain convergence study runner.  The study fixes the hull
length, grid resolution (dx=1), and step count, then varies the computational
domain size across at least three levels.  Each level runs a real D3Q19+MRT
bounce-back SUBOFF bare-hull simulation and produces a measured_candidate
evidence artifact with force/Ct time series.  The study collects Ct per level
and computes relative-change convergence indicators without claiming
convergence or physical validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm.suboff_domain_convergence_study import (
    DomainConvergenceStudyConfig,
    DomainLevel,
    run_suboff_domain_convergence_study,
)


def _small_study_config() -> DomainConvergenceStudyConfig:
    """Three small domain levels for fast test execution.

    Hull length is fixed at 12.0 lattice units across all levels.  Only the
    domain size (nx, ny, nz) varies, changing the blockage ratio.
    """
    return DomainConvergenceStudyConfig(
        domain_levels=(
            DomainLevel("small", 24, 12, 12),
            DomainLevel("medium", 32, 16, 16),
            DomainLevel("large", 40, 20, 20),
        ),
        hull_length=12.0,
        n_steps=5,
        warmup=2,
        u_in=0.06,
        re=200.0,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_fewer_than_three_domain_levels() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        DomainConvergenceStudyConfig(
            domain_levels=(
                DomainLevel("a", 24, 12, 12),
                DomainLevel("b", 32, 16, 16),
            ),
        )


def test_config_rejects_non_bare_hull() -> None:
    with pytest.raises(ValueError, match="bare_hull"):
        DomainConvergenceStudyConfig(
            domain_levels=(
                DomainLevel("a", 24, 12, 12),
                DomainLevel("b", 32, 16, 16),
                DomainLevel("c", 40, 20, 20),
            ),
            hull_type="with_sail",
        )


def test_config_rejects_non_d3q19() -> None:
    with pytest.raises(ValueError, match="D3Q19"):
        DomainConvergenceStudyConfig(
            domain_levels=(
                DomainLevel("a", 24, 12, 12),
                DomainLevel("b", 32, 16, 16),
                DomainLevel("c", 40, 20, 20),
            ),
            lattice="D3Q27",
        )


def test_config_rejects_non_mrt() -> None:
    with pytest.raises(ValueError, match="MRT"):
        DomainConvergenceStudyConfig(
            domain_levels=(
                DomainLevel("a", 24, 12, 12),
                DomainLevel("b", 32, 16, 16),
                DomainLevel("c", 40, 20, 20),
            ),
            collision="SRT",
        )


def test_config_rejects_non_positive_hull_length() -> None:
    with pytest.raises(ValueError, match="hull_length"):
        DomainConvergenceStudyConfig(
            domain_levels=(
                DomainLevel("a", 24, 12, 12),
                DomainLevel("b", 32, 16, 16),
                DomainLevel("c", 40, 20, 20),
            ),
            hull_length=0.0,
        )


def test_config_rejects_non_positive_n_steps() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        DomainConvergenceStudyConfig(
            domain_levels=(
                DomainLevel("a", 24, 12, 12),
                DomainLevel("b", 32, 16, 16),
                DomainLevel("c", 40, 20, 20),
            ),
            n_steps=0,
        )


def test_domain_level_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="nx"):
        DomainLevel("bad", 8, 12, 12)  # nx < 16


# ---------------------------------------------------------------------------
# Study execution
# ---------------------------------------------------------------------------

def test_study_runs_three_domain_levels_and_produces_diagnostic_artifact() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    assert artifact["artifact_kind"] == "suboff_domain_convergence_study"
    assert artifact["schema"] == "suboff-domain-convergence-study-r1"
    assert artifact["status"] == "diagnostic_only"
    assert artifact["physical_validation"] is False
    assert len(artifact["domain_levels"]) == 3
    assert len(artifact["Ct_per_level"]) == 3
    assert len(artifact["per_level_results"]) == 3


def test_study_artifact_domain_levels_match_config() -> None:
    config = _small_study_config()
    artifact = run_suboff_domain_convergence_study(config)

    for level_record, domain_level in zip(artifact["domain_levels"], config.domain_levels):
        assert level_record["level_id"] == domain_level.level_id
        assert level_record["nx"] == domain_level.nx
        assert level_record["ny"] == domain_level.ny
        assert level_record["nz"] == domain_level.nz


def test_study_hull_length_is_fixed_across_levels() -> None:
    config = _small_study_config()
    artifact = run_suboff_domain_convergence_study(config)

    for level_result in artifact["per_level_results"]:
        assert level_result["hull_length_lu"] == pytest.approx(config.hull_length)


def test_study_domain_length_varies_across_levels() -> None:
    config = _small_study_config()
    artifact = run_suboff_domain_convergence_study(config)

    domain_lengths = [lr["domain_length_lu"] for lr in artifact["per_level_results"]]
    # Domain lengths must be strictly increasing (domain grows)
    assert domain_lengths == sorted(domain_lengths)
    assert len(set(domain_lengths)) == 3  # all distinct


def test_study_blockage_ratio_decreases_as_domain_grows() -> None:
    config = _small_study_config()
    artifact = run_suboff_domain_convergence_study(config)

    blockage_ratios = [lr["blockage_ratio"] for lr in artifact["per_level_results"]]
    # Blockage ratio should decrease as domain grows (hull fixed, domain bigger)
    for i in range(len(blockage_ratios) - 1):
        assert blockage_ratios[i] > blockage_ratios[i + 1]


# ---------------------------------------------------------------------------
# Per-level results: force/Ct + measured_candidate
# ---------------------------------------------------------------------------

def test_study_per_level_has_force_time_series_and_measured_candidate() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    for level_result in artifact["per_level_results"]:
        assert level_result["evidence_status"] == "measured_candidate"
        assert level_result["physical_validation"] is False
        assert level_result["Ct"] is not None
        assert isinstance(level_result["Ct"], float)
        # Force time series: one entry per step
        assert len(level_result["force_time_series"]) == _small_study_config().n_steps
        for sample in level_result["force_time_series"]:
            assert "step" in sample
            assert "fx" in sample
            assert isinstance(sample["fx"], float)
        # Ct time series: one entry per step
        assert len(level_result["ct_time_series"]) == _small_study_config().n_steps
        for sample in level_result["ct_time_series"]:
            assert "step" in sample
            assert "ct" in sample
            assert isinstance(sample["ct"], float)


def test_study_per_level_has_runtime_and_geometry_info() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    for level_result in artifact["per_level_results"]:
        assert level_result["wetted_area"] > 0.0
        assert level_result["dynamic_pressure"] > 0.0
        assert level_result["runtime"]["completed_steps"] == _small_study_config().n_steps
        assert level_result["runtime"]["all_populations_finite"] is True
        assert level_result["blockage_ratio"] > 0.0


def test_study_ct_per_level_matches_per_level_results() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    for ct_value, level_result in zip(artifact["Ct_per_level"], artifact["per_level_results"]):
        assert ct_value == pytest.approx(level_result["Ct"])


# ---------------------------------------------------------------------------
# Convergence indicator
# ---------------------------------------------------------------------------

def test_study_convergence_indicator_has_required_fields() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

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
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    assert artifact["status"] == "diagnostic_only"
    assert artifact["physical_validation"] is False
    assert artifact["convergence_indicator"]["convergence_claim"] == "withheld"
    assert "converged" not in artifact["status"].lower()
    assert "validated" not in artifact["status"].lower()
    for level_result in artifact["per_level_results"]:
        assert level_result["evidence_status"] == "measured_candidate"
        assert level_result["physical_validation"] is False


# ---------------------------------------------------------------------------
# Determinism and provenance
# ---------------------------------------------------------------------------

def test_study_is_deterministic() -> None:
    config = _small_study_config()
    first = run_suboff_domain_convergence_study(config)
    second = run_suboff_domain_convergence_study(config)

    assert first["Ct_per_level"] == pytest.approx(second["Ct_per_level"])
    assert first["convergence_indicator"] == second["convergence_indicator"]
    assert first["provenance_hash"] == second["provenance_hash"]


def test_study_provenance_records_runner_and_model_identity() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    provenance = artifact["provenance"]
    assert provenance["runner_api"] == "tensorlbm.suboff_validation_runner.run_suboff_d3q19_mrt_validation"
    assert provenance["model_identity"]["lattice"] == "D3Q19"
    assert provenance["model_identity"]["collision"] == "MRT"
    assert provenance["model_identity"]["hull_type"] == "bare_hull"
    assert provenance["prohibition"] == "no_convergence_claim_or_physical_validation"


# ---------------------------------------------------------------------------
# JSON serializability and file output
# ---------------------------------------------------------------------------

def test_study_artifact_is_json_serializable() -> None:
    artifact = run_suboff_domain_convergence_study(_small_study_config())

    serialized = json.dumps(artifact, sort_keys=True, allow_nan=False)
    deserialized = json.loads(serialized)
    assert deserialized["artifact_kind"] == artifact["artifact_kind"]
    assert deserialized["Ct_per_level"] == artifact["Ct_per_level"]


def test_study_writes_machine_readable_artifact_file(tmp_path: Path) -> None:
    config = _small_study_config()
    artifact = run_suboff_domain_convergence_study(config, output_path=tmp_path / "domain_convergence.json")

    artifact_path = tmp_path / "domain_convergence.json"
    assert artifact_path.exists()
    loaded = json.loads(artifact_path.read_text())
    assert loaded["artifact_kind"] == "suboff_domain_convergence_study"
    assert loaded["status"] == "diagnostic_only"
    assert len(loaded["Ct_per_level"]) == 3
