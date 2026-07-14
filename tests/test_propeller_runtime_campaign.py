"""Actual dynamic-geometry multi-J CPU campaign contract."""
from __future__ import annotations

from pathlib import Path

import pytest

from tensorlbm.propeller_benchmark import PropellerBenchmarkConfig, run_propeller_benchmark
from tensorlbm.propeller_cad import PropellerGeometryConfig


def test_actual_dynamic_geometry_multi_j_campaign_writes_windowed_samples(tmp_path: Path) -> None:
    config = PropellerBenchmarkConfig(
        geometry=PropellerGeometryConfig(n_blades=3, diameter=12.0),
        inflow_velocities=(0.004, 0.006, 0.008),
        rpm=0.0005,
        nx=40,
        ny=20,
        nz=20,
        tau=0.8,
        warmup_steps=2,
        sampling_steps=32,
        n_revolutions=1,
        sample_window_steps=8,
        device="cpu",
        output_root=tmp_path,
        run_name="dynamic-multi-j",
    )

    summary = run_propeller_benchmark(config)

    assert len(summary["results"]) == 3
    assert [result["j_actual"] for result in summary["results"]] == [2 / 3, 1.0, 4 / 3]
    assert summary["campaign"]["n_j_cases"] == 3
    assert summary["campaign"]["status"] == "not_converged"
    assert [status["j_actual"] for status in summary["campaign"]["per_j_window_status"]] == [2 / 3, 1.0, 4 / 3]
    assert all(status["convergence"]["window_converged"] is False for status in summary["campaign"]["per_j_window_status"])
    for result in summary["results"]:
        assert result["dynamic_geometry"] is True
        assert len(result["samples"]) == result["sampling_steps"] == 32
        assert result["transient_discard_steps"] == 2
        assert result["window_report"]["discarded_transient_samples"] == 0
        assert result["window_report"]["convergence"]["available"] is True
        assert result["control_volume_cross_check"]["method"] == "discrete_full_control_volume_momentum_budget"
        assert result["control_volume_cross_check"]["available"] is True
        assert result["control_volume_cross_check"]["status"] == "comparable"
        assert result["control_volume_cross_check"]["same_operator_action_reaction_status"] == "comparable"
        assert result["control_volume_cross_check"]["me_vs_cv_comparison_status"] == "noncomparable"
        assert result["control_volume_cross_check"]["sample_count"] == 32
        assert all(sample["open_faces_available"] is True for sample in result["samples"])
        assert all("budget_residual_x" in sample for sample in result["samples"])
        assert all(sample["wall_action_reaction_signed_residual_norm"] == pytest.approx(0.0, abs=5e-6) for sample in result["samples"])
        assert all(sample["wall_action_reaction_absolute_residual_norm"] == pytest.approx(0.0, abs=5e-6) for sample in result["samples"])
        assert all(sample["wall_action_reaction_relative_residual"] < 1e-5 for sample in result["samples"])
        assert all(
            sample["wall_momentum_contribution_x"] + sample["wall_reaction_x"] == pytest.approx(0.0, abs=5e-6)
            for sample in result["samples"]
        )

    run_dir = tmp_path / "propeller_owt" / "dynamic-multi-j"
    assert (run_dir / "run_metadata.json").is_file()
    assert (run_dir / "open_water.csv").is_file()
