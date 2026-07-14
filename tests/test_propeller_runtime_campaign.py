"""Actual dynamic-geometry multi-J CPU campaign contract."""
from __future__ import annotations

from pathlib import Path

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
        assert result["control_volume_cross_check"]["method"] == "global_momentum_delta"
        assert result["control_volume_cross_check"]["available"] is True
        assert result["control_volume_cross_check"]["sample_count"] == 32

    run_dir = tmp_path / "propeller_owt" / "dynamic-multi-j"
    assert (run_dir / "run_metadata.json").is_file()
    assert (run_dir / "open_water.csv").is_file()
