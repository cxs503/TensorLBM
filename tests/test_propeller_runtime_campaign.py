"""Actual dynamic-geometry multi-J CPU campaign contract."""
from __future__ import annotations

from pathlib import Path

import pytest

from tensorlbm.propeller_benchmark import (
    PropellerBenchmarkConfig,
    run_propeller_benchmark,
    run_propeller_resolution_sensitivity,
)
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
        # The same moving-wall population delta must also provide a torque
        # action/reaction pair about the propeller axis for every sample.
        assert all("wall_fluid_torque_impulse_x" in sample for sample in result["samples"])
        assert all("wall_reaction_torque_x" in sample for sample in result["samples"])
        assert all(sample["wall_torque_action_reaction_signed_residual_norm"] == pytest.approx(0.0, abs=5e-6) for sample in result["samples"])
        assert all(sample["wall_torque_action_reaction_absolute_residual_norm"] == pytest.approx(0.0, abs=5e-6) for sample in result["samples"])
        assert all(sample["wall_torque_action_reaction_relative_residual"] < 1e-5 for sample in result["samples"])
        assert all(
            sample["wall_fluid_torque_impulse_x"] + sample["wall_reaction_torque_x"] == pytest.approx(0.0, abs=5e-6)
            for sample in result["samples"]
        )
        torque_check = result["control_volume_cross_check"]["same_operator_torque_action_reaction"]
        assert torque_check["status"] == "comparable"
        assert torque_check["sample_count"] == 32
        assert torque_check["absolute_residual_x_max"] == pytest.approx(0.0, abs=5e-6)

    run_dir = tmp_path / "propeller_owt" / "dynamic-multi-j"
    assert (run_dir / "run_metadata.json").is_file()
    assert (run_dir / "open_water.csv").is_file()


def test_actual_three_level_true_spatial_refinement_campaign_is_fail_closed(tmp_path: Path) -> None:
    """CPU evidence preserves physical tank/propeller while D cells increase."""
    def level(diameter: float, nx: int, rpm: float, tau: float, name: str) -> PropellerBenchmarkConfig:
        steps = round(1 / rpm)
        return PropellerBenchmarkConfig(
            geometry=PropellerGeometryConfig(n_blades=3, diameter=diameter),
            inflow_velocities=(0.0005,), rpm=rpm, nx=nx, ny=nx // 2, nz=nx // 2,
            tau=tau, warmup_steps=0, sampling_steps=None, n_revolutions=2,
            sample_window_steps=steps, device="cpu", output_root=tmp_path, run_name=name,
        )

    # D, rpm and nu scale as 1:1, 1/D and D respectively.  Hence physical
    # extents, J, Re_D and tip Mach are fixed while lattice spacing decreases.
    coarse = level(0.10, 40, 1 / 200, 0.8, "spatial-coarse")
    medium = level(0.15, 60, 1 / 300, 0.95, "spatial-medium")
    fine = level(0.20, 80, 1 / 400, 1.1, "spatial-fine")
    evidence = run_propeller_resolution_sensitivity((coarse, medium, fine), level_names=("coarse", "medium", "fine"))

    # Matching Ma/Re/J requires a different lattice rpm at every voxel level,
    # hence 200/300/400 updates per revolution. The direct voxel moving-boundary
    # implementation cannot call those histories the same temporal experiment.
    assert evidence["status"] == "withheld"
    assert evidence["reason"] == "incomparable_voxel_refinement_contract"
    assert evidence["metric_convergence"]["status"] == "withheld"
    basis = evidence["comparison_basis"]
    assert basis["kind"] == "same_physical_geometry_true_spatial_refinement"
    assert basis["low_mach_matched"] is basis["advance_ratios_matched"] is basis["re_d_matched"] is True
    assert basis["physical_domain_matched"] is True
    assert basis["exact_rotation_time_sampling_matched"] is False
    assert [level["diameter_lu"] for level in basis["levels"]] == [0.1, 0.15, 0.2]
    assert [level["cell_size_m"] for level in basis["levels"]] == [2.5, pytest.approx(5 / 3), 1.25]
    assert [level["domain_per_diameter"] for level in basis["levels"]] == [[400.0, 200.0, 200.0]] * 3
    assert [level["domain_physical_m"] for level in basis["levels"]] == [[100.0, 50.0, 50.0]] * 3
    assert [level["complete_windows"] for level in basis["levels"]] == [2, 2, 2]
    assert [level["campaign_status"] for level in evidence["levels"]] == ["not_run"] * 3
    assert (tmp_path / "propeller_owt" / "resolution_sensitivity.json").is_file()


def test_resolution_sensitivity_rejects_two_levels_before_running(tmp_path: Path) -> None:
    base = PropellerBenchmarkConfig(geometry=PropellerGeometryConfig(n_blades=3, diameter=0.1), inflow_velocities=(0.0005,), rpm=1 / 200, nx=40, ny=20, nz=20, tau=0.8, warmup_steps=0, n_revolutions=2, sample_window_steps=200, output_root=tmp_path)
    with pytest.raises(ValueError, match="at least 3"):
        run_propeller_resolution_sensitivity((base, base))


def test_resolution_sensitivity_rejects_rounded_rpm_angular_claim_before_run(tmp_path: Path) -> None:
    """The actual solver phase is step * 360*rpm, not 360/round(1/rpm)."""
    base = PropellerBenchmarkConfig(
        geometry=PropellerGeometryConfig(n_blades=3, diameter=0.1),
        inflow_velocities=(0.0005,), rpm=0.005, nx=40, ny=20, nz=20,
        tau=0.8, warmup_steps=0, sampling_steps=None, n_revolutions=2,
        sample_window_steps=200, device="cpu", output_root=tmp_path,
    )
    rounded_candidate = PropellerBenchmarkConfig(
        **{**base.__dict__, "rpm": 1.0 / 200.4}
    )

    with pytest.raises(
        ValueError,
        match=r"actual angular increment .* disagrees with claimed",
    ):
        run_propeller_resolution_sensitivity((base, rounded_candidate))


def test_resolution_sensitivity_requires_whole_rotation_windows(tmp_path: Path) -> None:
    base = PropellerBenchmarkConfig(
        geometry=PropellerGeometryConfig(n_blades=3, diameter=0.1),
        inflow_velocities=(0.0005,), rpm=0.005, nx=40, ny=20, nz=20,
        tau=0.8, warmup_steps=0, sampling_steps=None, n_revolutions=2,
        sample_window_steps=200, device="cpu", output_root=tmp_path,
    )
    partial_window = PropellerBenchmarkConfig(
        **{**base.__dict__, "sample_window_steps": 100}
    )

    with pytest.raises(ValueError, match="sample_window_steps.*exact whole number of rotations"):
        run_propeller_resolution_sensitivity((base, partial_window))
