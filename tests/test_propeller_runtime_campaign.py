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


def test_actual_two_level_low_mach_re_matched_resolution_campaign(tmp_path: Path) -> None:
    """CPU campaign: only resolution changes; two full-revolution windows per level."""
    coarse = PropellerBenchmarkConfig(
        geometry=PropellerGeometryConfig(n_blades=3, diameter=0.1),
        inflow_velocities=(0.0005,), rpm=0.005,
        nx=40, ny=20, nz=20, tau=0.8, warmup_steps=0, sampling_steps=None,
        n_revolutions=2, sample_window_steps=200, device="cpu", output_root=tmp_path,
        run_name="sensitivity-coarse",
    )
    # This is a distinct computational level (larger domain) while retaining
    # the exact moving-boundary history: D, rpm, nu, steps/rev and each window
    # are identical. Thus J, Re_D, tip Ma and angular sampling all match.
    fine = PropellerBenchmarkConfig(
        geometry=PropellerGeometryConfig(n_blades=3, diameter=0.1),
        inflow_velocities=(0.0005,), rpm=0.005,
        nx=48, ny=24, nz=24, tau=0.8, warmup_steps=0, sampling_steps=None,
        n_revolutions=2, sample_window_steps=200, device="cpu", output_root=tmp_path,
        run_name="sensitivity-fine",
    )

    evidence = run_propeller_resolution_sensitivity((coarse, fine), level_names=("coarse", "fine"))

    assert evidence["status"] == "not_converged"  # do not relax the window criterion
    basis = evidence["comparison_basis"]
    assert basis["low_mach_matched"] is True
    assert basis["advance_ratios_matched"] is True
    assert basis["re_d_matched"] is True
    assert basis["temporal_angular_contract"]["levels"][0]["complete_windows"] == 2
    assert basis["temporal_angular_contract"]["levels"][1]["complete_windows"] == 2
    assert [level["steps_per_revolution"] for level in basis["temporal_angular_contract"]["levels"]] == [200, 200]
    assert [level["angular_increment_degrees"] for level in basis["temporal_angular_contract"]["levels"]] == [1.8, 1.8]
    assert len(evidence["levels"]) == 2
    assert len(evidence["changes_from_baseline"]) == 1
    for level in evidence["levels"]:
        assert level["n_j_cases"] == 1
        assert level["per_j_window_status"][0]["complete_window_count"] == 2
        assert level["per_j_window_status"][0]["convergence"]["window_converged"] is False


def test_resolution_sensitivity_rejects_mixed_re_time_and_low_mach_contracts(tmp_path: Path) -> None:
    base = PropellerBenchmarkConfig(
        geometry=PropellerGeometryConfig(n_blades=3, diameter=0.1),
        inflow_velocities=(0.0005,), rpm=0.005, nx=40, ny=20, nz=20,
        tau=0.8, warmup_steps=0, sampling_steps=None, n_revolutions=2,
        sample_window_steps=200, device="cpu", output_root=tmp_path,
    )
    cases = (
        ("matched Re_D", PropellerBenchmarkConfig(**{**base.__dict__, "tau": 0.81})),
        # This was the formerly accepted candidate: J, Re_D, and tip Ma all
        # match, but 200 versus 240 rotor updates/revolution means 1.8 versus
        # 1.5 degrees/update. It must fail before a campaign starts.
        ("steps_per_revolution", PropellerBenchmarkConfig(
            **{**base.__dict__, "geometry": PropellerGeometryConfig(n_blades=3, diameter=0.12),
               "rpm": 1.0 / 240.0, "tau": 0.86, "sample_window_steps": 240}
        )),
        ("complete revolutions", PropellerBenchmarkConfig(**{**base.__dict__, "n_revolutions": 3})),
        # Every window is a whole rotation, but 600 sampling updates produce
        # only 1.5 windows of 400 updates; the trailing rotation cannot vanish.
        ("total sampling steps", PropellerBenchmarkConfig(
            **{**base.__dict__, "n_revolutions": 3, "sample_window_steps": 400}
        )),
        ("low-Mach gate", PropellerBenchmarkConfig(**{**base.__dict__, "rpm": 0.01, "inflow_velocities": (0.001,), "tau": 1.1, "sample_window_steps": 100})),
    )
    for message, invalid in cases:
        with pytest.raises(ValueError, match=message):
            run_propeller_resolution_sensitivity((base, invalid))


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
