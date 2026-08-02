from __future__ import annotations

import pytest

from tensorlbm.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    _initial_velocity,
    run_wall_resolved_channel3d,
)
import torch


def test_spectral_initialization_is_deterministic_solenoidal_and_no_slip() -> None:
    config = WallResolvedChannel3DConfig(
        nx=24,
        ny=18,
        nz=20,
        re_tau=40.0,
        u_tau=0.01,
        steps=2,
        warmup_steps=0,
        collision_model="cumulant",
        compile_natural_kbc=False,
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=0.75,
        spectral_mode_count=24,
        spectral_max_wavenumber=3,
        device="cpu",
    )
    first = _initial_velocity(config, torch.device("cpu"))
    second = _initial_velocity(config, torch.device("cpu"))
    for left, right in zip(first[1:4], second[1:4]):
        assert torch.equal(left, right)
    solid, ux, uy, uz, diagnostics = first
    assert torch.count_nonzero(ux[solid]) == 0
    assert torch.count_nonzero(uy[solid]) == 0
    assert torch.count_nonzero(uz[solid]) == 0
    assert diagnostics["total_rms_over_u_tau"] == pytest.approx(0.75, rel=2e-6)
    assert diagnostics["maximum_plane_mean_over_u_tau"] < 1e-5
    # The central-difference divergence converges to the analytic zero; the
    # residual is normalized by the perturbation velocity per lattice cell.
    base_y = torch.arange(config.ny, dtype=ux.dtype)[None, :, None]
    distance = torch.minimum(base_y - 0.5, config.height + 0.5 - base_y).clamp_min(0.0)
    from tensorlbm.spalding_wall_model import spalding_u_plus_from_y_plus
    base = spalding_u_plus_from_y_plus(distance * config.u_tau / config.nu) * config.u_tau
    px = ux - base.expand_as(ux)
    divergence = (
        0.5 * (torch.roll(px, -1, 2) - torch.roll(px, 1, 2))[:, 2:-2]
        + 0.5 * (uy[:, 3:-1] - uy[:, 1:-3])
        + 0.5 * (torch.roll(uz, -1, 0) - torch.roll(uz, 1, 0))[:, 2:-2]
    )
    assert float(torch.sqrt(divergence.square().mean())) < 2e-4 * config.u_tau


def test_channel_configuration_encodes_exact_momentum_balance(tmp_path) -> None:
    config = WallResolvedChannel3DConfig(
        nx=8,
        ny=10,
        nz=8,
        re_tau=20.0,
        u_tau=0.01,
        steps=2,
        warmup_steps=0,
        sample_interval=1,
        report_interval=1,
        checkpoint_interval=1,
        collision_model="cumulant",
        compile_natural_kbc=False,
        device="cpu",
        output=tmp_path / "result.json",
        checkpoint=tmp_path / "state.ckpt",
    )
    assert config.body_force_acceleration * config.height / 2 == pytest.approx(
        config.u_tau**2,
    )
    result = run_wall_resolved_channel3d(config)
    assert result["statistics"]["profile_samples"] == 2
    assert len(result["statistics"]["mean_velocity_profiles_xyz"]) == 3
    assert len(
        result["statistics"]["reynolds_stress_profiles_uu_vv_ww_uv"],
    ) == 4
    assert len(result["reports"]) == 2
    assert "crossflow_rms_over_u_tau" in result["reports"][0]
    assert "sustained_three_dimensional_fluctuations" in result["acceptance"]
    assert "minimum_two_eddy_turnover_statistics" in result["acceptance"]
    assert "stationarity_half_window_drift_below_2pct" in result["acceptance"]
    assert "domain_supports_full_dns_statistics" in result["acceptance"]
    assert config.output.exists()
    assert config.checkpoint.exists()


def test_channel_checkpoint_configuration_mismatch_fails_closed(tmp_path) -> None:
    common = dict(
        nx=8,
        ny=10,
        nz=8,
        re_tau=20.0,
        u_tau=0.01,
        warmup_steps=0,
        sample_interval=1,
        report_interval=1,
        checkpoint_interval=1,
        collision_model="cumulant",
        compile_natural_kbc=False,
        device="cpu",
        output=tmp_path / "result.json",
        checkpoint=tmp_path / "state.ckpt",
    )
    run_wall_resolved_channel3d(WallResolvedChannel3DConfig(steps=1, **common))
    with pytest.raises(ValueError, match="configuration"):
        run_wall_resolved_channel3d(
            WallResolvedChannel3DConfig(
                steps=2,
                resume=True,
                **(common | {"re_tau": 21.0}),
            ),
        )


def test_channel_checkpoint_allows_steps_only_extension(tmp_path) -> None:
    common = dict(
        nx=8,
        ny=10,
        nz=8,
        re_tau=20.0,
        u_tau=0.01,
        warmup_steps=0,
        sample_interval=1,
        report_interval=1,
        checkpoint_interval=1,
        collision_model="cumulant",
        compile_natural_kbc=False,
        device="cpu",
        output=tmp_path / "result.json",
        checkpoint=tmp_path / "state.ckpt",
    )
    run_wall_resolved_channel3d(WallResolvedChannel3DConfig(steps=1, **common))
    result = run_wall_resolved_channel3d(
        WallResolvedChannel3DConfig(steps=2, resume=True, **common),
    )
    assert result["reports"][-1]["step"] == 2


def test_channel_resume_can_reset_statistics_without_resetting_flow(tmp_path) -> None:
    common = dict(
        nx=8,
        ny=10,
        nz=8,
        re_tau=20.0,
        u_tau=0.01,
        warmup_steps=0,
        sample_interval=1,
        report_interval=1,
        checkpoint_interval=1,
        collision_model="cumulant",
        compile_natural_kbc=False,
        device="cpu",
        output=tmp_path / "result.json",
        checkpoint=tmp_path / "state.ckpt",
    )
    run_wall_resolved_channel3d(WallResolvedChannel3DConfig(steps=1, **common))
    result = run_wall_resolved_channel3d(WallResolvedChannel3DConfig(
        steps=2,
        resume=True,
        reset_statistics_on_resume=True,
        **common,
    ))
    assert result["statistics"]["profile_samples"] == 1
    assert result["statistics"]["statistics_reset_step"] == 1
