from __future__ import annotations

import pytest

from tensorlbm.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)


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
