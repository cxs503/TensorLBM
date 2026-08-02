from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tensorlbm.sphere_bfl_control_volume import (
    SphereBFLControlVolumeConfig,
    run_sphere_bfl_control_volume,
    schiller_naumann_cd,
)


def test_schiller_naumann_re100_reference() -> None:
    assert schiller_naumann_cd(100.0) == pytest.approx(1.0917310911)


def test_config_derived_viscosity_and_tau() -> None:
    cfg = SphereBFLControlVolumeConfig(radius=10, reynolds=100, lattice_speed=0.05)
    assert cfg.nu == pytest.approx(0.01)
    assert cfg.tau == pytest.approx(0.53)


def test_unknown_far_field_mode_is_rejected() -> None:
    cfg = SphereBFLControlVolumeConfig(far_field_mode="magic")
    with pytest.raises(ValueError, match="far_field_mode"):
        cfg.validate()


def test_unknown_collision_model_is_rejected() -> None:
    cfg = SphereBFLControlVolumeConfig(collision_model="magic")
    with pytest.raises(ValueError, match="collision_model"):
        cfg.validate()


def test_compiled_collision_requires_natural_kbc() -> None:
    cfg = SphereBFLControlVolumeConfig(compile_natural_kbc=True)
    with pytest.raises(ValueError, match="natural_kbc"):
        cfg.validate()


def test_negative_collision_chunk_is_rejected() -> None:
    cfg = SphereBFLControlVolumeConfig(collision_chunk_cells=-1)
    with pytest.raises(ValueError, match="collision_chunk_cells"):
        cfg.validate()


def test_short_sphere_composition_is_finite() -> None:
    cfg = SphereBFLControlVolumeConfig(
        nx=48, ny=32, nz=32, radius=4.0, center_x_fraction=0.35,
        reynolds=20.0, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
    )
    result = run_sphere_bfl_control_volume(cfg)
    measured = result["result"]
    assert measured["finite"] is True
    assert math.isfinite(measured["cd_control_volume"])
    assert math.isfinite(measured["cd_bfl_link"])
    assert measured["drag_stationarity"]["sufficiently_sampled"] is False
    assert result["acceptance"]["admitted"] is False
    assert result["schema"] == "tensorlbm-sphere-bfl-control-volume-v3"
    assert result["configuration"]["collision_model"] == "cumulant_d3q19_cs0"


def test_short_natural_kbc_sphere_composition_is_finite() -> None:
    cfg = SphereBFLControlVolumeConfig(
        nx=48, ny=32, nz=32, radius=4.0, center_x_fraction=0.35,
        reynolds=20.0, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
        collision_model="natural_kbc_d3q19",
        collision_chunk_cells=512,
    )

    result = run_sphere_bfl_control_volume(cfg)

    assert result["result"]["finite"] is True
    assert result["configuration"]["collision_model"] == "natural_kbc_d3q19"
    assert result["result"]["collision_execution"]["collision_calls"] == 128
    assert result["acceptance"]["admitted"] is False


def test_v3_checkpoint_requires_complete_physics_identity(tmp_path) -> None:
    checkpoint = tmp_path / "sphere.ckpt"
    base = SphereBFLControlVolumeConfig(
        nx=48, ny=32, nz=32, radius=4.0, center_x_fraction=0.35,
        reynolds=20.0, lattice_speed=0.04, steps=2, warmup_steps=1,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
        checkpoint_interval=1, checkpoint_path=str(checkpoint),
    )
    run_sphere_bfl_control_volume(base)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["schema"] == "tensorlbm-sphere-checkpoint-v3"

    resumed = run_sphere_bfl_control_volume(
        replace(base, steps=4, resume=True),
    )
    assert resumed["configuration"]["resumed_from_step"] == 2
    with pytest.raises(ValueError, match="configuration"):
        run_sphere_bfl_control_volume(
            replace(base, steps=4, resume=True, sponge_width=4),
        )


def test_v2_checkpoint_migration_is_explicit_and_hashed(tmp_path) -> None:
    checkpoint = tmp_path / "sphere-v2.ckpt"
    base = SphereBFLControlVolumeConfig(
        nx=48, ny=32, nz=32, radius=4.0, center_x_fraction=0.35,
        reynolds=20.0, lattice_speed=0.04, steps=2, warmup_steps=1,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
        checkpoint_interval=1, checkpoint_path=str(checkpoint),
    )
    run_sphere_bfl_control_volume(base)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["schema"] = "tensorlbm-sphere-checkpoint-v2"
    state["configuration"] = dict(state["configuration"])
    state["configuration"]["schema_version"] = 2
    state["configuration"].pop("statistics_window_steps")
    state.pop("migration_provenance")
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="configuration"):
        run_sphere_bfl_control_volume(replace(base, steps=4, resume=True))
    migrated = run_sphere_bfl_control_volume(replace(
        base, steps=4, resume=True, allow_v2_checkpoint=True,
    ))

    provenance = migrated["configuration"]["migration_provenance"]
    assert provenance["source_schema"] == "tensorlbm-sphere-checkpoint-v2"
    assert provenance["source_step"] == 2
    assert len(provenance["source_checkpoint_sha256"]) == 64
