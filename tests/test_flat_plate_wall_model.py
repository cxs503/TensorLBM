from __future__ import annotations

import math
from pathlib import Path

import pytest

from tensorlbm.flat_plate_wall_model import (
    FlatPlateWallModelConfig,
    ittc_1957_friction_coefficient,
    run_flat_plate_wall_model,
)


def test_flat_plate_transport_scaling_and_reference() -> None:
    cfg = FlatPlateWallModelConfig(plate_length=100, reynolds=1e6, resolved_reynolds=1e5)
    assert cfg.wall_nu == pytest.approx(6e-6)
    assert cfg.tau == pytest.approx(0.50018)
    assert ittc_1957_friction_coefficient(1e6) == pytest.approx(0.0046875)


def test_short_flat_plate_composition_is_finite() -> None:
    cfg = FlatPlateWallModelConfig(
        nx=64, ny=32, nz=3, plate_length=24,
        plate_start_fraction=0.25, reynolds=2e4,
        resolved_reynolds=2e3, lattice_speed=0.04,
        steps=4, warmup_steps=2, ramp_steps=2,
        sponge_width=3, cv_margin=3, device="cpu",
        stress_exchange_distance=2.0,
        wall_diagnostic_interval=1,
    )
    result = run_flat_plate_wall_model(cfg)["result"]
    assert result["finite"] is True
    assert math.isfinite(result["friction_coefficient"])
    assert result["drag_stationarity"]["sufficiently_sampled"] is False
    assert math.isfinite(result["maximum_positivity_limited_fraction"])
    applicability = result["wall_stress_applicability"]
    assert applicability["samples"] == 2
    assert applicability["y_plus_min"] > 0.0
    assert applicability["y_plus_max"] >= applicability["y_plus_min"]
    assert applicability["maximum_rejected_fraction"] == pytest.approx(0.0)


def test_flat_plate_rejects_nonpositive_exchange_distance() -> None:
    with pytest.raises(ValueError, match="stress_exchange_distance"):
        FlatPlateWallModelConfig(stress_exchange_distance=0.0).validate()


def test_flat_plate_checkpoint_can_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flat.ckpt"
    common = dict(
        nx=64, ny=32, nz=3, plate_length=24,
        plate_start_fraction=0.25, reynolds=2e4,
        resolved_reynolds=2e3, lattice_speed=0.04,
        warmup_steps=2, ramp_steps=2, sponge_width=3,
        cv_margin=3, report_interval=0, checkpoint_interval=2,
        checkpoint_path=str(checkpoint), device="cpu",
    )
    run_flat_plate_wall_model(FlatPlateWallModelConfig(**common, steps=4))
    resumed = run_flat_plate_wall_model(FlatPlateWallModelConfig(
        **common, steps=6, resume=True,
    ))
    assert checkpoint.exists()
    assert resumed["configuration"]["resumed_from_step"] == 4
    assert resumed["result"]["finite"] is True
