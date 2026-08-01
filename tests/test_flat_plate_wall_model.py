from __future__ import annotations

import math

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
    )
    result = run_flat_plate_wall_model(cfg)["result"]
    assert result["finite"] is True
    assert math.isfinite(result["friction_coefficient"])
    assert result["drag_stationarity"]["sufficiently_sampled"] is False
    assert math.isfinite(result["maximum_positivity_limited_fraction"])
