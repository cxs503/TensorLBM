from __future__ import annotations

import math

import pytest

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
