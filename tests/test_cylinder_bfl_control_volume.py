from __future__ import annotations

import math

from tensorlbm.cylinder_bfl_control_volume import (
    CYLINDER_RE100_CD_REFERENCE,
    CylinderBFLControlVolumeConfig,
    run_cylinder_bfl_control_volume,
)


def test_cylinder_reference_and_tau() -> None:
    cfg = CylinderBFLControlVolumeConfig(radius=10, reynolds=100, lattice_speed=0.05)
    assert CYLINDER_RE100_CD_REFERENCE == 1.33
    assert math.isclose(cfg.tau, 0.53)


def test_short_periodic_cylinder_composition_is_finite() -> None:
    cfg = CylinderBFLControlVolumeConfig(
        nx=56, ny=40, nz=3, radius=4, center_x_fraction=0.35,
        reynolds=20, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
    )
    result = run_cylinder_bfl_control_volume(cfg)["result"]
    assert result["finite"] is True
    assert math.isfinite(result["cd_control_volume"])
    assert math.isfinite(result["cd_bfl_link"])
