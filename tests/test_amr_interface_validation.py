from __future__ import annotations

import math

from tensorlbm.amr_interface_validation import (
    AMRInterfaceValidationConfig,
    run_amr_interface_validation,
)
from tensorlbm.refinement import BoxRegion


def test_short_uniform_fine_interface_comparison_is_finite() -> None:
    result = run_amr_interface_validation(AMRInterfaceValidationConfig(
        shape_zyx=(12, 14, 24),
        box=BoxRegion(x0=8, x1=16, y0=4, y1=10, z0=3, z1=9),
        pulse_radius=1.5,
        steps=3,
    ))
    metrics = result["result"]
    assert metrics["finite"] is True
    assert metrics["minimum_population"] > 0.0
    assert math.isfinite(metrics["density_rms_refined_amr"])
    assert metrics["relative_mass_drift"] < 1e-5
    assert result["mesh"]["saving_fraction"] > 0.0
    assert math.isfinite(metrics["refined_to_coarse_density_error_ratio"])
    assert math.isfinite(metrics["refined_to_coarse_velocity_x_error_ratio"])


def test_amr_interface_validation_is_public() -> None:
    import tensorlbm

    assert tensorlbm.AMRInterfaceValidationConfig is AMRInterfaceValidationConfig
    assert tensorlbm.run_amr_interface_validation is run_amr_interface_validation
