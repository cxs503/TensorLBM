"""Sub-grid-model switch on the wall-resolved 3D channel runner.

The campaign's C6 negative control (``Re_tau = 395`` without SGS must diverge or
err by more than 30 %) needs an SGS-on/off pair at fixed grid.  Until now the
production channel runner exposed no SGS switch at all, so the control could not
even be posed.  These tests pin the switch: molecular-only by default
(bit-identical behaviour), Smagorinsky only where the kernel supports it, and a
numerically live path when it is switched on.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.boundary.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)


def _config(tmp_path, *, smagorinsky_cs: float, collision_model: str = "cumulant"):
    return WallResolvedChannel3DConfig(
        nx=16,
        ny=20,
        nz=12,
        re_tau=180.0,
        u_tau=0.005,
        steps=40,
        warmup_steps=20,
        sample_interval=10,
        report_interval=40,
        checkpoint_interval=10**6,
        collision_model=collision_model,
        smagorinsky_cs=smagorinsky_cs,
        compile_natural_kbc=False,
        initialization_mode="spectral_solenoidal",
        device="cpu",
        output=tmp_path / "channel.json",
        checkpoint=tmp_path / "channel.ckpt",
    )


def test_default_config_is_molecular_only() -> None:
    config = WallResolvedChannel3DConfig()
    config.validate()
    assert config.smagorinsky_cs == 0.0


def test_sgs_requires_the_cumulant_kernel(tmp_path) -> None:
    with pytest.raises(ValueError, match="smagorinsky_cs > 0 requires"):
        _config(tmp_path, smagorinsky_cs=0.1, collision_model="natural_kbc").validate()
    # Zero stays valid for the natural-KBC path (unchanged production default).
    _config(tmp_path, smagorinsky_cs=0.0, collision_model="natural_kbc").validate()


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_sgs_strength_is_validated(tmp_path, value) -> None:
    with pytest.raises(ValueError, match="smagorinsky_cs must lie in"):
        _config(tmp_path, smagorinsky_cs=value).validate()


def test_sgs_run_is_finite_and_recorded(tmp_path) -> None:
    torch.set_num_threads(4)
    results = {}
    for cs in (0.0, 0.1):
        config = _config(tmp_path, smagorinsky_cs=cs)
        result = run_wall_resolved_channel3d(config)
        assert result["acceptance"]["population_health"] is True
        assert result["derived"]["smagorinsky_cs"] == pytest.approx(cs)
        assert result["reports"][-1]["minimum_population"] > 0.0
        results[cs] = result["reports"][-1]["measured_wall_shear"]
    # The SGS closure must actually enter the solver: same grid, same step count,
    # and a measurably different wall stress.
    assert results[0.1] != pytest.approx(results[0.0], rel=1.0e-6)