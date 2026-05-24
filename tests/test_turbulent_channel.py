"""Tests for the turbulent-channel benchmark."""
from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from tensorlbm import (
    TurbulentChannelConfig,
    log_law_velocity,
    run_turbulent_channel,
    viscous_sublayer_velocity,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLogLawVelocity:
    def test_monotonic_in_y_plus(self) -> None:
        assert log_law_velocity(30.0) > log_law_velocity(10.0)

    def test_matches_formula(self) -> None:
        expected = (1.0 / 0.41) * math.log(30.0) + 5.2
        assert math.isclose(log_law_velocity(30.0), expected)
        assert viscous_sublayer_velocity(3.0) == 3.0


class TestTurbulentChannelConfig:
    def test_valid_config_and_properties(self) -> None:
        config = TurbulentChannelConfig(nx=32, ny=20, re_tau=50.0, u_tau=0.01)
        config.validate()
        assert config.H == 18
        assert config.nu > 0.0
        assert config.tau > 0.5
        assert config.body_force > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"re_tau": 0.0}, "re_tau"),
            ({"u_tau": 0.0}, "u_tau"),
            ({"smagorinsky_cs": 0.5}, "smagorinsky_cs"),
            ({"averaging_start": 10, "n_steps": 10}, "averaging_start"),
        ],
    )
    def test_validate_raises(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            TurbulentChannelConfig(**kwargs).validate()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        config = TurbulentChannelConfig(nx=48, run_name="channel")
        path = tmp_path / "channel.json"
        config.save(path)
        loaded = TurbulentChannelConfig.load(path)
        assert loaded.nx == config.nx
        assert loaded.run_name == config.run_name


class TestRunTurbulentChannel:
    def test_smoke(self, tmp_path: Path) -> None:
        config = TurbulentChannelConfig(
            nx=32,
            ny=20,
            n_steps=10,
            averaging_start=5,
            output_interval=5,
            output_root=tmp_path / "outputs",
            run_name="smoke",
            overwrite=True,
        )
        run_dir = run_turbulent_channel(config)
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "velocity_profile.csv").exists()
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert metadata["config"]["n_steps"] == 10
        assert metadata["averaging_samples"] > 0
