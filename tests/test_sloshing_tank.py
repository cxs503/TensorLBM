"""Tests for the sloshing-tank benchmark."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import (
    SloshingTankConfig,
    faltinsen_natural_frequency,
    make_sloshing_wall_mask,
    run_sloshing_tank,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestFaltinsenNaturalFrequency:
    def test_positive_value(self) -> None:
        omega = faltinsen_natural_frequency(100, 40, 2e-5)
        assert omega > 0.0

    def test_doubling_length_reduces_frequency(self) -> None:
        omega_short = faltinsen_natural_frequency(100, 40, 2e-5)
        omega_long = faltinsen_natural_frequency(200, 40, 2e-5)
        assert omega_long < omega_short


class TestMakeSloshingWallMask:
    def test_shape_dtype_and_sides(self) -> None:
        mask = make_sloshing_wall_mask(12, 20, torch.device("cpu"))
        assert mask.shape == (12, 20)
        assert mask.dtype == torch.bool
        assert mask[0, :].all()
        assert mask[-1, :].all()
        assert mask[:, 0].all()
        assert mask[:, -1].all()


class TestSloshingTankConfig:
    def test_valid_config(self) -> None:
        config = SloshingTankConfig(nx=80, ny=60, water_level=20)
        config.validate()
        assert config.natural_omega > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"nx": 8}, "nx"),
            ({"ny": 8}, "ny"),
            ({"water_level": 0}, "water_level"),
            ({"tau": 0.5}, "tau"),
            ({"rho_water": 0.3, "rho_air": 0.4}, "rho_water"),
            ({"forcing_amp": -1.0}, "forcing_amp"),
        ],
    )
    def test_validate_raises(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            SloshingTankConfig(**kwargs).validate()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        config = SloshingTankConfig(nx=64, run_name="sloshing")
        path = tmp_path / "sloshing.json"
        config.save(path)
        loaded = SloshingTankConfig.load(path)
        assert loaded.nx == config.nx
        assert loaded.run_name == config.run_name


class TestRunSloshingTank:
    def test_smoke(self, tmp_path: Path) -> None:
        config = SloshingTankConfig(
            nx=60,
            ny=40,
            water_level=20,
            n_steps=5,
            output_interval=5,
            output_root=tmp_path / "outputs",
            run_name="smoke",
            overwrite=True,
        )
        run_dir = run_sloshing_tank(config)
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "elevation.csv").exists()
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert metadata["config"]["n_steps"] == 5
        assert "omega_theory" in metadata
