"""Tests for the multiphase sphere/cylinder water-entry benchmark module.

Targets ``tensorlbm.multiphase_water_entry`` which previously had only
~33% coverage because the ``run_multiphase_water_entry`` runner was not
exercised by any test.

Covers:
- ``MultiphaseWaterEntryConfig`` validation, derived properties and run-name.
- ``run_multiphase_water_entry`` 2-D smoke runs (CG and SC models).
- ``run_multiphase_water_entry`` 3-D smoke run (SC two-component).
- Output artifact contents (forces.csv, run_metadata.json, snapshots).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tensorlbm.multiphase_water_entry import (
    MultiphaseWaterEntryConfig,
    run_multiphase_water_entry,
)

# ---------------------------------------------------------------------------
# Config validation and properties
# ---------------------------------------------------------------------------


class TestMultiphaseWaterEntryConfig:
    def _default_kwargs(self) -> dict:
        return {
            "nx": 32,
            "ny": 32,
            "water_level": 12,
            "radius": 4.0,
            "n_steps": 4,
            "output_interval": 4,
        }

    def test_valid_config_does_not_raise(self) -> None:
        MultiphaseWaterEntryConfig(**self._default_kwargs()).validate()

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"mode": "4d"}, "mode"),
            ({"nx": 8}, ">= 16"),
            ({"ny": 8}, ">= 16"),
            ({"radius": 0.0}, "radius"),
            ({"tau": 0.4}, "tau"),
            ({"water_level": 0}, "water_level"),
            ({"water_level": 32}, "water_level"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        kwargs = self._default_kwargs()
        kwargs.update(overrides)
        cfg = MultiphaseWaterEntryConfig(**kwargs)
        with pytest.raises(ValueError, match=match):
            cfg.validate()

    def test_post_init_path_and_device_normalisation(self, tmp_path: Path) -> None:
        cfg = MultiphaseWaterEntryConfig(output_root=str(tmp_path), device="CPU")
        assert isinstance(cfg.output_root, Path)
        assert cfg.output_root == tmp_path
        assert cfg.device == "cpu"

    def test_resolved_run_name_default(self) -> None:
        cfg = MultiphaseWaterEntryConfig(nx=48, ny=32, radius=5.0, G=0.9, g=5e-5, n_steps=100)
        name = cfg.resolved_run_name()
        assert name.startswith("water_entry_2d_nx48_ny32_r5")
        assert "steps100" in name

    def test_resolved_run_name_custom(self) -> None:
        cfg = MultiphaseWaterEntryConfig(run_name="my_run")
        assert cfg.resolved_run_name() == "my_run"

    def test_sphere_center_2d(self) -> None:
        cfg = MultiphaseWaterEntryConfig(
            nx=40, ny=40, water_level=10, clearance=2, radius=3.0,
        )
        cx, cy = cfg.sphere_center_2d
        assert cx == 20.0
        # water_level + clearance + radius + 1 = 10 + 2 + 3 + 1 = 16
        assert cy == 16.0

    def test_sphere_center_3d(self) -> None:
        cfg = MultiphaseWaterEntryConfig(
            mode="3d", nx=40, ny=30, nz=40, water_level=10, clearance=2, radius=3.0,
        )
        cx, cy, cz = cfg.sphere_center_3d
        assert cx == 20.0
        assert cy == 15.0
        assert cz == 16.0


# ---------------------------------------------------------------------------
# 2-D runner smoke tests
# ---------------------------------------------------------------------------


def _read_forces_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestRunMultiphaseWaterEntry2D:
    """Smoke tests for the 2-D water-entry runner."""

    def _smoke_cfg(self, tmp_path: Path, **overrides) -> MultiphaseWaterEntryConfig:
        kwargs = {
            "mode": "2d",
            "model": "cg",
            "nx": 24,
            "ny": 24,
            "radius": 3.0,
            "water_level": 10,
            "clearance": 2,
            "n_steps": 2,
            "output_interval": 2,
            "g": 5e-5,
            "output_root": tmp_path,
            "run_name": "smoke",
            "overwrite": True,
        }
        kwargs.update(overrides)
        return MultiphaseWaterEntryConfig(**kwargs)

    def test_2d_cg_smoke_run(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path, model="cg", run_name="smoke_cg")
        run_dir = run_multiphase_water_entry(cfg)

        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "forces.csv").exists()
        # An output snapshot is written at the final step.
        assert (run_dir / "snapshot_000002.png").exists()

        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["config"]["model"] == "cg"
        assert metadata["config"]["mode"] == "2d"
        assert metadata["config"]["n_steps"] == 2
        assert isinstance(metadata["forces"], list)
        assert metadata["forces"], "forces list should not be empty"
        # One entry per output step.
        assert len(metadata["forces"]) == 1
        entry = metadata["forces"][0]
        for key in ("step", "fx", "fy", "mean_rho_water"):
            assert key in entry

    def test_2d_cg_forces_csv_columns(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path, model="cg", run_name="csv_cg")
        run_dir = run_multiphase_water_entry(cfg)
        rows = _read_forces_csv(run_dir / "forces.csv")
        assert rows, "forces.csv should not be empty"
        assert set(rows[0].keys()) == {"step", "fx", "fy", "mean_rho_water"}

    def test_2d_sc_smoke_run(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path, model="sc", run_name="smoke_sc")
        run_dir = run_multiphase_water_entry(cfg)

        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "forces.csv").exists()
        assert (run_dir / "snapshot_000002.png").exists()

        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["config"]["model"] == "sc"
        assert metadata["forces"], "forces list should not be empty"

    def test_2d_returns_existing_directory(self, tmp_path: Path) -> None:
        """The returned run_dir matches the resolved run-name path."""
        cfg = self._smoke_cfg(tmp_path, run_name="my_dir")
        run_dir = run_multiphase_water_entry(cfg)
        assert run_dir == tmp_path / "sphere_water_entry" / "my_dir"
        assert run_dir.is_dir()


# ---------------------------------------------------------------------------
# 3-D runner smoke test
# ---------------------------------------------------------------------------


class TestRunMultiphaseWaterEntry3D:
    def test_3d_sc_smoke_run(self, tmp_path: Path) -> None:
        cfg = MultiphaseWaterEntryConfig(
            mode="3d",
            nx=16,
            ny=16,
            nz=24,
            radius=2.5,
            water_level=8,
            clearance=2,
            n_steps=2,
            output_interval=2,
            g=5e-5,
            output_root=tmp_path,
            run_name="smoke_3d",
            overwrite=True,
        )
        run_dir = run_multiphase_water_entry(cfg)

        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "forces.csv").exists()
        # The 3-D branch does not write per-step PNG snapshots.
        assert not list(run_dir.glob("snapshot_*.png"))

        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["config"]["mode"] == "3d"
        assert metadata["forces"], "forces list should not be empty"
        entry = metadata["forces"][0]
        # 3-D entries also report fz.
        for key in ("step", "fx", "fy", "fz"):
            assert key in entry

        rows = _read_forces_csv(run_dir / "forces.csv")
        assert "fz" in rows[0]
