"""Tests for the sphere water-entry simulation module.

Covers:
- boundaries3d additions: zou_he_inlet_velocity_z, zou_he_outlet_pressure_z,
  make_tank_wall_mask_3d, apply_water_entry_boundaries_3d
- sphere_water_entry.py: SphereWaterEntryConfig validation, run smoke test
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorlbm import (
    SphereWaterEntryConfig,
    apply_water_entry_boundaries_3d,
    collide_bgk3d,
    equilibrium3d,
    macroscopic3d,
    make_tank_wall_mask_3d,
    run_sphere_water_entry,
    sphere_mask,
    stream3d,
    zou_he_inlet_velocity_z,
    zou_he_outlet_pressure_z,
)


# ---------------------------------------------------------------------------
# Zou/He z-direction boundary conditions
# ---------------------------------------------------------------------------


class TestZouHeInletVelocityZ:
    """Tests for the z=0 upward-flow Zou/He inlet BC."""

    def _make_f(self, nz: int, ny: int, nx: int) -> torch.Tensor:
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f = collide_bgk3d(f, tau=0.6)
        return stream3d(f)

    def test_preserves_shape(self) -> None:
        f = self._make_f(8, 6, 10)
        f_out = zou_he_inlet_velocity_z(f, uz_in=0.05)
        assert f_out.shape == f.shape

    def test_finite_values(self) -> None:
        f = self._make_f(8, 6, 10)
        f_out = zou_he_inlet_velocity_z(f, uz_in=0.05)
        assert torch.isfinite(f_out).all()

    def test_prescribes_uz_at_inlet(self) -> None:
        """After applying the BC, macroscopic uz at z=0 should equal uz_in."""
        nz, ny, nx = 10, 8, 12
        rho0 = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
        f = collide_bgk3d(f, tau=0.6)
        f = stream3d(f)

        uz_in = 0.06
        f_out = zou_he_inlet_velocity_z(f, uz_in=uz_in)
        _, _, _, uz_out = macroscopic3d(f_out)
        assert torch.allclose(
            uz_out[0, :, :], torch.full((ny, nx), uz_in), atol=2e-4
        )

    def test_zero_inlet_velocity(self) -> None:
        """uz_in=0 should not corrupt the distribution."""
        f = self._make_f(8, 6, 10)
        f_out = zou_he_inlet_velocity_z(f, uz_in=0.0)
        assert torch.isfinite(f_out).all()
        assert f_out.shape == f.shape


class TestZouHeOutletPressureZ:
    """Tests for the z=nz-1 pressure outlet BC."""

    def _make_f(self, nz: int, ny: int, nx: int, uz_in: float = 0.05) -> torch.Tensor:
        rho = torch.ones((nz, ny, nx))
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.full_like(rho, uz_in)
        f = equilibrium3d(rho, ux, uy, uz)
        f = collide_bgk3d(f, tau=0.6)
        return stream3d(f)

    def test_preserves_shape(self) -> None:
        f = self._make_f(8, 6, 10)
        f_out = zou_he_outlet_pressure_z(f)
        assert f_out.shape == f.shape

    def test_finite_values(self) -> None:
        f = self._make_f(8, 6, 10)
        f_out = zou_he_outlet_pressure_z(f)
        assert torch.isfinite(f_out).all()

    def test_prescribes_rho_at_outlet(self) -> None:
        """After applying the BC, rho at z=nz-1 should equal rho_out."""
        nz, ny, nx = 10, 8, 12
        f = self._make_f(nz, ny, nx)
        rho_out = 1.0
        f_out = zou_he_outlet_pressure_z(f, rho_out=rho_out)
        rho_field, _, _, _ = macroscopic3d(f_out)
        assert torch.allclose(
            rho_field[-1, :, :], torch.full((ny, nx), rho_out), atol=1e-4
        )


# ---------------------------------------------------------------------------
# make_tank_wall_mask_3d
# ---------------------------------------------------------------------------


class TestMakeTankWallMask3d:
    def test_shape_and_dtype(self) -> None:
        nz, ny, nx = 10, 8, 12
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask = make_tank_wall_mask_3d(nz, ny, nx, obstacle, device=torch.device("cpu"))
        assert mask.shape == (nz, ny, nx)
        assert mask.dtype == torch.bool

    def test_lateral_walls_marked(self) -> None:
        nz, ny, nx = 10, 8, 12
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask = make_tank_wall_mask_3d(nz, ny, nx, obstacle, device=torch.device("cpu"))
        assert mask[:, :, 0].all(), "x=0 wall should be True"
        assert mask[:, :, -1].all(), "x=nx-1 wall should be True"
        assert mask[:, 0, :].all(), "y=0 wall should be True"
        assert mask[:, -1, :].all(), "y=ny-1 wall should be True"

    def test_z_faces_not_marked(self) -> None:
        """z=0 and z=nz-1 faces (inlet/outlet) must not be marked as walls."""
        nz, ny, nx = 10, 8, 12
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask = make_tank_wall_mask_3d(nz, ny, nx, obstacle, device=torch.device("cpu"))
        # Interior x and y cells at z=0 and z=nz-1 should not be walls
        assert not mask[0, 1:-1, 1:-1].any(), "Interior of z=0 face should not be a wall"
        assert not mask[-1, 1:-1, 1:-1].any(), "Interior of z=nz-1 face should not be a wall"

    def test_obstacle_excluded_from_wall(self) -> None:
        nz, ny, nx = 10, 8, 12
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        obstacle[5, 0, :] = True  # force overlap with y=0 wall
        mask = make_tank_wall_mask_3d(nz, ny, nx, obstacle, device=torch.device("cpu"))
        # Obstacle cells should not be in wall mask
        assert not mask[obstacle].any()


# ---------------------------------------------------------------------------
# apply_water_entry_boundaries_3d
# ---------------------------------------------------------------------------


class TestApplyWaterEntryBoundaries3d:
    def test_preserves_shape_and_finite(self) -> None:
        nz, ny, nx = 12, 10, 10
        obstacle = sphere_mask(nx, ny, nz, nx // 2, ny // 2, nz // 2, 2.0,
                               device=torch.device("cpu"))
        wall_mask = make_tank_wall_mask_3d(nz, ny, nx, obstacle, device=torch.device("cpu"))
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = apply_water_entry_boundaries_3d(f, v_entry=0.05,
                                                wall_mask=wall_mask,
                                                obstacle_mask=obstacle)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# SphereWaterEntryConfig validation
# ---------------------------------------------------------------------------


class TestSphereWaterEntryConfig:
    def _default_kwargs(self) -> dict:
        return dict(
            nx=32, ny=32, nz=64,
            radius=4.0, sphere_z_frac=0.5,
            v_entry=0.05, re=100.0,
            n_steps=10, output_interval=5,
        )

    def test_valid_config_does_not_raise(self) -> None:
        cfg = SphereWaterEntryConfig(**self._default_kwargs())
        cfg.validate()  # should not raise

    def test_nu_and_tau(self) -> None:
        cfg = SphereWaterEntryConfig(**self._default_kwargs())
        assert cfg.nu == pytest.approx(0.05 * 8.0 / 100.0, rel=1e-6)
        assert cfg.tau > 0.5

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"nx": 4}, "at least 16"),
            ({"ny": 4}, "at least 16"),
            ({"nz": 4}, "at least 16"),
            ({"n_steps": 0}, "n_steps"),
            ({"output_interval": 0}, "output_interval"),
            ({"v_entry": -0.01}, "v_entry"),
            ({"re": 0.0}, "v_entry"),
            ({"radius": 0.0}, "v_entry"),
            ({"sphere_z_frac": 0.0}, "sphere_z_frac"),
            ({"sphere_z_frac": 1.0}, "sphere_z_frac"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        kwargs = self._default_kwargs()
        kwargs.update(overrides)
        cfg = SphereWaterEntryConfig(**kwargs)
        with pytest.raises(ValueError, match=match):
            cfg.validate()

    def test_tau_too_small_raises(self) -> None:
        cfg = SphereWaterEntryConfig(v_entry=1e-9, re=1e12, radius=4.0)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_resolved_run_name_contains_key_params(self) -> None:
        cfg = SphereWaterEntryConfig(nx=32, ny=32, nz=64, re=100.0, v_entry=0.05, n_steps=50)
        name = cfg.resolved_run_name()
        assert "nx32" in name
        assert "re100" in name
        assert "r" in name

    def test_sphere_too_close_to_inlet_raises(self) -> None:
        # sphere_z_frac=0.01 → cz ≈ 0.64, cz-r < 2 for radius=4
        cfg = SphereWaterEntryConfig(nx=32, ny=32, nz=64, radius=4.0, sphere_z_frac=0.01)
        with pytest.raises(ValueError, match="clearance"):
            cfg.validate()


# ---------------------------------------------------------------------------
# run_sphere_water_entry smoke test
# ---------------------------------------------------------------------------


class TestRunSphereWaterEntry:
    def test_smoke_run(self, tmp_path: Path) -> None:
        """A minimal smoke test: 5 steps, checks that output files exist."""
        cfg = SphereWaterEntryConfig(
            nx=24, ny=24, nz=48,
            radius=3.0, sphere_z_frac=0.5,
            v_entry=0.05, re=60.0,
            n_ramp=2,
            n_steps=5, output_interval=5,
            output_root=tmp_path,
            run_name="smoke",
            overwrite=True,
        )
        run_dir = run_sphere_water_entry(cfg)

        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "forces.csv").exists()
        assert (run_dir / "force_history.png").exists()
        assert (run_dir / "flow_step_000005.png").exists()

        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["config"]["n_steps"] == 5
        assert len(metadata["diagnostics"]) == 1

    def test_smoke_run_with_smagorinsky(self, tmp_path: Path) -> None:
        """Smoke test with Smagorinsky LES enabled."""
        cfg = SphereWaterEntryConfig(
            nx=24, ny=24, nz=48,
            radius=3.0, sphere_z_frac=0.5,
            v_entry=0.05, re=60.0, smagorinsky_cs=0.1,
            n_steps=4, output_interval=4,
            output_root=tmp_path,
            run_name="smoke_les",
            overwrite=True,
        )
        run_dir = run_sphere_water_entry(cfg)
        assert (run_dir / "forces.csv").exists()

    def test_forces_csv_has_correct_columns(self, tmp_path: Path) -> None:
        import csv as _csv
        cfg = SphereWaterEntryConfig(
            nx=24, ny=24, nz=48,
            radius=3.0, sphere_z_frac=0.5,
            v_entry=0.05, re=60.0,
            n_steps=3, output_interval=10,
            output_root=tmp_path, run_name="csv_check", overwrite=True,
        )
        run_dir = run_sphere_water_entry(cfg)
        with (run_dir / "forces.csv").open(encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            rows = list(reader)
        assert rows, "forces.csv should not be empty"
        assert "step" in rows[0]
        assert "fz" in rows[0]
        assert "cd" in rows[0]
        assert len(rows) == 3  # one row per step


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_sphere_water_entry_cli_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "outputs"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    cmd = [
        sys.executable,
        str(repo_root / "examples" / "sphere_water_entry.py"),
        "--nx", "24", "--ny", "24", "--nz", "48",
        "--radius", "3", "--v-entry", "0.05", "--re", "60",
        "--n-steps", "8", "--output-interval", "4",
        "--output-root", str(output_root),
        "--run-name", "smoke",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))

    run_dir = output_root / "sphere_water_entry" / "smoke"
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "forces.csv").exists()
    assert (run_dir / "flow_step_000008.png").exists()

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["n_steps"] == 8
    assert metadata["diagnostics"]
