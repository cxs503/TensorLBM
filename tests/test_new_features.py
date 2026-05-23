"""Tests for post-processing utilities, checkpoint, config I/O, and CFD benchmarks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import (
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    BackwardFacingStepConfig,
    LidDrivenCavityConfig,
    equilibrium3d,
    load_checkpoint,
    load_config_json,
    save_checkpoint,
    save_config_json,
)
from tensorlbm.postprocess import (
    compute_pressure_coefficient,
    compute_q_criterion,
    compute_recirculation_length,
    compute_vorticity_3d,
    extract_wake_profile,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Post-processing: compute_vorticity_3d
# ---------------------------------------------------------------------------

class TestComputeVorticity3D:
    def test_output_shape(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        wx, wy, wz = compute_vorticity_3d(ux, uy, uz)
        assert wx.shape == (nz, ny, nx)
        assert wy.shape == (nz, ny, nx)
        assert wz.shape == (nz, ny, nx)

    def test_zero_velocity_gives_zero_vorticity(self) -> None:
        nz, ny, nx = 6, 8, 10
        zeros = torch.zeros((nz, ny, nx))
        wx, wy, wz = compute_vorticity_3d(zeros, zeros, zeros)
        assert float(wx.abs().max()) == pytest.approx(0.0)
        assert float(wy.abs().max()) == pytest.approx(0.0)
        assert float(wz.abs().max()) == pytest.approx(0.0)

    def test_finite_values(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        wx, wy, wz = compute_vorticity_3d(ux, uy, uz)
        assert torch.isfinite(wx).all()
        assert torch.isfinite(wy).all()
        assert torch.isfinite(wz).all()


# ---------------------------------------------------------------------------
# Post-processing: extract_wake_profile
# ---------------------------------------------------------------------------

class TestExtractWakeProfile:
    def test_2d_output_shape(self) -> None:
        ny, nx = 10, 20
        ux = torch.rand((ny, nx))
        profile = extract_wake_profile(ux, x_wake=10)
        assert profile.shape == (ny,)

    def test_3d_returns_mid_z_slice(self) -> None:
        nz, ny, nx = 8, 10, 20
        ux = torch.rand((nz, ny, nx))
        profile = extract_wake_profile(ux, x_wake=10)
        assert profile.shape == (ny,)


# ---------------------------------------------------------------------------
# Post-processing: compute_recirculation_length
# ---------------------------------------------------------------------------

class TestComputeRecirculationLength:
    def test_no_recirculation_gives_zero(self) -> None:
        ny, nx = 10, 20
        ux = torch.full((ny, nx), 0.05)
        # obstacle mask: all False (no obstacle)
        obstacle = torch.zeros((ny, nx), dtype=torch.bool)
        length = compute_recirculation_length(ux, obstacle)
        assert length == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Post-processing: compute_q_criterion
# ---------------------------------------------------------------------------

class TestComputeQCriterion:
    def test_output_shape(self) -> None:
        nz, ny, nx = 4, 5, 6
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        Q = compute_q_criterion(ux, uy, uz)
        assert Q.shape == (nz, ny, nx)
        assert torch.isfinite(Q).all()

    def test_uniform_flow_gives_zero(self) -> None:
        nz, ny, nx = 4, 5, 6
        ux = torch.full((nz, ny, nx), 0.05)
        zeros = torch.zeros((nz, ny, nx))
        Q = compute_q_criterion(ux, zeros, zeros)
        # Uniform flow → no strain, no rotation → Q = 0 in interior
        interior = Q[1:-1, 1:-1, 1:-1]
        assert float(interior.abs().max()) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Post-processing: compute_pressure_coefficient
# ---------------------------------------------------------------------------

def test_compute_pressure_coefficient_scalar() -> None:
    rho = torch.full((4, 5, 6), 1.0)
    # At reference density with non-zero u_in, Cp should be 0
    cp = compute_pressure_coefficient(rho, u_in=0.1, rho_ref=1.0)
    assert torch.allclose(cp, torch.zeros_like(cp), atol=1e-6)


# ---------------------------------------------------------------------------
# Checkpoint: save and load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_round_trip(self, tmp_path: Path) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        step = 42
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_checkpoint(f, step, run_dir)
        f_loaded, step_loaded, _ = load_checkpoint(run_dir)
        assert step_loaded == step
        assert torch.allclose(f_loaded, f, atol=1e-6)

    def test_checkpoint_creates_file(self, tmp_path: Path) -> None:
        f = torch.rand(9, 4, 5)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        ckpt_path = save_checkpoint(f, 10, run_dir)
        assert ckpt_path.exists()


# ---------------------------------------------------------------------------
# Config I/O: save_config_json / load_config_json
# ---------------------------------------------------------------------------

class TestConfigIO:
    def test_cylinder_flow_config_round_trip(self, tmp_path: Path) -> None:
        from tensorlbm import CylinderFlowConfig
        cfg = CylinderFlowConfig(nx=64, ny=32, re=100.0, n_steps=20, run_name="test_io")
        path = tmp_path / "cfg.json"
        cfg.save(path)
        cfg2 = CylinderFlowConfig.load(path)
        assert cfg2.nx == cfg.nx
        assert cfg2.ny == cfg.ny
        assert cfg2.re == cfg.re
        assert cfg2.run_name == cfg.run_name

    def test_sphere_flow_config_round_trip(self, tmp_path: Path) -> None:
        from tensorlbm import SphereFlowConfig
        cfg = SphereFlowConfig(nx=40, ny=20, nz=20, re=30.0, n_steps=5)
        path = tmp_path / "sphere_cfg.json"
        save_config_json(cfg, path)
        cfg2 = load_config_json(SphereFlowConfig, path)
        assert cfg2.nx == cfg.nx
        assert cfg2.re == cfg.re

    def test_backward_facing_step_round_trip(self, tmp_path: Path) -> None:
        cfg = BackwardFacingStepConfig(nx=100, ny=30, step_h=8, re=50.0, n_steps=5)
        path = tmp_path / "bfs_cfg.json"
        cfg.save(path)
        cfg2 = BackwardFacingStepConfig.load(path)
        assert cfg2.step_h == cfg.step_h
        assert cfg2.re == cfg.re

    def test_json_is_valid(self, tmp_path: Path) -> None:
        from tensorlbm import CylinderFlowConfig
        cfg = CylinderFlowConfig(nx=32, re=50.0)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        data = json.loads(path.read_text())
        assert data["nx"] == 32
        assert data["re"] == 50.0


# ---------------------------------------------------------------------------
# LidDrivenCavityConfig
# ---------------------------------------------------------------------------

class TestLidDrivenCavityConfig:
    def test_valid_config(self) -> None:
        cfg = LidDrivenCavityConfig(nx=32, re=100.0, n_steps=10)
        cfg.validate()

    def test_ghia_tables_present(self) -> None:
        for table in (GHIA_RE100, GHIA_RE400, GHIA_RE1000):
            assert "y" in table
            assert "u" in table
            assert len(table["y"]) == len(table["u"])

    def test_config_round_trip(self, tmp_path: Path) -> None:
        cfg = LidDrivenCavityConfig(nx=64, re=400.0, n_steps=50)
        path = tmp_path / "lid_cfg.json"
        cfg.save(path)
        cfg2 = LidDrivenCavityConfig.load(path)
        assert cfg2.nx == cfg.nx
        assert cfg2.re == cfg.re


# ---------------------------------------------------------------------------
# BackwardFacingStepConfig
# ---------------------------------------------------------------------------

class TestBackwardFacingStepConfig:
    def test_valid_config(self) -> None:
        cfg = BackwardFacingStepConfig(nx=200, ny=60, step_h=20, n_steps=10)
        cfg.validate()

    def test_config_round_trip(self, tmp_path: Path) -> None:
        cfg = BackwardFacingStepConfig(nx=200, ny=60, step_h=20, n_steps=10)
        path = tmp_path / "bfs_cfg.json"
        cfg.save(path)
        cfg2 = BackwardFacingStepConfig.load(path)
        assert cfg2.step_h == cfg.step_h
        assert cfg2.re == cfg.re
