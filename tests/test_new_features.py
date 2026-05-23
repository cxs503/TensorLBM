"""Tests for post-processing utilities, checkpoint, config I/O, and CFD benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

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
    compute_pressure,
    compute_q_criterion,
    compute_recirculation_length,
    compute_vorticity_3d,
    extract_wake_profile,
)

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
        profile = extract_wake_profile(ux, x_probe=10)
        assert profile.shape == (ny,)

    def test_3d_returns_mid_z_slice(self) -> None:
        nz, ny, nx = 8, 10, 20
        ux = torch.rand((nz, ny, nx))
        profile = extract_wake_profile(ux, x_probe=10)
        assert profile.shape == (ny,)
        assert torch.equal(profile, ux[nz // 2, :, 10])


# ---------------------------------------------------------------------------
# Post-processing: compute_recirculation_length
# ---------------------------------------------------------------------------

class TestComputeRecirculationLength:
    def test_no_negative_gives_zero(self) -> None:
        ny, nx = 10, 20
        ux = torch.full((ny, nx), 0.05)
        length = compute_recirculation_length(ux, x_start=5)
        assert length == 0.0

    def test_detects_recirculation_region(self) -> None:
        ny, nx = 10, 20
        ux = torch.full((ny, nx), 0.05)
        ux[5, 5:12] = -0.01  # negative region from x=5 to x=11
        length = compute_recirculation_length(ux, x_start=0, y_mid=5)
        assert length > 0.0


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
        # Uniform flow → no strain, no rotation → Q = 0
        interior = Q[1:-1, 1:-1, 1:-1]
        assert float(interior.abs().max()) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Post-processing: compute_pressure
# ---------------------------------------------------------------------------

def test_compute_pressure_scalar() -> None:
    rho = torch.full((4, 5, 6), 1.0)
    p = compute_pressure(rho)
    expected = 1.0 / 3.0
    assert torch.allclose(p, torch.full_like(p, expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Checkpoint: save and load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_round_trip(self, tmp_path: Path) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        step = 42
        path = tmp_path / "ckpt.npz"
        save_checkpoint(path, f, step)
        f_loaded, step_loaded = load_checkpoint(path)
        assert step_loaded == step
        assert torch.allclose(f_loaded, f, atol=1e-6)

    def test_loaded_tensor_on_device(self, tmp_path: Path) -> None:
        f = torch.rand(9, 4, 5)
        path = tmp_path / "ckpt.npz"
        save_checkpoint(path, f, step=10)
        f_loaded, _ = load_checkpoint(path, device="cpu")
        assert f_loaded.device.type == "cpu"


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
        cfg = BackwardFacingStepConfig(nx=100, ny=30, step_height=8, re=50.0, n_steps=5)
        path = tmp_path / "bfs_cfg.json"
        cfg.save(path)
        cfg2 = BackwardFacingStepConfig.load(path)
        assert cfg2.step_height == cfg.step_height
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
        cfg = LidDrivenCavityConfig(n=32, re=100.0, n_steps=10)
        cfg.validate()

    def test_ghia_tables_present(self) -> None:
        for table in (GHIA_RE100, GHIA_RE400, GHIA_RE1000):
            assert "y" in table
            assert "u" in table
            assert len(table["y"]) == len(table["u"])

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"n": 2}, "n must be"),
            ({"n_steps": 0}, "n_steps"),
            ({"u_lid": -0.1}, "u_lid"),
            ({"re": 0.0}, "u_lid"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        base = dict(n=32, re=100.0, n_steps=10, u_lid=0.1)
        base.update(overrides)
        cfg = LidDrivenCavityConfig(**base)
        with pytest.raises(ValueError, match=match):
            cfg.validate()

    def test_config_round_trip(self, tmp_path: Path) -> None:
        cfg = LidDrivenCavityConfig(n=64, re=400.0, n_steps=50)
        path = tmp_path / "lid_cfg.json"
        cfg.save(path)
        cfg2 = LidDrivenCavityConfig.load(path)
        assert cfg2.n == cfg.n
        assert cfg2.re == cfg.re


# ---------------------------------------------------------------------------
# BackwardFacingStepConfig
# ---------------------------------------------------------------------------

class TestBackwardFacingStepConfig:
    def test_valid_config(self) -> None:
        cfg = BackwardFacingStepConfig(nx=100, ny=30, step_height=8, n_steps=10)
        cfg.validate()

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"nx": 10}, "nx >= 32"),
            ({"ny": 4}, "ny >= 8"),  # ny=4 → step_height=10 >= ny//2
            ({"step_height": 0}, "step_height"),
            ({"n_steps": 0}, "n_steps"),
            ({"u_in": -0.01}, "u_in"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        base = dict(nx=100, ny=30, step_height=8, n_steps=10, u_in=0.05, re=50.0)
        base.update(overrides)
        cfg = BackwardFacingStepConfig(**base)
        with pytest.raises(ValueError, match=match):
            cfg.validate()
