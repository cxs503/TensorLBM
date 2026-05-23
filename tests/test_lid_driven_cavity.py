"""Tests for the Lid-Driven Cavity benchmark.

Covers:
* zou_he_moving_lid – BC correctness
* make_cavity_wall_mask – geometry
* LidDrivenCavityConfig – validation, properties, round-trip
* run_lid_driven_cavity – smoke test (small domain, few steps)
* compare_ghia – interpolation and error calculation
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import (
    GHIA_RE100,
    LidDrivenCavityConfig,
    compare_ghia,
    equilibrium,
    macroscopic,
    run_lid_driven_cavity,
)
from tensorlbm.lid_driven_cavity import make_cavity_wall_mask, zou_he_moving_lid

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# zou_he_moving_lid
# ---------------------------------------------------------------------------


class TestZouHeMovingLid:
    """Unit tests for the moving-lid Zou/He boundary condition."""

    def test_preserves_shape(self) -> None:
        ny, nx = 12, 16
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = zou_he_moving_lid(f, u_lid=0.1)
        assert f_out.shape == (9, ny, nx)

    def test_finite_values(self) -> None:
        ny, nx = 12, 16
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = zou_he_moving_lid(f, u_lid=0.1)
        assert torch.isfinite(f_out).all()

    def test_prescribes_ux_at_lid(self) -> None:
        """After applying the BC, interior top-wall cells should have ux ≈ u_lid."""
        ny, nx = 16, 24
        u_lid = 0.1
        rho0 = torch.ones((ny, nx))
        f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        # Run several times so the BC is in a realistic state
        from tensorlbm import bounce_back_cells, collide_bgk, stream
        from tensorlbm.lid_driven_cavity import make_cavity_wall_mask

        wall = make_cavity_wall_mask(ny, nx, torch.device("cpu"))
        for _ in range(50):
            f = collide_bgk(f, tau=0.8)
            f = stream(f)
            f = bounce_back_cells(f, wall)
            f = zou_he_moving_lid(f, u_lid)

        _, ux, _ = macroscopic(f)
        # Interior top-wall cells (x=1 … nx-2) should have ux close to u_lid
        ux_lid = ux[-1, 1:-1]
        assert torch.allclose(ux_lid, torch.full_like(ux_lid, u_lid), atol=1e-3), (
            f"ux at lid not close to u_lid: mean={ux_lid.mean().item():.4f}"
        )

    def test_uy_near_zero_at_lid(self) -> None:
        """uy should remain near zero at the interior lid cells."""
        ny, nx = 16, 24
        u_lid = 0.1
        rho0 = torch.ones((ny, nx))
        f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        from tensorlbm import bounce_back_cells, collide_bgk, stream
        from tensorlbm.lid_driven_cavity import make_cavity_wall_mask

        wall = make_cavity_wall_mask(ny, nx, torch.device("cpu"))
        for _ in range(50):
            f = collide_bgk(f, tau=0.8)
            f = stream(f)
            f = bounce_back_cells(f, wall)
            f = zou_he_moving_lid(f, u_lid)

        _, _, uy = macroscopic(f)
        uy_lid = uy[-1, 1:-1]
        assert torch.allclose(uy_lid, torch.zeros_like(uy_lid), atol=1e-3)

    def test_zero_lid_velocity_gives_no_slip(self) -> None:
        """u_lid=0 should give zero velocity at the top wall (stationary wall)."""
        ny, nx = 16, 24
        rho0 = torch.ones((ny, nx))
        ux0 = torch.full_like(rho0, 0.05)
        f = equilibrium(rho0, ux0, torch.zeros_like(rho0))

        from tensorlbm import bounce_back_cells, collide_bgk, stream
        from tensorlbm.lid_driven_cavity import make_cavity_wall_mask

        wall = make_cavity_wall_mask(ny, nx, torch.device("cpu"))
        for _ in range(100):
            f = collide_bgk(f, tau=0.8)
            f = stream(f)
            f = bounce_back_cells(f, wall)
            f = zou_he_moving_lid(f, 0.0)

        _, ux, _ = macroscopic(f)
        assert torch.allclose(ux[-1, 1:-1], torch.zeros(nx - 2), atol=2e-3)


# ---------------------------------------------------------------------------
# make_cavity_wall_mask
# ---------------------------------------------------------------------------


class TestMakeCavityWallMask:
    def test_corners_are_solid(self) -> None:
        ny, nx = 20, 30
        mask = make_cavity_wall_mask(ny, nx, torch.device("cpu"))
        assert mask[0, 0].item()
        assert mask[0, -1].item()
        assert mask[-1, 0].item()
        assert mask[-1, -1].item()

    def test_walls_fully_marked(self) -> None:
        ny, nx = 20, 30
        mask = make_cavity_wall_mask(ny, nx, torch.device("cpu"))
        assert mask[0, :].all()
        assert mask[-1, :].all()
        assert mask[:, 0].all()
        assert mask[:, -1].all()

    def test_interior_is_fluid(self) -> None:
        ny, nx = 20, 30
        mask = make_cavity_wall_mask(ny, nx, torch.device("cpu"))
        interior = mask[1:-1, 1:-1]
        assert not interior.any()


# ---------------------------------------------------------------------------
# LidDrivenCavityConfig
# ---------------------------------------------------------------------------


class TestLidDrivenCavityConfig:
    def test_defaults(self) -> None:
        cfg = LidDrivenCavityConfig()
        assert cfg.nx == 128
        assert cfg.re == 100.0

    def test_ny_equals_nx(self) -> None:
        cfg = LidDrivenCavityConfig(nx=64)
        assert cfg.ny == 64

    def test_nu_property(self) -> None:
        cfg = LidDrivenCavityConfig(nx=100, u_lid=0.1, re=100.0)
        assert abs(cfg.nu - 0.1 * 100 / 100.0) < 1e-10

    def test_tau_property(self) -> None:
        cfg = LidDrivenCavityConfig(nx=100, u_lid=0.1, re=100.0)
        expected_tau = 3.0 * cfg.nu + 0.5
        assert abs(cfg.tau - expected_tau) < 1e-10

    def test_validate_small_nx(self) -> None:
        with pytest.raises(ValueError, match="nx"):
            LidDrivenCavityConfig(nx=4).validate()

    def test_validate_zero_u_lid_raises(self) -> None:
        with pytest.raises(ValueError, match="u_lid"):
            LidDrivenCavityConfig(nx=8, u_lid=0.0, re=100.0).validate()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        cfg = LidDrivenCavityConfig(nx=64, re=200.0, run_name="test_rt")
        p = tmp_path / "cfg.json"
        cfg.save(p)
        cfg2 = LidDrivenCavityConfig.load(p)
        assert cfg2.nx == cfg.nx
        assert cfg2.re == cfg.re
        assert cfg2.run_name == cfg.run_name


# ---------------------------------------------------------------------------
# compare_ghia
# ---------------------------------------------------------------------------


class TestCompareGhia:
    def test_returns_dict_with_keys(self) -> None:
        ny, nx = 32, 32
        ux = torch.zeros((ny, nx))
        uy = torch.zeros((ny, nx))
        result = compare_ghia(ux, uy, u_lid=0.1, reference=GHIA_RE100)
        assert "rmse_u" in result
        assert "rmse_v" in result

    def test_perfect_match_gives_zero_error(self) -> None:
        """If we supply exactly the Ghia values at the reference positions, error = 0."""
        import numpy as np

        ny, nx = 129, 129
        y_pos = np.linspace(0.0, 1.0, ny)
        x_pos = np.linspace(0.0, 1.0, nx)

        u_lid = 1.0
        # Reconstruct an ux field by interpolating Ghia values at all y positions
        ux_col = np.interp(y_pos, GHIA_RE100["y"][::-1], GHIA_RE100["u"][::-1])
        ux_field = np.tile(ux_col.reshape(-1, 1), (1, nx)).astype(np.float32)

        vy_row = np.interp(x_pos, GHIA_RE100["x"], GHIA_RE100["v"])
        uy_field = np.zeros((ny, nx), dtype=np.float32)
        uy_field[ny // 2, :] = vy_row

        ux_t = torch.from_numpy(ux_field)
        uy_t = torch.from_numpy(uy_field)
        result = compare_ghia(ux_t, uy_t, u_lid=u_lid, reference=GHIA_RE100)
        # RMSE_u should be very small (just interpolation round-trip error)
        assert result["rmse_u"] < 0.02


# ---------------------------------------------------------------------------
# run_lid_driven_cavity – smoke test
# ---------------------------------------------------------------------------


def test_lid_driven_cavity_smoke(tmp_path: Path) -> None:
    """Smoke test: run a very small cavity for a few steps and check outputs."""
    config = LidDrivenCavityConfig(
        nx=16,
        u_lid=0.1,
        re=50.0,
        n_steps=4,
        output_interval=2,
        output_root=tmp_path / "outputs",
        run_name="smoke",
        overwrite=True,
    )
    run_dir = run_lid_driven_cavity(config)
    assert run_dir.exists()

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["config"]["n_steps"] == 4
    assert meta["diagnostics"]
    assert (run_dir / "snapshot_000004.png").exists()
    assert (run_dir / "ghia_comparison.csv").exists()


def test_lid_driven_cavity_ghia_comparison_re100(tmp_path: Path) -> None:
    """Re=100 comparison is written in metadata when re=100."""
    config = LidDrivenCavityConfig(
        nx=16,
        u_lid=0.1,
        re=100.0,
        n_steps=2,
        output_interval=2,
        output_root=tmp_path / "outputs",
        run_name="ghia",
        overwrite=True,
    )
    run_dir = run_lid_driven_cavity(config)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "ghia_errors" in meta


def test_lid_driven_cavity_overwrite(tmp_path: Path) -> None:
    """Running twice with overwrite=True does not raise."""
    config = LidDrivenCavityConfig(
        nx=16,
        u_lid=0.1,
        re=50.0,
        n_steps=2,
        output_interval=2,
        output_root=tmp_path / "out",
        run_name="ow",
        overwrite=True,
    )
    run_lid_driven_cavity(config)
    run_lid_driven_cavity(config)  # should not raise
