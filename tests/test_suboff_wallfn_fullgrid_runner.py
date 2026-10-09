"""TDD tests for SUBOFF 7-collision × 2-lattice × wall_function full-grid runner.

Tests verify:
  1. Config validation (lattice, collision, wall_law, grid, parameters)
  2. Collision dispatch for all 7 families × 2 lattices
  3. Drag computation (friction + pressure) from wall-function state
  4. Runner produces valid artifact with required fields (CPU, small grid)
  5. All 11 combinations are executable (CPU, tiny grid)
  6. SDAA smoke test (sdaa:0, tiny grid)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.suboff_wallfn_fullgrid_runner import (
    COLLISION_FAMILIES,
    COMBINATIONS,
    LATTICES,
    SuboffWallFnFullGridConfig,
    _collide,
    _compute_drags,
    run_suboff_wallfn_fullgrid,
    write_artifact,
)

# --------------------------------------------------------------------------- #
# Small-grid config factory
# --------------------------------------------------------------------------- #


def _small_config(**overrides: Any) -> SuboffWallFnFullGridConfig:
    defaults: dict[str, Any] = dict(
        re=200.0,
        lattice="D3Q19",
        collision="BGK",
        nx=32,
        ny=16,
        nz=16,
        n_steps=5,
        u_in=0.06,
        hull_length=16.0,
        device="cpu",
    )
    defaults.update(overrides)
    return SuboffWallFnFullGridConfig(**defaults)


# --------------------------------------------------------------------------- #
# 1. Config validation
# --------------------------------------------------------------------------- #


class TestConfigValidation:
    def test_default_re_is_2e6(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.re == 2_000_000.0

    def test_default_grid_is_480x240x240(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.nx == 480
        assert cfg.ny == 240
        assert cfg.nz == 240

    def test_default_steps_is_1000(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.n_steps == 1000

    def test_default_device_is_sdaa0(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.device == "sdaa:0"

    def test_default_lattice_is_d3q27(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.lattice == "D3Q27"

    def test_default_collision_is_cumulant(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.collision == "CUMULANT"

    def test_default_hull_length_is_240(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.hull_length == 240.0

    def test_default_wall_law_is_log(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.wall_law == "log"

    def test_default_y_val_is_0_5(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.y_val == 0.5

    def test_invalid_lattice_raises(self) -> None:
        with pytest.raises(ValueError, match="lattice"):
            SuboffWallFnFullGridConfig(lattice="D2Q9")

    def test_invalid_collision_raises(self) -> None:
        with pytest.raises(ValueError, match="collision"):
            SuboffWallFnFullGridConfig(collision="ELB")

    def test_invalid_wall_law_raises(self) -> None:
        with pytest.raises(ValueError, match="wall_law"):
            SuboffWallFnFullGridConfig(wall_law="power")

    def test_invalid_u_in_raises(self) -> None:
        with pytest.raises(ValueError, match="u_in"):
            SuboffWallFnFullGridConfig(u_in=0.2)

    def test_invalid_re_raises(self) -> None:
        with pytest.raises(ValueError, match="re"):
            SuboffWallFnFullGridConfig(re=-1.0)

    def test_tau_property(self) -> None:
        cfg = SuboffWallFnFullGridConfig(re=200.0, u_in=0.06, hull_length=16.0)
        expected_nu = 0.06 * 16.0 / 200.0
        expected_tau = 3.0 * expected_nu + 0.5
        assert cfg.nu == pytest.approx(expected_nu)
        assert cfg.tau == pytest.approx(expected_tau)

    def test_re_2e6_tau_is_near_half(self) -> None:
        """At Re=2e6, tau is very close to 0.5 (stability limit)."""
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.tau > 0.5
        assert cfg.tau < 0.51

    def test_reference_values(self) -> None:
        cfg = SuboffWallFnFullGridConfig()
        assert cfg.reference_Ct == 0.00405
        assert cfg.reference_source == "ITTC-1957"

    def test_combinations_count(self) -> None:
        assert len(COMBINATIONS) == 11

    def test_all_collision_families_present(self) -> None:
        assert set(COLLISION_FAMILIES) == {
            "CUMULANT",
            "BGK",
            "MRT",
            "TRT",
            "CM",
            "KBC",
            "RLBM",
        }

    def test_all_lattices_present(self) -> None:
        assert set(LATTICES) == {"D3Q27", "D3Q19"}


# --------------------------------------------------------------------------- #
# 2. Collision dispatch
# --------------------------------------------------------------------------- #


class TestCollisionDispatch:
    def _make_f_d3q19(self, shape=(4, 5, 6)) -> torch.Tensor:
        rho = torch.ones(shape)
        ux = torch.full(shape, 0.03)
        uy = torch.zeros(shape)
        uz = torch.zeros(shape)
        return equilibrium3d(rho, ux, uy, uz)

    def _make_f_d3q27(self, shape=(4, 5, 6)) -> torch.Tensor:
        rho = torch.ones(shape)
        ux = torch.full(shape, 0.03)
        uy = torch.zeros(shape)
        uz = torch.zeros(shape)
        return equilibrium27(rho, ux, uy, uz)

    @pytest.mark.parametrize("collision", ["CUMULANT", "BGK", "MRT", "TRT", "CM", "KBC", "RLBM"])
    def test_d3q19_collision_preserves_shape_and_finite(self, collision: str) -> None:
        f = self._make_f_d3q19()
        tau = 0.55
        out = _collide("D3Q19", collision, f, tau)
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("collision", ["CUMULANT", "BGK", "MRT", "TRT", "CM", "KBC", "RLBM"])
    def test_d3q27_collision_preserves_shape_and_finite(self, collision: str) -> None:
        f = self._make_f_d3q27()
        tau = 0.55
        out = _collide("D3Q27", collision, f, tau)
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("collision", ["CUMULANT", "BGK", "MRT", "TRT", "CM", "KBC", "RLBM"])
    def test_d3q19_equilibrium_is_fixed_point(self, collision: str) -> None:
        """At equilibrium, collision should be approximately a no-op."""
        f = self._make_f_d3q19()
        tau = 0.55
        out = _collide("D3Q19", collision, f, tau)
        assert torch.allclose(out, f, atol=2e-5)

    @pytest.mark.parametrize("collision", ["CUMULANT", "BGK", "MRT", "TRT", "CM", "KBC", "RLBM"])
    def test_d3q27_equilibrium_is_fixed_point(self, collision: str) -> None:
        """At equilibrium, collision should be approximately a no-op."""
        f = self._make_f_d3q27()
        tau = 0.55
        out = _collide("D3Q27", collision, f, tau)
        assert torch.allclose(out, f, atol=2e-5)


# --------------------------------------------------------------------------- #
# 3. Drag computation
# --------------------------------------------------------------------------- #


class TestDragComputation:
    def test_compute_drags_returns_floats(self) -> None:
        """Drag computation should return finite floats."""
        nz, ny, nx = 8, 8, 16
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.06)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        f = equilibrium3d(rho, ux, uy, uz)

        # Simple solid block
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solid[:, :, 6:10] = True
        ux[solid] = 0.0
        f = equilibrium3d(rho, ux, uy, uz)

        # Compute u_tau (will be near-zero at equilibrium, but function should work)
        from tensorlbm.wall_function_common import compute_u_tau

        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        u_tau = compute_u_tau(u_mag, nu=0.01, y_val=0.5, wall_law="log")

        drag_fric, drag_pres = _compute_drags(f, solid, u_tau, "D3Q19")
        assert isinstance(drag_fric, float)
        assert isinstance(drag_pres, float)
        assert math.isfinite(drag_fric)
        assert math.isfinite(drag_pres)

    def test_compute_drags_d3q27(self) -> None:
        """Drag computation should work for D3Q27."""
        nz, ny, nx = 8, 8, 16
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.06)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        f = equilibrium27(rho, ux, uy, uz)

        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solid[:, :, 6:10] = True
        ux[solid] = 0.0
        f = equilibrium27(rho, ux, uy, uz)

        from tensorlbm.wall_function_common import compute_u_tau

        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        u_tau = compute_u_tau(u_mag, nu=0.01, y_val=0.5, wall_law="log")

        drag_fric, drag_pres = _compute_drags(f, solid, u_tau, "D3Q27")
        assert isinstance(drag_fric, float)
        assert isinstance(drag_pres, float)
        assert math.isfinite(drag_fric)
        assert math.isfinite(drag_pres)


# --------------------------------------------------------------------------- #
# 4. Runner produces valid artifact
# --------------------------------------------------------------------------- #


class TestRunnerArtifact:
    def test_run_produces_required_fields(self) -> None:
        cfg = _small_config(collision="BGK")
        artifact = run_suboff_wallfn_fullgrid(cfg)
        # Required fields from task spec
        assert artifact["status"] == "diagnostic_only"
        assert artifact["physical_validation"] is False
        assert artifact["lattice"] == "D3Q19"
        assert artifact["collision"] == "BGK"
        assert "wall_function" in artifact
        assert "log-law" in artifact["wall_function"]
        assert "Ct_fric" in artifact
        assert isinstance(artifact["Ct_fric"], float)
        assert "Ct_pres" in artifact
        assert isinstance(artifact["Ct_pres"], float)
        assert "Ct_total" in artifact
        assert isinstance(artifact["Ct_total"], float)
        assert "finite" in artifact
        assert isinstance(artifact["finite"], bool)
        assert artifact["steps_completed"] == cfg.n_steps
        assert artifact["grid"] == {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz}
        assert artifact["hull_length"] == cfg.hull_length
        assert artifact["reference_Ct"] == 0.00405
        assert artifact["reference_source"] == "ITTC-1957"
        assert artifact["Re"] == cfg.re

    def test_run_has_force_and_ct_time_series(self) -> None:
        cfg = _small_config()
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert len(artifact["force_time_series"]) == cfg.n_steps
        assert len(artifact["ct_time_series"]) == cfg.n_steps
        # Each entry has step and values
        f_entry = artifact["force_time_series"][0]
        assert "step" in f_entry
        assert "drag_fric" in f_entry
        assert "drag_pres" in f_entry
        assert "drag_total" in f_entry
        ct_entry = artifact["ct_time_series"][0]
        assert "step" in ct_entry
        assert "ct_fric" in ct_entry
        assert "ct_pres" in ct_entry
        assert "ct_total" in ct_entry

    def test_run_finite_on_small_grid(self) -> None:
        """Small-grid run should complete with finite populations."""
        cfg = _small_config(collision="BGK")
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == cfg.n_steps

    def test_write_artifact(self, tmp_path: Path) -> None:
        cfg = _small_config()
        artifact = run_suboff_wallfn_fullgrid(cfg)
        path = tmp_path / "artifact.json"
        write_artifact(artifact, path)
        loaded = json.loads(path.read_text())
        assert loaded["status"] == "diagnostic_only"
        assert loaded["collision"] == "BGK"
        assert loaded["lattice"] == "D3Q19"

    def test_ct_total_equals_fric_plus_pres(self) -> None:
        """Ct_total must equal Ct_fric + Ct_pres."""
        cfg = _small_config()
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert artifact["Ct_total"] == pytest.approx(artifact["Ct_fric"] + artifact["Ct_pres"])


# --------------------------------------------------------------------------- #
# 5. All 11 combinations executable (CPU, tiny grid)
# --------------------------------------------------------------------------- #


class TestAllCombinations:
    """Smoke-test all 11 collision×lattice combinations on a tiny CPU grid."""

    @pytest.mark.parametrize("lattice, collision", COMBINATIONS)
    def test_combination_runs(self, lattice: str, collision: str) -> None:
        cfg = _small_config(
            lattice=lattice,
            collision=collision,
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert artifact["lattice"] == lattice
        assert artifact["collision"] == collision.upper()
        assert artifact["steps_completed"] == 3
        # Must be finite (tiny grid, few steps)
        assert artifact["finite"] is True


# --------------------------------------------------------------------------- #
# 6. SDAA smoke test
# --------------------------------------------------------------------------- #


class TestSdaaSmokeTest:
    """Verify the runner works on SDAA hardware."""

    @pytest.fixture
    def sdaa_available(self) -> bool:
        return hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0

    @pytest.mark.skipif(
        not (hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0),
        reason="No SDAA device available",
    )
    def test_sdaa_bgk_d3q19(self, sdaa_available: bool) -> None:
        cfg = _small_config(
            lattice="D3Q19",
            collision="BGK",
            device="sdaa:0",
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == 3
        assert artifact["device"] == "sdaa"

    @pytest.mark.skipif(
        not (hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0),
        reason="No SDAA device available",
    )
    def test_sdaa_cumulant_d3q27(self, sdaa_available: bool) -> None:
        cfg = _small_config(
            lattice="D3Q27",
            collision="CUMULANT",
            device="sdaa:0",
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == 3
        assert artifact["device"] == "sdaa"

    @pytest.mark.skipif(
        not (hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0),
        reason="No SDAA device available",
    )
    @pytest.mark.parametrize("lattice, collision", COMBINATIONS)
    def test_sdaa_all_combinations(
        self,
        lattice: str,
        collision: str,
        sdaa_available: bool,
    ) -> None:
        cfg = _small_config(
            lattice=lattice,
            collision=collision,
            device="sdaa:0",
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_wallfn_fullgrid(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == 3
