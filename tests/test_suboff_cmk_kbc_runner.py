"""TDD tests for SUBOFF CM/CUMULANT/KBC × SGS runner.

Tests verify:
  1. Config validation (collision, turbulence_model, lattice)
  2. SGS tau_eff computation for all 3 models
  3. Collision with SGS produces finite output (CPU, small grid)
  4. Runner produces valid artifact with required fields (CPU, small grid)
  5. All 9 combinations are executable (CPU, tiny grid)
  6. SDAA smoke test (sdaa:0, tiny grid)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.suboff_cmk_kbc_runner import (
    COMBINATIONS,
    SuboffCmkKbcConfig,
    _compute_sgs_tau_eff,
    _collide_with_sgs,
    run_suboff_cmk_kbc,
    write_artifact,
)


# ---------------------------------------------------------------------------
# Small-grid config factory
# ---------------------------------------------------------------------------

def _small_config(**overrides: Any) -> SuboffCmkKbcConfig:
    defaults: dict[str, Any] = dict(
        re=200.0,
        collision="CM",
        turbulence_model="smagorinsky",
        nx=32,
        ny=16,
        nz=16,
        n_steps=5,
        u_in=0.06,
        hull_length=16.0,
        device="cpu",
    )
    defaults.update(overrides)
    return SuboffCmkKbcConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_default_re_is_2e6(self) -> None:
        cfg = SuboffCmkKbcConfig()
        assert cfg.re == 2_000_000.0

    def test_default_grid_is_320x160x160(self) -> None:
        cfg = SuboffCmkKbcConfig()
        assert cfg.nx == 320
        assert cfg.ny == 160
        assert cfg.nz == 160

    def test_default_steps_is_1000(self) -> None:
        cfg = SuboffCmkKbcConfig()
        assert cfg.n_steps == 1000

    def test_default_device_is_sdaa0(self) -> None:
        cfg = SuboffCmkKbcConfig()
        assert cfg.device == "sdaa:0"

    def test_default_boundary_is_farfield(self) -> None:
        cfg = SuboffCmkKbcConfig()
        assert cfg.boundary_type == "farfield"

    def test_invalid_collision_raises(self) -> None:
        with pytest.raises(ValueError, match="collision"):
            SuboffCmkKbcConfig(collision="BGK")

    def test_invalid_turbulence_model_raises(self) -> None:
        with pytest.raises(ValueError, match="turbulence_model"):
            SuboffCmkKbcConfig(turbulence_model="dynamic_smagorinsky")

    def test_invalid_lattice_raises(self) -> None:
        with pytest.raises(ValueError, match="lattice"):
            SuboffCmkKbcConfig(lattice="D3Q27")

    def test_tau_property(self) -> None:
        cfg = SuboffCmkKbcConfig(re=200.0, u_in=0.06, hull_length=16.0)
        expected_nu = 0.06 * 16.0 / 200.0
        expected_tau = 3.0 * expected_nu + 0.5
        assert cfg.nu == pytest.approx(expected_nu)
        assert cfg.tau == pytest.approx(expected_tau)

    def test_re_2e6_tau_is_near_half(self) -> None:
        """At Re=2e6, tau is very close to 0.5 (stability limit)."""
        cfg = SuboffCmkKbcConfig()
        assert cfg.tau > 0.5
        assert cfg.tau < 0.51


# ---------------------------------------------------------------------------
# 2. SGS tau_eff computation
# ---------------------------------------------------------------------------

class TestSgsTauEff:
    def _make_f(self, shape=(4, 5, 6)) -> torch.Tensor:
        rho = torch.ones(shape)
        ux = torch.full(shape, 0.03)
        uy = torch.zeros(shape)
        uz = torch.zeros(shape)
        return equilibrium3d(rho, ux, uy, uz)

    def test_smagorinsky_tau_eff_is_tensor_and_finite(self) -> None:
        f = self._make_f()
        cfg = _small_config(turbulence_model="smagorinsky")
        tau_eff = _compute_sgs_tau_eff(f, cfg, cfg.tau)
        assert isinstance(tau_eff, torch.Tensor)
        assert torch.isfinite(tau_eff).all()
        # At equilibrium, f_neq=0, so tau_eff == tau_base
        assert torch.allclose(tau_eff, torch.full_like(tau_eff, cfg.tau))

    def test_wale_tau_eff_is_tensor_and_finite(self) -> None:
        f = self._make_f()
        cfg = _small_config(turbulence_model="wale")
        tau_eff = _compute_sgs_tau_eff(f, cfg, cfg.tau)
        assert isinstance(tau_eff, torch.Tensor)
        assert torch.isfinite(tau_eff).all()
        # At uniform velocity, gradients are zero, so nu_t=0, tau_eff=tau_base
        assert torch.allclose(tau_eff, torch.full_like(tau_eff, cfg.tau))

    def test_vreman_tau_eff_is_tensor_and_finite(self) -> None:
        f = self._make_f()
        cfg = _small_config(turbulence_model="vreman")
        tau_eff = _compute_sgs_tau_eff(f, cfg, cfg.tau)
        assert isinstance(tau_eff, torch.Tensor)
        assert torch.isfinite(tau_eff).all()
        # At uniform velocity, gradients are zero, so nu_t=0, tau_eff=tau_base
        assert torch.allclose(tau_eff, torch.full_like(tau_eff, cfg.tau))

    def test_tau_eff_increases_with_shear(self) -> None:
        """Non-equilibrium should increase tau_eff above tau_base."""
        rho = torch.ones((4, 5, 6))
        ux = torch.full((4, 5, 6), 0.03)
        # Add velocity gradient
        ux[:, :, 0] = 0.06
        ux[:, :, -1] = 0.0
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)
        # Add perturbation to create non-equilibrium
        f = f + 1e-3 * torch.randn_like(f)

        cfg = _small_config(turbulence_model="smagorinsky")
        tau_eff = _compute_sgs_tau_eff(f, cfg, cfg.tau)
        # Some cells should have tau_eff > tau_base
        assert (tau_eff > cfg.tau).any()


# ---------------------------------------------------------------------------
# 3. Collision with SGS
# ---------------------------------------------------------------------------

class TestCollideWithSgs:
    def _make_f(self, shape=(4, 5, 6)) -> torch.Tensor:
        rho = torch.ones(shape)
        ux = torch.full(shape, 0.03)
        uy = torch.zeros(shape)
        uz = torch.zeros(shape)
        return equilibrium3d(rho, ux, uy, uz)

    @pytest.mark.parametrize("collision", ["CM", "CUMULANT", "KBC"])
    @pytest.mark.parametrize("sgs", ["smagorinsky", "wale", "vreman"])
    def test_collide_preserves_shape_and_finite(
        self, collision: str, sgs: str,
    ) -> None:
        f = self._make_f()
        cfg = _small_config(collision=collision, turbulence_model=sgs)
        out = _collide_with_sgs(f, cfg, cfg.tau)
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("collision", ["CM", "CUMULANT", "KBC"])
    def test_equilibrium_is_fixed_point(self, collision: str) -> None:
        """At equilibrium, collision should be a no-op (within tolerance)."""
        f = self._make_f()
        cfg = _small_config(collision=collision, turbulence_model="smagorinsky")
        out = _collide_with_sgs(f, cfg, cfg.tau)
        assert torch.allclose(out, f, atol=2e-5)


# ---------------------------------------------------------------------------
# 4. Runner produces valid artifact
# ---------------------------------------------------------------------------

class TestRunnerArtifact:
    def test_run_produces_required_fields(self) -> None:
        cfg = _small_config(collision="CM", turbulence_model="smagorinsky")
        artifact = run_suboff_cmk_kbc(cfg)
        # Required fields from task spec
        assert artifact["status"] == "diagnostic_only"
        assert artifact["physical_validation"] is False
        assert artifact["Re"] == cfg.re
        assert artifact["collision"] == "CM"
        assert artifact["turbulence_model"] == "smagorinsky"
        assert "Ct" in artifact
        assert isinstance(artifact["Ct"], float)
        assert "finite" in artifact
        assert isinstance(artifact["finite"], bool)
        assert artifact["steps_completed"] == cfg.n_steps
        assert artifact["boundary_type"] == "farfield"
        assert artifact["device"] == "sdaa"
        assert artifact["reference_Ct"] == 0.00405
        assert artifact["reference_source"] == "ITTC-1957"

    def test_run_has_force_and_ct_time_series(self) -> None:
        cfg = _small_config()
        artifact = run_suboff_cmk_kbc(cfg)
        assert len(artifact["force_time_series"]) == cfg.n_steps
        assert len(artifact["ct_time_series"]) == cfg.n_steps
        # Each entry has step and values
        entry = artifact["force_time_series"][0]
        assert "step" in entry
        assert "fx" in entry
        assert "fy" in entry
        assert "fz" in entry
        ct_entry = artifact["ct_time_series"][0]
        assert "step" in ct_entry
        assert "ct" in ct_entry

    def test_run_finite_on_small_grid(self) -> None:
        """Small-grid run should complete with finite populations."""
        cfg = _small_config(collision="CM", turbulence_model="smagorinsky")
        artifact = run_suboff_cmk_kbc(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == cfg.n_steps

    def test_write_artifact(self, tmp_path: Path) -> None:
        cfg = _small_config()
        artifact = run_suboff_cmk_kbc(cfg)
        path = tmp_path / "artifact.json"
        write_artifact(artifact, path)
        loaded = json.loads(path.read_text())
        assert loaded["status"] == "diagnostic_only"
        assert loaded["collision"] == "CM"


# ---------------------------------------------------------------------------
# 5. All 9 combinations executable (CPU, tiny grid)
# ---------------------------------------------------------------------------

class TestAllNineCombinations:
    """Smoke-test all 9 collision×SGS combinations on a tiny CPU grid."""

    @pytest.mark.parametrize("collision,sgs", COMBINATIONS)
    def test_combination_runs(self, collision: str, sgs: str) -> None:
        cfg = _small_config(
            collision=collision,
            turbulence_model=sgs,
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_cmk_kbc(cfg)
        assert artifact["collision"] == collision.upper()
        assert artifact["turbulence_model"] == sgs.lower()
        assert artifact["steps_completed"] == 3
        # Must be finite (tiny grid, few steps)
        assert artifact["finite"] is True


# ---------------------------------------------------------------------------
# 6. SDAA smoke test
# ---------------------------------------------------------------------------

class TestSdaaSmokeTest:
    """Verify the runner works on SDAA hardware."""

    @pytest.fixture
    def sdaa_available(self) -> bool:
        return hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0

    @pytest.mark.skipif(
        not (hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0),
        reason="No SDAA device available",
    )
    def test_sdaa_cm_smagorinsky(self, sdaa_available: bool) -> None:
        cfg = _small_config(
            collision="CM",
            turbulence_model="smagorinsky",
            device="sdaa:0",
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_cmk_kbc(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == 3
        assert artifact["device"] == "sdaa"

    @pytest.mark.skipif(
        not (hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0),
        reason="No SDAA device available",
    )
    @pytest.mark.parametrize("collision,sgs", COMBINATIONS)
    def test_sdaa_all_combinations(
        self, collision: str, sgs: str, sdaa_available: bool,
    ) -> None:
        cfg = _small_config(
            collision=collision,
            turbulence_model=sgs,
            device="sdaa:0",
            nx=16,
            ny=8,
            nz=8,
            n_steps=3,
            hull_length=8.0,
        )
        artifact = run_suboff_cmk_kbc(cfg)
        assert artifact["finite"] is True
        assert artifact["steps_completed"] == 3
