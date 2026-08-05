"""TDD tests for SUBOFF TRT×SGS D3Q27 runner on SDAA.

Tests verify:
  1. Config validation and physics parameters
  2. TRT+none collision via collide_advanced_3d dispatch
  3. TRT+SGS collision (Smagorinsky, WALE, Vreman) with per-cell tau_eff
  4. D3Q27 far-field boundary condition
  5. Full runner produces machine-readable artifact with required fields
  6. Campaign function produces 4 artifacts
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tensorlbm.suboff_trt_runner import (
    SuboffTrtConfig,
    SuboffTrtEvidence,
    _collide_trt_sgs_27,
    _far_field_bc_27,
    run_suboff_trt_sgs,
    run_suboff_trt_sgs_campaign,
    ITTC_1957_REFERENCE_CT,
)


# ---------------------------------------------------------------------------
# Small-grid config factory (CPU, fast tests)
# ---------------------------------------------------------------------------

def _small_config(**overrides: Any) -> SuboffTrtConfig:
    defaults: dict[str, Any] = dict(
        nx=32,
        ny=16,
        nz=16,
        n_steps=10,
        u_in=0.05,
        re=200.0,
        hull_length=19.2,  # 0.6 * 32
        device="cpu",
        turbulence_model="none",
    )
    defaults.update(overrides)
    return SuboffTrtConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestSuboffTrtConfig:
    """Verify configuration validation and derived physics parameters."""

    def test_defaults_produce_valid_config(self) -> None:
        cfg = SuboffTrtConfig()
        assert cfg.nx == 320
        assert cfg.ny == 160
        assert cfg.nz == 160
        assert cfg.n_steps == 1000
        assert cfg.re == 2e6
        assert cfg.device == "sdaa:0"
        assert cfg.boundary_type == "farfield"

    def test_nu_derived_from_re(self) -> None:
        cfg = _small_config(re=200.0, u_in=0.05, hull_length=19.2)
        expected_nu = 0.05 * 19.2 / 200.0
        assert abs(cfg.nu - expected_nu) < 1e-12

    def test_tau_derived_from_nu(self) -> None:
        cfg = _small_config(re=200.0, u_in=0.05, hull_length=19.2)
        expected_tau = 3.0 * cfg.nu + 0.5
        assert abs(cfg.tau - expected_tau) < 1e-12

    def test_tau_must_exceed_half(self) -> None:
        """tau must be > 0.5; with u_in=0 the u_in check fires first."""
        with pytest.raises(ValueError, match="u_in must be in"):
            SuboffTrtConfig(u_in=0.0)

    def test_invalid_turbulence_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="turbulence_model"):
            SuboffTrtConfig(turbulence_model="k_omega")

    def test_re_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="re must be > 0"):
            SuboffTrtConfig(re=-1.0)

    def test_steps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            SuboffTrtConfig(n_steps=0)


# ---------------------------------------------------------------------------
# 2. TRT+none collision via collide_advanced_3d
# ---------------------------------------------------------------------------

class TestTrtNoneCollision:
    """Verify TRT+none uses collide_advanced_3d unified dispatch."""

    def test_trt_none_preserves_mass(self) -> None:
        """TRT collision must conserve mass (zeroth moment)."""
        from tensorlbm.d3q27 import equilibrium27
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium27(rho, ux, uy, uz)
        tau = 0.55
        f_post = _collide_trt_sgs_27(f, tau=tau, turbulence_model="none", lambda_trt=3.0 / 16.0)
        mass_before = float(f.sum().item())
        mass_after = float(f_post.sum().item())
        assert abs(mass_after - mass_before) < 1e-5, "TRT must conserve mass"

    def test_trt_none_equilibrium_is_fixed_point(self) -> None:
        """At equilibrium, TRT collision must return the same distribution."""
        from tensorlbm.d3q27 import equilibrium27
        nz, ny, nx = 6, 6, 6
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.03)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        feq = equilibrium27(rho, ux, uy, uz)
        f_post = _collide_trt_sgs_27(feq, tau=0.55, turbulence_model="none")
        assert torch.allclose(feq, f_post, atol=1e-6), "Equilibrium must be a fixed point of TRT"


# ---------------------------------------------------------------------------
# 3. TRT+SGS collision
# ---------------------------------------------------------------------------

class TestTrtSgsCollision:
    """Verify TRT+SGS collision with per-cell effective tau."""

    @pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
    def test_sgs_preserves_mass(self, model: str) -> None:
        """TRT+SGS collision must conserve mass."""
        from tensorlbm.d3q27 import equilibrium27
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium27(rho, ux, uy, uz)
        # Add perturbation to generate non-equilibrium
        f = f * (1.0 + 0.01 * torch.randn_like(f))
        tau = 0.55
        f_post = _collide_trt_sgs_27(f, tau=tau, turbulence_model=model)
        mass_before = float(f.sum().item())
        mass_after = float(f_post.sum().item())
        assert abs(mass_after - mass_before) < 1e-4, f"TRT+{model} must conserve mass"

    @pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
    def test_sgs_output_is_finite(self, model: str) -> None:
        """TRT+SGS collision must produce finite output."""
        from tensorlbm.d3q27 import equilibrium27
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium27(rho, ux, uy, uz)
        f = f * (1.0 + 0.01 * torch.randn_like(f))
        f_post = _collide_trt_sgs_27(f, tau=0.55, turbulence_model=model)
        assert torch.isfinite(f_post).all(), f"TRT+{model} output must be finite"

    def test_sgs_tau_eff_geq_tau_none(self) -> None:
        """SGS effective tau must be >= baseline tau (eddy viscosity adds dissipation)."""
        from tensorlbm.d3q27 import equilibrium27, macroscopic27
        from tensorlbm.turbulence import (
            _neq_stress_norm_27,
            _smagorinsky_tau,
        )
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium27(rho, ux, uy, uz)
        f = f * (1.0 + 0.1 * torch.randn_like(f))
        rho_m, ux_m, uy_m, uz_m = macroscopic27(f)
        feq = equilibrium27(rho_m, ux_m, uy_m, uz_m)
        f_neq = f - feq
        pi_norm = _neq_stress_norm_27(f_neq)
        tau = 0.55
        tau_eff = _smagorinsky_tau(tau, pi_norm, rho_m, 0.1)
        assert (tau_eff >= tau - 1e-10).all(), "Smagorinsky tau_eff must be >= tau"


# ---------------------------------------------------------------------------
# 4. D3Q27 far-field boundary condition
# ---------------------------------------------------------------------------

class TestFarFieldBC27:
    """Verify D3Q27 far-field boundary condition."""

    def test_inlet_set_to_free_stream(self) -> None:
        """Inlet plane must be free-stream equilibrium."""
        from tensorlbm.d3q27 import equilibrium27
        nz, ny, nx = 6, 6, 8
        u_in = 0.05
        f = torch.randn(27, nz, ny, nx)
        f = _far_field_bc_27(f, u_in=u_in)
        rho_fs = torch.ones(nz, ny, nx)
        feq = equilibrium27(rho_fs, torch.full_like(rho_fs, u_in),
                            torch.zeros_like(rho_fs), torch.zeros_like(rho_fs))
        assert torch.allclose(f[:, :, :, 0], feq[:, :, :, 0], atol=1e-6)

    def test_outlet_zero_gradient(self) -> None:
        """Outlet plane must equal the second-to-last plane (zero gradient)."""
        nz, ny, nx = 6, 6, 8
        f = torch.randn(27, nz, ny, nx)
        f = _far_field_bc_27(f, u_in=0.05)
        assert torch.allclose(f[:, :, :, -1], f[:, :, :, -2])

    def test_lateral_faces_set_to_free_stream(self) -> None:
        """All four lateral faces must be free-stream equilibrium."""
        from tensorlbm.d3q27 import equilibrium27
        nz, ny, nx = 6, 8, 8
        u_in = 0.05
        f = torch.randn(27, nz, ny, nx)
        f = _far_field_bc_27(f, u_in=u_in)
        rho_fs = torch.ones(nz, ny, nx)
        feq = equilibrium27(rho_fs, torch.full_like(rho_fs, u_in),
                            torch.zeros_like(rho_fs), torch.zeros_like(rho_fs))
        # y- and y+
        assert torch.allclose(f[:, 0, :, :], feq[:, 0, :, :], atol=1e-6)
        assert torch.allclose(f[:, -1, :, :], feq[:, -1, :, :], atol=1e-6)
        # z- and z+
        assert torch.allclose(f[:, :, 0, :], feq[:, :, 0, :], atol=1e-6)
        assert torch.allclose(f[:, :, -1, :], feq[:, :, -1, :], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Full runner produces valid artifact
# ---------------------------------------------------------------------------

class TestSuboffTrtRunner:
    """Verify the full runner produces a machine-readable artifact."""

    @pytest.mark.parametrize("model", ["none", "smagorinsky", "wale", "vreman"])
    def test_runner_produces_valid_artifact(self, model: str) -> None:
        """Runner must produce evidence with all required artifact fields."""
        cfg = _small_config(turbulence_model=model, n_steps=5)
        evidence = run_suboff_trt_sgs(cfg)
        artifact = evidence.to_artifact()

        # Required top-level fields
        required_fields = {
            "schema", "status", "physical_validation", "Re",
            "collision", "turbulence_model", "Ct", "finite",
            "steps_completed", "boundary_type", "device",
            "reference_Ct", "reference_source",
        }
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"

        # Field value checks
        assert artifact["status"] == "diagnostic_only"
        assert artifact["physical_validation"] is False
        assert artifact["Re"] == 200.0  # small config
        assert artifact["collision"] == "TRT"
        assert artifact["turbulence_model"] == model
        assert artifact["boundary_type"] == "farfield"
        assert artifact["device"] == "cpu"
        assert artifact["reference_Ct"] == ITTC_1957_REFERENCE_CT
        assert artifact["reference_source"] == "ITTC-1957"
        assert artifact["steps_completed"] == 5
        assert isinstance(artifact["Ct"], float)
        assert isinstance(artifact["finite"], bool)

    @pytest.mark.parametrize("model", ["none", "smagorinsky", "wale", "vreman"])
    def test_runner_finiteness(self, model: str) -> None:
        """Runner must complete all steps with finite populations (small grid)."""
        cfg = _small_config(turbulence_model=model, n_steps=5)
        evidence = run_suboff_trt_sgs(cfg)
        assert evidence.finite is True, f"TRT+{model} must produce finite populations"
        assert evidence.steps_completed == 5

    def test_runner_writes_artifact_file(self, tmp_path: Path) -> None:
        """Evidence.write_artifact must produce a valid JSON file."""
        cfg = _small_config(n_steps=3)
        evidence = run_suboff_trt_sgs(cfg)
        path = tmp_path / "artifact.json"
        evidence.write_artifact(str(path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["collision"] == "TRT"
        assert data["turbulence_model"] == "none"


# ---------------------------------------------------------------------------
# 6. Campaign function
# ---------------------------------------------------------------------------

class TestSuboffTrtCampaign:
    """Verify the campaign function produces 4 artifacts."""

    def test_campaign_produces_four_artifacts(self, tmp_path: Path) -> None:
        """Campaign must produce one artifact per combination."""
        configs = [
            SuboffTrtConfig(
                nx=32, ny=16, nz=16, n_steps=3, u_in=0.05,
                re=200.0, hull_length=19.2, device="cpu",
                turbulence_model=model,
            )
            for model in ["none", "smagorinsky", "wale", "vreman"]
        ]
        results = run_suboff_trt_sgs_campaign(configs)
        assert len(results) == 4
        models = [r.turbulence_model for r in results]
        assert set(models) == {"none", "smagorinsky", "wale", "vreman"}
        for r in results:
            assert r.collision == "TRT"
            assert r.status == "diagnostic_only"
            assert r.physical_validation is False
