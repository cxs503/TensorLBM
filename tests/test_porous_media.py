"""Tests for porous-media gas-water displacement benchmarks.

Covers:
    - Geometry helpers: make_random_cylinder_medium, make_tube_array_medium
    - Wettability: apply_wall_wettability_sc
    - LaplaceTestConfig validation + short smoke run
    - CapillaryInvasionConfig validation + short smoke run
    - TwoPhasePoiseuilleConfig validation + short smoke run
    - PorousDrainageConfig validation + short smoke run (SC and CG)
    - Analytical Poiseuille profile sanity check
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from tensorlbm import (
    CapillaryInvasionConfig,
    LaplaceTestConfig,
    PorousDrainageConfig,
    TwoPhasePoiseuilleConfig,
    apply_wall_wettability_sc,
    make_random_cylinder_medium,
    make_tube_array_medium,
    run_capillary_invasion,
    run_laplace_test,
    run_porous_drainage,
    run_two_phase_poiseuille,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestMakeRandomCylinderMedium:
    def test_shape(self) -> None:
        solid = make_random_cylinder_medium(30, 40, n_cylinders=5, r_min=2, r_max=4, seed=0)
        assert solid.shape == (30, 40)
        assert solid.dtype == torch.bool

    def test_has_walls(self) -> None:
        solid = make_random_cylinder_medium(30, 40, n_cylinders=5, r_min=2, r_max=4, seed=0)
        # Top and bottom rows must be solid (wall)
        assert solid[0, :].all()
        assert solid[-1, :].all()

    def test_has_fluid(self) -> None:
        solid = make_random_cylinder_medium(30, 40, n_cylinders=3, r_min=2, r_max=3, seed=1)
        assert (~solid).any(), "Domain must contain at least some fluid nodes"

    def test_porosity_reasonable(self) -> None:
        ny, nx = 50, 60
        solid = make_random_cylinder_medium(ny, nx, n_cylinders=5, r_min=3, r_max=5, seed=7)
        porosity = float((~solid).float().mean().item())
        assert 0.3 < porosity < 1.0, f"Unexpected porosity {porosity:.3f}"

    def test_reproducible(self) -> None:
        s1 = make_random_cylinder_medium(30, 40, n_cylinders=5, r_min=2, r_max=4, seed=42)
        s2 = make_random_cylinder_medium(30, 40, n_cylinders=5, r_min=2, r_max=4, seed=42)
        assert torch.equal(s1, s2)

    def test_device(self) -> None:
        solid = make_random_cylinder_medium(20, 30, n_cylinders=2, r_min=2, r_max=3,
                                            seed=0, device=DEVICE)
        assert solid.device.type == "cpu"


class TestMakeTubeArrayMedium:
    def test_shape(self) -> None:
        solid = make_tube_array_medium(30, 40, n_tubes=2, tube_width=5)
        assert solid.shape == (30, 40)
        assert solid.dtype == torch.bool

    def test_has_fluid(self) -> None:
        solid = make_tube_array_medium(30, 40, n_tubes=2, tube_width=5)
        assert (~solid).any()

    def test_porosity(self) -> None:
        ny, nx = 30, 40
        solid = make_tube_array_medium(ny, nx, n_tubes=2, tube_width=5)
        porosity = float((~solid).float().mean().item())
        assert porosity > 0.0

    def test_single_tube(self) -> None:
        solid = make_tube_array_medium(20, 30, n_tubes=1, tube_width=4)
        # There must be at least one open row
        open_rows = (~solid).any(dim=1).sum().item()
        assert open_rows >= 1


# ---------------------------------------------------------------------------
# Wall wettability
# ---------------------------------------------------------------------------

class TestApplyWallWettabilitySC:
    def test_shape_preserved(self) -> None:
        ny, nx = 20, 30
        rho1 = torch.rand((ny, nx)) + 0.5
        rho2 = torch.rand((ny, nx)) + 0.3
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[0, :] = True
        solid[-1, :] = True
        rho1_out, rho2_out = apply_wall_wettability_sc(rho1, rho2, solid)
        assert rho1_out.shape == (ny, nx)
        assert rho2_out.shape == (ny, nx)

    def test_solid_nodes_modified(self) -> None:
        ny, nx = 10, 12
        rho1 = torch.ones((ny, nx)) * 0.7
        rho2 = torch.ones((ny, nx)) * 0.3
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[0, :] = True
        G_ads1, G_ads2 = 0.3, 0.1
        rho1_out, rho2_out = apply_wall_wettability_sc(
            rho1, rho2, solid, G_ads1=G_ads1, G_ads2=G_ads2
        )
        # Solid nodes should carry the adsorption pseudo-density
        assert torch.allclose(rho1_out[solid], torch.full_like(rho1_out[solid], G_ads1))
        assert torch.allclose(rho2_out[solid], torch.full_like(rho2_out[solid], G_ads2))

    def test_fluid_nodes_unchanged(self) -> None:
        ny, nx = 10, 12
        rho1 = torch.rand((ny, nx)) + 0.5
        rho2 = torch.rand((ny, nx)) + 0.3
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[0, :] = True
        rho1_out, rho2_out = apply_wall_wettability_sc(rho1, rho2, solid)
        fluid = ~solid
        assert torch.allclose(rho1_out[fluid], rho1[fluid])
        assert torch.allclose(rho2_out[fluid], rho2[fluid])

    def test_no_solid_is_identity(self) -> None:
        ny, nx = 8, 10
        rho1 = torch.rand((ny, nx)) + 0.5
        rho2 = torch.rand((ny, nx)) + 0.3
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        rho1_out, rho2_out = apply_wall_wettability_sc(rho1, rho2, solid)
        assert torch.allclose(rho1_out, rho1)
        assert torch.allclose(rho2_out, rho2)

    def test_negative_g_ads_raises(self) -> None:
        rho1 = torch.ones((5, 5))
        rho2 = torch.ones((5, 5))
        solid = torch.zeros((5, 5), dtype=torch.bool)
        with pytest.raises(ValueError, match="non-negative"):
            apply_wall_wettability_sc(rho1, rho2, solid, G_ads1=-0.1, G_ads2=0.0)


# ---------------------------------------------------------------------------
# LaplaceTestConfig
# ---------------------------------------------------------------------------

class TestLaplaceTestConfig:
    def test_valid_config(self) -> None:
        cfg = LaplaceTestConfig(nx=60, ny=60, bubble_radius=10.0, n_steps=2,
                                output_interval=2)
        cfg.validate()

    def test_invalid_small_domain(self) -> None:
        cfg = LaplaceTestConfig(nx=10, ny=60, bubble_radius=5.0)
        with pytest.raises(ValueError, match="nx and ny"):
            cfg.validate()

    def test_invalid_bubble_radius_zero(self) -> None:
        cfg = LaplaceTestConfig(nx=60, ny=60, bubble_radius=0.0)
        with pytest.raises(ValueError, match="bubble_radius"):
            cfg.validate()

    def test_invalid_bubble_radius_too_large(self) -> None:
        cfg = LaplaceTestConfig(nx=60, ny=60, bubble_radius=35.0)
        with pytest.raises(ValueError, match="bubble_radius"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = LaplaceTestConfig(nx=60, ny=60, bubble_radius=10.0, tau1=0.4)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_density_order(self) -> None:
        cfg = LaplaceTestConfig(rho_water=0.2, rho_gas=0.8)
        with pytest.raises(ValueError, match="rho_water"):
            cfg.validate()

    def test_invalid_G12_negative(self) -> None:
        cfg = LaplaceTestConfig(G_12=-0.5)
        with pytest.raises(ValueError, match="G_12"):
            cfg.validate()

    def test_run_name_default(self) -> None:
        cfg = LaplaceTestConfig()
        name = cfg.resolved_run_name()
        assert "laplace" in name

    def test_smoke_run(self) -> None:
        """Short smoke run — check output dict keys and delta_p is positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = LaplaceTestConfig(
                nx=40, ny=40, bubble_radius=8.0,
                G_12=0.9, tau1=1.0, tau2=1.0,
                rho_water=0.7, rho_gas=0.3,
                n_steps=10, output_interval=10,
                output_root=Path(tmpdir), overwrite=True,
            )
            result = run_laplace_test(cfg)

        assert "final_delta_p" in result
        assert "sigma_eff" in result
        assert "diagnostics" in result
        assert len(result["diagnostics"]) >= 1
        # After only 10 steps the bubble may not be fully formed, but delta_p should be finite
        delta_p = result["final_delta_p"]
        assert math.isfinite(float(delta_p))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CapillaryInvasionConfig
# ---------------------------------------------------------------------------

class TestCapillaryInvasionConfig:
    def test_valid_config(self) -> None:
        cfg = CapillaryInvasionConfig(nx=60, ny=20, tube_width=10,
                                      n_steps=2, output_interval=2)
        cfg.validate()

    def test_invalid_small_domain(self) -> None:
        cfg = CapillaryInvasionConfig(nx=10, ny=20, tube_width=10)
        with pytest.raises(ValueError, match="nx must be"):
            cfg.validate()

    def test_invalid_tube_width(self) -> None:
        cfg = CapillaryInvasionConfig(nx=60, ny=20, tube_width=1)
        with pytest.raises(ValueError, match="tube_width"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = CapillaryInvasionConfig(nx=60, ny=20, tube_width=10, tau_water=0.3)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_density_order(self) -> None:
        cfg = CapillaryInvasionConfig(rho_water=0.1, rho_gas=0.9)
        with pytest.raises(ValueError, match="rho_water"):
            cfg.validate()

    def test_run_name_default(self) -> None:
        cfg = CapillaryInvasionConfig()
        name = cfg.resolved_run_name()
        assert "capillary" in name

    def test_smoke_run(self) -> None:
        """Short smoke run — check output dict keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CapillaryInvasionConfig(
                nx=60, ny=20, tube_width=12,
                G_12=0.9, tau_water=1.0, tau_gas=1.0,
                n_steps=10, output_interval=10,
                output_root=Path(tmpdir), overwrite=True,
            )
            result = run_capillary_invasion(cfg)

        assert "invasion_series" in result
        assert "washburn_exponent" in result


# ---------------------------------------------------------------------------
# TwoPhasePoiseuilleConfig
# ---------------------------------------------------------------------------

class TestTwoPhasePoiseuilleConfig:
    def test_valid_config(self) -> None:
        cfg = TwoPhasePoiseuilleConfig(ny=20, n_steps=2, output_interval=2)
        cfg.validate()

    def test_invalid_small_ny(self) -> None:
        cfg = TwoPhasePoiseuilleConfig(ny=5)
        with pytest.raises(ValueError, match="ny"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = TwoPhasePoiseuilleConfig(ny=20, tau_water=0.3)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_zero_density(self) -> None:
        cfg = TwoPhasePoiseuilleConfig(ny=20, rho_water=0.0)
        with pytest.raises(ValueError, match="densities"):
            cfg.validate()

    def test_nu_water(self) -> None:
        cfg = TwoPhasePoiseuilleConfig(tau_water=1.0)
        # ν = cs² (τ − 0.5) = (1/3)(0.5)
        expected = (1.0 / 3.0) * (1.0 - 0.5)
        assert abs(cfg.nu_water() - expected) < 1e-10

    def test_run_name_default(self) -> None:
        cfg = TwoPhasePoiseuilleConfig()
        name = cfg.resolved_run_name()
        assert "poiseuille" in name

    def test_smoke_run(self) -> None:
        """Short smoke run — check output dict keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TwoPhasePoiseuilleConfig(
                nx=4, ny=20,
                tau_water=1.0, tau_gas=0.7,
                G_x=5e-5, G_12=0.9,
                n_steps=10, output_interval=10,
                output_root=Path(tmpdir), overwrite=True,
            )
            result = run_two_phase_poiseuille(cfg)

        assert "velocity_profile" in result
        assert "analytical_profile" in result
        assert "l2_error_rel" in result
        assert len(result["velocity_profile"]) == 20  # type: ignore[arg-type]
        # After 10 steps the simulation may not be steady; just check the value is a float
        assert isinstance(result["l2_error_rel"], float)

    def test_analytical_profile_no_slip(self) -> None:
        """Analytical profile must satisfy no-slip at walls."""
        from tensorlbm.porous_media import _two_phase_poiseuille_analytical  # noqa: PLC0415
        ny, half = 30, 15
        profile = _two_phase_poiseuille_analytical(
            ny, half, G_x=5e-5, nu_w=1 / 6, nu_g=1 / 12, mu_ratio=2.0
        )
        assert len(profile) == ny
        # Wall nodes should be (approximately) zero
        assert abs(profile[0]) < 1e-10
        assert abs(profile[-1]) < 1e-10

    def test_analytical_profile_positive_interior(self) -> None:
        """Interior velocities must be non-negative (body force in +x)."""
        from tensorlbm.porous_media import _two_phase_poiseuille_analytical  # noqa: PLC0415
        ny, half = 20, 10
        profile = _two_phase_poiseuille_analytical(
            ny, half, G_x=5e-5, nu_w=1 / 6, nu_g=1 / 12, mu_ratio=2.0
        )
        # All interior values should be ≥ 0
        assert all(v >= -1e-12 for v in profile[1:-1])


# ---------------------------------------------------------------------------
# PorousDrainageConfig
# ---------------------------------------------------------------------------

class TestPorousDrainageConfig:
    def test_valid_config_random_cylinders(self) -> None:
        cfg = PorousDrainageConfig(nx=60, ny=40, geometry="random_cylinders",
                                   n_steps=2, output_interval=2)
        cfg.validate()

    def test_valid_config_tube_array(self) -> None:
        cfg = PorousDrainageConfig(nx=60, ny=40, geometry="tube_array",
                                   n_steps=2, output_interval=2)
        cfg.validate()

    def test_invalid_small_domain(self) -> None:
        cfg = PorousDrainageConfig(nx=10, ny=40)
        with pytest.raises(ValueError, match="nx must be"):
            cfg.validate()

    def test_invalid_geometry(self) -> None:
        cfg = PorousDrainageConfig.__new__(PorousDrainageConfig)
        object.__setattr__(cfg, "nx", 60)
        object.__setattr__(cfg, "ny", 40)
        object.__setattr__(cfg, "geometry", "invalid")
        object.__setattr__(cfg, "model", "sc")
        object.__setattr__(cfg, "tau_water", 1.0)
        object.__setattr__(cfg, "tau_gas", 1.0)
        object.__setattr__(cfg, "rho_water", 0.7)
        object.__setattr__(cfg, "rho_gas", 0.3)
        object.__setattr__(cfg, "n_cylinders", 5)
        object.__setattr__(cfg, "r_min", 3.0)
        object.__setattr__(cfg, "r_max", 6.0)
        object.__setattr__(cfg, "n_tubes", 3)
        object.__setattr__(cfg, "tube_width", 8)
        object.__setattr__(cfg, "seed", 42)
        object.__setattr__(cfg, "G_12", 0.9)
        object.__setattr__(cfg, "G_ads_water", 0.3)
        object.__setattr__(cfg, "G_ads_gas", 0.0)
        object.__setattr__(cfg, "n_steps", 10)
        object.__setattr__(cfg, "output_interval", 10)
        object.__setattr__(cfg, "output_root", Path("outputs"))
        object.__setattr__(cfg, "run_name", None)
        object.__setattr__(cfg, "device", "cpu")
        object.__setattr__(cfg, "overwrite", False)
        with pytest.raises(ValueError, match="geometry"):
            cfg.validate()

    def test_invalid_model(self) -> None:
        cfg = PorousDrainageConfig.__new__(PorousDrainageConfig)
        for attr, val in [
            ("nx", 60), ("ny", 40), ("geometry", "random_cylinders"), ("model", "xyz"),
            ("tau_water", 1.0), ("tau_gas", 1.0), ("rho_water", 0.7), ("rho_gas", 0.3),
            ("n_cylinders", 5), ("r_min", 3.0), ("r_max", 6.0), ("n_tubes", 3),
            ("tube_width", 8), ("seed", 42), ("G_12", 0.9), ("G_ads_water", 0.3),
            ("G_ads_gas", 0.0), ("n_steps", 10), ("output_interval", 10),
            ("output_root", Path("outputs")), ("run_name", None), ("device", "cpu"),
            ("overwrite", False),
        ]:
            object.__setattr__(cfg, attr, val)
        with pytest.raises(ValueError, match="model"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = PorousDrainageConfig(nx=60, ny=40, tau_water=0.3)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_density_order(self) -> None:
        cfg = PorousDrainageConfig(rho_water=0.1, rho_gas=0.9)
        with pytest.raises(ValueError, match="rho_water"):
            cfg.validate()

    def test_run_name_default(self) -> None:
        cfg = PorousDrainageConfig()
        name = cfg.resolved_run_name()
        assert "porous" in name

    def test_smoke_run_sc_random_cylinders(self) -> None:
        """SC model, random cylinders — short smoke run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PorousDrainageConfig(
                nx=60, ny=40, geometry="random_cylinders",
                n_cylinders=3, r_min=3.0, r_max=5.0, seed=99,
                model="sc", G_12=0.9,
                tau_water=1.0, tau_gas=1.0,
                rho_water=0.7, rho_gas=0.3,
                n_steps=5, output_interval=5,
                output_root=Path(tmpdir), overwrite=True,
            )
            result = run_porous_drainage(cfg)

        assert "saturation_series" in result
        assert "porosity" in result
        assert 0.0 < result["porosity"] < 1.0  # type: ignore[operator]
        assert len(result["saturation_series"]) >= 1  # type: ignore[arg-type]
        s0 = result["saturation_series"][0]
        assert "S_water" in s0
        assert "S_gas" in s0
        assert abs(float(s0["S_water"]) + float(s0["S_gas"]) - 1.0) < 1e-4

    def test_smoke_run_cg_tube_array(self) -> None:
        """CG model, tube array — short smoke run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PorousDrainageConfig(
                nx=60, ny=40, geometry="tube_array",
                n_tubes=2, tube_width=8,
                model="cg", G_12=0.9,
                tau_water=1.0, tau_gas=1.0,
                rho_water=0.7, rho_gas=0.3,
                n_steps=5, output_interval=5,
                output_root=Path(tmpdir), overwrite=True,
            )
            result = run_porous_drainage(cfg)

        assert "saturation_series" in result
        assert "porosity" in result

    def test_saturation_sums_to_one(self) -> None:
        """S_water + S_gas should sum to 1 at each diagnostic step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PorousDrainageConfig(
                nx=60, ny=40, geometry="tube_array",
                n_tubes=2, tube_width=8,
                model="cg",   # CG is more numerically stable for smoke tests
                n_steps=5, output_interval=5,
                output_root=Path(tmpdir), overwrite=True,
            )
            result = run_porous_drainage(cfg)

        for entry in result["saturation_series"]:  # type: ignore[union-attr]
            sw = float(entry["S_water"])  # type: ignore[index]
            sg = float(entry["S_gas"])  # type: ignore[index]
            assert abs(sw + sg - 1.0) < 1e-3, f"S_w + S_g = {sw + sg} ≠ 1"
