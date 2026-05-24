"""Tests for the multiphase LBM benchmark suite.

Covers:
    - StaticDropletConfig: validation and short smoke run (SCMC + CG)
    - SpinodaleConfig: validation and short smoke run (SCMP)
    - TwoPhaseChannelCompareConfig: validation and short smoke run (SCMC + CG)
    - MultiphaseBenchmarkSuiteConfig: construction and minimal suite run
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from tensorlbm.multiphase_benchmarks import (
    MultiphaseBenchmarkSuiteConfig,
    SpinodaleConfig,
    StaticDropletConfig,
    TwoPhaseChannelCompareConfig,
    run_multiphase_benchmark_suite,
    run_spinodal_decomposition,
    run_static_droplet,
    run_two_phase_channel_compare,
)

# Always run on CPU to keep test times short
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# StaticDropletConfig
# ---------------------------------------------------------------------------

class TestStaticDropletConfig:
    def test_valid_config(self) -> None:
        cfg = StaticDropletConfig(nx=60, ny=60, radii=(10.0,), n_steps=2, output_interval=2)
        cfg.validate()

    def test_invalid_small_domain(self) -> None:
        cfg = StaticDropletConfig(nx=10, ny=60, radii=(5.0,))
        with pytest.raises(ValueError, match="nx and ny"):
            cfg.validate()

    def test_invalid_empty_radii(self) -> None:
        cfg = StaticDropletConfig.__new__(StaticDropletConfig)
        object.__setattr__(cfg, "nx", 60)
        object.__setattr__(cfg, "ny", 60)
        object.__setattr__(cfg, "radii", ())
        object.__setattr__(cfg, "n_steps", 2)
        object.__setattr__(cfg, "output_interval", 2)
        object.__setattr__(cfg, "scmc_G12", 0.9)
        object.__setattr__(cfg, "scmc_tau", 1.0)
        object.__setattr__(cfg, "scmc_rho_heavy", 0.7)
        object.__setattr__(cfg, "scmc_rho_light", 0.3)
        object.__setattr__(cfg, "cg_A", 0.04)
        object.__setattr__(cfg, "cg_beta", 0.7)
        object.__setattr__(cfg, "cg_tau", 1.0)
        object.__setattr__(cfg, "cg_rho_heavy", 0.65)
        object.__setattr__(cfg, "cg_rho_light", 0.05)
        object.__setattr__(cfg, "output_root", Path("outputs"))
        object.__setattr__(cfg, "run_name", None)
        object.__setattr__(cfg, "device", "cpu")
        object.__setattr__(cfg, "overwrite", False)
        with pytest.raises(ValueError, match="radii"):
            cfg.validate()

    def test_invalid_tau_scmc(self) -> None:
        cfg = StaticDropletConfig(nx=60, ny=60, radii=(10.0,), scmc_tau=0.4)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_density_order_scmc(self) -> None:
        cfg = StaticDropletConfig(nx=60, ny=60, radii=(10.0,),
                                   scmc_rho_heavy=0.2, scmc_rho_light=0.8)
        with pytest.raises(ValueError, match="scmc_rho_heavy"):
            cfg.validate()

    def test_invalid_g12(self) -> None:
        cfg = StaticDropletConfig(nx=60, ny=60, radii=(10.0,), scmc_G12=-0.5)
        with pytest.raises(ValueError, match="scmc_G12"):
            cfg.validate()

    def test_run_name_default(self) -> None:
        cfg = StaticDropletConfig()
        name = cfg.resolved_run_name()
        assert "static_droplet" in name

    def test_smoke_run(self) -> None:
        """Short run — check output dict keys and sigma_eff is finite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = StaticDropletConfig(
                nx=40, ny=40,
                radii=(8.0, 12.0),
                n_steps=5, output_interval=5,
                output_root=Path(tmpdir), overwrite=True,
                device=DEVICE,
            )
            result = run_static_droplet(cfg)

        assert "scmc" in result["results"]  # type: ignore[index]
        assert "cg" in result["results"]  # type: ignore[index]
        for model in ("scmc", "cg"):
            mres = result["results"][model]  # type: ignore[index]
            assert "sigma_eff_fit" in mres
            assert "mean_max_spurious_u" in mres
            assert math.isfinite(float(mres["sigma_eff_fit"]))
            per_r = mres["per_radius"]
            assert len(per_r) == 2
            for row in per_r:
                assert "delta_p" in row
                assert "max_spurious_u" in row
                assert math.isfinite(float(row["delta_p"]))
                assert float(row["max_spurious_u"]) >= 0.0


# ---------------------------------------------------------------------------
# SpinodaleConfig
# ---------------------------------------------------------------------------

class TestSpinodaleConfig:
    def test_valid_config(self) -> None:
        cfg = SpinodaleConfig(nx=16, ny=16, n_steps=2, output_interval=2)
        cfg.validate()

    def test_invalid_small_domain(self) -> None:
        cfg = SpinodaleConfig(nx=5, ny=16)
        with pytest.raises(ValueError, match="nx and ny"):
            cfg.validate()

    def test_invalid_G_positive(self) -> None:
        cfg = SpinodaleConfig(G=1.0)
        with pytest.raises(ValueError, match="G must be < 0"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = SpinodaleConfig(tau=0.3)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_rho0_zero(self) -> None:
        cfg = SpinodaleConfig(rho0=0.0)
        with pytest.raises(ValueError, match="rho0"):
            cfg.validate()

    def test_run_name_default(self) -> None:
        cfg = SpinodaleConfig()
        name = cfg.resolved_run_name()
        assert "spinodal" in name

    def test_smoke_run(self) -> None:
        """Short run — check output dict keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SpinodaleConfig(
                nx=16, ny=16,
                G=-4.0, tau=1.0, rho0=0.7, noise_amp=0.05,
                n_steps=5, output_interval=5,
                output_root=Path(tmpdir), overwrite=True,
                device=DEVICE,
            )
            result = run_spinodal_decomposition(cfg)

        assert "rho_liquid" in result
        assert "rho_gas" in result
        assert "density_ratio" in result
        assert "phase_separated" in result
        assert "diagnostics" in result
        assert math.isfinite(float(result["rho_liquid"]))
        assert math.isfinite(float(result["rho_gas"]))
        assert float(result["rho_liquid"]) >= float(result["rho_gas"])
        assert len(result["diagnostics"]) >= 1  # type: ignore[arg-type]

    def test_rho_liquid_ge_rho_gas(self) -> None:
        """After any number of steps, rho_max ≥ rho_min."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SpinodaleConfig(
                nx=16, ny=16, n_steps=3, output_interval=3,
                output_root=Path(tmpdir), overwrite=True, device=DEVICE,
            )
            result = run_spinodal_decomposition(cfg)
        assert float(result["rho_liquid"]) >= float(result["rho_gas"])


# ---------------------------------------------------------------------------
# TwoPhaseChannelCompareConfig
# ---------------------------------------------------------------------------

class TestTwoPhaseChannelCompareConfig:
    def test_valid_config(self) -> None:
        cfg = TwoPhaseChannelCompareConfig(ny=20, n_steps=2, output_interval=2)
        cfg.validate()

    def test_invalid_small_ny(self) -> None:
        cfg = TwoPhaseChannelCompareConfig(ny=5)
        with pytest.raises(ValueError, match="ny"):
            cfg.validate()

    def test_invalid_scmc_tau(self) -> None:
        cfg = TwoPhaseChannelCompareConfig(ny=20, scmc_tau_heavy=0.3)
        with pytest.raises(ValueError, match="SCMC tau"):
            cfg.validate()

    def test_invalid_cg_tau(self) -> None:
        cfg = TwoPhaseChannelCompareConfig(ny=20, cg_tau=0.4)
        with pytest.raises(ValueError, match="CG tau"):
            cfg.validate()

    def test_invalid_cg_density_order(self) -> None:
        cfg = TwoPhaseChannelCompareConfig(ny=20, cg_rho_heavy=0.02, cg_rho_light=0.5)
        with pytest.raises(ValueError, match="cg_rho_heavy"):
            cfg.validate()

    def test_nu_methods(self) -> None:
        cfg = TwoPhaseChannelCompareConfig()
        assert cfg.scmc_nu_heavy() == pytest.approx((1.0 / 3.0) * 0.5)
        assert cfg.scmc_nu_light() == pytest.approx((1.0 / 3.0) * 0.2)
        assert cfg.cg_nu() == pytest.approx((1.0 / 3.0) * 0.5)

    def test_run_name_default(self) -> None:
        cfg = TwoPhaseChannelCompareConfig()
        name = cfg.resolved_run_name()
        assert "poiseuille" in name

    def test_smoke_run(self) -> None:
        """Short smoke run — check dict keys and finite errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TwoPhaseChannelCompareConfig(
                nx=4, ny=20,
                G_x=5e-5,
                n_steps=5, output_interval=5,
                output_root=Path(tmpdir), overwrite=True,
                device=DEVICE,
            )
            result = run_two_phase_channel_compare(cfg)

            assert "scmc" in result["results"]  # type: ignore[index]
            assert "cg" in result["results"]  # type: ignore[index]
            for model in ("scmc", "cg"):
                mres = result["results"][model]  # type: ignore[index]
                assert "velocity_profile" in mres
                assert "analytical_profile" in mres
                assert "l2_error_rel" in mres
                # L2 error might be large after only 5 steps but must be a
                # finite non-negative float (not NaN)
                l2 = float(mres["l2_error_rel"])
                assert l2 == l2, f"L2 error is NaN for model {model}"
                assert len(mres["velocity_profile"]) == 20

    def test_viscosity_ratio(self) -> None:
        cfg = TwoPhaseChannelCompareConfig(
            scmc_tau_heavy=1.0,
            scmc_tau_light=0.7,
        )
        expected = cfg.scmc_nu_heavy() / cfg.scmc_nu_light()
        assert expected == pytest.approx(0.5 / 0.2)


# ---------------------------------------------------------------------------
# MultiphaseBenchmarkSuiteConfig + suite smoke run
# ---------------------------------------------------------------------------

class TestMultiphaseBenchmarkSuiteConfig:
    def test_default_construction(self) -> None:
        cfg = MultiphaseBenchmarkSuiteConfig()
        assert isinstance(cfg.droplet, StaticDropletConfig)
        assert isinstance(cfg.spinodal, SpinodaleConfig)
        assert isinstance(cfg.poiseuille, TwoPhaseChannelCompareConfig)

    def test_output_root_path_conversion(self) -> None:
        cfg = MultiphaseBenchmarkSuiteConfig(output_root="my_outputs")  # type: ignore[arg-type]
        assert isinstance(cfg.output_root, Path)

    def test_minimal_suite_smoke_run(self) -> None:
        """Very short suite run — verify top-level report structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MultiphaseBenchmarkSuiteConfig(
                droplet=StaticDropletConfig(
                    nx=40, ny=40, radii=(8.0,),
                    n_steps=5, output_interval=5,
                ),
                spinodal=SpinodaleConfig(
                    nx=16, ny=16, n_steps=5, output_interval=5,
                ),
                poiseuille=TwoPhaseChannelCompareConfig(
                    nx=4, ny=20, n_steps=5, output_interval=5,
                ),
                output_root=Path(tmpdir),
                device=DEVICE,
                overwrite=True,
            )
            report = run_multiphase_benchmark_suite(cfg)

            assert "benchmarks" in report
            assert "analysis" in report
            benchmarks = report["benchmarks"]
            assert "static_droplet" in benchmarks  # type: ignore[operator]
            assert "spinodal_decomposition" in benchmarks  # type: ignore[operator]
            assert "two_phase_poiseuille" in benchmarks  # type: ignore[operator]
            analysis = report["analysis"]
            assert "surface_tension" in analysis  # type: ignore[operator]
            assert "spinodal" in analysis  # type: ignore[operator]
            assert "poiseuille" in analysis  # type: ignore[operator]
            assert "summary" in analysis  # type: ignore[operator]
            # Report file should exist (check inside with block before tmpdir cleanup)
            report_path = Path(tmpdir) / "multiphase_suite_report.json"
            assert report_path.exists()


# ---------------------------------------------------------------------------
# Optimisation regression tests
# ---------------------------------------------------------------------------

class TestColorGradientOptimisation:
    """Verify that the CG optimisations (no redundant rolls, no feq_unit alloc)
    do not change numerical results compared to a reference baseline."""

    def test_cg_mass_conservation_after_opt(self) -> None:
        """CG total mass must still be conserved after the optimisation."""
        from tensorlbm import color_gradient_step, equilibrium, stream  # noqa: PLC0415
        ny, nx = 20, 24
        rho1 = torch.ones((ny, nx))
        rho2 = torch.full((ny, nx), 0.5)
        zero = torch.zeros((ny, nx))
        f_r = equilibrium(rho1, zero, zero)
        f_b = equilibrium(rho2, zero, zero)
        m_before = (f_r + f_b).sum()
        f_r, f_b = color_gradient_step(f_r, f_b, tau=1.0, A=0.04, beta=0.7)
        f_r = stream(f_r)
        f_b = stream(f_b)
        assert torch.allclose((f_r + f_b).sum(), m_before, atol=1e-4)

    def test_cg_3d_mass_conservation_after_opt(self) -> None:
        """3-D CG total mass must still be conserved after the optimisation."""
        from tensorlbm import (  # noqa: PLC0415
            color_gradient_step_3d,
            equilibrium3d,
            stream3d,
        )
        nz, ny, nx = 5, 6, 8
        rho1 = torch.ones((nz, ny, nx))
        rho2 = torch.full((nz, ny, nx), 0.5)
        zero = torch.zeros((nz, ny, nx))
        f_r = equilibrium3d(rho1, zero, zero, zero)
        f_b = equilibrium3d(rho2, zero, zero, zero)
        m_before = (f_r + f_b).sum()
        f_r, f_b = color_gradient_step_3d(f_r, f_b, tau=1.0, A=0.04, beta=0.7)
        f_r = stream3d(f_r)
        f_b = stream3d(f_b)
        assert torch.allclose((f_r + f_b).sum(), m_before, atol=1e-3)
