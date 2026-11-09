"""Pytest tests for IN99-N03: LBM turbulent channel flow validation."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensorlbm.turbulence.turbulent_channel3d import (
    TurbulentChannel3D,
    TurbulentChannel3DConfig,
    log_law_u_plus,
    reichardt_u_plus,
)


class TestTurbulentChannel3DConfig:
    """Test configuration validation."""

    def test_default_config(self):
        cfg = TurbulentChannel3DConfig()
        assert cfg.re_tau == 180.0
        assert cfg.height == 62
        assert cfg.nu > 0
        assert cfg.tau_plus > 0.5

    def test_tau_plus_calculation(self):
        cfg = TurbulentChannel3DConfig(re_tau=180, u_tau=0.01, ny=64)
        expected_nu = 0.01 * 31 / 180
        expected_tau = 0.5 + 3.0 * expected_nu
        assert abs(cfg.nu - expected_nu) < 1e-10
        assert abs(cfg.tau_plus - expected_tau) < 1e-10

    def test_smagorinsky_config_valid(self):
        """Smagorinsky config should be valid for TurbulentChannel3D."""
        cfg = TurbulentChannel3DConfig(
            re_tau=180, u_tau=0.01, ny=64,
            smagorinsky_cs=0.1,
        )
        assert cfg.smagorinsky_cs == 0.1

    def test_smagorinsky_zero(self):
        """Zero Smagorinsky should be valid (molecular only)."""
        cfg = TurbulentChannel3DConfig(
            re_tau=180, u_tau=0.01, ny=64,
            smagorinsky_cs=0.0,
        )
        assert cfg.smagorinsky_cs == 0.0


class TestAnalyticReferences:
    """Test analytic reference functions."""

    def test_log_law(self):
        """Log law should give u+ = y+ in viscous sublayer limit."""
        # At y+=100, log law should be approximately correct
        y_plus = 100.0
        u_plus = log_law_u_plus(y_plus)
        expected = np.log(100) / 0.41 + 5.2
        assert abs(float(u_plus) - expected) < 0.01

    def test_reichardt(self):
        """Reichardt law should be smooth and positive."""
        y_plus = np.linspace(1, 1000, 100)
        u_plus = reichardt_u_plus(y_plus)
        u_plus_np = np.array(u_plus) if hasattr(u_plus, 'numpy') else np.asarray(u_plus)
        assert (u_plus_np > 0).all()
        assert (np.diff(u_plus_np) > 0).all()  # Monotonically increasing


class TestTurbulentChannel3DSmoke:
    """Smoke tests for the solver (small grids, few steps)."""

    @pytest.fixture
    def small_config(self):
        return TurbulentChannel3DConfig(
            nx=32, nz=16, ny=20, re_tau=180, u_tau=0.01,
            steps=200, warmup_steps=100, sample_interval=5,
            perturbation_amplitude=0.35, perturbation_modes=4,
            seed=42, device="cpu", dtype="float32",
            smagorinsky_cs=0.0, wall_scheme="interpolated",
            force_scheme="trt_split", label="smoke_test",
        )

    def test_solver_runs(self, small_config):
        """Solver should complete without errors."""
        sim = TurbulentChannel3D(small_config)
        result = sim.run()
        assert result is not None
        assert "u_tau_routes" in result
        assert "wall_units" in result

    def test_u_tau_routes_exist(self, small_config):
        """All three u_tau routes should be present."""
        sim = TurbulentChannel3D(small_config)
        result = sim.run()
        routes = result["u_tau_routes"]
        assert "body_force" in routes
        assert "stress_fit" in routes
        assert "wall_exchange" in routes

    def test_profiles_exist(self, small_config):
        """Wall unit profiles should be present."""
        sim = TurbulentChannel3D(small_config)
        result = sim.run()
        wu = result["wall_units"]
        assert "y_plus" in wu
        assert "u_plus" in wu
        assert "uu_plus" in wu
        assert "vv_plus" in wu
        assert "ww_plus" in wu
        assert "uv_plus" in wu

    def test_smagorinsky_runs(self):
        """Solver with Smagorinsky should complete."""
        cfg = TurbulentChannel3DConfig(
            nx=32, nz=16, ny=20, re_tau=180, u_tau=0.01,
            steps=200, warmup_steps=100, sample_interval=5,
            perturbation_amplitude=0.35, perturbation_modes=4,
            seed=42, device="cpu", dtype="float32",
            smagorinsky_cs=0.1, wall_scheme="interpolated",
            force_scheme="trt_split", label="smag_smoke",
            collision_model="cumulant",
        )
        sim = TurbulentChannel3D(cfg)
        result = sim.run()
        assert result is not None


class TestDNSReference:
    """Test DNS reference data loading."""

    def test_mkm180_exists(self):
        dns_dir = Path(__file__).parent.parent / "benchmarks" / "in99_n03_channel_turb" / "dns_ref"
        assert (dns_dir / "mkm180.csv").exists()

    def test_mkm395_exists(self):
        dns_dir = Path(__file__).parent.parent / "benchmarks" / "in99_n03_channel_turb" / "dns_ref"
        assert (dns_dir / "mkm395.csv").exists()

    def test_hj590_exists(self):
        dns_dir = Path(__file__).parent.parent / "benchmarks" / "in99_n03_channel_turb" / "dns_ref"
        assert (dns_dir / "hj590.csv").exists()

    def test_mkm180_format(self):
        import csv
        dns_dir = Path(__file__).parent.parent / "benchmarks" / "in99_n03_channel_turb" / "dns_ref"
        with open(dns_dir / "mkm180.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 10
        assert "y_plus" in rows[0]
        assert "u_plus" in rows[0]
        assert "uu_plus" in rows[0]
        assert "vv_plus" in rows[0]
        assert "ww_plus" in rows[0]
        assert "uv_plus" in rows[0]


class TestValidationResults:
    """Test validation results if they exist."""

    @pytest.fixture
    def summary_path(self):
        return Path(__file__).parent / "runs" / "summary.json"

    def test_summary_exists(self, summary_path):
        """Summary file should exist after validation run."""
        # This test will be skipped if validation hasn't been run yet
        if not summary_path.exists():
            pytest.skip("Validation results not yet generated")
        assert summary_path.exists()

    def test_summary_format(self, summary_path):
        """Summary should have expected structure."""
        if not summary_path.exists():
            pytest.skip("Validation results not yet generated")
        with open(summary_path) as f:
            summary = json.load(f)
        assert "mission" in summary
        assert summary["mission"] == "IN99-N03"
