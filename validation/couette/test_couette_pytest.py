"""Pytest tests for D2Q9 plane Couette flow validation.

Acceptance criteria (machine判据):
    1. Linear profile L2 relative error <= 2%
    2. Top wall no-slip error |u(ny-1) - U| / U <= 3%
    3. Bottom wall no-slip error |u(0)| / U <= 3%
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from couette_flow import CouetteConfig, run_couette

# Acceptance thresholds
L2_ERROR_THRESHOLD = 0.02       # 2%
NO_SLIP_THRESHOLD = 0.03        # 3%


@pytest.fixture(scope="module")
def couette_results():
    """Run Couette simulation once for all tests."""
    cfg = CouetteConfig(
        nx=16,
        ny=16,
        u_top=0.05,
        nu=0.02,
        n_steps=5000,
        n_warmup=2000,
        device="cpu",
    )
    return run_couette(cfg), cfg


class TestCouetteL2Error:
    """Test L2 error of velocity profile vs analytical solution."""

    def test_l2_error_below_threshold(self, couette_results):
        results, cfg = couette_results
        assert results["l2_error"] <= L2_ERROR_THRESHOLD, (
            f"L2 error {results['l2_error']*100:.4f}% exceeds threshold {L2_ERROR_THRESHOLD*100:.1f}%"
        )

    def test_l2_error_value(self, couette_results):
        results, _ = couette_results
        # Should be well below 2%
        assert results["l2_error"] < 0.015, (
            f"L2 error {results['l2_error']*100:.4f}% is too high"
        )


class TestCouetteNoSlip:
    """Test no-slip boundary conditions."""

    def test_bottom_wall_no_slip(self, couette_results):
        results, cfg = couette_results
        rel_error = results["no_slip_bottom"] / cfg.u_top
        assert rel_error <= NO_SLIP_THRESHOLD, (
            f"Bottom wall no-slip error {rel_error*100:.4f}% exceeds threshold {NO_SLIP_THRESHOLD*100:.1f}%"
        )

    def test_top_wall_no_slip(self, couette_results):
        results, cfg = couette_results
        rel_error = results["no_slip_top"] / cfg.u_top
        assert rel_error <= NO_SLIP_THRESHOLD, (
            f"Top wall no-slip error {rel_error*100:.4f}% exceeds threshold {NO_SLIP_THRESHOLD*100:.1f}%"
        )

    def test_bottom_velocity_zero(self, couette_results):
        results, _ = couette_results
        assert abs(results["ux_profile"][0].item()) < 1e-6, (
            f"Bottom wall velocity {results['ux_profile'][0].item()} should be ~0"
        )

    def test_top_velocity_equals_u_top(self, couette_results):
        results, cfg = couette_results
        assert abs(results["ux_profile"][-1].item() - cfg.u_top) < 1e-6, (
            f"Top wall velocity {results['ux_profile'][-1].item()} should be {cfg.u_top}"
        )


class TestCouetteProfile:
    """Test velocity profile properties."""

    def test_profile_monotonic(self, couette_results):
        """Velocity should increase monotonically from bottom to top."""
        results, _ = couette_results
        ux = results["ux_profile"].numpy()
        # Allow small numerical noise
        diffs = ux[1:] - ux[:-1]
        assert all(d > -1e-6 for d in diffs), (
            "Velocity profile should be monotonically increasing"
        )

    def test_transverse_velocity_near_zero(self, couette_results):
        """u_y should be approximately zero everywhere."""
        results, _ = couette_results
        uy = results["uy_profile"].numpy()
        assert max(abs(u) for u in uy) < 0.001, (
            f"Transverse velocity max {max(abs(u) for u in uy)} should be ~0"
        )

    def test_profile_shape(self, couette_results):
        """Profile should be approximately linear."""
        results, cfg = couette_results
        ux = results["ux_profile"].numpy()
        y = results["y_coords"].numpy()
        h_eff = cfg.ny - 1
        expected = cfg.u_top * y / h_eff
        # Max pointwise error should be small
        max_error = max(abs(ux[i] - expected[i]) for i in range(len(ux)))
        assert max_error < 0.005, (
            f"Max pointwise error {max_error:.6f} is too large"
        )
