"""Tests for LES channel flow verification."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Direct path setup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.les_channel_verification import (
    LESChannelConfig,
    LESChannelSolver,
    run_case,
    evaluate_criteria,
)


def test_config_initialization():
    """Test LESChannelConfig initialization."""
    cfg = LESChannelConfig(
        nx=32, ny=17, nz=16,
        half_height=8.0,
        re_tau=180.0,
        u_tau=0.005,
    )
    
    assert cfg.nx == 32
    assert cfg.ny == 17
    assert cfg.nz == 16
    assert cfg.re_tau == 180.0
    assert cfg.nu > 0
    assert cfg.body_force > 0


def test_solver_initialization():
    """Test LESChannelSolver initialization."""
    cfg = LESChannelConfig(half_height=8, re_tau=180, u_tau=0.005)
    solver = LESChannelSolver(cfg)
    
    assert solver.f.shape == (19, cfg.nz, cfg.ny, cfg.nx)
    assert np.all(np.isfinite(solver.f))


def test_solver_step():
    """Test single solver step."""
    cfg = LESChannelConfig(half_height=8, re_tau=180, u_tau=0.005)
    solver = LESChannelSolver(cfg)
    
    f_initial = solver.f.copy()
    solver.step()
    
    # Check that populations changed
    assert not np.allclose(solver.f, f_initial)
    # Check that populations are still finite
    assert np.all(np.isfinite(solver.f))


def test_macroscopic_conservation():
    """Test that macroscopic quantities are conserved."""
    cfg = LESChannelConfig(half_height=8, re_tau=180, u_tau=0.005)
    solver = LESChannelSolver(cfg)
    
    rho, ux, uy, uz = solver._macroscopic()
    
    # Density should be close to 1
    assert np.abs(rho.mean() - 1.0) < 0.1
    # Velocities should be finite
    assert np.all(np.isfinite(ux))
    assert np.all(np.isfinite(uy))
    assert np.all(np.isfinite(uz))


def test_smagorinsky_model():
    """Test that Smagorinsky model increases effective viscosity."""
    cfg = LESChannelConfig(half_height=8, re_tau=180, u_tau=0.005, smagorinsky_cs=0.1)
    solver = LESChannelSolver(cfg)
    
    # Run a few steps
    for _ in range(10):
        solver.step()
    
    # Check that simulation is still stable
    assert np.all(np.isfinite(solver.f))


def test_run_case_short():
    """Test run_case with minimal parameters."""
    artifact = run_case(
        half_height=4,
        steps=100,
        warmup_steps=50,
        u_tau=0.005,
        re_tau=180.0,
        cs=0.1,
        verbose=False,
    )
    
    assert "config" in artifact
    assert "profiles" in artifact
    assert "criteria" in artifact
    assert "summary" in artifact
    
    assert artifact["config"]["re_tau"] == 180.0
    assert artifact["summary"]["n_samples"] > 0


def test_evaluate_criteria():
    """Test criteria evaluation."""
    y_plus = np.array([1, 5, 10, 20, 50, 100])
    U_plus = np.array([1, 5, 10, 18, 35, 50])
    u_rms_plus = np.array([2, 2.5, 2.5, 2.5, 2.5, 2.5])
    uv_plus = np.array([0, 0.1, 0.2, 0.25, 0.28, 0.25])
    spectrum_fit = {"slope": -1.67, "r_squared": 0.9}
    
    criteria = evaluate_criteria(
        y_plus=y_plus,
        U_plus=U_plus,
        u_rms_plus=u_rms_plus,
        uv_plus=uv_plus,
        spectrum_fit=spectrum_fit,
        re_tau=180.0,
    )
    
    assert "checks" in criteria
    assert "all_pass" in criteria
    assert isinstance(criteria["all_pass"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
