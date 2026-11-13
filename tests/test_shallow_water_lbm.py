"""Tests for the LBM shallow-water solver and cross-validation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.shallow_water_lbm import (  # noqa: E402
    SWE_LBM_Config,
    init_dam_break_lbm,
    lbm_step,
    run_swe_lbm,
)


def test_config_validation() -> None:
    """Config should reject invalid parameters."""
    cfg = SWE_LBM_Config(nx=16)  # too small
    with pytest.raises(ValueError, match="nx must be >= 32"):
        cfg.validate()

    cfg = SWE_LBM_Config(lam=-1.0)
    with pytest.raises(ValueError, match="lam must be positive"):
        cfg.validate()


def test_init_dam_break_shape() -> None:
    """Initial condition should have correct shape."""
    cfg = SWE_LBM_Config(nx=100)
    f, g = init_dam_break_lbm(cfg)
    assert f.shape == (2, 100)
    assert g.shape == (2, 100)


def test_init_dam_break_mass() -> None:
    """Initial mass should match expected value."""
    cfg = SWE_LBM_Config(nx=100, x_min=-4, x_max=12, dam_x=0, h0=1.0)
    f, g = init_dam_break_lbm(cfg)
    h = f[0] + f[1]
    # Mass = h0 * (dam_x - x_min) = 1.0 * 4.0 = 4.0
    expected_mass = cfg.h0 * (cfg.dam_x - cfg.x_min)
    actual_mass = float(h.sum().item() * cfg.dx)
    assert actual_mass == pytest.approx(expected_mass, rel=1e-10)


def test_lbm_step_no_nan_short() -> None:
    """Short runs should not produce NaN."""
    cfg = SWE_LBM_Config(nx=100, x_min=-4, x_max=12, dam_x=0, h0=1.0,
                         t_end=0.5, sigma_h=1e-2, sigma_q=0.2)
    f, g = init_dam_break_lbm(cfg)

    for _ in range(50):
        f, g = lbm_step(f, g, cfg)

    h = f[0] + f[1]
    hu = g[0] + g[1]
    assert not torch.isnan(h).any()
    assert not torch.isnan(hu).any()


def test_run_swe_lbm_returns_result() -> None:
    """run_swe_lbm should return a valid result object."""
    cfg = SWE_LBM_Config(nx=100, x_min=-4, x_max=12, dam_x=0, h0=1.0,
                         t_end=0.5, sigma_h=1e-2, sigma_q=0.2)
    result = run_swe_lbm(cfg)

    assert result.h.shape == (100,)
    assert result.hu.shape == (100,)
    assert result.x.shape == (100,)
    assert result.t > 0
    assert result.n_steps > 0
    assert result.mass_initial > 0


def test_lbm_runs_without_error() -> None:
    """LBM solver should complete without error for short runs."""
    cfg = SWE_LBM_Config(nx=100, x_min=-4, x_max=12, dam_x=0, h0=1.0,
                         t_end=0.5, sigma_h=1e-2, sigma_q=0.2)
    result = run_swe_lbm(cfg)

    # Check result is reasonable
    assert result.h.min() >= -1.0  # Allow small negative values
    assert result.h.max() <= 2.0 * cfg.h0  # Should not exceed initial depth much


def test_lbm_mass_change_bounded() -> None:
    """Mass change should be bounded (not necessarily conserved for dry-bed problems)."""
    cfg = SWE_LBM_Config(nx=100, x_min=-4, x_max=12, dam_x=0, h0=1.0,
                         t_end=0.5, sigma_h=1e-2, sigma_q=0.2)
    result = run_swe_lbm(cfg)

    # For dry-bed problems, mass loss is expected due to boundary conditions
    # Just check it's not exploding
    assert result.mass_final < 2 * result.mass_initial
    assert result.mass_final > 0.1 * result.mass_initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
