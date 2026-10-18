"""Physical acceptance checks for the Shan-Chen spinodal benchmarks.

Fast tests check mass accounting and report serialization. Slow tests require
actual phase separation in 2-D and 3-D and run on main/nightly CI, not PR smoke
jobs. A density ratio alone is a coarse morphology criterion, not a coexistence
curve validation; EOS/Maxwell and grid-convergence studies remain separate work.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tensorlbm.multiphase_benchmarks import (
    Spinodal3DConfig,
    SpinodaleConfig,
    run_spinodal_decomposition,
    run_spinodal_decomposition_3d,
)


def test_spinodal_2d_reports_mass_conservation(tmp_path: Path) -> None:
    result = run_spinodal_decomposition(
        SpinodaleConfig(
            nx=16,
            ny=16,
            n_steps=5,
            output_interval=5,
            output_root=tmp_path,
            device="cpu",
        )
    )
    assert "relative_mass_drift" in result
    assert math.isfinite(float(result["relative_mass_drift"]))
    assert float(result["relative_mass_drift"]) < 1e-5
    assert (tmp_path / "spinodal" / "spinodal_G-4.0_rho0.70_nx16_steps5" / "spinodal.json").exists()


def test_spinodal_3d_reports_mass_conservation(tmp_path: Path) -> None:
    result = run_spinodal_decomposition_3d(
        Spinodal3DConfig(
            nx=12,
            ny=12,
            nz=12,
            n_steps=3,
            output_interval=3,
            output_root=tmp_path,
            device="cpu",
        )
    )
    assert "relative_mass_drift" in result
    assert math.isfinite(float(result["relative_mass_drift"]))
    assert float(result["relative_mass_drift"]) < 1e-5
    assert (tmp_path / "spinodal_3d" / "spinodal_3d_G-4.0_rho0.70_n12x12x12_steps3" / "spinodal_3d.json").exists()


@pytest.mark.slow
def test_spinodal_2d_physically_separates(tmp_path: Path) -> None:
    """Longer 2-D run must cross the benchmark's documented density-ratio gate."""
    result = run_spinodal_decomposition(
        SpinodaleConfig(
            nx=64,
            ny=64,
            G=-4.0,
            tau=1.0,
            rho0=0.7,
            noise_amp=0.05,
            n_steps=1200,
            output_interval=200,
            seed=42,
            output_root=tmp_path,
            device="cpu",
        )
    )
    assert float(result["relative_mass_drift"]) < 1e-5
    assert float(result["density_ratio"]) > 2.0
    assert result["phase_separated"] is True


@pytest.mark.slow
def test_spinodal_3d_physically_separates(tmp_path: Path) -> None:
    """Longer D3Q19 run must cross the same explicit density-ratio gate."""
    result = run_spinodal_decomposition_3d(
        Spinodal3DConfig(
            nx=24,
            ny=24,
            nz=24,
            G=-4.0,
            tau=1.0,
            rho0=0.7,
            noise_amp=0.05,
            n_steps=500,
            output_interval=100,
            seed=42,
            output_root=tmp_path,
            device="cpu",
        )
    )
    assert float(result["relative_mass_drift"]) < 1e-5
    assert float(result["density_ratio"]) > 2.0
    assert result["phase_separated"] is True
