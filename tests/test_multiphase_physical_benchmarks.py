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
import torch

from tensorlbm.multiphase_benchmarks import (
    Spinodal3DConfig,
    SpinodaleConfig,
    run_spinodal_decomposition,
    run_spinodal_decomposition_3d,
)


@pytest.mark.parametrize("cls", [SpinodaleConfig, Spinodal3DConfig])
@pytest.mark.parametrize("G", [-5.0, 0.0, float("nan")])
def test_spinodal_rejects_nonattractive_or_nonfinite_coupling(cls, G):
    with pytest.raises(ValueError, match="G must be > 0"):
        cls(G=G).validate()


def test_single_component_attraction_points_toward_density_peak():
    from tensorlbm.d3q19 import C, equilibrium3d
    from tensorlbm.multiphase import psi_exp, sc_single_component_force
    from tensorlbm.multiphase3d import collide_sc_single_component_3d

    x = torch.arange(32, dtype=torch.float64)
    rho = (0.7 + 0.02 * torch.cos(2 * torch.pi * x / 32))[None, :].expand(16, 32)
    fx, _ = sc_single_component_force(rho, G=5.0, psi_fn=psi_exp)
    # Between the maximum at x=0 and minimum at x=16, attraction points left.
    assert (fx[:, 1:16] < 0).all()
    reversed_fx, _ = sc_single_component_force(rho, G=-5.0, psi_fn=psi_exp)
    torch.testing.assert_close(reversed_fx, -fx)
    rho3 = rho[None].expand(12, 16, 32)
    z = torch.zeros_like(rho3)
    f = equilibrium3d(rho3, z, z, z)
    out = collide_sc_single_component_3d(f, G=5.0, psi_fn=psi_exp)
    fx3 = ((out - f) * C[:, 0, None, None, None]).sum(0)
    assert (fx3[:, :, 1:16] < 0).all()


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
    assert (tmp_path / "spinodal" / "spinodal_G5.0_rho0.70_nx16_steps5" / "spinodal.json").exists()


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
    assert (
        tmp_path / "spinodal_3d" / "spinodal_3d_G5.0_rho0.70_n12x12x12_steps3" / "spinodal_3d.json"
    ).exists()


@pytest.mark.slow
def test_spinodal_2d_physically_separates(tmp_path: Path) -> None:
    """Longer 2-D run must cross the benchmark's documented density-ratio gate."""
    result = run_spinodal_decomposition(
        SpinodaleConfig(
            nx=64,
            ny=64,
            G=5.0,
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
            G=5.0,
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
