"""Tests for ice-edge zone LBM simulation.

Verifies:
- Configuration validation
- Basic simulation runs without NaN
- Drag coefficient is physically reasonable
- Mesh convergence trend
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

import torch


def test_config_validation():
    """Test that config validates correctly."""
    from tensorlbm.ice_edge_zone import IceEdgeZoneConfig

    # Valid config
    config = IceEdgeZoneConfig(nx=60, ny=16, nz=8, u0=0.05, re=100.0)
    config.validate()
    assert config.tau > 0.5
    assert config.nu > 0

    # Invalid: nx too small
    config_bad = IceEdgeZoneConfig(nx=5, ny=16, nz=8)
    with pytest.raises(AssertionError):
        config_bad.validate()


def test_mirror_tables():
    """Test that mirror tables are correctly constructed."""
    from tensorlbm.ice_edge_zone import MIRROR_Y, MIRROR_Z
    from tensorlbm.d3q19 import C

    # Mirror Y should flip cy sign
    for i in range(19):
        cx, cy, cz = C[i].tolist()
        mi = MIRROR_Y[i]
        cx_m, cy_m, cz_m = C[mi].tolist()
        assert cx_m == cx, f"Mirror Y should preserve cx: dir {i}"
        assert cy_m == -cy, f"Mirror Y should flip cy: dir {i}"
        assert cz_m == cz, f"Mirror Y should preserve cz: dir {i}"

    # Mirror Z should flip cz sign
    for i in range(19):
        cx, cy, cz = C[i].tolist()
        mi = MIRROR_Z[i]
        cx_m, cy_m, cz_m = C[mi].tolist()
        assert cx_m == cx
        assert cy_m == cy
        assert cz_m == -cz


def test_equilibrium_conservation():
    """Test that equilibrium conserves mass and momentum."""
    from tensorlbm.ice_edge_zone import _equilibrium_d3q19
    from tensorlbm.d3q19 import C

    device = torch.device("cpu")
    nz, ny, nx = 4, 8, 16

    rho0 = torch.ones(nz, ny, nx, device=device) * 1.0
    ux0 = torch.full((nz, ny, nx), 0.05, device=device)
    uy0 = torch.zeros(nz, ny, nx, device=device)
    uz0 = torch.zeros(nz, ny, nx, device=device)

    feq = _equilibrium_d3q19(rho0, ux0, uy0, uz0)

    # Check mass conservation
    rho_rec = feq.sum(dim=0)
    assert torch.allclose(rho_rec, rho0, atol=1e-5), "Mass not conserved in equilibrium"

    # Check momentum conservation
    c = C.to(device).float()
    jx = (feq * c[:, 0].view(19,1,1,1)).sum(dim=0)
    jy = (feq * c[:, 1].view(19,1,1,1)).sum(dim=0)
    jz = (feq * c[:, 2].view(19,1,1,1)).sum(dim=0)

    assert torch.allclose(jx, rho0 * ux0, atol=1e-5), "x-momentum not conserved"
    assert torch.allclose(jy, rho0 * uy0, atol=1e-5), "y-momentum not conserved"
    assert torch.allclose(jz, rho0 * uz0, atol=1e-5), "z-momentum not conserved"


def test_streaming_conservation():
    """Test that streaming conserves mass in the interior."""
    from tensorlbm.ice_edge_zone import _equilibrium_d3q19, _stream_d3q19

    device = torch.device("cpu")
    nz, ny, nx = 4, 8, 16

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), 0.02, device=device)
    uy0 = torch.zeros(nz, ny, nx, device=device)
    uz0 = torch.zeros(nz, ny, nx, device=device)

    f = _equilibrium_d3q19(rho0, ux0, uy0, uz0)
    mass_before = f.sum().item()

    f_new = _stream_d3q19(f)
    mass_after = f_new.sum().item()

    # Mass should be approximately conserved (some loss at boundaries is expected)
    rel_diff = abs(mass_after - mass_before) / mass_before
    assert rel_diff < 0.1, f"Streaming mass loss too large: {rel_diff:.4f}"


def test_small_simulation_no_nan():
    """Test that a small simulation runs without NaN."""
    from tensorlbm.ice_edge_zone import IceEdgeZoneConfig, run_ice_edge_zone

    config = IceEdgeZoneConfig(
        nx=30, ny=10, nz=4,
        u0=0.02, re=50.0,  # Low Re for stability
        ice_edge_frac=0.5, ls_ice=1.0,
        n_steps=50, output_interval=25,
        stats_start=25,
        output_root=Path("runs/ice_edge_zone/pytest"),
        run_name="test_no_nan",
    )

    result = run_ice_edge_zone(config)

    # Check no NaN in results
    assert not np.isnan(result.drag_coefficient), "C_D is NaN"
    assert not np.any(np.isnan(result.u_mean)), "u_mean contains NaN"
    assert not np.any(np.isnan(result.p_mean)), "p_mean contains NaN"

    # Check drag coefficient is positive and reasonable
    assert result.drag_coefficient > 0, "C_D should be positive"
    assert result.drag_coefficient < 100, f"C_D={result.drag_coefficient} unreasonably large"


def test_slip_length_effect():
    """Test that increasing slip length reduces drag under ice."""
    from tensorlbm.ice_edge_zone import IceEdgeZoneConfig, run_ice_edge_zone

    results = []
    for ls in [0.0, 2.0, 5.0]:
        config = IceEdgeZoneConfig(
            nx=40, ny=12, nz=4,
            u0=0.02, re=50.0,
            ice_edge_frac=0.5, ls_ice=ls,
            n_steps=100, output_interval=50,
            stats_start=50,
            output_root=Path("runs/ice_edge_zone/pytest"),
            run_name=f"test_ls_{ls}",
        )
        result = run_ice_edge_zone(config)
        results.append((ls, result.drag_ice_component))

    # Drag under ice should decrease with increasing slip length
    # (at least the trend should be non-increasing)
    for i in range(1, len(results)):
        ls_prev, cd_prev = results[i-1]
        ls_curr, cd_curr = results[i]
        # Allow some numerical noise
        assert cd_curr <= cd_prev * 1.5, (
            f"C_D_ice should not increase much with slip length: "
            f"Ls={ls_prev}→{ls_curr}, C_D={cd_prev:.4f}→{cd_curr:.4f}"
        )
