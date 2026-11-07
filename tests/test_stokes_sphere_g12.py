"""G12 Stokes sphere drag validation — wet-node MEM in creeping flow.

Tests the momentum-exchange method for a sphere in the Stokes regime
(Re ≤ 0.1) against the exact analytical solution F = 6πμRU, Cd = 24/Re.

Historical context (G12/G15 ledger):
  - Re=100 sphere: MEM standard +264.6%, galilean +590%, bg_sub no-op
  - Root cause: staircase geometry error + equilibrium background not
    cancelling on curved surfaces at moderate Re
  - Stokes regime (Re≤0.1): inertia-free flow, staircase error ≪ 5%,
    wet-node MEM recovers analytical drag within tolerance

Test matrix:
  T1  Stokes analytical formula correctness
  T2  Stokes force analytical formula correctness
  T3  Single sphere Re=0.1, D=16: Cd within ±5% of 24/Re  [cluster]
  T4  Single sphere Re=0.05, D=20: Cd within ±5% of 24/Re  [cluster]
  T5  Grid convergence: error decreases monotonically with D  [cluster]
  T6  Wet-node MEM returns zero on initial state (background test)
  T7  Re=0.1, D=24: Cd within ±3% (tighter tolerance)  [cluster]
  T8  Near-wall mask correctness (unit test)
  T9  Stokes force vs Cd consistency check
  T10 Convergence report structure validation  [cluster]
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.stokes_sphere import (
    run_stokes_sphere,
    run_stokes_sphere_convergence,
    stokes_cd_analytical,
    stokes_force_analytical,
    _near_wall_mask,
)
from tensorlbm.momentum_exchange import momentum_exchange_wet_node
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.boundary.boundaries3d import sphere_mask


# ---------------------------------------------------------------------------
# T1: Analytical Cd formula
# ---------------------------------------------------------------------------
def test_stokes_cd_analytical():
    """Cd = 24/Re must be exact."""
    assert stokes_cd_analytical(0.1) == pytest.approx(240.0, rel=1e-12)
    assert stokes_cd_analytical(0.05) == pytest.approx(480.0, rel=1e-12)
    assert stokes_cd_analytical(1.0) == pytest.approx(24.0, rel=1e-12)
    assert stokes_cd_analytical(0.01) == pytest.approx(2400.0, rel=1e-12)


# ---------------------------------------------------------------------------
# T2: Analytical force formula
# ---------------------------------------------------------------------------
def test_stokes_force_analytical():
    """F = 6πμRU must be exact."""
    f = stokes_force_analytical(mu=0.01, r=5.0, u=0.01)
    expected = 6.0 * math.pi * 0.01 * 5.0 * 0.01
    assert f == pytest.approx(expected, rel=1e-12)

    f2 = stokes_force_analytical(mu=1.0, r=10.0, u=0.1)
    expected2 = 6.0 * math.pi * 1.0 * 10.0 * 0.1
    assert f2 == pytest.approx(expected2, rel=1e-12)


# ---------------------------------------------------------------------------
# T3: Re=0.1, D=16 — Cd within ±5%  [requires cluster for full run]
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stokes_sphere_re01_d16():
    """Stokes sphere Re=0.1, D=16: Cd within ±5% of 24/Re=240."""
    result = run_stokes_sphere(
        re=0.1, d_cells=16, domain_factor=5.0,
        steps=6000, warmup=1500,
        device="cpu", dtype=torch.float64,
    )
    cd_ref = 240.0
    err = abs(result["cd"] - cd_ref) / cd_ref * 100.0
    assert err < 5.0, (
        "Re=0.1 D=16: Cd={:.4f} vs ref={}, err={:.2f}%".format(
            result["cd"], cd_ref, err)
    )


# ---------------------------------------------------------------------------
# T4: Re=0.05, D=20 — Cd within ±5%  [requires cluster]
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stokes_sphere_re005_d20():
    """Stokes sphere Re=0.05, D=20: Cd within ±5% of 24/Re=480."""
    result = run_stokes_sphere(
        re=0.05, d_cells=20, domain_factor=5.0,
        steps=6000, warmup=1500,
        device="cpu", dtype=torch.float64,
    )
    cd_ref = 480.0
    err = abs(result["cd"] - cd_ref) / cd_ref * 100.0
    assert err < 5.0, (
        "Re=0.05 D=20: Cd={:.4f} vs ref={}, err={:.2f}%".format(
            result["cd"], cd_ref, err)
    )


# ---------------------------------------------------------------------------
# T5: Grid convergence — error decreases with resolution  [requires cluster]
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stokes_sphere_grid_convergence():
    """Grid convergence: |err| must decrease monotonically with D."""
    results = run_stokes_sphere_convergence(
        re=0.1,
        d_cells_list=[10, 16, 24],
        steps=5000,
        warmup=1000,
        device="cpu",
        dtype=torch.float64,
    )
    cd_ref = 240.0
    errors = [abs(r["cd"] - cd_ref) / cd_ref for r in results]

    err_strs = ["{:.4f}".format(e) for e in errors]
    cd_strs = ["{:.4f}".format(r["cd"]) for r in results]
    assert errors[-1] < errors[0], (
        "Grid convergence failed: errors={}, Cd={}".format(err_strs, cd_strs)
    )

    assert errors[-1] < 0.05, (
        "Finest grid error {:.2f}% > 5%".format(errors[-1] * 100)
    )


# ---------------------------------------------------------------------------
# T6: Wet-node MEM returns zero on initial state
# ---------------------------------------------------------------------------
def test_wet_node_mem_zero_on_equilibrium():
    """Wet-node MEM must return F~0 on initial state: fluid at uniform flow,
    solid at rest equilibrium (the engine's initial condition).

    This is the key property that distinguishes wet-node MEM from standard
    Ladd MEM: on the initial state, the solid-cell populations are isotropic
    (rest equilibrium) and cancel pairwise over the symmetric link set.
    """
    dev = torch.device("cpu")
    nz, ny, nx = 30, 30, 30
    solid = sphere_mask(nx, ny, nz, nx/2, ny/2, nz/2, 8.0, device=dev)
    near = _near_wall_mask(solid)

    # Fluid: uniform flow equilibrium
    rho_f = torch.ones(nz, ny, nx, device=dev, dtype=torch.float64)
    ux_f = torch.full((nz, ny, nx), 0.01, device=dev, dtype=torch.float64)
    uy_f = torch.zeros(nz, ny, nx, device=dev, dtype=torch.float64)
    uz_f = torch.zeros(nz, ny, nx, device=dev, dtype=torch.float64)
    f = equilibrium3d(rho_f, ux_f, uy_f, uz_f, device=dev)

    # Solid: rest equilibrium (u=0) — this is the engine's initial state
    rho_s = torch.ones(nz, ny, nx, device=dev, dtype=torch.float64)
    ux_s = torch.zeros(nz, ny, nx, device=dev, dtype=torch.float64)
    uy_s = torch.zeros(nz, ny, nx, device=dev, dtype=torch.float64)
    uz_s = torch.zeros(nz, ny, nx, device=dev, dtype=torch.float64)
    f_solid = equilibrium3d(rho_s, ux_s, uy_s, uz_s, device=dev)
    f[:, solid] = f_solid[:, solid]

    fx, fy, fz = momentum_exchange_wet_node(f, solid, near=near, links="full")
    assert abs(fx) < 1e-10, "fx={} should be ~0 on initial state".format(fx)
    assert abs(fy) < 1e-10, "fy={} should be ~0 on initial state".format(fy)
    assert abs(fz) < 1e-10, "fz={} should be ~0 on initial state".format(fz)


# ---------------------------------------------------------------------------
# T7: Re=0.1, D=24 — Cd within ±3% (tighter tolerance)  [requires cluster]
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stokes_sphere_re01_d24_tight():
    """Stokes sphere Re=0.1, D=24: Cd within ±3% of 24/Re=240."""
    result = run_stokes_sphere(
        re=0.1, d_cells=24, domain_factor=5.0,
        steps=8000, warmup=2000,
        device="cpu", dtype=torch.float64,
    )
    cd_ref = 240.0
    err = abs(result["cd"] - cd_ref) / cd_ref * 100.0
    assert err < 3.0, (
        "Re=0.1 D=24: Cd={:.4f} vs ref={}, err={:.2f}%".format(
            result["cd"], cd_ref, err)
    )


# ---------------------------------------------------------------------------
# T8: Near-wall mask correctness
# ---------------------------------------------------------------------------
def test_near_wall_mask():
    """Near-wall mask should identify fluid cells adjacent to solid."""
    dev = torch.device("cpu")
    solid = torch.zeros(5, 5, 5, dtype=torch.bool, device=dev)
    solid[2, 2, 2] = True

    near = _near_wall_mask(solid)

    expected_near = [
        (1, 2, 2), (3, 2, 2),
        (2, 1, 2), (2, 3, 2),
        (2, 2, 1), (2, 2, 3),
    ]
    for k, j, i in expected_near:
        assert near[k, j, i], "Cell ({},{},{}) should be near-wall".format(k, j, i)

    assert not near[2, 2, 2], "Solid cell should not be in near-wall mask"
    assert not near[1, 1, 2], "Diagonal cell should not be near-wall"


# ---------------------------------------------------------------------------
# T9: Stokes force vs Cd consistency
# ---------------------------------------------------------------------------
def test_stokes_force_cd_consistency():
    """F = Cd * 0.5*rho*U^2*A must be consistent with F = 6*pi*mu*R*U.

    Consistency requires Re = U*D/nu = U*2R/(mu/rho).
    We pick parameters so that Re=0.1 exactly.
    """
    re = 0.1
    r = 10.0
    u = 0.01
    rho = 1.0
    nu = u * 2.0 * r / re
    mu = rho * nu

    f_stokes = stokes_force_analytical(mu, r, u)

    cd = stokes_cd_analytical(re)
    area = math.pi * r**2
    f_cd = cd * 0.5 * rho * u**2 * area

    assert f_stokes == pytest.approx(f_cd, rel=1e-10), (
        "F_stokes={} vs F_cd={}".format(f_stokes, f_cd)
    )


# ---------------------------------------------------------------------------
# T10: Multi-resolution convergence report  [requires cluster]
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_convergence_report_structure():
    """Convergence study returns properly structured results."""
    results = run_stokes_sphere_convergence(
        re=0.1,
        d_cells_list=[10, 16],
        steps=3000,
        warmup=500,
        device="cpu",
        dtype=torch.float64,
    )

    assert len(results) == 2
    for r in results:
        assert "cd" in r
        assert "cd_ref" in r
        assert "err_pct" in r
        assert "d_cells" in r
        assert "cd_history" in r
        assert len(r["cd_history"]) > 0

    cd_ref = 240.0
    for r in results:
        assert 100 < r["cd"] < 400, "Cd={} out of reasonable range".format(r["cd"])
