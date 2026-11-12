"""Zero-gradient (Neumann) outlet BC verification against Kovasznay analytic N-S.

This test closes the "生态摩擦" gap registered in the Kovasznay report
(15179/15194): the zero-gradient outlet ``f[:,:,-1]=f[:,:,-2]`` was a 3-line
inline in ``benchmarks/verified/kovasznay_2d/run.py`` with no library function.
It adds ``tensorlbm.boundaries.zero_gradient_outlet_2d`` and re-verifies the
merged Kovasznay (1948) steady-state analytic solution using the library
function instead of the inline copy.

Physical setup (Kovasznay 1948, 2-D steady incompressible N-S exact solution):
    u(x',y') = U0(1 - e^{lambda x'} cos 2pi y')
    v(x',y') = U0(lambda/2pi) e^{lambda x'} sin 2pi y'
    lambda = Re/2 - sqrt(Re^2/4 + 4pi^2)   (negative, decaying branch)
    Re = U0*L/nu, L = y-period = ny cells.
    Domain x' in [0,3] (3 y-periods), y periodic (1 period).
    Inlet x=0: analytic Zou-He Dirichlet; outlet x=nx-1: zero-gradient Neumann.

Verification criteria (from the merged benchmark, pre-declared 3 % gate):
    max(u L2, v L2, u masked-max rel, v masked-max rel) <= 3 %
    at ny=64/128, with monotone 64->128 convergence.

References:
    Kovasznay, L.I.G. (1948). Laminar flow behind a two-dimensional grid.
    Proc. Camb. Phil. Soc. 44, 58-62.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # <repo>

from tensorlbm.boundaries import zero_gradient_outlet_2d
from tensorlbm.boundary_capability_contract import (
    boundary_capability_matrix,
    require_boundary_condition_capability,
)
from tensorlbm.d2q9 import equilibrium, macroscopic


def _kov_lambda(re: float) -> float:
    return re / 2.0 - float(np.sqrt(re * re / 4.0 + 4.0 * np.pi * np.pi))


def _analytic_field(nx: int, ny: int, u0: float, lam: float, xmax: float, dev, dt):
    y = torch.arange(ny, device=dev, dtype=dt)
    x = torch.arange(nx, device=dev, dtype=dt)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xp = xx / nx * xmax
    yp = yy / ny
    e = torch.exp(lam * xp)
    u = u0 * (1.0 - e * torch.cos(2.0 * np.pi * yp))
    v = u0 * (lam / (2.0 * np.pi)) * e * torch.sin(2.0 * np.pi * yp)
    return u, v


def _run_kovasznay(ny: int, re: float, u0: float, xmax: float, n_steps: int):
    """Run the Kovasznay steady-state with the library zero-gradient outlet."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    dev = torch.device("cpu")
    dt = torch.float32
    nx = int(round(ny * xmax))
    lam = _kov_lambda(re)
    nu = u0 * ny / re
    tau = 0.5 + 3.0 * nu

    u_ana, v_ana = _analytic_field(nx, ny, u0, lam, xmax, dev, dt)
    f = equilibrium(torch.ones((ny, nx), device=dev, dtype=dt), u_ana, v_ana)
    u_in = u_ana[:, 0].contiguous()
    v_in = v_ana[:, 0].contiguous()

    from tensorlbm.boundaries import zou_he_inlet_velocity
    from tensorlbm.solver import collide_bgk, stream

    for _ in range(n_steps):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = zero_gradient_outlet_2d(f)  # library function (was inline)
        f = zou_he_inlet_velocity(f, u_in, v_in)

    # time-average last 20 steps
    acc_u = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    acc_v = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    for _ in range(20):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = zero_gradient_outlet_2d(f)
        f = zou_he_inlet_velocity(f, u_in, v_in)
        _, ux, uy = macroscopic(f)
        acc_u += ux.to(torch.float64)
        acc_v += uy.to(torch.float64)
    acc_u /= 20.0
    acc_v /= 20.0

    u_num = acc_u.cpu().numpy()
    v_num = acc_v.cpu().numpy()
    u_a = u_ana.cpu().numpy().astype(np.float64)
    v_a = v_ana.cpu().numpy().astype(np.float64)

    def l2_rel(num, ana):
        return float(np.linalg.norm(num - ana) / np.linalg.norm(ana))

    def max_rel_masked(num, ana, thresh):
        mask = np.abs(ana) > thresh
        if mask.sum() == 0:
            return float("nan")
        return float(np.max(np.abs(num[mask] - ana[mask]) / np.abs(ana[mask])) * 100.0)

    return {
        "u_l2": l2_rel(u_num, u_a),
        "v_l2": l2_rel(v_num, v_a),
        "u_max_rel": max_rel_masked(u_num, u_a, 0.1 * u0),
        "v_max_rel": max_rel_masked(v_num, v_a, 0.1 * float(np.max(np.abs(v_a)))),
    }


# ---------------------------------------------------------------------------
# 1. Library function is a pure upstream-column copy (mechanics)
# ---------------------------------------------------------------------------


def test_zero_gradient_outlet_is_pure_copy():
    """Outlet column must equal the upstream column (all 9 directions)."""
    f = torch.randn(9, 16, 32, dtype=torch.float64)
    f_out = zero_gradient_outlet_2d(f)
    assert torch.equal(f_out[:, :, -1], f[:, :, -2])
    # interior untouched
    assert torch.equal(f_out[:, :, :-1], f[:, :, :-1])
    # input not mutated in place
    assert not torch.equal(f[:, :, -1], f[:, :, -2])


def test_zero_gradient_outlet_rejects_wrong_directions():
    with pytest.raises(ValueError):
        zero_gradient_outlet_2d(torch.zeros(19, 8, 8))


# ---------------------------------------------------------------------------
# 2. Capability contract registration
# ---------------------------------------------------------------------------


def test_zero_gradient_outlet_registered_in_contract():
    """D2Q9 zero-gradient outlet must be PHYSICS_VALIDATED in the contract."""
    matrix = boundary_capability_matrix()
    assert "zero_gradient_outlet" in matrix
    cap = matrix["zero_gradient_outlet"]["D2Q9"]
    assert cap.implementation_status == "PHYSICS_VALIDATED"
    assert cap.entrypoint == "tensorlbm.boundaries.zero_gradient_outlet_2d"
    # D3Q19/D3Q27 honestly withheld (no implementation)
    assert matrix["zero_gradient_outlet"]["D3Q19"].implementation_status == "NO_IMPLEMENTATION"
    assert matrix["zero_gradient_outlet"]["D3Q27"].implementation_status == "NO_IMPLEMENTATION"


# ---------------------------------------------------------------------------
# 3. Kovasznay re-verification with the library function (verification debt)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ny", [64, 128])
def test_kovasznay_library_outlet_within_3pct(ny):
    """Kovasznay steady-state with library zero-gradient outlet <= 3 % (long domain)."""
    res = _run_kovasznay(ny=ny, re=40.0, u0=0.03, xmax=3.0, n_steps=2000)
    worst = max(res["u_l2"], res["v_l2"], res["u_max_rel"] / 100.0, res["v_max_rel"] / 100.0)
    assert worst < 0.03, f"ny={ny}: worst={worst*100:.3f}% > 3% gate ({res})"


def test_kovasznay_library_outlet_converges_64_to_128():
    """64->128 must be monotone (convergence), matching the merged benchmark."""
    r64 = _run_kovasznay(ny=64, re=40.0, u0=0.03, xmax=3.0, n_steps=2000)
    r128 = _run_kovasznay(ny=128, re=40.0, u0=0.03, xmax=3.0, n_steps=2000)
    for key in ("u_l2", "v_l2", "u_max_rel", "v_max_rel"):
        assert r128[key] <= r64[key], (
            f"{key} not monotone: 64={r64[key]:.4f} -> 128={r128[key]:.4f}"
        )
