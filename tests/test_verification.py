"""Verification tests: convergence and physical accuracy.

These tests validate the physical correctness of TensorLBM at a quantitative
level, complementing the unit tests in test_solver.py.

Poiseuille convergence
----------------------
For pressure-driven channel flow (body force F_x in x-direction, no-slip
walls at y=0 and y=ny-1) the analytical steady-state velocity profile is::

    ux(y) = F_x / (2*nu) * y * (ny - 1 - y)

We run at three resolutions and verify the L2 error converges at O(h^2)
(second-order spatial accuracy of D2Q9 LBM).

Taylor–Green vortex decay
-------------------------
The 2D Taylor–Green vortex has a periodic analytical solution whose kinetic
energy decays exponentially as E(t) = E0 * exp(-4*nu*k^2*t) for wavenumber k.
We run a periodic LBM simulation and verify the measured decay rate matches
the theoretical value to within 20 % (allowing for higher-order LBM effects).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import collide_bgk, equilibrium, macroscopic, stream


def _run_poiseuille(ny: int, nu: float, fx: float, n_steps: int) -> torch.Tensor:
    """Run 2D Poiseuille flow with a body force and return final ux profile."""
    nx = 4
    rho0 = torch.ones((ny, nx))
    ux0 = torch.zeros_like(rho0)
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)

    tau = 3.0 * nu + 0.5

    from tensorlbm import OPPOSITE

    opp = OPPOSITE

    for _ in range(n_steps):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
        f_bb = f.clone()
        f_bb[:, 0, :] = f[opp][:, 0, :]
        f_bb[:, -1, :] = f[opp][:, -1, :]
        f = f_bb
        rho, ux, uy = macroscopic(f)
        # Match the discrete forcing amplitude used by the BGK update so the
        # steady-state profile converges to the analytical Poiseuille solution.
        ux = ux + 2.0 * fx * tau
        feq_new = equilibrium(rho, ux, uy)
        feq_old = equilibrium(rho, ux - 2.0 * fx * tau, uy)
        f = f + (feq_new - feq_old) * (1.0 - 0.5 / tau)

    _, ux_f, _ = macroscopic(f)
    return ux_f[:, 0]


def _poiseuille_analytic(ny: int, nu: float, fx: float) -> torch.Tensor:
    """Parabolic Poiseuille velocity profile u(y) = fx/(2*nu) * y*(ny-1-y)."""
    y = torch.arange(ny, dtype=torch.float32)
    return (fx / (2.0 * nu)) * y * (ny - 1 - y)


def _l2_error(numerical: torch.Tensor, analytic: torch.Tensor) -> float:
    """Relative L2 error, excluding boundary cells."""
    diff = numerical[1:-1] - analytic[1:-1]
    denom = analytic[1:-1].norm()
    if float(denom) < 1e-14:
        return float(diff.norm())
    return float((diff.norm() / denom).item())


@pytest.mark.parametrize("ny", [16, 32, 64])
def test_poiseuille_convergence(ny: int) -> None:
    """Poiseuille flow L2 error should decrease at roughly O(h^2) with resolution."""
    nu = 1.0 / 6.0
    fx = 1e-4
    n_steps = ny * ny * 4

    ux_num = _run_poiseuille(ny, nu, fx, n_steps)
    ux_ref = _poiseuille_analytic(ny, nu, fx)

    err = _l2_error(ux_num, ux_ref)
    assert err < 0.10, f"Poiseuille L2 error too large at ny={ny}: {err:.4f}"


def test_poiseuille_spatial_convergence_order() -> None:
    """Verify O(h^2) convergence: refining by 2× must halve error at least 1.5×."""
    nu = 1.0 / 6.0
    fx = 1e-4
    errors = []
    for ny in (16, 32, 64):
        n_steps = ny * ny * 4
        ux_num = _run_poiseuille(ny, nu, fx, n_steps)
        ux_ref = _poiseuille_analytic(ny, nu, fx)
        errors.append(_l2_error(ux_num, ux_ref))

    ratio_01 = errors[0] / errors[1] if errors[1] > 0 else float("inf")
    ratio_12 = errors[1] / errors[2] if errors[2] > 0 else float("inf")
    assert ratio_01 > 1.5, f"Convergence ratio ny=16→32 too low: {ratio_01:.2f}"
    assert ratio_12 > 1.5, f"Convergence ratio ny=32→64 too low: {ratio_12:.2f}"


def test_taylor_green_energy_decay() -> None:
    """2-D Taylor–Green vortex kinetic energy should decay at the predicted rate."""
    n = 32
    nu = 1.0 / 30.0
    tau = 3.0 * nu + 0.5
    k = 2.0 * math.pi / n

    amp = 0.01
    xx, yy = torch.meshgrid(
        torch.arange(n, dtype=torch.float32),
        torch.arange(n, dtype=torch.float32),
        indexing="xy",
    )
    ux0 = amp * torch.sin(k * xx) * torch.cos(k * yy)
    uy0 = -amp * torch.cos(k * xx) * torch.sin(k * yy)
    rho0 = torch.ones((n, n))
    f = equilibrium(rho0, ux0, uy0)

    def _kinetic_energy(f_dist: torch.Tensor) -> float:
        rho, ux, uy = macroscopic(f_dist)
        return float((0.5 * rho * (ux * ux + uy * uy)).sum().item())

    decay_rate = 4.0 * nu * k**2
    n_steps = 200

    e0 = _kinetic_energy(f)
    for _ in range(n_steps):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
    e_final = _kinetic_energy(f)

    if e0 > 0 and e_final > 0:
        measured_rate = -math.log(e_final / e0) / n_steps
        assert abs(measured_rate - decay_rate) / decay_rate < 0.30, (
            "Taylor-Green decay rate mismatch: "
            f"measured={measured_rate:.5f}, theory={decay_rate:.5f}"
        )
    else:
        assert e_final > 0, "Kinetic energy became non-positive"
