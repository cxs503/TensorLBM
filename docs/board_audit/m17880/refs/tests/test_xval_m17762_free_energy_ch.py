"""Regression tests for the XVAL m17762 free-energy Cahn-Hilliard core.

Guards the two findings of ``docs/XVAL_M17762_LBM_FREE_ENERGY_CAHN_HILLIARD_CORE.md``:

1. ``geq_convention="swift"`` (historical default) carries a spurious
   ``cs^2 (tau_g - 1/2) grad^2 phi`` diffusion that is independent of ``Gamma``,
   so it does **not** solve the Cahn-Hilliard equation.
2. ``geq_convention="lee_lin"`` (structure transplanted from the reference
   free-energy implementation) carries no such term and reproduces the analytic
   Cahn-Hilliard linear growth rate ``omega = Gamma (tau_g - 1/2) k^2 (A - kappa k^2)``.

The single-mode growth measurement is the same protocol as
``scripts/xval_m17762_free_energy_ch_core.py``: a z-invariant box
``(nz=1, ny=N, nx=N)``, frozen momentum, amplitude followed in time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.multiphase3d import free_energy_step_3d, init_free_energy_g_3d
from tensorlbm.solver3d import stream3d

TAU_G = 1.0 / (3.0 - math.sqrt(3.0))
CS2 = 1.0 / 3.0


def _run_mode(*, N=48, m=6, convention, Gamma, A=0.075, kappa=0.0375, eps=0.01, steps=260,
              lo=1.2, hi=6.0):
    torch.set_num_threads(1)
    k = 2.0 * math.pi * m / N
    x = np.arange(N)
    phi = torch.tensor(eps * np.sin(k * x)[None, None, :], dtype=torch.float32)
    g = init_free_energy_g_3d(phi)
    z = torch.zeros_like(phi)
    f = equilibrium3d(torch.ones_like(phi), z, z, z)
    amps = []
    for _ in range(steps):
        g = stream3d(g)
        _, g = free_energy_step_3d(
            f,
            g,
            tau_f=1.0,
            tau_g=TAU_G,
            A=A,
            B=A,
            kappa=kappa,
            Gamma=Gamma,
            geq_convention=convention,
        )
        amps.append(2.0 / N * abs(np.sum(g.sum(dim=0)[0].numpy() * np.exp(-1j * k * x))))
        if amps[-1] > 8.0 * eps:
            break
    amps = np.asarray(amps)
    ok = (amps > lo * eps) & (amps < hi * eps)
    if ok.sum() < 6:
        return float("nan")
    return float(np.polyfit(np.arange(len(amps))[ok], np.log(amps[ok]), 1)[0])


def test_lee_lin_reproduces_analytic_cahn_hilliard_growth():
    """mobility-carrying convention: omega within 15% of the CH relation."""
    k = 2.0 * math.pi * 6 / 48
    A, kappa = 0.075, 0.0375
    analytic = Gamma = 1.0
    analytic = Gamma * (TAU_G - 0.5) * k**2 * (A - kappa * k**2)
    measured = _run_mode(convention="lee_lin", Gamma=Gamma)
    assert analytic > 0.0
    assert abs(measured - analytic) / abs(analytic) < 0.15, (measured, analytic)


def test_lee_lin_has_no_gamma_independent_phi_diffusion():
    """Gamma = 0 must freeze the order parameter (transport only via mu)."""
    k = 2.0 * math.pi * 6 / 48
    measured = _run_mode(convention="lee_lin", Gamma=0.0, lo=0.15, hi=1.5)
    spurious = CS2 * (TAU_G - 0.5) * k**2
    assert abs(measured) < 0.05 * spurious, (measured, spurious)


def test_swift_convention_keeps_spurious_diffusion():
    """Historical convention: the decay rate at Gamma = 0 is cs^2 (tau-1/2) k^2."""
    k = 2.0 * math.pi * 6 / 48
    measured = _run_mode(convention="swift", Gamma=0.0, lo=0.15, hi=1.5)
    spurious = CS2 * (TAU_G - 0.5) * k**2
    assert abs(measured + spurious) / spurious < 0.15, (measured, spurious)


def test_free_energy_parameters_invert_the_closed_form_surface_tension():
    from tensorlbm.allen_cahn_lbm import free_energy_parameters

    sigma, W = 0.05, 2.0
    params = free_energy_parameters(sigma, W=W)
    recovered = (2.0 * math.sqrt(2.0) / 3.0) * math.sqrt(params.kappa * params.beta)
    assert abs(recovered - sigma) / sigma < 1e-12


def test_invalid_geq_convention_is_rejected():
    phi = torch.zeros((1, 4, 4))
    g = init_free_energy_g_3d(phi)
    z = torch.zeros_like(phi)
    f = equilibrium3d(torch.ones_like(phi), z, z, z)
    with pytest.raises(ValueError, match="geq_convention"):
        free_energy_step_3d(f, g, geq_convention="nope")


def test_lee_lin_equilibrium_has_exact_first_and_zeroth_moments():
    """Sum_i g_i^eq = phi and the mobility tensor is isotropic (M = Gamma (tau-1/2))."""
    from tensorlbm.d3q19 import C, W

    phi = torch.zeros((1, 5, 6))
    g = init_free_energy_g_3d(phi)
    assert torch.allclose(g.sum(dim=0), phi, atol=1e-6)

    # tensor Sigma_i w_i lambda_i c_ia c_ib = delta_ab  (lambda = (3/2)(3|c|^2-3))
    w = W.double()
    c = C.double()
    c_sq = (c**2).sum(dim=1)
    lam = 1.5 * (3.0 * c_sq - 3.0)
    tensor = torch.zeros((3, 3), dtype=torch.float64)
    for i in range(c.shape[0]):
        tensor += w[i] * lam[i] * torch.outer(c[i], c[i])
    assert torch.allclose(tensor, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert abs(float((w * lam).sum())) < 1e-6  # zero sum: phi moment untouched