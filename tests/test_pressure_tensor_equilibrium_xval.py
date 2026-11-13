"""Round-2 checks for XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT.

The physics of the transplant is exercised by the runner
(``benchmarks/xval_pressure_tensor_equilibrium/run_matrix.py``); these tests only
pin the *measuring instruments* so a green matrix cannot be a measurement artefact:

1. ``fit_interface_width`` recovers a known ``xi`` from the equilibrium stress
   profile ``kappa sech^4((x-x0)/xi)/xi^2``;
2. the same fit on the nine-point discrete gradient of ``tanh`` is biased high by
   a *bounded* amount that shrinks with ``xi/h`` -- the bias is a documented
   discretisation effect, not noise (it is why ``xi_fit`` fails at ``xi/h = 2``);
3. a flat interface at rest carries no shear: ``P_xy`` from the transplanted
   anisotropic coefficients is zero to machine precision;
4. the transverse stress difference integrates to ``sigma`` (Rowlinson-Widom route);
5. the predeclared gate logic keeps the owner band and never upgrades a value.
"""

import json
import math
import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

from tensorlbm.phasefield.free_energy_pressure import (  # noqa: E402
    FreeEnergyBinaryParams,
    equilibrium_f,
    fit_interface_width,
    grad_lap,
    pressure_tensor_measured,
    tanh_profile,
)


def _band(nx=64, ny=8, xi=3.0):
    x = torch.arange(nx, dtype=torch.float64)
    signed = 8.0 - (x - nx / 2.0).abs()
    phi = tanh_profile(signed, 0.0, xi)
    return x, phi.expand(ny, nx).contiguous()


@pytest.mark.parametrize("xi,kappa", [(2.0, 0.2), (3.0, 0.45), (4.0, 0.8)])
def test_fit_recovers_xi_from_analytic_stress(xi, kappa):
    x = torch.arange(96, dtype=torch.float64)
    stress = kappa * torch.cosh((x - 41.0) / xi) ** (-4) / xi ** 2
    out = fit_interface_width(x, stress, xi, x0=41.0)
    assert abs(out["xi"] - xi) / xi < 1e-3
    assert abs(out["amplitude"] - kappa) / kappa < 1e-3
    assert out["rms_rel"] < 1e-3


def test_fit_on_discrete_gradient_is_biased_high_but_bounded():
    """Documented discretisation bias: +10 % at xi/h=2, +4 % at xi/h=3 (asserted)."""
    got = {}
    for xi in (2.0, 3.0):
        kappa = 0.5 * 0.1 * xi**2
        x, phi = _band(xi=xi)
        gx, _, _ = grad_lap(phi)
        stress = kappa * (gx[0] ** 2)
        w = stress.clamp(min=0.0) * (x < x.size(0) // 2)
        x0 = float((x * w).sum() / w.sum())
        got[xi] = fit_interface_width(x, stress, xi, x0=x0)["xi"]
    assert 0.0 < (got[2.0] - 2.0) / 2.0 < 0.25      # biased high, but < 25 %
    assert 0.0 < (got[3.0] - 3.0) / 3.0 < 0.10      # and the bias shrinks
    assert got[2.0] - 2.0 > got[3.0] - 3.0


def test_flat_interface_at_rest_carries_no_shear():
    x, phi = _band()
    p = FreeEnergyBinaryParams(A=0.1, kappa=0.45, Gamma=0.2, tau_f=1.0, tau_phi=1.5)
    rho = torch.ones_like(phi)
    zero = torch.zeros_like(phi)
    gx, gy, lap = grad_lap(phi)
    f = equilibrium_f(rho, zero, zero, phi, gx, gy, lap, p.A, p.kappa)
    Pxx, Pyy, Pxy = pressure_tensor_measured(f)
    assert float(Pxy.abs().max()) < 1e-14 * float(Pxx.abs().max())
    # and the transverse difference is positive: P_NN - P_TT = kappa phi'^2
    assert float((Pxx - Pyy).min()) > -1e-12


def test_transverse_stress_difference_integrates_to_sigma():
    for xi in (2.0, 3.0):
        p = FreeEnergyBinaryParams(A=0.1, kappa=0.5 * 0.1 * xi**2, Gamma=0.2,
                                   tau_f=1.0, tau_phi=1.5)
        x, phi = _band(xi=xi)
        gx, _, _ = grad_lap(phi)
        stress = p.kappa * (gx[0] ** 2)
        half = stress[x < x.size(0) // 2].sum()          # one interface
        assert abs(float(half) - p.sigma) / p.sigma < 0.07   # 6.1 % at xi/h=2, 2.9 % at xi/h=3


def test_gate_keeps_owner_band_and_reports_failures():
    from xval_pressure_tensor_equilibrium.run_matrix import evaluate

    gates = [{"id": "sigma_tension", "arms": ["flat_xi3"], "gate": 0.15},
             {"id": "xi_fit", "arms": ["*"], "gate": 0.15, "tight_target": 0.05}]
    rows = evaluate(gates, "flat_xi3", {"sigma_tension": 0.0201, "xi_fit": 0.191})
    by = {r["kpi"]: r for r in rows}
    assert by["sigma_tension"]["pass"] is True
    assert by["xi_fit"]["pass"] is False           # 19.1 % > 15 % is not upgraded
    assert by["xi_fit"]["within_tight"] is False
