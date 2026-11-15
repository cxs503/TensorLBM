"""Tests for the Shan-Chen Laplace pressure calibration (XVAL anchor).

Covers:
* sign/coefficient of the SC lattice pressure (verified against Maxwell),
* Maxwell coexistence reproduces the library's documented constants,
* the guarded radius estimator (the reference implementation's UB),
* Laplace fit machinery.
"""
import math

import pytest
import torch

from tensorlbm.d2q9 import CS2
from tensorlbm.multiphase import RHO_L_DEFAULT, RHO_V_DEFAULT, psi_exp
from tensorlbm.multiphase.laplace import (RadiusEstimationError, fit_laplace,
                                          interface_radius,
                                          laplace_pressure_calibration,
                                          sc_maxwell_coexistence, sc_pressure)

torch.set_default_dtype(torch.float64)


def _droplet(rl=RHO_L_DEFAULT, rv=RHO_V_DEFAULT, R=14.0, n=64, width=1.5):
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    d = torch.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) - R
    return rv + (rl - rv) * 0.5 * (1.0 - torch.tanh(d / width))


def test_pressure_formula_matches_definition():
    rho = torch.tensor([0.2, 1.0, 2.0], dtype=torch.float64)
    G = 5.0
    expected = CS2 * rho - 0.5 * G * CS2 * psi_exp(rho) ** 2
    assert torch.allclose(sc_pressure(rho, G), expected)


def test_pressure_sign_is_attractive():
    """Attraction must lower the pressure: p(rho_l) > p(rho_v) at coexistence."""
    p_l = float(sc_pressure(torch.tensor(RHO_L_DEFAULT), 5.0))
    p_v = float(sc_pressure(torch.tensor(RHO_V_DEFAULT), 5.0))
    assert p_l > p_v, "attractive interaction must give p_liquid > p_vapour"


def test_maxwell_reproduces_documented_coexistence():
    mw = sc_maxwell_coexistence(5.0)
    assert abs(mw["rho_l"] - RHO_L_DEFAULT) / RHO_L_DEFAULT < 0.15
    assert abs(mw["rho_v"] - RHO_V_DEFAULT) / RHO_V_DEFAULT < 0.15


def test_maxwell_reproduces_reference_cpp_densities():
    """Reference C++ (chapter9/shanchen.cpp, g=-4.7) measured 1.6707 / 0.18473."""
    mw = sc_maxwell_coexistence(4.7)
    assert abs(mw["rho_l"] - 1.6707) / 1.6707 < 0.15
    assert abs(mw["rho_v"] - 0.18473) / 0.18473 < 0.15


def test_interface_radius_recovers_synthetic_value():
    rho = _droplet(R=14.0, width=1.5)
    rho_av = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)
    r = interface_radius(rho, rho_av)
    assert abs(r - 14.0) < 1.0, f"radius estimator off by {r - 14.0:.2f} cells"


def test_interface_radius_raises_instead_of_undefined():
    """Upstream UB guard: no crossing must raise, never return garbage."""
    uniform = torch.full((32, 32), 1.0, dtype=torch.float64)
    with pytest.raises(RadiusEstimationError):
        interface_radius(uniform, 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT))


def test_calibration_on_synthetic_droplet():
    rho = _droplet(R=14.0, width=1.5)
    cal = laplace_pressure_calibration(rho, G=5.0, box=4)
    assert cal["dp"] > 0 and cal["gamma"] > 0
    assert cal["rho_l"] > cal["rho_v"]
    assert math.isfinite(cal["radius"])
    # Laplace: gamma = dp * radius, and dp must be far below the +G/6 value
    assert cal["gamma"] == pytest.approx(cal["dp"] * cal["radius"])
    assert cal["dp"] < 0.05, "sign error check: +G/6 would give dp ~ 1.2"


def test_fit_laplace_recovers_exact_sigma():
    sigma, radii = 0.05, [10.0, 12.0, 14.0, 18.0]
    dps = [sigma / r for r in radii]
    fit = fit_laplace(dps, radii)
    assert fit["sigma"] == pytest.approx(sigma, rel=1e-10)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-10)


def test_fit_laplace_requires_two_points():
    with pytest.raises(ValueError):
        fit_laplace([0.01], [12.0])
